# Previously skipped optimization techniques

## Outcome

Three researched techniques produced retained improvements:

1. shape-specific FP16 cuDNN SDPA dispatch for cases 9 and 13;
2. independent B64 candidate microbatching for case 6 while the strict reference
   retains B250;
3. a D128 Triton kernel that fuses FFN output projection, residual addition, and
   the following FP32 LayerNorm for cases 4, 5, 12, and 13.

Expanded D128 projection tiles were retained only for the 2,048-row family used
by cases 4 and 12. Configurations that regressed the compiled end-to-end model
were reverted even when they won isolated microbenchmarks.

Cases 1-13 pass the official correctness rule with zero failed elements. Case 14
remains an input-allocation OOM before either model executes.

## Technique 1 — FP16 SDPA backend dispatch

The earlier backend study covered FP32 only. On this PyTorch build, FP16 cuDNN
attention is eligible for most D128 shapes while automatic dispatch selects the
CUTLASS efficient-attention backend.

### Isolated attention results

| Case | Efficient SDPA | cuDNN SDPA | Efficient/cuDNN | Decision |
|---:|---:|---:|---:|:---|
| 1 | 0.078608 ms | 0.104928 ms | 0.749x | reject cuDNN |
| 4 | 0.040320 ms | 0.054608 ms | 0.738x | reject cuDNN |
| 5 | 0.136256 ms | 0.149392 ms | 0.912x | reject cuDNN |
| 9 | 0.093936 ms | 0.080016 ms | 1.174x | retain cuDNN |
| 10 | 0.081888 ms | 0.099904 ms | 0.820x | reject cuDNN |
| 11 | 0.197952 ms | 0.221600 ms | 0.893x | reject cuDNN |
| 12 | 0.090208 ms | 0.051312 ms | 1.758x | reject end to end |
| 13 | 1.854080 ms | 1.464096 ms | 1.266x | retain cuDNN |

All isolated cuDNN outputs had zero failed elements relative to efficient SDPA.
Case 12 was rejected after its compiled full model regressed from 0.2039 ms to
0.2635 ms. Case 9 passed three full-model trials at 0.6132 ms before the later
thermal-validation runs. Case 13 passed three trials at 12.4907 ms before deeper
fusion was added.

## Technique 2 — mixed case-6 microbatch retuning

The earlier B250 choice was tuned for the FP32 candidate. The mixed candidate
has a different memory and kernel-efficiency profile.

| Candidate microbatch | Candidate median |
|---:|---:|
| 125 | 161.4021 ms |
| 250 | 186.2925 ms |
| 500 | 209.6925 ms |
| 1000 | 221.9420 ms |

A broader candidate-only screen selected B64. The controlled validation passed
three trials with maximum absolute error 0.00193438 and zero failed elements out
of 491,520,000. Its candidate median was 150.4044 ms, versus 182.5907 ms in the
preceding official suite. The strict baseline keeps its established B250
microbatching; only the optimized execution schedule changed.

## Technique 3 — expanded D128 projection tuning

The tuner now covers the actual FP16 QKV output path, `BLOCK_K=64`, two-warp
tiles, and group size 4. Isolated winners did not reliably transfer to the
compiled Transformer. Broader schedules regressed or were noise-level for the
8,192-, 16,384-, and 65,536-row families and were reverted.

The 2,048-row family was retained. Controlled candidate medians improved to
0.1953 ms for case 4 before deeper fusion and 0.2015 ms for case 12. Both passed
three accuracy trials.

## Technique 4 — fused D128 FFN output, residual, and LayerNorm

The new kernel keeps the FP16 Tensor-Core projection accumulator in FP32, adds
the FP32 residual, calculates LayerNorm statistics in FP32, and writes both the
residual and normalized output. It removes the standalone FP32 FFN-output tensor
write/read and replaces two device kernels with one.

### Isolated two-kernel versus fused results

| Rows | Previous sequence | Best fused | Kernel-level gain |
|---:|---:|---:|---:|
| 2,048 | 0.056800 ms | 0.033936 ms | 1.674x |
| 8,192 | 0.066640 ms | 0.049792 ms | 1.338x |
| 16,384 | 0.119376 ms | 0.066096 ms | 1.806x |
| 65,536 | 0.631664 ms | 0.394496 ms | 1.601x |

The 8,192-row fusion regressed the compiled full model and is disabled. Retained
full-model shapes are cases 4, 5, 12, and 13. Three-trial controlled medians were
0.1880 ms, 1.4415 ms, 0.1930 ms, and 12.0212 ms respectively, all with zero
failed elements.

## Official full-suite protocol

```text
--accuracy-trials 1 --warmup 3 --repeats 10 --benchmark-rounds 1
```

| Case | Status | Max abs | Strict FP32 baseline ms | Mixed candidate ms | Speedup |
|---:|:---:|---:|---:|---:|---:|
| 1 | PASS | 0.0015206 | 3.3806 | 0.7166 | 4.717x |
| 2 | PASS | 0.00127742 | 1.9480 | 0.1202 | 16.205x |
| 3 | PASS | 0.00170898 | 1.9779 | 0.1524 | 12.983x |
| 4 | PASS | 0.00142139 | 1.6095 | 0.1893 | 8.501x |
| 5 | PASS | 0.001551 | 11.6970 | 1.2984 | 9.009x |
| 6 | PASS | 0.00193438 | 880.1353 | 150.5037 | 5.848x |
| 7 | PASS | 0.00171909 | 2.4881 | 0.3882 | 6.410x |
| 8 | PASS | 0.00119174 | 90.8256 | 24.0089 | 3.783x |
| 9 | PASS | 0.00120425 | 2.4916 | 0.6680 | 3.730x |
| 10 | PASS | 0.00120425 | 2.7428 | 0.6782 | 4.044x |
| 11 | PASS | 0.00120425 | 14.2696 | 1.3790 | 10.348x |
| 12 | PASS | 0.00128376 | 1.8112 | 0.1951 | 9.286x |
| 13 | PASS | 0.00118756 | 276.5473 | 15.6564 | 17.664x |
| 14 | OOM | n/a | n/a | n/a | n/a |

## Timing interpretation

The full suite uses one timing round and experienced substantial clock/thermal
variation late in the sequence. Cases 10 and 11 use their preceding unchanged
execution paths despite slower official values. Case 13 measured 12.0212 ms in
the controlled three-round validation but 15.6564 ms late in the full sweep;
its strict baseline simultaneously rose from the preceding 219.5173 ms to
276.5473 ms. The exact official script outputs are retained above, while the
controlled values are the better implementation comparison.

## Reproducibility

- Raw logs: `benchmark_logs/28_skipped_techniques_retained/`
- Parsed CSV: `benchmark_logs/28_skipped_techniques_retained/results.csv`
- Previous comparison: `benchmark_logs/27_mixed_d128_dispatch_results.md`
- Strict baseline SHA256: `E1925A209BBC2A536B0DE96870585D8422018107D0696D4EA1D196BC9A4BEE4A`
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, SM120
- PyTorch: 2.13.0+cu130
- CUDA runtime: 13.0
- Triton: 3.7.1

FP8/MXFP8 and SM120 CUTLASS/cuBLASLt kernels were not attempted in this pass.
They require a compiler/CUTLASS toolchain that is absent on this machine and
carry substantially higher numerical or engineering risk than the retained
techniques.
