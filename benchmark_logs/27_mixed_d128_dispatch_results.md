# Shape-dispatched D128 mixed-precision kernels

## Outcome

The mixed candidate now uses tuned Triton projection kernels for selected D128
benchmark shapes. Cases 1-13 pass the official correctness rule with zero failed
elements. Case 14 remains an expected input-allocation OOM.

The strict reference remains FP32 throughout with TF32 disabled. This change did
not modify the original baseline implementation.

## What changed

- Added an isolated D128 projection tuner: `mixed_precision/tune_linear_d128.py`.
- Added shape-dispatched Triton packed-QKV, output, fused FFN-input-plus-exact-GELU,
  and FFN-output projections for cases 1, 4, 5, 9, 10, 11, 12, and 13.
- Reused the existing custom FP32 causal-attention kernel for cases 1, 4, 5, and
  13, where the D128/head-4 shape is supported.
- Kept PyTorch fused SDPA for cases 9-12; only their projections use the new
  Triton dispatch.
- Kept cases 2, 3, 6, and 7 on the conservative vendor mixed path.
- Kept the existing D1024 case-8 custom projection path unchanged.

The D128 kernels use FP16 weights and Tensor-Core operands with FP32
accumulation. Residuals, normalization, exact GELU evaluation, sensitive
attention work, and the returned output stay FP32 according to the established
mixed-precision policy.

## Rejected D128 case-6 route

The custom D128 path was tested on case 6 but rejected. One of three randomized
accuracy trials produced one failed value out of 163,840,000, with maximum
absolute error 0.00201623. Restoring case 6 to the vendor mixed path passed three
trials with zero failures out of 491,520,000 and maximum absolute error
0.00193438. The retained case-6 candidate median was 182.5247 ms in that
controlled validation.

## Controlled validation

Selected retained D128 paths were checked with three randomized accuracy trials
and a longer timing protocol (`10` warmups, `30` repeats, `3` rounds):

| Case | Candidate median | Max abs across 3 trials | Failed values | Decision |
|---:|---:|---:|---:|:---:|
| 1 | 0.6965 ms | 0.001553 | 0 / 3,145,728 | RETAIN |
| 4 | 0.2024 ms | 0.00154722 | 0 / 786,432 | RETAIN |
| 5 | 1.5224 ms | 0.001553 | 0 / 6,291,456 | RETAIN |
| 9 | 0.7105 ms | 0.00123921 | 0 / 3,145,728 | RETAIN |
| 10 | 0.6485 ms | 0.00137335 | 0 / 3,145,728 | RETAIN |
| 11 | 1.1438 ms | 0.00122869 | 0 / 3,145,728 | RETAIN |
| 12 | 0.2039 ms | 0.00128667 | 0 / 786,432 | RETAIN |
| 13 | 12.7930 ms | 0.00173271 | 0 / 25,165,824 | RETAIN |

Case 8's unchanged custom Triton path previously measured 19.7395 ms in the
controlled three-round run, with maximum absolute error 0.00120211 and zero
failed values out of 25,165,824.

## Official full-suite protocol

```text
--accuracy-trials 1 --warmup 3 --repeats 10 --benchmark-rounds 1
```

| Case | B | D | H | S | L | F | Status | Max abs | Strict FP32 baseline ms | Mixed candidate ms | Speedup |
|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| 1 | 64 | 128 | 4 | 128 | 4 | 128 | PASS | 0.0015206 | 3.3342 | 0.7036 | 4.739x |
| 2 | 1 | 128 | 4 | 128 | 4 | 128 | PASS | 0.00127742 | 1.8424 | 0.1304 | 14.125x |
| 3 | 4 | 128 | 4 | 128 | 4 | 128 | PASS | 0.00170898 | 1.8467 | 0.1522 | 12.135x |
| 4 | 16 | 128 | 4 | 128 | 4 | 128 | PASS | 0.00142139 | 1.6929 | 0.2025 | 8.359x |
| 5 | 128 | 128 | 4 | 128 | 4 | 128 | PASS | 0.001551 | 10.4372 | 1.4803 | 7.051x |
| 6 | 10000 | 128 | 4 | 128 | 4 | 128 | PASS | 0.00193438 | 837.8748 | 182.5907 | 4.589x |
| 7 | 64 | 32 | 4 | 128 | 4 | 32 | PASS | 0.00171909 | 2.5029 | 0.3685 | 6.793x |
| 8 | 64 | 1024 | 4 | 128 | 4 | 1024 | PASS | 0.00119174 | 78.7416 | 25.4779 | 3.091x |
| 9 | 64 | 128 | 1 | 128 | 4 | 128 | PASS | 0.00120425 | 2.4444 | 0.6908 | 3.538x |
| 10 | 64 | 128 | 2 | 128 | 4 | 128 | PASS | 0.00120425 | 2.8232 | 0.5807 | 4.862x |
| 11 | 64 | 128 | 16 | 128 | 4 | 128 | PASS | 0.00120425 | 14.4284 | 1.2044 | 11.980x |
| 12 | 64 | 128 | 4 | 32 | 4 | 128 | PASS | 0.00128376 | 2.0548 | 0.2046 | 10.045x |
| 13 | 64 | 128 | 4 | 1024 | 4 | 128 | PASS | 0.00145179 | 219.5173 | 12.9883 | 16.901x |
| 14 | 32 | 1024 | 16 | 100000 | 2 | 1024 | OOM | n/a | n/a | n/a | n/a |

## Comparison with the preceding mixed suite

This comparison uses two independent single-round runs, so small changes on
unchanged paths are timing noise rather than implementation gains.

| Case | Previous mixed ms | New mixed ms | Ratio | Interpretation |
|---:|---:|---:|---:|:---|
| 1 | 0.8478 | 0.7036 | 1.205x | dispatched D128 gain |
| 2 | 0.1470 | 0.1304 | 1.127x | unchanged path; run variance |
| 3 | 0.1793 | 0.1522 | 1.178x | unchanged path; run variance |
| 4 | 0.2274 | 0.2025 | 1.123x | dispatched D128 gain |
| 5 | 1.9128 | 1.4803 | 1.292x | dispatched D128 gain |
| 6 | 183.9077 | 182.5907 | 1.007x | unchanged path; effectively flat |
| 7 | 0.4017 | 0.3685 | 1.090x | unchanged path; run variance |
| 8 | 22.6195 | 25.4779 | 0.888x | unchanged path; noisy single round |
| 9 | 0.8160 | 0.6908 | 1.181x | dispatched D128 gain |
| 10 | 0.7928 | 0.5807 | 1.365x | dispatched D128 gain |
| 11 | 1.3338 | 1.2044 | 1.107x | dispatched D128 gain |
| 12 | 0.2262 | 0.2046 | 1.106x | dispatched D128 gain |
| 13 | 16.5254 | 12.9883 | 1.272x | dispatched D128 gain |

## Reproducibility

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, SM120
- PyTorch: 2.13.0+cu130
- CUDA runtime: 13.0
- Triton: 3.7.1
- Raw full-suite logs: `benchmark_logs/26_mixed_d128_shape_dispatch/`
- Parsed CSV: `benchmark_logs/26_mixed_d128_shape_dispatch/results.csv`
- Complete mixed report: `benchmark_logs/27_mixed_d128_dispatch_results.md`
- Runner: `mixed_precision_benchmark.py`
- Suite script: `run_mixed_suite.ps1`
- D128 tuner: `mixed_precision/tune_linear_d128.py`

The official case-8 value is retained exactly as emitted by the script. Its
controlled three-round result is the better estimate of steady-state performance
because the one-round protocol is sensitive to clock and thermal variance.
