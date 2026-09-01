# Current mixed-candidate GPU profile

## Scope and method

This profile covers the production mixed implementation after the case-11
schedule tournament. No production source was changed. The optimized model was
compiled with `torch.compile(mode="reduce-overhead")`, warmed up before capture,
and profiled for one steady-state forward with PyTorch/Kineto. The profiled
production snapshot was commit `5db371cca121bd8cb7ed84a8930e6223f86359f5`;
later concurrent experiments in the shared worktree are not part of these data.

Commands:

```text
.\.venv\Scripts\python.exe analysis_profile_mixed.py --case 6 --warmup 2 --iterations 1 --top 40
.\.venv\Scripts\python.exe analysis_profile_mixed.py --case 8 --warmup 5 --iterations 1 --top 30
.\.venv\Scripts\python.exe analysis_profile_mixed.py --case 10 --warmup 5 --iterations 1 --top 25
.\.venv\Scripts\python.exe analysis_profile_mixed.py --case 11 --warmup 5 --iterations 1 --top 25
.\.venv\Scripts\python.exe analysis_profile_mixed.py --case 13 --warmup 5 --iterations 1 --top 35
```

Kineto emits a synthetic `Call CompiledFxGraph` device record that duplicates
the enclosed GPU duration. It is excluded from the launch counts and percentages
below. Profiling perturbs absolute latency, especially for the 7,850-launch case
6, so CUDA-event benchmark results remain the source of truth for runtime.

## Kernel-time breakdown

| Case | Real launches | Dominant kernel families (share of self CUDA time) |
|---:|---:|:---|
| 6 | 7,850 | CUTLASS FP16 GEMMs 41.65%; CUTLASS efficient attention 24.78%; residual/LayerNorm 18.97%; conversion/layout/GELU kernels 11.53%; graph-input and output copies 2.29%; final LayerNorm 0.77% |
| 8 | 30 | custom mixed projections 75.82%; residual/LayerNorm 14.56%; CUTLASS efficient attention 8.13%; final LayerNorm 0.76%; CUDA-Graph input copy 0.73% |
| 10 | 30 | custom mixed projections 55.72%; CUTLASS efficient attention 21.64%; residual/LayerNorm 18.15%; CUDA-Graph input copy 3.19%; final LayerNorm 1.29% |
| 11 | 30 | CUTLASS efficient attention 61.28%; custom mixed projections 26.89%; residual/LayerNorm 9.50%; CUDA-Graph input copy 1.59%; final LayerNorm 0.75% |
| 13 | 26 | cuDNN flash SDPA 46.32%; custom mixed projections 20.78%; post-attention residual/LayerNorm 15.94%; fused FFN-output/residual/LayerNorm 13.85%; final LayerNorm 1.65%; CUDA-Graph input copy 1.46% |

Per-forward launch structure for cases 8, 10, and 11 is 16 projection kernels,
four attention kernels, eight residual/LayerNorm kernels, one final LayerNorm,
and one CUDA-Graph input copy. Case 13 uses 12 standalone projections, four
cuDNN attention kernels, four post-attention residual/LayerNorm kernels, four
fused FFN-output/residual/LayerNorm kernels, one final LayerNorm, and one graph
input copy.

Case 6 executes 157 microbatches. Each microbatch has 16 vendor GEMMs, four
attention kernels, eight residual/LayerNorm kernels, 19 conversion/layout/GELU
kernels, one final LayerNorm, one CUDA-Graph input copy, and one copy into the
preallocated full output: 50 launches per chunk and 7,850 per full forward.
There is no host synchronization inside the loop; the large launch count and
repeated global-memory passes are nevertheless real.

## Workload diagnosis

- Case 8 is projection/compute dominated. Its D1024 GEMMs account for over
  three quarters of runtime and execute about 412 GFLOP per forward. The current
  case-8 tuner covers only six tile combinations, so projection schedule search
  is not exhausted. Residual/LayerNorm is the secondary, memory-bound component;
  its eight calls move roughly 1 GiB before cache effects.
- Case 11 is attention-efficiency dominated. Cases 10 and 11 perform the same
  nominal attention FLOPs, but changing from two heads of width 64 to sixteen
  heads of width 8 raises measured four-layer attention time from 0.129 ms to
  0.667 ms in the profiler. The selected PyTorch backend is already fused CUTLASS
  efficient attention, but the tiny head dimension uses the GPU poorly.
- Case 13 is split between cuDNN flash attention and memory traffic. cuDNN is
  already the fastest tested PyTorch backend for this exact shape. The remaining
  obvious low-risk gap is the standalone attention-output projection followed
  by residual/LayerNorm; the analogous FFN sequence is already fused.
- Case 6 combines all costs at scale. Vendor GEMMs and attention are 66.4% of
  GPU time, while normalization and conversion/layout kernels are another 30.5%.
  Profiling inflated its absolute duration from the normal 144-150 ms range to
  419 ms, so only its percentages and exact launch inventory should guide work.

## Best-known comparable current runtimes

| Case | Strongest controlled current-route evidence | Best official one-round result | Latest official result |
|---:|---:|---:|---:|
| 6 | 150.4044 ms (B64 candidate microbatch validation) | 144.0220 ms | 144.2230 ms |
| 8 | 19.7395 ms (three-round custom-projection validation) | 22.6195 ms | 23.2569 ms |
| 10 | 0.6485 ms (three-round D128 validation) | 0.5714 ms | 0.7027 ms |
| 11 | 1.1569 ms (three-round retained-route validation) | 1.2044 ms on an older schedule; 1.2453 ms for the current schedule | 1.4400 ms |
| 13 | 12.0212 ms (three-round cuDNN + FFN-fusion validation) | 11.5377 ms | 11.8053 ms |

The controlled and official columns are not interchangeable. The controlled
runs use more warmups, repeats, and rounds and are preferred for route choice;
official one-round results are retained because they are the exact benchmark
outputs and reveal the expected thermal/clock range.

## Ranked experiments

1. **Broaden case-8 projection autotuning and compare cuBLASLt/CUTLASS.** Search
   operation-specific `BLOCK_M/N/K`, group size, warps, and stages rather than
   sharing the six existing schedules. This attacks 75.82% of case-8 time and
   should remain on FP16 Tensor Cores with FP32 accumulation. A 15% projection
   improvement would improve the whole case by about 12%.
2. **Build a head-dimension-8 attention candidate for case 11.** Pack or process
   multiple heads per CTA so H16/Dh8 supplies enough work, while retaining online
   softmax and FP32 statistics. Attention is 61.28% of current time; a 1.5x
   attention gain would improve the full case by about 26%. Exact-shape dispatch
   limits correctness and regression risk.
3. **Prototype a case-6 microbatch-specialized graph progressively.** Construct
   the core for its actual B64 slice and replace safe vendor sequences one at a
   time, beginning with projection epilogues/conversions and only then attention.
   The goal is to remove some of the 19 conversion/layout kernels per chunk and
   reduce 7,850 launches without repeating the previously rejected all-at-once
   D128 route, which missed correctness by one value. Validate on all 163,840,000
   outputs before retention.

The next lower-risk experiment is to fuse case 13's attention-output D128
projection with residual addition and `norm2`, mirroring the retained FFN fusion.
It removes four kernels and one FP32 intermediate round trip per forward and
targets roughly the 16% post-attention normalization component plus part of the
21% projection component.

## Environment

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, SM120, 26 SMs, 8,123 MiB
- L2 cache reported by PyTorch: 32 MiB
- PyTorch: 2.13.0+cu130
- CUDA runtime: 13.0
- Triton: 3.7.1
- Nsight Systems / Nsight Compute: not installed; no hardware-counter claims
  are made for achieved occupancy, DRAM bandwidth, or Tensor Core utilization.
