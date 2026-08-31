# Best Verified Implementation

This file records the best currently retained implementation, not the lowest
single timing sample. Routes are retained only after a same-process comparison
and correctness checks; unsupported shapes use the safe PyTorch fallback.

## Verification checkpoint

- Verification date: 2026-09-01
- Base checkpoint: `5db371c`
- Strict FP32 reference SHA256:
  `E1925A209BBC2A536B0DE96870585D8422018107D0696D4EA1D196BC9A4BEE4A`
- Full-suite results: `benchmark_logs/43_final_optimized_suite/results.csv`
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8,518,041,600 bytes
- Software: PyTorch 2.13.0+cu130, CUDA 13.0, Triton 3.7.1

The strict reference source and scoring rule were not modified. Compilation,
input generation, and accuracy checks are outside the CUDA-event timed region.

## Numerical policy

- Projection weights and Tensor-Core operands: FP16.
- GEMM accumulators: FP32.
- Attention online-softmax maximum, sum, and output accumulator: FP32.
- Residual stream and LayerNorm arithmetic: FP32.
- GELU: exact-erf form evaluated from FP32 values.
- Intermediate normalized activation storage: FP16 only for cases 4, 5, and
  7–13; it is FP32 for cases 1–3.
- Returned model output: FP32.

## Retained dispatch

| Cases | Retained route |
|:---|:---|
| 1 | D128 Triton projections, FP32 custom attention, both fusion paths, FP32 normalized stream |
| 2 | D128 Triton projections and both fusion paths, efficient SDPA, FP32 normalized stream |
| 3 | Small-batch D128 Triton projections and full FFN fusion, efficient SDPA, FP32 normalized stream |
| 4 | D128 Triton projections, FP32 custom attention, both fusion paths, FP16 normalized storage |
| 5 | D128 Triton projections, FP32 custom attention, attention-output fusion, FP16 normalized storage; full FFN fusion disabled |
| 6 | B64 compiled microbatches, preallocated output, custom FP16 causal attention |
| 7 | Specialized D32 projections and fused projection/residual/LayerNorm epilogues |
| 8 | Hand-tuned D1024 Triton projections and FP16 normalized storage |
| 9–11 | D128 projections, custom FP16 causal attention, both fusion paths, FP16 normalized storage |
| 12 | D128 projections, both fusion paths, FP16 normalized storage |
| 13 | D128 projections, long-sequence custom FP16 causal attention, both fusion paths, FP16 normalized storage |
| 14 | Cannot allocate the official nominal input on the 8 GB development GPU |

The custom FP16 attention kernel uses shape-specific tiles: head dimension 8
uses 64x32, head dimension 64 uses 32x32, head dimension 128 uses 64x32, and
the long-sequence head-dimension-32 route uses 128x32. All use four warps and
two stages.

## Final official results

| Case | Strict FP32 | Candidate | Speedup | Max abs. error | Failed | Status |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 4.6147 ms | 0.6424 ms | 7.184x | 0.00152054 | 0 / 1,048,576 | PASS |
| 2 | 2.4440 ms | 0.1252 ms | 19.515x | 0.000883877 | 0 / 16,384 | PASS |
| 3 | 2.3736 ms | 0.1285 ms | 18.476x | 0.00107079 | 0 / 65,536 | PASS |
| 4 | 1.9381 ms | 0.1692 ms | 11.454x | 0.00142145 | 0 / 262,144 | PASS |
| 5 | 12.5268 ms | 1.1638 ms | 10.764x | 0.00155100 | 0 / 2,097,152 | PASS |
| 6 | 982.7582 ms | 137.6488 ms | 7.140x | 0.00193438 | 0 / 163,840,000 | PASS |
| 7 | 2.9817 ms | 0.2895 ms | 10.298x | 0.00131706 | 0 / 262,144 | PASS |
| 8 | 99.1962 ms | 23.3075 ms | 4.256x | 0.00119174 | 0 / 8,388,608 | PASS |
| 9 | 3.3562 ms | 0.4924 ms | 6.816x | 0.00120437 | 0 / 1,048,576 | PASS |
| 10 | 3.5996 ms | 0.5122 ms | 7.028x | 0.00120437 | 0 / 1,048,576 | PASS |
| 11 | 17.0789 ms | 0.6353 ms | 26.882x | 0.00120437 | 0 / 1,048,576 | PASS |
| 12 | 2.5019 ms | 0.1701 ms | 14.712x | 0.00128353 | 0 / 262,144 | PASS |
| 13 | 262.4509 ms | 8.9474 ms | 29.333x | 0.00118732 | 0 / 8,388,608 | PASS |
| 14 | n/a | n/a | n/a | n/a | n/a | OOM |

The official suite uses one accuracy trial, three warmups, ten repeats, and one
round. A separate high-confidence case-13 run used three accuracy trials, ten
warmups, thirty repeats, and three rounds: 8.2941 ms candidate versus 246.5003
ms strict FP32 (29.720x), with zero failed values out of 25,165,824.

The same high-confidence protocol also validated the two other newest
architecture-specific routes. Case 11 measured 0.6477 ms, passed 0 / 3,145,728
failed values, and had maximum absolute error 0.00122881. Case 7 measured
0.2907 ms, passed 0 / 786,432 failed values, and had maximum absolute error
0.00136477.

## Reproduce

```powershell
powershell -ExecutionPolicy Bypass -File .\run_mixed_suite.ps1 `
  -RunId reproduction_run
```

Close GPU-heavy applications first. Laptop clock and thermal state can move
absolute timings, so use same-process controlled experiments for decisions near
the noise floor.

## Remaining constraints and next research

Case 8 remains projection-GEMM dominated (75.82% of profiled GPU time), and the
nearby Triton tile sweep did not beat the current route. Meaningful progress
would require a deeper SM120-specific GEMM schedule or lower-level CUDA work.
Case 6 still pays 157 launches per microbatch forward and 40 microbatch calls;
more whole-block fusion could help, but would be considerably higher risk.
Case 14 cannot be solved by attention tiling alone because its nominal input
itself is about 13.1 GB in FP32; passing it requires changing the input contract
to streamed/chunked generation or running on a larger-memory GPU.
