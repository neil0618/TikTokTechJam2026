# Final optimized benchmark results

## Test contract

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8,151 MiB, compute capability 12.0)
- PyTorch: 2.13.0+cu130
- CUDA runtime: 13.0
- CUDA driver UMD: 13.4 (driver 616.56)
- Triton module: 3.7.1 (`triton-windows` 3.7.1.post27)
- Implementation SHA256: `1A99AA9C10BDEB07CB45E71F01D0C3E7DC0CA1671C34336B68543BD465497819`
- Dtype: float32
- Correctness: absolute error <= 0.002 **or** relative error <= 2%
- All recorded feasible cases: 3 correctness trials
- Cases 1, 8, 11, 13: 5 warmups, 30 repeats, 2 rounds
- Other feasible cases: 3 warmups, 10 repeats, 1 round

## Results

| Case | B | D | H | S | L | F | Baseline (ms) | Optimized (ms) | Speedup | Max abs error | Result |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 64 | 128 | 4 | 128 | 4 | 128 | 3.3807 | 1.7922 | 1.886x | 0.00111634 | PASS |
| 2 | 1 | 128 | 4 | 128 | 4 | 128 | 2.0165 | 0.8065 | 2.500x | 0.00071144 | PASS |
| 3 | 4 | 128 | 4 | 128 | 4 | 128 | 1.9362 | 0.8045 | 2.407x | 0.00082052 | PASS |
| 4 | 16 | 128 | 4 | 128 | 4 | 128 | 1.9848 | 0.7444 | 2.666x | 0.00090277 | PASS |
| 5 | 128 | 128 | 4 | 128 | 4 | 128 | 10.4376 | 3.2374 | 3.224x | 0.00111634 | PASS |
| 6 | 10000 | 128 | 4 | 128 | 4 | 128 | — | — | — | — | NOT COMPLETED¹ |
| 7 | 64 | 32 | 4 | 128 | 4 | 32 | 2.2974 | 0.9728 | 2.362x | 0.00119746 | PASS |
| 8 | 64 | 1024 | 4 | 128 | 4 | 1024 | 42.5285 | 34.3235 | 1.239x | 0.00115755 | PASS |
| 9 | 64 | 128 | 1 | 128 | 4 | 128 | 2.0994 | 1.6037 | 1.309x | 0.00110120 | PASS |
| 10 | 64 | 128 | 2 | 128 | 4 | 128 | 2.5878 | 1.5552 | 1.664x | 0.00102055 | PASS |
| 11 | 64 | 128 | 16 | 128 | 4 | 128 | 13.5892 | 3.5051 | 3.877x | 0.00110716 | PASS |
| 12 | 64 | 128 | 4 | 32 | 4 | 128 | 2.3949 | 0.7267 | 3.296x | 0.00099731 | PASS |
| 13 | 64 | 128 | 4 | 1024 | 4 | 128 | 201.9509 | 32.9384 | 6.131x | 0.00112346 | PASS |
| 14 | 32 | 1024 | 16 | 100000 | 2 | 1024 | — | — | — | — | OOM² |

¹ Case 6 drove the 8 GiB GPU into paging and did not finish after more than
100 minutes during the original baseline attempt, so it was not repeated after
each optimization.

² Case 14 cannot allocate its input: `[32, 100000, 1024]` float32 alone is
about 12.21 GiB, before model weights or intermediates.

## Optimization stack

1. Skip unnecessary all-valid padding-mask work and cache causal masks for the
   padded fallback.
2. Replace explicit score materialization, masking, softmax, and value matmul
   with PyTorch scaled-dot-product attention (CUTLASS FMHA on this GPU).
3. Pack Q/K/V into one projection.
4. Keep float32 after float16 and bfloat16 failed the unchanged correctness
   contract.
5. Fuse residual additions with their consuming LayerNorm using Triton.

## Overall conclusion

Every feasible official case passes with zero failed output elements. Final
speedups range from 1.239x for the compute-heavy D=1024 case to 6.131x for the
long-sequence attention-heavy case. The remaining dominant work is GEMM and
fused attention rather than mask construction, score materialization, or
standalone normalization traffic.
