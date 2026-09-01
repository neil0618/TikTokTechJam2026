# Current results after lower-risk optimization phase

## Configuration

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8 GiB
- PyTorch: 2.13.0+cu130
- CUDA runtime: 13.0
- Triton: 3.7.1 (`triton-windows` 3.7.1.post27)
- Dtype: float32 with TF32 enabled
- Compilation: `reduce-overhead` for ordinary CUDA shapes
- Correctness: absolute error <= 0.002 **or** relative error <= 2%
- Implementation SHA256:
  `E1925A209BBC2A536B0DE96870585D8422018107D0696D4EA1D196BC9A4BEE4A`

## Official cases

| Case | B | D | H | S | Baseline (ms) | Optimized (ms) | Speedup | Max abs error | Result |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 64 | 128 | 4 | 128 | 3.1794 | 1.4285 | 2.226x | 0.00112993 | PASS |
| 2 | 1 | 128 | 4 | 128 | 1.9968 | 0.1226 | 16.286x | 0.00071144 | PASS |
| 3 | 4 | 128 | 4 | 128 | 1.9231 | 0.1668 | 11.529x | 0.00080031 | PASS |
| 4 | 16 | 128 | 4 | 128 | 1.9233 | 0.3719 | 5.171x | 0.00090277 | PASS |
| 5 | 128 | 128 | 4 | 128 | 8.9373 | 2.8784 | 3.105x | 0.00111634 | PASS |
| 6 | 10000 | 128 | 4 | 128 | 741.2646 | 265.1999 | 2.795x | 0.00117791 | PASS¹ |
| 7 | 64 | 32 | 4 | 128 | 2.2916 | 0.8141 | 2.815x | 0.00119746 | PASS |
| 8 | 64 | 1024 | 4 | 128 | 42.4865 | 34.6794 | 1.225x | 0.00115755 | PASS |
| 9 | 64 | 128 | 1 | 128 | 2.0995 | 1.2850 | 1.634x | 0.00110114 | PASS |
| 10 | 64 | 128 | 2 | 128 | 2.5364 | 1.2124 | 2.092x | 0.00102055 | PASS |
| 11 | 64 | 128 | 16 | 128 | 13.3369 | 2.8815 | 4.628x | 0.00110719 | PASS |
| 12 | 64 | 128 | 4 | 32 | 2.0656 | 0.3358 | 6.151x | 0.00104997 | PASS |
| 13 | 64 | 128 | 4 | 1024 | 202.2207 | 32.9443 | 6.138x | 0.00112352 | PASS |
| 14 | 32 | 1024 | 16 | 100000 | — | — | — | — | OOM² |

¹ Case 6 uses exact batch-dimension microbatching. Its validation used one
full correctness trial and three timed repeats because every output contains
163,840,000 values.

² Case 14 cannot allocate its 12.21 GiB input on an 8 GiB GPU. Its attention
work is also quadratic in the 100,000-token sequence length, so merely
streaming the input would not make the supplied computation practical.

## Additional padded validation

| Shape | Padding | Baseline | Optimized | Speedup | Result |
|---|---:|---:|---:|---:|:---:|
| B64 D128 H4 S128 L4 | 25% | 3.1196 ms | 1.4340 ms | 2.175x | PASS |
| B16 D128 H4 S1024 L4 | 25% | 48.6172 ms | 8.7551 ms | 5.553x | PASS |

## Verification

- All runnable official cases pass with zero failed elements.
- Padded and unpadded paths pass.
- CPU and explicit eager fallbacks passed earlier regression checks.
- Python bytecode compilation passes.
- `git diff --check` passes.

