# Current optimized benchmark results

## Configuration

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- PyTorch: 2.13.0+cu130
- CUDA runtime: 13.0
- Triton: 3.7.1 (`triton-windows` 3.7.1.post27)
- Dtype: float32
- Optimized model: `torch.compile(mode="reduce-overhead")` on all-valid CUDA
  inputs
- Correctness: absolute error <= 0.002 **or** relative error <= 2%
- Correctness trials: 3
- Timing: 5 warmups, 30 repeats, 2 alternating rounds
- Implementation SHA256:
  `101D8BB14B13A68B8C9516BAB4D2E77BF00222AFDA945A46778A37FBC88ED488`

## Results

| Case | B | D | H | S | Baseline (ms) | Optimized (ms) | Speedup | Max abs error | Result |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 64 | 128 | 4 | 128 | 3.3136 | 1.4488 | 2.287x | 0.00111634 | PASS |
| 2 | 1 | 128 | 4 | 128 | 1.9968 | 0.1226 | 16.286x | 0.00071144 | PASS |
| 3 | 4 | 128 | 4 | 128 | 1.9231 | 0.1668 | 11.529x | 0.00080031 | PASS |
| 4 | 16 | 128 | 4 | 128 | 1.9233 | 0.3719 | 5.171x | 0.00090277 | PASS |
| 5 | 128 | 128 | 4 | 128 | 8.9373 | 2.8784 | 3.105x | 0.00111634 | PASS |
| 6 | 10000 | 128 | 4 | 128 | — | — | — | — | NOT COMPLETED¹ |
| 7 | 64 | 32 | 4 | 128 | 2.2916 | 0.8141 | 2.815x | 0.00119746 | PASS |
| 8 | 64 | 1024 | 4 | 128 | 41.6244 | 34.4322 | 1.209x | 0.00115755 | PASS |
| 9 | 64 | 128 | 1 | 128 | 2.0995 | 1.2850 | 1.634x | 0.00110114 | PASS |
| 10 | 64 | 128 | 2 | 128 | 2.5364 | 1.2124 | 2.092x | 0.00102055 | PASS |
| 11 | 64 | 128 | 16 | 128 | 13.3369 | 2.8815 | 4.628x | 0.00110719 | PASS |
| 12 | 64 | 128 | 4 | 32 | 2.0656 | 0.3358 | 6.151x | 0.00104997 | PASS |
| 13 | 64 | 128 | 4 | 1024 | 202.1641 | 32.6212 | 6.197x | 0.00112352 | PASS |
| 14 | 32 | 1024 | 16 | 100000 | — | — | — | — | OOM² |

¹ Case 6 previously entered severe GPU paging and failed to complete after more
than 100 minutes. It was not repeated during incremental validation.

² Case 14's input alone requires about 12.21 GiB, exceeding this GPU's 8 GiB
before the model executes.

## Current optimization stack

1. All-valid mask fast path and cached padded causal mask.
2. PyTorch efficient scaled-dot-product attention.
3. Packed QKV projection.
4. Triton fused residual-add + LayerNorm.
5. Shape-aware LayerNorm warp dispatch.
6. TorchInductor reduce-overhead compilation and CUDA Graph replay for the
   all-valid CUDA path.

The baseline implementation and correctness contract remain unchanged.
