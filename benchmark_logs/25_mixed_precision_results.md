# Mixed FP16/FP32 candidate with strict-FP32 reference

## Outcome

The separate mixed-precision candidate passes cases 1-13 with zero failed output elements. Case 14 still fails before either model executes because the FP32 input alone requires about 12.21 GiB on an 8 GiB GPU.

The original baseline implementation was not modified. Its SHA256 remains:

```text
E1925A209BBC2A536B0DE96870585D8422018107D0696D4EA1D196BC9A4BEE4A
```

## Precision policy

### Strict reference

- Input, weights, activations, reductions, and output: FP32
- `torch.set_float32_matmul_precision("highest")`
- `torch.backends.cuda.matmul.allow_tf32 = False`
- `torch.backends.cudnn.allow_tf32 = False`

### Mixed candidate

- Persistent residual stream: FP32
- LayerNorm parameters, means, variances, and outputs: FP32
- Exact GELU evaluation: FP32
- Attention softmax statistics: FP32 inside fused SDPA
- Projection weights and Tensor-Core operands: FP16
- Projection accumulation: FP32
- Output returned to the benchmark: FP32
- Case 8: custom Triton mixed projection kernels

For case 8, the model contains 25,190,400 FP16 projection parameters and 18,432 FP32 normalization parameters.

## Custom case-8 Triton kernel

The exact case-8 shape uses custom projection kernels for:

- packed QKV: FP32 activations converted to FP16 on-chip, FP16 weights, FP32 accumulation, FP16 output for attention;
- attention output projection: FP16 context/weights, FP32 accumulation and output;
- FFN input projection: FP32 activations converted on-chip, FP16 weights, FP32 accumulation, exact FP32 GELU epilogue, FP16 output;
- FFN output projection: FP16 activation/weights, FP32 accumulation and output.

Selected tiles use `BLOCK_K=32`, four warps, and three stages. QKV and fused FFN-input use `BLOCK_M=64, BLOCK_N=128`; FP32-output projections use `BLOCK_M=128, BLOCK_N=64`.

### Isolated case-8 projection results

| Operation | PyTorch mixed path | Custom Triton | Incremental speedup | Correctness |
|---|---:|---:|---:|:---:|
| Packed QKV | 1.849120 ms | 1.690688 ms | 1.094x | PASS |
| FFN input + exact GELU | 1.149888 ms | 0.615440 ms | 1.869x | PASS |
| Projection with FP32 output | 0.722640 ms | 0.560672 ms | 1.289x | PASS |

The full case-8 mixed model improved from 25.7359 ms with vendor projections to 19.7395 ms with the custom Triton path in the controlled three-round run, an incremental 1.304x. Three accuracy trials passed with maximum absolute error 0.00120211 and zero failed elements out of 25,165,824.

## Official full-suite protocol

```text
--accuracy-trials 1 --warmup 3 --repeats 10 --benchmark-rounds 1
```

Every baseline value below is strict IEEE FP32. Every candidate output is FP32.

| Case | B | D | H | S | L | F | Status | Max abs | Strict FP32 baseline ms | Mixed ms | Speedup |
|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| 1 | 64 | 128 | 4 | 128 | 4 | 128 | PASS | 0.00146234 | 3.3261 | 0.8478 | 3.923x |
| 2 | 1 | 128 | 4 | 128 | 4 | 128 | PASS | 0.00127742 | 4.4513 | 0.1470 | 30.289x |
| 3 | 4 | 128 | 4 | 128 | 4 | 128 | PASS | 0.00170898 | 2.1089 | 0.1793 | 11.764x |
| 4 | 16 | 128 | 4 | 128 | 4 | 128 | PASS | 0.00146234 | 1.7294 | 0.2274 | 7.606x |
| 5 | 128 | 128 | 4 | 128 | 4 | 128 | PASS | 0.00147408 | 10.1363 | 1.9128 | 5.299x |
| 6 | 10000 | 128 | 4 | 128 | 4 | 128 | PASS | 0.00193438 | 837.7858 | 183.9077 | 4.555x |
| 7 | 64 | 32 | 4 | 128 | 4 | 32 | PASS | 0.00171909 | 2.6691 | 0.4017 | 6.644x |
| 8 | 64 | 1024 | 4 | 128 | 4 | 1024 | PASS | 0.00119174 | 83.1712 | 22.6195 | 3.677x |
| 9 | 64 | 128 | 1 | 128 | 4 | 128 | PASS | 0.00128931 | 2.6158 | 0.8160 | 3.206x |
| 10 | 64 | 128 | 2 | 128 | 4 | 128 | PASS | 0.00134480 | 2.7174 | 0.7928 | 3.428x |
| 11 | 64 | 128 | 16 | 128 | 4 | 128 | PASS | 0.00128931 | 14.4268 | 1.3338 | 10.816x |
| 12 | 64 | 128 | 4 | 32 | 4 | 128 | PASS | 0.00147390 | 2.4265 | 0.2262 | 10.726x |
| 13 | 64 | 128 | 4 | 1024 | 4 | 128 | PASS | 0.00146246 | 209.8313 | 16.5254 | 12.698x |
| 14 | 32 | 1024 | 16 | 100000 | 2 | 1024 | OOM | n/a | n/a | n/a | n/a |

## Reproducibility

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, SM120
- PyTorch: 2.13.0+cu130
- CUDA runtime: 13.0
- Triton: 3.7.1
- Raw logs: `benchmark_logs/24_mixed_fp16_fp32_strict_baseline/`
- Parsed CSV: `benchmark_logs/24_mixed_fp16_fp32_strict_baseline/results.csv`
- Runner: `mixed_precision_benchmark.py`
- Suite script: `run_mixed_suite.ps1`
- Mixed model: `mixed_precision/transformer.py`
- Custom projections: `mixed_precision/triton_linear.py`
- Isolated tuner: `mixed_precision/tune_linear.py`

## Important comparison note

These speedups use the user-requested strict-FP32 baseline. They must not be compared directly with older tables whose baseline allowed TF32. The custom case-8 kernel's incremental comparison against the vendor mixed path is the relevant measure of kernel-specific improvement.
