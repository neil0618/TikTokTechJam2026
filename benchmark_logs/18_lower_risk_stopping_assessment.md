# Lower-risk optimization stopping assessment

## Final compiled profile

| Operation family | Case 1 | Case 8 (D1024) | Case 13 (S1024) |
|---|---:|---:|---:|
| CUTLASS GEMMs | 45.8% | 77.9% | 14.3% |
| CUTLASS fused attention | 41.9% | 10.0% | 72.8% |
| Triton residual + LayerNorm | 8.4% | 9.1% | 9.5% |
| Compiled GELU pointwise | 1.8% | 1.8% | 2.3% |

Case 1's fused residual/LayerNorm work is approximately 110 microseconds total
for all eight calls after warp tuning. CUDA Graph replay has already reduced
host dispatch to one graph launch per forward.

## Lower-risk ideas exhausted

- Unnecessary masks and repeated invalid-query writes removed.
- Efficient attention and every available alternative backend benchmarked.
- QKV projections packed.
- Residual addition and LayerNorm fused and shape-tuned.
- CUDA Graph replay enabled wherever it is beneficial.
- Huge-batch paging replaced with exact microbatching.
- Compile modes benchmarked.
- TF32 and matmul precision settings benchmarked.
- fp16 and bf16 rejected under the unchanged correctness rule.
- Approximate GELU tested and reverted because the gain was insignificant.

## Why this is the stopping point

The remaining dominant time is inside already optimized CUTLASS Tensor-Core
GEMMs and fused attention. Configuration changes, extra pointwise fusion, or
launch reduction cannot materially change those percentages.

A further considerable improvement now requires at least one higher-risk
project:

1. a specialized GEMM epilogue that fuses FFN activation or output projection
   with residual/normalization;
2. a custom attention kernel specialized for the benchmark's long-sequence and
   head-dimension shapes; or
3. a broader persistent fused Transformer block.

These require custom kernel tiling, register/shared-memory management, and
substantial numerical validation. They are not extensions of the remaining
low-risk work.
