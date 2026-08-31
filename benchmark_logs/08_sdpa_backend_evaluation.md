# Stage 7 — SDPA backend evaluation (no dispatch change adopted)

## Goal

Test whether shape-specific SDPA backend dispatch can improve the optimized
attention path for normal, unusual-head-count, and long-sequence cases.

The microbenchmark used the exact Q/K/V shape and stride produced by the packed
QKV projection:

- logical shape `[B, H, S, head_dim]`
- case 1 stride `(49152, 32, 384, 1)`
- case 13 stride `(393216, 32, 384, 1)`

## Backend availability

| Backend | Float32 status |
|---|---|
| Efficient attention / CUTLASS | Available |
| cuDNN attention | Unavailable; this build requires fp16 or bf16 |
| FlashAttention | Unavailable; PyTorch was not compiled with it |
| Math SDPA | Available, but substantially slower |

## Attention-only median latency

| Case | Shape summary | Efficient (ms) | Math (ms) | Efficient advantage |
|---:|---|---:|---:|---:|
| 1 | B64 H4 S128 Dh32 | 0.1540 | 0.5450 | 3.54x |
| 9 | B64 H1 S128 Dh128 | 0.1240 | 0.2296 | 1.85x |
| 11 | B64 H16 S128 Dh8 | 0.5582 | 2.3559 | 4.22x |
| 13 | B64 H4 S1024 Dh32 | 6.0940 | 33.9558 | 5.57x |

## Decision

Keep PyTorch's current efficient-attention path. There is no second optimized
float32 backend available to dispatch to, and forcing the math backend regresses
every tested shape. Mixed precision is still excluded because Stage 4 showed it
violates the unchanged correctness contract.

No implementation change was made in this stage.
