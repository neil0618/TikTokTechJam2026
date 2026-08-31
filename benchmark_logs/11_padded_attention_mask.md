# Stage 9 — Fused causal + padding-mask attention

## Change

The padded attention path now passes the broadcast key-padding mask directly to
scaled-dot-product attention while retaining `is_causal=True`.

Previously it materialized a combined boolean mask with shape `[B, 1, S, S]`
and disabled the causal hint. The new form is exactly equivalent on this
PyTorch build and keeps the padding mask at broadcast shape `[B, 1, 1, S]`.

## Attention-only comparison

Both tests used 25% padding and the exact packed-QKV strides.

| Shape | Materialized combined mask | Direct key mask + causal hint | Improvement | Avoided mask |
|---|---:|---:|---:|---:|
| B64 H4 S128 Dh32 | 0.2547 ms | 0.1709 ms | 1.49x | 1 MiB/call |
| B64 H4 S1024 Dh32 | 14.1022 ms | 6.3825 ms | 2.21x | 64 MiB/call |

## End-to-end validation

| Shape | Padding | Baseline | Optimized | Speedup | Max abs error | Result |
|---|---:|---:|---:|---:|---:|:---:|
| B64 D128 H4 S128 L4 | 25% | 3.1960 ms | 2.2991 ms | 1.390x | 0.00111628 | PASS |
| B16 D128 H4 S1024 L4 | 25% | 48.9708 ms | 8.9819 ms | 5.452x | 0.00095809 | PASS |

The all-valid compiled case 1 was rerun and also passed, confirming no
regression to the primary fast path.

