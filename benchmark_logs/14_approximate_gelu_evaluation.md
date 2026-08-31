# Stage 12 — Approximate GELU evaluation (rejected)

## Trial

The optimized all-valid FFN temporarily used `GELU(approximate="tanh")` while
the baseline retained exact GELU. Five correctness trials and steady-state
timing were run on cases 1 and 8.

| Case | Approximate-GELU latency | Max abs error | Correctness |
|---:|---:|---:|:---:|
| 1 | 1.3580 ms | 0.00139296 | PASS |
| 8 | 33.7180 ms | 0.00114757 | PASS |

## Decision

Reverted to exact GELU. The apparent improvement was only 0–2%, within normal
GPU clock variance, while case 1's maximum error increased from approximately
0.00112 to 0.00139. This is not a considerable or sufficiently robust gain.

Replacing the vendor GEMM with a custom GEMM+GELU kernel would be a materially
higher-risk project and is outside the current lower-risk phase.
