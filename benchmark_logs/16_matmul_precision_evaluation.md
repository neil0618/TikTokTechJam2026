# Stage 14 — Float32 matmul-setting evaluation (current setting retained)

## Case 8 results

| Setting | Optimized latency | Correctness |
|---|---:|:---:|
| `matmul_precision=medium`, TF32 enabled | 33.9745 ms | PASS |
| `matmul_precision=high`, TF32 enabled | approximately 34 ms | PASS |
| `matmul_precision=highest`, TF32 enabled | 34.3364 ms | PASS |
| TF32 disabled | 66.8228 ms | PASS |

The enabled-TF32 settings are within roughly 1% run-to-run variance. Disabling
TF32 improves agreement between the two implementations but nearly doubles
latency, so it is not a viable performance choice under the already-passing
correctness contract.

## Decision

Keep the existing `high` precision setting with TF32 enabled. No code change
was made in this stage.
