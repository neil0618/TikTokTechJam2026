# Stage 4 — Mixed-precision evaluation (rejected)

## Decision

Neither float16 nor bfloat16 satisfies the benchmark's existing correctness
criterion for the optimized SDPA path. No mixed-precision mode was adopted and
the default remains float32. The tolerance was not relaxed.

## Test configuration

- Representative cases: 1, 8, 11, and 13
- Correctness trials: 3 per case
- Criterion: absolute error <= 0.002 **or** relative error <= 2%
- Same dtype used for the baseline and optimized models

## Float16

| Case | Result | Max abs error | Failed / total elements (3 trials) |
|---:|---|---:|---:|
| 1 | FAIL | 0.00585938 | 5 / 3,145,728 |
| 8 | FAIL | 0.00781250 | 22 / 25,165,824 |
| 11 | FAIL | 0.00781250 | 6 / 3,145,728 |
| 13 | FAIL | 0.00781250 | 13 / 25,165,824 |

Float16 is close, but a correctness failure remains a failure for this
benchmark. Performance timing was automatically skipped after validation.

## Bfloat16

| Case | Result | Max abs error | Failed / total elements (3 trials) |
|---:|---|---:|---:|
| 1 | FAIL | 0.0625 | 146,754 / 3,145,728 |
| 8 | FAIL | 0.0625 | 1,009,967 / 25,165,824 |
| 11 | FAIL | 0.046875 | 146,374 / 3,145,728 |
| 13 | FAIL | 0.0625 | 859,400 / 25,165,824 |

Bfloat16 fails by a wide margin under the supplied tolerances. Performance
timing was automatically skipped after validation.

## Conclusion

Mixed precision could still be useful under an explicitly looser application
error budget, especially float16, but it is not a valid optimization for the
current scoring contract. Stage 5 therefore continues in float32.
