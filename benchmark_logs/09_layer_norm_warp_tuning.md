# Stage 8 — Shape-aware residual/LayerNorm warp tuning

## Change

The fused Triton residual-add + LayerNorm kernel now selects its CTA warp count
from the feature width instead of using four warps for every width up to 256:

| Padded feature block | Previous | New |
|---:|---:|---:|
| <= 64 | 4 warps | 2 warps |
| 65–256 | 4 warps | 1 warp |
| > 256 | 8 warps | 8 warps |

This removes under-utilized warps and cross-warp reduction barriers for the
common `D=128` cases.

## Kernel microbenchmark

Test setup: 8,192 rows, float32, 100 launches per sample, median of five
samples.

| D | 1 warp (ms) | 2 warps (ms) | 4 warps (ms) | 8 warps (ms) | Selected |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.0199¹ | 0.0152 | 0.0206 | — | 2 |
| 128 | 0.0165 | 0.0308 | 0.0352 | 0.0580 | 1 |
| 1024 | — | — | 0.3988 | 0.3983 | 8 |

¹ The one-warp D=32 samples showed clock/warmup instability; two warps were the
stable winner.

## Representative end-to-end results

All results include the Stage 6 default reduce-overhead compilation.

| Case | Before tuning (ms) | After tuning (ms) | Incremental change | Correctness |
|---:|---:|---:|---:|:---:|
| 1 | 1.5276 | 1.4488 | 1.054x faster | PASS |
| 7 (`D=32`) | 0.8006 | 0.8141 | 1.7% slower/noise | PASS |
| 8 (`D=1024`) | 33.9872 | 34.4322 | 1.3% slower/noise | PASS |
| 13 | 32.7985 | 32.6212 | 1.005x faster | PASS |

The D=32 and D=1024 changes are small relative to run-to-run variance. The
repeatable D=128 kernel-level improvement and the 5.4% case 1 gain justify the
new dispatch.

## Regression coverage

- Cases 1–5 and 7–13: PASS with zero failed elements.
- Representative cases: 5 warmups, 30 repeats, 2 rounds.
- Remaining cases were first smoke-tested, then rerun with 5 warmups, 30
  repeats, and 2 rounds.
- Python bytecode compilation: PASS.
- `git diff --check`: PASS.

