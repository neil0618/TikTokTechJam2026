# Stage 6 — Reduce-overhead compilation and CUDA Graph capture

## Change

The optimized model now uses `torch.compile(mode="reduce-overhead")` by default
for all-valid CUDA inputs. This lets TorchInductor compile the fixed-shape
forward and use CUDA Graph replay to substantially reduce Python and driver
launch overhead.

- The baseline remains eager and unchanged.
- CPU execution remains eager by default.
- `--no-compile-user` restores the eager optimized path.
- Padded inputs deliberately remain eager after a padded regression exposed a
  CUDA-Graph lifetime conflict with the dynamically cached mask.

## Validation configuration

- Correctness trials: 3
- Warmups: 5
- Timed repeats: 30
- Timing rounds: 2
- Dtype: float32
- Compilation time is incurred during correctness/warmup and excluded from the
  steady-state latency measurement.

## Candidate latency comparison

| Case | Stage 5 eager (ms) | Compiled (ms) | Incremental gain | Correctness |
|---:|---:|---:|---:|:---:|
| 1 | 1.7922 | 1.5276 | 1.173x | PASS |
| 2 | 0.8065 | 0.1239 | 6.509x | PASS |
| 3 | 0.8045 | 0.1751 | 4.594x | PASS |
| 4 | 0.7444 | 0.4044 | 1.841x | PASS |
| 5 | 3.2374 | 3.2764 | 0.988x | PASS |
| 7 | 0.9728 | 0.8006 | 1.215x | PASS |
| 8 | 34.3235 | 33.9872 | 1.010x | PASS |
| 9 | 1.6037 | 1.3614 | 1.178x | PASS |
| 10 | 1.5552 | 1.3398 | 1.161x | PASS |
| 11 | 3.5051 | 2.9869 | 1.174x | PASS |
| 12 | 0.7267 | 0.3679 | 1.975x | PASS |
| 13 | 32.9384 | 32.7985 | 1.004x | PASS |

All feasible official cases again passed with zero failed elements. Case 5 is
within approximately 1.2% of the eager checkpoint and is the only measured
regression; the much larger gains on launch-bound cases justify enabling this
path by default.

## Fallback validation

- Padded causal input with 25% padding: PASS on the automatic eager fallback.
- CPU input: PASS with compilation disabled automatically.
- Explicit `--no-compile-user`: PASS.

## Interpretation

Compilation is most effective when individual kernels are small and host
launch latency is a large fraction of the forward. It is nearly neutral for
the D=1024 GEMM-heavy case and the S=1024 attention-heavy case, where GPU work
rather than launch dispatch dominates.
