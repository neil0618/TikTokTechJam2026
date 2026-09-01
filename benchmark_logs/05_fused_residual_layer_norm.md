# Stage 5 — Triton fused residual-add + LayerNorm

## Change

The optimized all-valid inference path now fuses each residual addition with
the LayerNorm that immediately consumes that residual. One Triton program
produces both values:

1. the residual tensor needed by the next skip connection; and
2. the normalized tensor needed by the next attention or FFN projection.

The padded path retains the already-validated PyTorch block implementation.
The baseline is unchanged. CPU and non-contiguous inputs use a native PyTorch
fallback.

## Local Triton setup note

The portable Python distribution initially lacked `Python.h` and
`python312.lib`, which Triton-Windows needs to build its driver helper. The
official CPython 3.12.10 NuGet development package was installed into the
git-ignored local Python/venv directories. Triton's generated cache is directed
to the system temporary directory.

## Representative validation

- Correctness trials: 3
- Warmups: 5
- Timed repeats: 30
- Timing rounds: 2
- Dtype: float32

| Case | Baseline (ms) | Stage 3 (ms) | Stage 5 (ms) | Final speedup | Max abs error | Failed elements |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.3807 | 1.8716 | 1.7922 | 1.886x | 0.00111634 | 0 |
| 8 | 42.5285 | 35.0227 | 34.3235 | 1.239x | 0.00115755 | 0 |
| 11 | 13.5892 | 3.3629 | 3.5051 | 3.877x | 0.00110716 | 0 |
| 13 | 201.9509 | 35.1778 | 32.9384 | 6.131x | 0.00112346 | 0 |

Case 11's standalone checkpoint is slightly slower than the earlier Stage 3
run. A same-process five-round A/B check removed cross-run variance and measured
the fused path at 3.4288 ms versus 4.0114 ms for the unfused candidate, so the
fusion remains enabled for that shape.

## Profile comparison (case 1)

| Metric | Stage 3 | Stage 5 |
|---|---:|---:|
| Compute-kernel launches | 41 | 33 |
| Native LayerNorm calls | 9 | 1 |
| Fused residual+LayerNorm calls | 0 | 8 |

Stage 5 candidate CUDA attribution:

| Operation | Calls | Approx. CUDA share |
|---|---:|---:|
| GEMM kernels | 16 | 49.43% |
| Fused CUTLASS FMHA | 4 | 30.23% |
| Triton residual+LayerNorm | 8 | 14.83% |
| Initial native LayerNorm | 1 | 4.11% |
| GELU | 4 | 1.39% |

Profiler self CUDA time was approximately 1.820 ms. The eight standalone
residual-add kernels are gone, and eight native LayerNorm calls have been
replaced by eight fused Triton kernels.

## Additional coverage

- Cases 1–5 and 7–13: PASS
- Padded causal fallback (25% padding): PASS
- Non-causal CUDA path: PASS
- Non-causal CPU fallback: PASS
- Python bytecode compilation: PASS

