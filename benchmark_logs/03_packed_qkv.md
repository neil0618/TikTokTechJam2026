# Stage 3 — Packed QKV projection

## Change

The optimized attention path now computes Q, K, and V with one packed
`Linear(D, 3D)` projection. The baseline remains unchanged. Baseline Q/K/V
weights and biases are concatenated when weights are copied into the optimized
model.

This removes two GEMM launches per Transformer layer and avoids materializing
three separate contiguous Q/K/V copies before attention.

## Validation configuration

- Correctness trials: 3
- Warmups: 5
- Timed repeats: 30
- Timing rounds: 2
- Default dtype: float32
- Correctness thresholds: benchmark defaults

## Representative results

| Case | Shape summary | Baseline (ms) | Optimized (ms) | Speedup | Max abs error | Failed elements |
|---:|---|---:|---:|---:|---:|---:|
| 1 | B64 D128 H4 S128 L4 F128 | 3.2853 | 1.8716 | 1.755x | 0.00111628 | 0 |
| 8 | B64 D1024 H4 S128 L4 F1024 | 42.0908 | 35.0227 | 1.202x | 0.00104547 | 0 |
| 11 | B64 D128 H16 S128 L4 F128 | 13.9162 | 3.3629 | 4.138x | 0.00110716 | 0 |
| 13 | B64 D128 H4 S1024 L4 F128 | 201.9226 | 35.1778 | 5.740x | 0.00112349 | 0 |

## Regression coverage

- Cases 1–5 and 7–13: PASS
- Small unpadded causal check: PASS
- Small causal check with 25% padding: PASS
- Case 6 remains impractical on this machine because it pages at the memory
  limit.
- Case 14 remains impossible on this 8 GiB GPU because the input allocation
  alone requires about 12.21 GiB.

The smoke timings for the non-representative cases were intentionally not
recorded as performance claims because they used a single repeat.

## Profile comparison (case 1)

| Metric | Original baseline | Stage 2 SDPA | Stage 3 packed QKV |
|---|---:|---:|---:|
| Compute-kernel launches | 83 | 61 | 41 |

Stage 3 candidate CUDA time attribution:

| Operation | Calls | Approx. CUDA share |
|---|---:|---:|
| Fused CUTLASS FMHA | 4 | 27.79% |
| Packed QKV `addmm` | 4 | 22.01% |
| Other `addmm` | 12 | 23.23% |
| LayerNorm | 9 | 23.16% |
| Residual add | 8 | 2.59% |
| GELU | 4 | 1.22% |

Profiler self CUDA time was approximately 1.982 ms. After attention fusion and
packed QKV, LayerNorm and the remaining projections are the largest visible
costs. The next code-level opportunity is therefore residual-add + LayerNorm
fusion.
