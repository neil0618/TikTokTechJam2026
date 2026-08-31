# Stage 2 - fused scaled-dot-product attention

## Change

- Replaced the candidate's explicit `QK^T`, scale, causal mask, padding mask, softmax, and probability-V chain with `torch.nn.functional.scaled_dot_product_attention`.
- The all-valid path passes `is_causal=True` without an explicit mask.
- Padded causal inputs use a cached combined boolean mask.
- The baseline remains unchanged.

PyTorch dispatched `_efficient_attention_forward` using an `fmha_cutlassF_f32` fused kernel on the FP32 case-1 profile.

## Validation

- Cases 1-5, 7-13: PASS with no failed elements.
- Padded and unpadded small regressions: PASS.
- Representative maximum absolute error stayed between approximately 0.00096 and 0.00112, below `atol=0.002`.
- Very large relative-error summaries occur at reference values close to zero; those elements pass the absolute-error branch of the required criterion.

## Representative timing

Timing configuration: 3 accuracy trials, 5 warmups, 30 repeats, 2 alternating rounds. Median CUDA-event latency.

| Case | Baseline (ms) | Candidate (ms) | Speedup | Max abs | Failed elements |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.0672 | 1.9897 | 1.542x | 0.00111628 | 0 / 3,145,728 |
| 8 | 41.7209 | 38.0470 | 1.097x | 0.00104547 | 0 / 25,165,824 |
| 11 | 13.4397 | 3.3966 | 3.957x | 0.00110716 | 0 / 3,145,728 |
| 13 | 201.8141 | 37.8862 | 5.327x | 0.00112349 | 0 / 25,165,824 |

## Profile delta, case 1

- Candidate compute-kernel launches: 61, down from 83.
- The explicit attention score copies, scaling kernels, masking kernels, standalone softmax, and two BMM launches per layer disappeared into four fused FMHA calls.
- Fused attention accounted for about 25.3% of profiled candidate CUDA time; GEMMs and LayerNorm became the leading remaining costs.
