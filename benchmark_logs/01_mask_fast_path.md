# Stage 1 - padding fast path and cached causal masks

## Change

- The candidate skips padding-key and output masking when `padding_ratio=0` guarantees that all tokens are valid.
- Each candidate attention layer creates its causal mask once and reuses it for subsequent calls of the same shape/device.
- The baseline implementation remains unchanged.

## Validation

- Cases 1-5, 7-13: PASS. The four representative cases used 3 full accuracy trials; the other safe cases used one regression trial.
- Padded regression (`B=4,S=32,D=32,H=4,F=32,L=2,padding_ratio=0.25`): PASS, max absolute and relative error both zero.
- Unpadded small regression: PASS, max absolute and relative error both zero.
- Cases 6 and 14 were not repeated for the reasons in `00_baseline.md`.

## Representative timing

Timing configuration: 3 accuracy trials, 5 warmups, 30 repeats, 2 alternating rounds. Median CUDA-event latency.

| Case | Baseline (ms) | Candidate (ms) | Speedup | Max abs | Max rel |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.2611 | 2.6585 | 1.227x | 0 | 0 |
| 8 | 44.1340 | 38.3815 | 1.150x | 0 | 0 |
| 11 | 13.1574 | 10.5041 | 1.253x | 0 | 0 |
| 13 | 201.9549 | 147.8047 | 1.366x | 0 | 0 |

Single-repeat timings from the broad regression sweep were treated only as smoke tests and are deliberately not used as performance claims.
