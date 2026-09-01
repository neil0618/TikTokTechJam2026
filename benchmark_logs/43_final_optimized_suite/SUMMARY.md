# Final optimized suite — 2026-09-01

Environment: NVIDIA GeForce RTX 5060 Laptop GPU; PyTorch 2.13.0+cu130;
CUDA 13.0; Triton 3.7.1.

Protocol: `--accuracy-trials 1 --warmup 3 --repeats 10
--benchmark-rounds 1`. Latencies are CUDA-event medians and exclude compilation
and input generation.

| Case | Baseline ms | Candidate ms | Speedup | Max abs. | Result |
|---:|---:|---:|---:|---:|:---:|
| 1 | 4.6147 | 0.6424 | 7.184x | 0.00152054 | PASS |
| 2 | 2.4440 | 0.1252 | 19.515x | 0.000883877 | PASS |
| 3 | 2.3736 | 0.1285 | 18.476x | 0.00107079 | PASS |
| 4 | 1.9381 | 0.1692 | 11.454x | 0.00142145 | PASS |
| 5 | 12.5268 | 1.1638 | 10.764x | 0.00155100 | PASS |
| 6 | 982.7582 | 137.6488 | 7.140x | 0.00193438 | PASS |
| 7 | 2.9817 | 0.2895 | 10.298x | 0.00131706 | PASS |
| 8 | 99.1962 | 23.3075 | 4.256x | 0.00119174 | PASS |
| 9 | 3.3562 | 0.4924 | 6.816x | 0.00120437 | PASS |
| 10 | 3.5996 | 0.5122 | 7.028x | 0.00120437 | PASS |
| 11 | 17.0789 | 0.6353 | 26.882x | 0.00120437 | PASS |
| 12 | 2.5019 | 0.1701 | 14.712x | 0.00128353 | PASS |
| 13 | 262.4509 | 8.9474 | 29.333x | 0.00118732 | PASS |
| 14 | n/a | n/a | n/a | n/a | OOM at input allocation |

All feasible cases had zero failed output elements. Exact stdout and complete
error metrics are in `case_1.txt` through `case_14.txt`; machine-readable data
is in `results.csv`.
