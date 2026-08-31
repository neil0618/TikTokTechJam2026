# Six-experiment optimization pass

## Outcome

Six proposed optimization directions were implemented or evaluated independently. One produced a repeatable end-to-end improvement and was retained. Five were rejected and reverted because they regressed performance, were noise-level, or violated correctness.

| # | Experiment | Result | Decision |
|---:|---|---|---|
| 1 | Split causal prefix and diagonal attention loops | S128 regressed; S1024 effectively flat | Rejected |
| 2 | D128 multi-row residual plus LayerNorm | 0-2.6% isolated differences; no material large-shape gain | Rejected |
| 3 | Compile B250 microbatch core and preallocate output | Case 6: 237.4574 to 228.1437 ms, 1.041x | Retained |
| 4 | Fuse FFN input GEMM, bias, and exact GELU | Fast mode failed 2 elements; accurate mode regressed case 13 | Rejected |
| 5 | Attention register/tile tuning | Smaller tiles lost reuse; full-model alternatives regressed | Rejected |
| 6 | Vendor GEMM/compiler-mode dispatch | max-autotune 37.5770 vs existing 37.4218 ms | Rejected |

## Retained full-suite results

Protocol: one accuracy trial, three warmups, ten timed repeats, one benchmark round. All values below are exact medians printed by the benchmark.

| Case | Status | Max abs | Baseline ms | Candidate ms | Speedup |
|---:|:---:|---:|---:|---:|---:|
| 1 | PASS | 0.00131249 | 3.2333 | 0.9217 | 3.508x |
| 2 | PASS | 0.00062263 | 4.6192 | 0.2426 | 19.042x |
| 3 | PASS | 0.00125545 | 4.1041 | 0.3887 | 10.559x |
| 4 | PASS | 0.00130025 | 4.2592 | 0.2625 | 16.225x |
| 5 | PASS | 0.00143442 | 10.4423 | 1.8772 | 5.563x |
| 6 | PASS | 0.00194860 | 918.1219 | 229.2808 | 4.004x |
| 7 | PASS | 0.00109220 | 4.6795 | 0.7516 | 6.226x |
| 8 | PASS | 0.00104547 | 46.3504 | 47.6307 | 0.973x |
| 9 | PASS | 0.000884891 | 4.4777 | 1.2358 | 3.623x |
| 10 | PASS | 0.00102055 | 5.0040 | 1.1879 | 4.213x |
| 11 | PASS | 0.000924885 | 16.2578 | 2.8934 | 5.619x |
| 12 | PASS | 0.00104997 | 6.0944 | 0.3410 | 17.870x |
| 13 | PASS | 0.00147647 | 255.0505 | 18.4238 | 13.844x |
| 14 | OOM | n/a | n/a | n/a | n/a |

Case 8 was slower than its baseline in this particular thermally variable sweep. The candidate path for case 8 was not changed by the retained optimization; the previous controlled compiler-mode comparison remained a statistical tie.

## Artifacts

- Detailed decision log: `optimization_log.md`
- Raw output: `benchmark_logs/22_case6_compiled_preallocated/case_1.txt` through `case_14.txt`
- Parsed results: `benchmark_logs/22_case6_compiled_preallocated/results.csv`
- Reusable suite runner: `run_custom_suite.ps1`
