# Controlled per-case candidate tournament

## Decision

The tournament retained the expanded D128 projection schedule for case 11 and
rejected the older FP32 custom-attention route for case 3.

Both tournaments ran the incumbent and challenger in the same Python process,
on the same fixed input, with alternating measurement order. Compilation time
was excluded. The strict comparison settings were FP32 matmul precision
`highest`, cuBLAS TF32 disabled, and cuDNN TF32 disabled.

```text
--accuracy-trials 3 --warmup 10 --repeats 30 --rounds 5
```

## Case 3

| Route | Median | Mean | Minimum | Accuracy |
|:---|---:|---:|---:|:---:|
| Current mixed/vendor route | 0.152096 ms | 0.160243 ms | 0.148928 ms | PASS |
| Historical FP32 custom attention | 0.280896 ms | 0.300567 ms | 0.249440 ms | PASS |

The challenger reached only 0.541x the incumbent's speed and was rejected. Its
historical 0.1299 ms result came from the older custom suite with TF32 enabled;
it is not comparable to the strict no-TF32 mixed suite. The current mixed route
remains selected.

## Case 11

| Route | Median | Mean | Minimum | Accuracy |
|:---|---:|---:|---:|:---:|
| Current D128 tiles | 1.267904 ms | 1.330283 ms | 1.176576 ms | PASS |
| Expanded case-11 tiles | 1.224176 ms | 1.287681 ms | 1.199776 ms | PASS |

The challenger was 1.0357x faster overall and won every one of the five
alternating rounds. It produced exactly the same recorded maximum errors as the
incumbent on all three accuracy trials, with zero failed elements. The expanded
schedule is now enabled only for the exact case-11 shape.

The retained production route was then validated with three accuracy trials,
10 warmups, 30 repeats, and three rounds. It measured 1.1569 ms and 12.611x
versus the strict FP32 baseline, with maximum absolute error 0.00122869 and zero
failed elements out of 3,145,728.

## Official complete-suite result

The official protocol remained:

```text
--accuracy-trials 1 --warmup 3 --repeats 10 --benchmark-rounds 1
```

| Case | Status | Max abs | Strict FP32 baseline ms | Mixed candidate ms | Speedup |
|---:|:---:|---:|---:|---:|---:|
| 1 | PASS | 0.0015206 | 3.8655 | 0.7161 | 5.398x |
| 2 | PASS | 0.00127742 | 2.0946 | 0.1275 | 16.432x |
| 3 | PASS | 0.00170898 | 1.7854 | 0.1525 | 11.707x |
| 4 | PASS | 0.00142139 | 3.5132 | 0.1896 | 18.526x |
| 5 | PASS | 0.001551 | 11.6215 | 1.3459 | 8.635x |
| 6 | PASS | 0.00193438 | 843.7256 | 144.0220 | 5.858x |
| 7 | PASS | 0.00171909 | 2.3464 | 0.3698 | 6.345x |
| 8 | PASS | 0.00119174 | 76.8208 | 25.3370 | 3.032x |
| 9 | PASS | 0.00120425 | 2.4589 | 0.6085 | 4.041x |
| 10 | PASS | 0.00120425 | 2.7893 | 0.5714 | 4.882x |
| 11 | PASS | 0.00120425 | 14.5020 | 1.2453 | 11.645x |
| 12 | PASS | 0.00128376 | 1.7016 | 0.1932 | 8.810x |
| 13 | PASS | 0.00118756 | 218.8180 | 11.5377 | 18.965x |
| 14 | OOM | n/a | n/a | n/a | n/a |

Cases 1-13 all passed with zero failed output elements. Case 14 still fails
during allocation of the enormous input, before either implementation runs.
The single-round suite remains sensitive to clocks and thermals, particularly
case 8; the implementation path for case 8 was not changed in this experiment.

## Reproducibility

- Tournament raw logs: `benchmark_logs/30_controlled_candidate_tournament/`
- Official raw logs and CSV: `benchmark_logs/31_case11_tournament_winner/`
- Tournament driver: `tournament_candidates.py`
- Strict baseline SHA256: `E1925A209BBC2A536B0DE96870585D8422018107D0696D4EA1D196BC9A4BEE4A`
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, SM120
- PyTorch: 2.13.0+cu130
- CUDA runtime: 13.0
- Triton: 3.7.1
