# Latest implementation test run

## Protocol

```text
--accuracy-trials 1 --warmup 3 --repeats 10 --benchmark-rounds 1
```

CUDA latency is the value reported by the benchmark's CUDA events. Compilation,
random input generation, and accuracy checking are outside the timed region.

| Case | Status | Max abs | Failed | Strict FP32 baseline ms | Latest candidate ms | Speedup |
|---:|:---:|---:|---:|---:|---:|---:|
| 1 | PASS | 0.0015206 | 0 / 1,048,576 | 3.4552 | 0.6187 | 5.584x |
| 2 | PASS | 0.00127742 | 0 / 16,384 | 1.9367 | 0.1192 | 16.243x |
| 3 | PASS | 0.00170898 | 0 / 65,536 | 2.0727 | 0.1523 | 13.613x |
| 4 | PASS | 0.00142139 | 0 / 262,144 | 1.6174 | 0.1881 | 8.599x |
| 5 | PASS | 0.001551 | 0 / 2,097,152 | 11.7145 | 1.3764 | 8.511x |
| 6 | PASS | 0.00193438 | 0 / 163,840,000 | 841.1934 | 144.2230 | 5.833x |
| 7 | PASS | 0.00171909 | 0 / 262,144 | 2.4569 | 0.3933 | 6.246x |
| 8 | PASS | 0.00119174 | 0 / 8,388,608 | 86.6236 | 23.2569 | 3.725x |
| 9 | PASS | 0.00120425 | 0 / 1,048,576 | 2.4585 | 0.5848 | 4.204x |
| 10 | PASS | 0.00120425 | 0 / 1,048,576 | 2.7901 | 0.7027 | 3.970x |
| 11 | PASS | 0.00120425 | 0 / 1,048,576 | 14.5829 | 1.4400 | 10.127x |
| 12 | PASS | 0.00128376 | 0 / 262,144 | 1.7567 | 0.1924 | 9.130x |
| 13 | PASS | 0.00118756 | 0 / 8,388,608 | 214.1676 | 11.8053 | 18.142x |
| 14 | OOM | n/a | n/a | n/a | n/a | n/a |

Cases 1-13 passed with zero failed elements. Case 14 exhausted memory while
allocating its extremely large input before either implementation ran.

Cases 10 and 11 were slower than their controlled measurements even though the
selected routes are unchanged. This run intentionally records the exact
single-round script output; the five-round tournament remains the stronger
route-selection evidence.

## Reproducibility

- Raw per-case output and CSV: `benchmark_logs/33_latest_implementation_tests/`
- Controlled route comparison: `benchmark_logs/32_controlled_tournament_results.md`
- Strict baseline SHA256: `E1925A209BBC2A536B0DE96870585D8422018107D0696D4EA1D196BC9A4BEE4A`
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, SM120
- PyTorch: 2.13.0+cu130
- CUDA runtime: 13.0
- Triton: 3.7.1
