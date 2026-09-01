  # Full test suite — latest implementation

Run date: 2026-08-30 (Asia/Singapore)

## Environment

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- Reported usable capacity: 7.93 GiB
- PyTorch: 2.13.0+cu130
- CUDA runtime: 13.0
- Triton: 3.7.1 (`triton-windows` 3.7.1.post27)
- Dtype: float32
- TF32: enabled
- Optimized compilation: `reduce-overhead` except case 6 microbatching
- Implementation SHA256:
  `E1925A209BBC2A536B0DE96870585D8422018107D0696D4EA1D196BC9A4BEE4A`

## Test protocol

- Cases 1–5 and 7–13:
  - 5 correctness trials
  - 5 warmups
  - 30 timed repeats
  - 2 alternating timing rounds
- Case 6:
  - 3 correctness trials
  - 3 warmups
  - 10 timed repeats
  - 2 alternating timing rounds
  - automatic microbatch size 250
- Case 14:
  - attempted with 1 correctness trial, no warmup, and 1 requested repeat
  - failed during input generation before a model forward
- Correctness rule for every element:
  - absolute error <= 0.002, **or**
  - relative error <= 2%

## Results

| Case | B | D | H | S | L | F | Baseline median | Optimized median | Speedup | Max abs error | Failed elements | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 64 | 128 | 4 | 128 | 4 | 128 | 3.1110 ms | 1.3742 ms | 2.264x | 0.00112993 | 0 / 5,242,880 | PASS |
| 2 | 1 | 128 | 4 | 128 | 4 | 128 | 1.8515 ms | 0.1220 ms | 15.176x | 0.00071144 | 0 / 81,920 | PASS |
| 3 | 4 | 128 | 4 | 128 | 4 | 128 | 1.9532 ms | 0.1665 ms | 11.728x | 0.00081426 | 0 / 327,680 | PASS |
| 4 | 16 | 128 | 4 | 128 | 4 | 128 | 1.8950 ms | 0.3700 ms | 5.121x | 0.00090277 | 0 / 1,310,720 | PASS |
| 5 | 128 | 128 | 4 | 128 | 4 | 128 | 8.8392 ms | 2.8161 ms | 3.139x | 0.00112993 | 0 / 10,485,760 | PASS |
| 6 | 10000 | 128 | 4 | 128 | 4 | 128 | 744.8266 ms | 262.1124 ms | 2.842x | 0.00124328 | 0 / 491,520,000 | PASS |
| 7 | 64 | 32 | 4 | 128 | 4 | 32 | 2.1636 ms | 0.7580 ms | 2.854x | 0.00131610 | 0 / 1,310,720 | PASS |
| 8 | 64 | 1024 | 4 | 128 | 4 | 1024 | 41.7433 ms | 34.0807 ms | 1.225x | 0.00115755 | 0 / 41,943,040 | PASS |
| 9 | 64 | 128 | 1 | 128 | 4 | 128 | 2.1480 ms | 1.3366 ms | 1.607x | 0.00110114 | 0 / 5,242,880 | PASS |
| 10 | 64 | 128 | 2 | 128 | 4 | 128 | 2.5419 ms | 1.2477 ms | 2.037x | 0.00108171 | 0 / 5,242,880 | PASS |
| 11 | 64 | 128 | 16 | 128 | 4 | 128 | 13.3578 ms | 3.0036 ms | 4.447x | 0.00110719 | 0 / 5,242,880 | PASS |
| 12 | 64 | 128 | 4 | 32 | 4 | 128 | 2.8546 ms | 0.3350 ms | 8.522x | 0.00104997 | 0 / 1,310,720 | PASS |
| 13 | 64 | 128 | 4 | 1024 | 4 | 128 | 202.1930 ms | 32.7119 ms | 6.181x | 0.00112352 | 0 / 41,943,040 | PASS |
| 14 | 32 | 1024 | 16 | 100000 | 2 | 1024 | — | — | — | — | — | OOM |

## Case 6 details

- Baseline throughput: 1,718,521 token/s
- Optimized throughput: 4,883,401 token/s
- Accuracy covered 491,520,000 output elements across three trials.
- All three trials passed with zero failed elements.
- This case previously entered severe GPU paging and did not finish; automatic
  batch-only microbatching now makes it practical.

## Case 14 failure

The run attempted to allocate 13,107,200,000 bytes (`12.21 GiB`) while creating
the float32 input `[32, 100000, 1024]`. The GPU exposes only 7.93 GiB, so input
generation raised `torch.OutOfMemoryError` before either Transformer executed.

Even if input streaming were added, exact dense attention at sequence length
100,000 has quadratic work and is not practical on this laptop GPU under the
provided algorithm and dtype.

## Conclusion

- Runnable official cases passed: **13 / 13**
- Failed correctness elements: **0**
- Unrunnable due to input allocation: **case 14**
- Observed optimized speedup range: **1.225x–15.176x**
- No implementation changes were made during this final test run.
