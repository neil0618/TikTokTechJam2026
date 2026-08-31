# Baseline benchmark log

Date: 2026-08-29 (Asia/Singapore)

Implementation SHA-256: `8451EA2F58F5B2B0C9E7CD0515CDCC94A73AE843634A61282C3F1DD9375E1472`

## Environment

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8,151 MiB, compute capability 12.0
- NVIDIA driver: 616.56
- CUDA driver/UMD: 13.4
- PyTorch: 2.13.0+cu130
- PyTorch CUDA runtime: 13.0
- cuDNN: 9.20.0
- Triton: 3.7.1 (`triton-windows` 3.7.1.post27)
- Dtype: FP32; matmul precision `high`; TF32 allowed
- Correctness: 5 trials, `atol=0.002 OR rtol=0.02`
- Timing: 20 warmups, 100 repeats, 3 alternating rounds; CUDA events; median shown

## Results

| Case | B | S | D | H | F | L | Baseline median (ms) | Candidate median (ms) | Correctness | Max abs | Max rel |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| 1 | 64 | 128 | 128 | 4 | 128 | 4 | 3.4430 | 3.6272 | PASS | 0 | 0 |
| 2 | 1 | 128 | 128 | 4 | 128 | 4 | 1.9796 | 1.9720 | PASS | 0 | 0 |
| 3 | 4 | 128 | 128 | 4 | 128 | 4 | 1.9332 | 1.9264 | PASS | 0 | 0 |
| 4 | 16 | 128 | 128 | 4 | 128 | 4 | 2.0059 | 1.9609 | PASS | 0 | 0 |
| 5 | 128 | 128 | 128 | 4 | 128 | 4 | 9.1381 | 9.0050 | PASS | 0 | 0 |
| 6 | 10000 | 128 | 128 | 4 | 128 | 4 | not completed | not completed | unavailable | unavailable | unavailable |
| 7 | 64 | 128 | 32 | 4 | 32 | 4 | 2.3521 | 2.2069 | PASS | 0 | 0 |
| 8 | 64 | 128 | 1024 | 4 | 1024 | 4 | 42.5340 | 42.5582 | PASS | 0 | 0 |
| 9 | 64 | 128 | 128 | 1 | 128 | 4 | 2.2369 | 2.1542 | PASS | 0 | 0 |
| 10 | 64 | 128 | 128 | 2 | 128 | 4 | 2.6620 | 2.8466 | PASS | 0 | 0 |
| 11 | 64 | 128 | 128 | 16 | 128 | 4 | 13.1410 | 13.1101 | PASS | 0 | 0 |
| 12 | 64 | 32 | 128 | 4 | 128 | 4 | 2.3116 | 2.3385 | PASS | 0 | 0 |
| 13 | 64 | 1024 | 128 | 4 | 128 | 4 | 202.6814 | 202.5678 | PASS | 0 | 0 |
| 14 | 32 | 100000 | 1024 | 16 | 1024 | 2 | OOM | OOM | not reached | not reached | not reached |

The candidate was identical to the baseline at this stage, so non-1.0 speedups are measurement noise.

## Stress-case notes

- Case 6 ran for more than 100 minutes at 100% GPU utilization with approximately 7-8 GiB VRAM in use and substantial WDDM paging. It did not provide a usable report before the workflow moved to profiling, so no estimated latency is recorded.
- Case 14 failed while generating the input. Multiplying the input by `input_scale` attempted a 12.21 GiB allocation on a 7.93 GiB GPU. Dense attention would require vastly more memory.

## Representative case-1 profile

One warmed four-layer forward launched 83 CUDA compute kernels, 17 device-to-device copies, and 24 async memsets. Prominent parent-operator attribution was: linear GEMMs 24.2%, LayerNorm 16.8%, attention BMMs 12.2%, score-mask copies 8.9%, key mask 6.4%, causal mask 5.6%, softmax 5.1%, and scale 4.5%.
