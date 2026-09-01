  # Custom causal-attention kernel: implementation and full-suite results

## Reproducibility

- Date: 2026-08-30
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, compute capability 12.0 (SM120)
- Driver: 616.56
- VRAM reported by `nvidia-smi`: 8151 MiB
- PyTorch: 2.13.0+cu130
- CUDA runtime: 13.0
- Triton: 3.7.1
- Accuracy trials: 1
- Warmup: 3
- Timed repeats: 10
- Benchmark rounds: 1
- Dtype: float32
- TF32: enabled
- Compilation: `torch.compile(mode="reduce-overhead")`, except microbatched case 6

Every case used:

```text
--accuracy-trials 1 --warmup 3 --repeats 10 --benchmark-rounds 1
```

Raw console logs are in `benchmark_logs/custom_kernel_raw/case_1.txt` through
`case_14.txt`.

## Separation from the previous implementation

The existing `torch_transformer_benchmark.py` was not edited during this custom
kernel stage. Its SHA256 before and after this work is:

```text
E1925A209BBC2A536B0DE96870585D8422018107D0696D4EA1D196BC9A4BEE4A
```

New implementation files:

- `custom_kernel/triton_attention.py`: custom tiled online-softmax kernel
- `custom_kernel/transformer.py`: shape dispatcher and separate Transformer model
- `custom_kernel/tune_attention.py`: reproducible isolated tuner
- `custom_kernel_benchmark.py`: separate benchmark entry point

The machine has no `nvcc`, MSVC compiler, or CUTLASS checkout, so the planned
CUDA/CuTe implementation could not be compiled locally. The same custom
online-softmax design was implemented in Triton, which is the installed GPU
kernel toolchain. It is a genuine standalone GPU kernel rather than a composition
of PyTorch attention operators.

## Kernel design

- One program instance handles a `(batch, head, 64-query-row)` tile.
- It streams 32 key/value rows per iteration.
- Future causal key tiles are skipped entirely.
- QK scores and softmax probabilities are never written to global memory.
- Online row maximum and normalization sum remain in FP32.
- QK and probability-V products use TF32 dot operations with FP32 accumulation.
- The result is written directly in contiguous `[B,S,H,Dhead]` order, eliminating
  the usual post-attention transpose/copy before output projection.
- Selected launch: `BLOCK_M=64`, `BLOCK_N=32`, four warps, two stages.
- Specialized shapes: float32, causal, all-valid input, `Dhead=32`, and sequence
  length 128 or 1024.
- Batch 1 / sequence 128 and every unsupported shape use the previous SDPA path.

## Isolated tuning result

At `B=64, H=4, S=1024, Dhead=32`, repeated tuner runs measured the selected
kernel near 1.35 ms versus about 6.14 ms for PyTorch SDPA. All tested tile
variants had zero failures under the benchmark's element-wise correctness rule.
Eight-warp variants were about twice as slow, and larger tiles were not stable
winners across repeated runs.

## Full-suite results

| Case | B | D | H | S | L | F | Custom active | Correct | Failed elements | Max abs | Baseline ms | Candidate ms | Speedup |
|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|---:|---:|
| 1 | 64 | 128 | 4 | 128 | 4 | 128 | yes | PASS | 0 / 1,048,576 | 0.00131249 | 4.4333 | 0.8356 | 5.305x |
| 2 | 1 | 128 | 4 | 128 | 4 | 128 | no | PASS | 0 / 16,384 | 0.00062263 | 1.9060 | 0.1231 | 15.485x |
| 3 | 4 | 128 | 4 | 128 | 4 | 128 | yes | PASS | 0 / 65,536 | 0.00125545 | 1.9437 | 0.1299 | 14.966x |
| 4 | 16 | 128 | 4 | 128 | 4 | 128 | yes | PASS | 0 / 262,144 | 0.00130025 | 1.9450 | 0.2516 | 7.730x |
| 5 | 128 | 128 | 4 | 128 | 4 | 128 | yes | PASS | 0 / 2,097,152 | 0.00143442 | 10.4546 | 1.8634 | 5.611x |
| 6 | 10000 | 128 | 4 | 128 | 4 | 128 | yes¹ | PASS | 0 / 163,840,000 | 0.00194860 | 744.2282 | 198.7912 | 3.744x |
| 7 | 64 | 32 | 4 | 128 | 4 | 32 | no | PASS | 0 / 262,144 | 0.00109220 | 2.2496 | 0.8276 | 2.718x |
| 8 | 64 | 1024 | 4 | 128 | 4 | 1024 | no | PASS | 0 / 8,388,608 | 0.00104547 | 41.3009 | 35.3370 | 1.169x |
| 9 | 64 | 128 | 1 | 128 | 4 | 128 | no | PASS | 0 / 1,048,576 | 0.000884891 | 2.9469 | 1.1950 | 2.466x |
| 10 | 64 | 128 | 2 | 128 | 4 | 128 | no | PASS | 0 / 1,048,576 | 0.00102055 | 2.3103 | 1.1658 | 1.982x |
| 11 | 64 | 128 | 16 | 128 | 4 | 128 | no | PASS | 0 / 1,048,576 | 0.000924885 | 13.7108 | 2.9309 | 4.678x |
| 12 | 64 | 128 | 4 | 32 | 4 | 128 | no | PASS | 0 / 262,144 | 0.00104997 | 2.0798 | 0.3368 | 6.175x |
| 13 | 64 | 128 | 4 | 1024 | 4 | 128 | yes | PASS | 0 / 8,388,608 | 0.00147647 | 204.2207 | 15.7809 | 12.941x |
| 14 | 32 | 1024 | 16 | 100000 | 2 | 1024 | no | OOM | n/a | n/a | n/a | n/a | n/a |

¹ Case 6 runs the custom kernel on batch slices of 250 to cap activation memory.

Large relative-error maxima occur where the reference is extremely close to
zero. The benchmark uses `absolute <= 0.002 OR relative <= 2%`; every element in
cases 1-13 passed that exact rule.

## Change versus the previous optimized implementation

The comparison below uses the previous full-suite medians from
`benchmark_logs/19_full_suite_latest.md`. It is cross-run and therefore includes
clock/thermal variance, but the custom-active shapes show a consistent gain.

| Case | Previous candidate ms | Custom candidate ms | Incremental speedup |
|---:|---:|---:|---:|
| 1 | 1.3742 | 0.8356 | 1.644x |
| 3 | 0.1665 | 0.1299 | 1.282x |
| 4 | 0.3700 | 0.2516 | 1.471x |
| 5 | 2.8161 | 1.8634 | 1.511x |
| 6 | 262.1124 | 198.7912 | 1.319x |
| 13 | 32.7119 | 15.7809 | 2.073x |

Case 13 is the main result: the new kernel approximately halves the runtime of
the already-optimized candidate and improves the original baseline by 12.941x.

## Case 14

Case 14 fails while generating the input, before either Transformer executes.
The float32 input multiplication requests 12.21 GiB on a GPU with 7.93 GiB usable
capacity. A custom attention kernel cannot repair this without changing the
input-generation or workload contract. Dense sequence length 100,000 would also
have quadratic attention work even if the input allocation were changed.
