# Transformer GPU Kernel Optimization

This project benchmarks and optimizes the forward pass of a causal Transformer
on an NVIDIA GPU. It contains an unchanged PyTorch FP32 reference, a standalone
custom Triton attention implementation, and the latest shape-dispatched
mixed-precision implementation.

The optimized implementation keeps numerically sensitive work in FP32 while
using FP16 Tensor Core operations where they provide a measurable benefit.
Every performance run checks its output against the reference before reporting
a speedup.

## Current status

- Cases 1–13 pass the benchmark correctness rule with zero failed elements.
- Case 14 cannot allocate its input on the test GPU and OOMs before either model
  executes.
- The latest official run produced speedups from **3.725x to 18.142x**.
- The strict reference remains IEEE FP32 with TF32 disabled in the mixed
  benchmark.

Results below were measured on an NVIDIA GeForce RTX 5060 Laptop GPU (SM120)
with PyTorch 2.13.0+cu130, CUDA 13.0, and Triton 3.7.1.

| Case | Latest candidate | Speedup | Correctness |
|---:|---:|---:|:---:|
| 1 | 0.6187 ms | 5.584x | PASS |
| 2 | 0.1192 ms | 16.243x | PASS |
| 3 | 0.1523 ms | 13.613x | PASS |
| 4 | 0.1881 ms | 8.599x | PASS |
| 5 | 1.3764 ms | 8.511x | PASS |
| 6 | 144.2230 ms | 5.833x | PASS |
| 7 | 0.3933 ms | 6.246x | PASS |
| 8 | 23.2569 ms | 3.725x | PASS |
| 9 | 0.5848 ms | 4.204x | PASS |
| 10 | 0.7027 ms | 3.970x | PASS |
| 11 | 1.4400 ms | 10.127x | PASS |
| 12 | 0.1924 ms | 9.130x | PASS |
| 13 | 11.8053 ms | 18.142x | PASS |
| 14 | n/a | n/a | OOM during input allocation |

These are the exact outputs from a single official suite run. Laptop GPU clock
and thermal variation can affect individual timings, so the controlled
multi-round logs should be used when comparing close alternatives.

## Setup

The repository is configured for Windows, Python 3.12, CUDA 13.0, and
`triton-windows`.

From PowerShell in the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the environment:

```powershell
python -c "import torch, triton; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('Triton:', triton.__version__); print('GPU:', torch.cuda.get_device_name(0))"
```

The pinned environment expects:

- PyTorch `2.13.0+cu130`
- Triton for Windows `3.7.1.post27`
- NumPy `2.5.2`

An NVIDIA GPU and compatible driver are required for the Triton kernels and GPU
timings.

## Running one benchmark case

The recommended entry point is `mixed_precision_benchmark.py`. For example,
case 13 is:

```powershell
python mixed_precision_benchmark.py `
  --batch-size 64 `
  --d-model 128 `
  --heads 4 `
  --seq-len 1024 `
  --ffn-dim 128 `
  --layers 4 `
  --causal `
  --accuracy-trials 1 `
  --warmup 3 `
  --repeats 10 `
  --benchmark-rounds 1
```

The program prints:

1. the selected implementation paths;
2. correctness and error measurements;
3. strict FP32 baseline latency;
4. optimized latency and throughput;
5. speedup based on median GPU latency.

`torch.compile` compilation time is **not** included in the reported latency.
Input generation and correctness checking are also outside the timed region.

## Running the complete suite

Run all 14 configured shapes and save one text file per case plus a CSV summary:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_mixed_suite.ps1 `
  -RunId my_test_run
```

The output is written to:

```text
benchmark_logs/my_test_run/
├── case_1.txt
├── ...
├── case_14.txt
└── results.csv
```

The suite script uses:

```text
--accuracy-trials 1 --warmup 3 --repeats 10 --benchmark-rounds 1
```

For a more stable comparison between close implementations, increase the
warmups, repeats, and rounds. The benchmark alternates model timing order
between rounds to reduce ordering and thermal bias.

## Implementations

### Strict FP32 reference

`torch_transformer_benchmark.py` contains the reference Transformer and shared
benchmarking utilities. The mixed benchmark forces this model to FP32 and
disables TF32. Do not use older TF32-enabled results as direct comparisons with
the strict mixed benchmark.

### Custom causal attention

`custom_kernel/` contains a standalone Triton online-softmax causal-attention
kernel and its Transformer wrapper. It can be tested separately with:

```powershell
python custom_kernel_benchmark.py `
  --batch-size 64 --d-model 128 --heads 4 --seq-len 128 `
  --ffn-dim 128 --layers 4 --causal
```

Run its historical full suite with:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_custom_suite.ps1 `
  -RunId custom_test_run
```

### Latest mixed FP16/FP32 candidate

`mixed_precision/` contains the current optimized implementation. Its policy is:

- FP16 projection operands and weights for Tensor Core use;
- FP32 accumulation and output where required;
- FP32 residual stream and LayerNorm;
- exact GELU evaluated with FP32 values;
- shape-specific Triton projection tiles;
- custom FP32 causal attention on selected shapes;
- cuDNN FP16 SDPA where profiling showed an end-to-end win;
- fused FFN-output projection, residual addition, and LayerNorm for selected
  D128 shapes;
- preallocated B64 microbatch execution for the very large case 6.

Unsupported shapes safely fall back to PyTorch operations.

## Correctness rule

For every output element, the candidate passes if either condition is true:

```text
absolute_error <= 0.002
OR
absolute_error <= 0.02 * abs(reference)
```

The benchmark reports maximum absolute error, maximum relative error, failed
elements, and the worst output location. Performance is skipped after a failed
accuracy test unless `--benchmark-on-failure` is supplied.

## Tested shapes

| Case | Batch | D model | Heads | Sequence | Layers | FFN | Causal |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 64 | 128 | 4 | 128 | 4 | 128 | yes |
| 2 | 1 | 128 | 4 | 128 | 4 | 128 | yes |
| 3 | 4 | 128 | 4 | 128 | 4 | 128 | yes |
| 4 | 16 | 128 | 4 | 128 | 4 | 128 | yes |
| 5 | 128 | 128 | 4 | 128 | 4 | 128 | yes |
| 6 | 10,000 | 128 | 4 | 128 | 4 | 128 | yes |
| 7 | 64 | 32 | 4 | 128 | 4 | 32 | yes |
| 8 | 64 | 1,024 | 4 | 128 | 4 | 1,024 | yes |
| 9 | 64 | 128 | 1 | 128 | 4 | 128 | yes |
| 10 | 64 | 128 | 2 | 128 | 4 | 128 | yes |
| 11 | 64 | 128 | 16 | 128 | 4 | 128 | yes |
| 12 | 64 | 128 | 4 | 32 | 4 | 128 | yes |
| 13 | 64 | 128 | 4 | 1,024 | 4 | 128 | yes |
| 14 | 32 | 1,024 | 16 | 100,000 | 2 | 1,024 | yes |

Case 14's input alone contains more than 3.2 billion FP32 values, before QKV
and attention intermediates are considered. It therefore cannot be allocated on
the 8 GB development GPU.

## Repository layout

| Path | Purpose |
|:---|:---|
| `torch_transformer_benchmark.py` | FP32 reference, correctness checks, timing, and CLI |
| `mixed_precision_benchmark.py` | Strict-reference versus latest mixed candidate runner |
| `mixed_precision/transformer.py` | Shape dispatcher and mixed Transformer |
| `mixed_precision/triton_linear.py` | Triton projection and fused projection/normalization kernels |
| `custom_kernel/triton_attention.py` | Standalone causal-attention kernel |
| `run_mixed_suite.ps1` | Runs all official shapes and writes logs/CSV |
| `tournament_candidates.py` | Same-process comparison of close candidate routes |
| `benchmark_logs/` | Raw measurements and experiment reports |
| `optimization_log.md` | Chronological engineering decisions and rejected experiments |

## Useful reports

- `benchmark_logs/34_latest_implementation_test_results.md` — latest complete
  test results.
- `benchmark_logs/32_controlled_tournament_results.md` — controlled case 3 and
  case 11 route selection.
- `benchmark_logs/29_skipped_techniques_results.md` — cuDNN SDPA,
  microbatching, projection tuning, and fusion experiments.
- `optimization_log.md` — complete optimization history.

## License

This project is released under the MIT License. See `LICENSE`.
