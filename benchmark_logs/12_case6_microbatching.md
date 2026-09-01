# Stage 10 — Bounded-memory execution for case 6

## Change

CUDA batches of 4,096 or more now default to inference microbatches of 250.
Both baseline and optimized models are wrapped identically, preserving the
benchmark comparison and output order. Users can override the size with
`--microbatch-size` or disable automatic slicing with
`--microbatch-size 0`.

Compilation is disabled for microbatched execution because direct testing found
CUDA Graph replay neutral-to-slower and more memory-hungry for this workload.

## Why this is exact

Self-attention operates independently for each batch element. Splitting only
the batch dimension does not change attention, normalization, residuals, or
FFN mathematics. Chunk outputs are concatenated back in original batch order.

## Case 6 result

Configuration: B10000 D128 H4 S128 L4 F128, causal, float32.

| Metric | Baseline | Optimized |
|---|---:|---:|
| Median latency | 741.2646 ms | 265.1999 ms |
| Throughput | 1,726,779 token/s | 4,826,548 token/s |
| Speedup | — | 2.795x |
| Approx. tested peak allocation | 1.5–2.1 GiB | 2.0–2.5 GiB |

Correctness: **PASS**, max absolute error `0.00117791`, zero failed elements out
of 163,840,000.

Validation used one full correctness trial, one warmup, and three timed repeats
because each output contains more than 163 million float32 values. Before this
change, the full-batch baseline entered GPU paging and did not finish after more
than 100 minutes.

## Microbatch-size evaluation

| Model | 250 | 500 | 1000 | Larger |
|---|---:|---:|---:|---:|
| Baseline | 796 ms | 842 ms | 842 ms | — |
| Optimized eager | 247 ms | 290 ms | 303 ms | 298–322 ms |

The size-250 policy was the fastest tested setting and left substantial memory
headroom on the 8 GiB GPU.
