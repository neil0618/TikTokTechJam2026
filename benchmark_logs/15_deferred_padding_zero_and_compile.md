# Stage 13 — Deferred padded-query zeroing and compiled padded execution

## Change 1: zero invalid queries once

The candidate previously zeroed invalid query positions after every attention
and every Transformer block. This is redundant:

- invalid keys remain masked in every attention call;
- LayerNorm and FFN operate independently per token;
- invalid query positions therefore cannot affect valid output positions.

The candidate now allows invalid query rows to flow through intermediate
per-token work and applies the required zero mask once at the final output.
This removes eight masked-fill passes for a four-layer model.

## Eager padded-path result

| Shape | Before | After | Improvement | Correctness |
|---|---:|---:|---:|:---:|
| B64 D128 H4 S128, 25% padding | 2.2991 ms | 2.0067 ms | 1.146x | PASS, 5 trials |
| B16 D128 H4 S1024, 25% padding | 8.9819 ms | 8.8913 ms | 1.010x | PASS, 5 trials |

## Change 2: re-enable padded compilation

The earlier CUDA-Graph failure originated from the candidate's dynamically
cached combined causal mask. Stage 9 removed that object entirely. Repeated
compiled calls with five different inputs and masks then passed, so automatic
CUDA compilation is enabled for padded inputs as well.

| Shape | Eager optimized | Compiled optimized | Compile gain | Final speedup vs baseline |
|---|---:|---:|---:|---:|
| B64 D128 H4 S128, 25% padding | 2.0067 ms | 1.4340 ms | 1.399x | 2.175x |
| B16 D128 H4 S1024, 25% padding | 8.8913 ms | 8.7551 ms | 1.016x | 5.553x |

All measured outputs passed the unchanged elementwise correctness rule with
zero failed elements.
