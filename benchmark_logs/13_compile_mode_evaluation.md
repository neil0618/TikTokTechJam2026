# Stage 11 — Compiler-mode evaluation (current mode retained)

## Goal

Compare TorchInductor `default`, `reduce-overhead`, and `max-autotune` modes on
the normal D=128 case and the GEMM-heavy D=1024 case.

## Results

| Shape | default | reduce-overhead checkpoint | max-autotune | Decision |
|---|---:|---:|---:|---|
| Case 1, D128 | 1.4232 ms | 1.36–1.45 ms | 1.3983 ms | Keep reduce-overhead |
| Case 8, D1024 | 33.9381 ms | 33.99–34.43 ms | 33.9379 ms | Statistical tie |

Every mode passed correctness. TorchInductor emitted `Not enough SMs to use
max_autotune_gemm mode` for this 26-SM GPU, so max-autotune did not provide its
intended GEMM search. The D=1024 differences are within normal thermal and clock
variance.

## Decision

Keep `reduce-overhead` as the default because it provides the proven CUDA Graph
benefits on small shapes without sacrificing compute-heavy shapes. No code
change was made in this stage.
