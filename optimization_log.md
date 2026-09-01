# Custom-kernel optimization log

## Measurement protocol

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (SM120)
- PyTorch: 2.13.0+cu130
- CUDA runtime: 13.0
- Triton: 3.7.1
- Full-suite reference: `benchmark_logs/21_custom_kernel_results.md`
- Full-model checks use `--accuracy-trials 1 --warmup 3 --repeats 10 --benchmark-rounds 1` unless noted.
- Correctness requirement: absolute error <= 0.002 OR relative error <= 0.02 for every element.
- A change is retained only after a repeatable measured improvement.

## Reference before this optimization pass

Cases 1-13 passed. Case 14 could not allocate its 12.21 GiB input on the 8 GiB GPU.

| Case | Candidate median (ms) |
|---:|---:|
| 1 | 0.8356 |
| 2 | 0.1231 |
| 3 | 0.1299 |
| 4 | 0.2516 |
| 5 | 1.8634 |
| 6 | 198.7912 |
| 7 | 0.8276 |
| 8 | 35.3370 |
| 9 | 1.1950 |
| 10 | 1.1658 |
| 11 | 2.9309 |
| 12 | 0.3368 |
| 13 | 15.7809 |
| 14 | OOM before model execution |

## Experiment 1 — split causal prefix and diagonal loops

- Date: 2026-08-31
- Motivation: avoid element-wise causal predicates on key tiles strictly before the query tile.
- Bottleneck addressed: long-sequence custom attention instruction and predicate overhead.
- Implementation tested: separate mask-free prefix loop and masked diagonal loop; remove boundary predicates for exact supported sequence lengths.
- Affected shapes: custom-attention paths with head dimension 32 and sequence length 128 or 1024.
- Correctness: PASS, case 13 maximum absolute error 0.00147647, failed elements 0 / 8,388,608.
- Selected-kernel isolated result, B64/H4/S128/Dh32: 0.104176 ms before, 0.144048 ms after (0.723x; regression).
- Selected-kernel isolated result, B64/H4/S1024/Dh32: 1.580640 ms before, 1.569536 ms after (1.007x; effectively flat).
- Case 13 full model: 15.7809 ms retained reference, 18.0203 ms experiment median (0.876x cross-run); minimum was 16.3640 ms.
- GPU observation: splitting the loop increases control-flow/code complexity and did not reduce the selected kernel's effective cost on SM120.
- Decision: REJECTED and reverted.
- Reason: no repeatable long-sequence benefit and a material short-sequence regression.

## Experiment 2 — D128 multi-row residual plus LayerNorm

- Date: 2026-08-31
- Motivation: amortize program scheduling and assign multiple independent 128-wide rows to one Triton program.
- Bottleneck addressed: memory-bound residual-add plus LayerNorm, especially cases 6 and 13.
- Implementation tested: 1, 2, 4, and 8 rows per program with matching 1, 2, 4, and 8 warp launches.
- Affected shapes: every D=128 benchmark case.
- Correctness: bit-identical residuals and normalized outputs in the isolated tests; zero failed values.
- 128 rows: retained 0.079104 ms; best experiment 0.078224 ms (1.011x, noise-level).
- 8,192 rows: retained 0.083744 ms; best experiment 0.081648 ms (1.026x, noise-level).
- 32,000 rows: retained 0.298432 ms; best experiment 0.298032 ms (1.001x).
- 65,536 rows: retained 0.565248 ms; best experiment 0.564032 ms (1.002x).
- GPU observation: runtime scales with bytes moved at larger row counts; grouping rows does not reduce the required two input reads and two output writes.
- Decision: REJECTED; inactive experiment files removed.
- Reason: no material or repeatable gain over the current one-row/one-warp kernel.

## Experiment 3 — compiled fixed microbatch core plus preallocated output

- Date: 2026-08-31
- Motivation: remove case-6 eager execution overhead, repeated output-list allocation, and final concatenation.
- Bottleneck addressed: 40 eager microbatch invocations, about 1,321 device launches, 280 observed allocator calls, and the final full-output `torch.cat`.
- Implementation retained: compile the fixed B=250 custom model with `torch.compile(mode="reduce-overhead")`; copy each completed slice directly into a preallocated output tensor.
- Affected shape: case 6 only. Non-microbatched dispatch is unchanged.
- Correctness: PASS; maximum absolute error 0.0019486; failed elements 0 / 163,840,000.
- Same-condition case-6 comparison: original eager/list/cat 237.4574 ms; compiled/preallocated 228.1437 ms; incremental speedup 1.041x.
- Repeated full-suite case-6 median: 229.2808 ms.
- GPU observation: preallocation alone was neutral (237.2658 ms); compiling the fixed microbatch core supplied the measurable gain. Direct copies remain necessary because CUDA Graph output storage can be reused on the next slice.
- Decision: RETAINED.
- Full-suite raw logs and CSV: `benchmark_logs/22_case6_compiled_preallocated/`.
- Full-suite outcome: cases 1-13 passed; case 14 remained an expected input-allocation OOM.

## Experiment 4 — fused FFN input GEMM, bias, and exact GELU

- Date: 2026-08-31
- Motivation: remove the separate GELU launch and the global-memory round trip for the pre-activation tensor.
- Bottleneck addressed: FFN projection/activation traffic and four GELU launches per forward.
- Implementation tested: shape-aware Triton Tensor-Core `128x128` GEMM with bias and exact-erf GELU in its FP32 epilogue.
- Affected shapes: FP32 cases with D=F=128.
- Isolated TF32 result at 65,536 rows: PyTorch 0.651264 ms; fused 0.306224 ms (2.127x).
- TF32 correctness: REJECTED; case 13 failed 2 / 8,388,608 elements, maximum absolute error 0.00236905.
- Higher-accuracy `tf32x3` isolated result: PyTorch 0.662336 ms; fused 0.535792 ms (1.236x), maximum isolated error 0.000002.
- `tf32x3` full-model correctness: PASS; maximum absolute error 0.00158155; zero failed elements.
- `tf32x3` case-13 performance: 19.0734 ms versus the retained preceding-sweep result of 18.4238 ms (0.966x; regression).
- Decision: REJECTED; PyTorch FFN restored and inactive experiment files removed.
- Reason: the fast mode violates correctness, while the accurate mode loses end-to-end performance.

## Experiment 5 — attention tile and register-pressure tuning

- Date: 2026-08-31
- Motivation: reduce the custom attention kernel's 96-register footprint and improve shape-specific scheduling.
- Bottleneck addressed: attention occupancy and tile efficiency.
- Implementation tested: added BLOCK_M=32 candidates plus BLOCK_N 16/32/64, 2-4 stages, existing 64/128-row candidates, and 4/8-warp configurations.
- Affected shapes: custom attention at sequence lengths 128 and 1024.
- Correctness: all isolated variants produced zero failed elements under the official rule.
- Register-reduction result: every BLOCK_M=32 long-sequence configuration was about 2.83-3.04 ms, versus 1.458 ms for the retained 64x32/4-warp/2-stage kernel.
- B128/S128 64-key candidate: isolated measurements appeared favorable, but full case 5 regressed from 1.8772 ms to 2.0398 ms.
- B64/S128 three-stage candidate: full case 1 measured 0.9836 ms versus the preceding-sweep 0.9217 ms.
- Long-sequence winner: the retained 64x32/4-warp/2-stage configuration remained best at 1.458112 ms in the expanded run.
- Decision: REJECTED; runtime dispatch restored. The expanded offline tuner is retained for reproducibility.
- Reason: smaller accumulator tiles lose too much reuse, and apparent isolated short-shape winners did not improve the full compiled Transformer.

## Experiment 6 — vendor GEMM algorithm/compiler-mode dispatch

- Date: 2026-08-31
- Motivation: improve the dominant QKV, output, and FFN GEMMs without replacing optimized vendor kernels.
- Bottleneck addressed: GEMMs, especially the D1024 case.
- Implementation tested: current `reduce-overhead` dispatch versus `max-autotune` on case 8, using 20 repeats and two benchmark rounds.
- Correctness: both modes PASS; maximum absolute error 0.00104547; zero failed elements.
- `max-autotune`: 37.5770 ms.
- `reduce-overhead`: 37.4218 ms.
- Relative result: 0.996x for max-autotune; existing mode is 0.4% faster, within noise.
- GPU observation: PyTorch already dispatches CUTLASS Tensor-Core kernels, and its earlier diagnostic reports that this 26-SM GPU is below the threshold for the intended max-autotune GEMM search.
- Decision: REJECTED; keep `reduce-overhead`.
- Reason: no measured improvement and no local CUDA/CUTLASS toolchain for a credible lower-level vendor-algorithm search.

## Final retained state

- Retained change: compiled fixed B=250 custom microbatch core with direct copies into a preallocated final output.
- Runtime kernel dispatch for cases 1-5 and 7-13 remains unchanged from the prior custom-kernel version.
- Expanded attention tuner retained as a diagnostic tool; it does not affect model execution.
- Full validation: `benchmark_logs/22_case6_compiled_preallocated/`.
- Cases 1-13: PASS.
- Case 14: expected input-allocation OOM before model execution.

## Experiment 7 — hybrid FP16 projections with FP32-sensitive operations

- Date: 2026-08-31
- Reference contract: original baseline remains strict IEEE FP32 with TF32 disabled and matmul precision `highest`.
- Mixed policy: FP16 projection operands/weights; FP32 accumulation, exact GELU, residual stream, LayerNorm, softmax statistics, and returned output.
- Case-8 custom implementation: Triton packed-QKV, FP32-output projections, and fused FFN-input-plus-exact-GELU kernels.
- Case-8 controlled result: vendor mixed 25.7359 ms; custom Triton mixed 19.7395 ms; incremental speedup 1.304x.
- Case-8 correctness: three trials PASS; maximum absolute error 0.00120211; failed 0 / 25,165,824.
- Full-suite outcome: cases 1-13 PASS with zero failed elements; case 14 remains an input-allocation OOM.
- Decision: RETAINED as a separate mixed-precision candidate.
- Complete report: `benchmark_logs/25_mixed_precision_results.md`.
- Raw logs and CSV: `benchmark_logs/24_mixed_fp16_fp32_strict_baseline/`.

## Experiment 8 — shape-dispatched D128 Triton projections

- Date: 2026-08-31
- Motivation: extend the successful mixed Tensor-Core projection strategy to the D128 cases while retaining FP32-sensitive computation.
- Implementation retained: tuned Triton packed-QKV, FP32-output projection, fused FFN-input-plus-exact-FP32-GELU, and FFN-output kernels for cases 1, 4, 5, and 9-13.
- Attention policy: reuse custom FP32 causal attention for cases 1, 4, 5, and 13; retain PyTorch fused FP16 SDPA for cases 9-12.
- Conservative fallbacks: cases 2, 3, 6, and 7 remain on vendor mixed projections; case 8 retains its existing D1024 Triton path.
- Case-6 rejection: custom D128 route failed 1 / 163,840,000 values in one of three trials (max abs 0.00201623); vendor mixed passed 0 / 491,520,000 (max abs 0.00193438).
- Controlled retained-path validation: cases 1, 4, 5, and 9-13 each passed three randomized accuracy trials with zero failed elements.
- Official full-suite result: cases 1-13 PASS with zero failed elements; case 14 remains the expected input-allocation OOM.
- Largest single-round incremental improvements versus the preceding mixed suite: case 10 1.365x, case 5 1.292x, case 13 1.272x, and case 1 1.205x.
- Decision: RETAINED with exact-shape dispatch and conservative fallback.
- Complete report: `benchmark_logs/27_mixed_d128_dispatch_results.md`.
- Raw logs and CSV: `benchmark_logs/26_mixed_d128_shape_dispatch/`.

## Experiment 9 — FP16 SDPA backend dispatch

- Date: 2026-08-31
- Previously skipped gap: the earlier backend comparison tested FP32 only.
- Retained: cuDNN FP16 SDPA for cases 9 and 13.
- Rejected: cases 1, 4, 5, 10, and 11 lost in isolated tests; case 12 won in isolation but regressed end to end from 0.2039 ms to 0.2635 ms.
- Correctness: retained paths passed three full-model trials with zero failed elements.
- Decision: RETAIN exact-shape dispatch for cases 9 and 13.

## Experiment 10 — mixed-specific case-6 microbatching

- Date: 2026-08-31
- Retained: strict baseline B250; mixed candidate B64 with direct output copies.
- Controlled candidate result: 150.4044 ms versus the preceding 182.5907 ms.
- Correctness: three trials PASS; maximum absolute error 0.00193438; failed 0 / 491,520,000.
- Decision: RETAIN independent candidate execution schedule.

## Experiment 11 — expanded D128 projection configuration search

- Date: 2026-08-31
- Added search coverage: actual FP16 QKV output, BLOCK_K=64, two-warp tiles, and GROUP_M=4.
- Retained: 2,048-row configurations for cases 4 and 12.
- Rejected: broader replacements for the 8,192-, 16,384-, and 65,536-row families because isolated gains did not transfer reliably to the compiled full model.
- Decision: RETAIN only exact 2,048-row dispatch.

## Experiment 12 — fused D128 FFN output, residual, and LayerNorm

- Date: 2026-08-31
- Implementation: one Triton kernel performs FP16 Tensor-Core FFN output projection with FP32 accumulation, residual addition, FP32 LayerNorm statistics, and dual residual/normalized stores.
- Memory effect: removes the intermediate FP32 FFN-output global write and read.
- Launch effect: replaces the FFN-output projection plus residual/LayerNorm pair with one kernel.
- Retained shapes: cases 4, 5, 12, and 13.
- Rejected shape family: 8,192 rows (cases 1 and 9-11) because it regressed end-to-end despite an isolated win.
- Controlled candidate medians: case 4 0.1880 ms; case 5 1.4415 ms; case 12 0.1930 ms; case 13 12.0212 ms.
- Correctness: all retained cases passed three trials with zero failed elements.
- Official suite: cases 1-13 PASS; case 14 remains input-allocation OOM.
- Complete report: `benchmark_logs/29_skipped_techniques_results.md`.
- Raw logs and CSV: `benchmark_logs/28_skipped_techniques_retained/`.

## Experiment 13 — controlled per-case route tournament

- Goal: verify whether historical per-case minima can be safely merged into
  the current mixed candidate under one consistent, strict no-TF32 protocol.
- Method: same-process incumbent/challenger comparisons on the same input, five
  alternating timing rounds, 30 repeats per round, 10 warmups, and three
  accuracy trials.
- Case 3 FP32 custom-attention route: REJECTED. It measured 0.280896 ms versus
  0.152096 ms for the current mixed route. The earlier 0.1299 ms custom result
  used TF32 and was therefore not comparable to the strict mixed suite.
- Case 11 expanded D128 projection schedule: RETAINED. It measured 1.224176 ms
  versus 1.267904 ms for the incumbent, a 1.0357x gain, and won every timing
  round with identical recorded accuracy metrics.
- Retained production validation: 1.1569 ms over three timing rounds, 12.611x
  versus strict FP32, zero failed elements out of 3,145,728.
- Complete official suite: cases 1-13 PASS with zero failed elements; case 14
  remains an input-allocation OOM. Case 11 measured 1.2453 ms and 11.645x in
  that single-round sweep.
- Complete report: `benchmark_logs/32_controlled_tournament_results.md`.
- Tournament raw logs: `benchmark_logs/30_controlled_candidate_tournament/`.
- Full-suite raw logs: `benchmark_logs/31_case11_tournament_winner/`.

## Latest implementation validation

- Official one-trial/one-round suite rerun after retaining the case-11 route.
- Cases 1-13: PASS with zero failed elements.
- Case 14: input-allocation OOM before model execution.
- Exact script outputs: `benchmark_logs/34_latest_implementation_test_results.md`.
- Raw logs and CSV: `benchmark_logs/33_latest_implementation_tests/`.
- Strict baseline SHA256 remained unchanged.
