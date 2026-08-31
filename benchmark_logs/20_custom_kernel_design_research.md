# Custom-kernel design research (no implementation)

## Decision

Build one shape-dispatched, forward-only **causal FlashAttention kernel specialized
for NVIDIA SM120 and float32 inputs**, using CUDA C++ with CUTLASS/CuTe building
blocks. Target case 13 (`B=64, H=4, S=1024, head_dim=32`) first. Keep the current
PyTorch SDPA kernel as the fallback for every unsupported or slower shape.

This is the best first custom kernel because attention accounts for 72.8% of the
profiled CUDA time in case 13. By contrast, attention is only 10.0% of case 8,
where vendor GEMMs dominate, and 41.9% of case 1. A whole Transformer-block
kernel is not recommended: the projections mix all model dimensions, attention
mixes sequence positions independently per head, and LayerNorm performs row
reductions. Their incompatible tilings and cross-tile dependencies would force
large intermediate storage or duplicated work.

## Proposed kernel

### Interface and scope

- Inference/forward only; no backward pass.
- Float32 Q, K, and V with FP32 softmax statistics and FP32 output.
- Causal self-attention, initially all-valid tokens.
- Directly consume the existing packed-QKV strided views; do not make Q/K/V
  contiguous copies.
- Directly write the output layout expected by the output projection.
- Register as a PyTorch CUDA custom operator with a FakeTensor implementation so
  it remains compatible with `torch.compile` and CUDA Graph capture.
- Dispatch only on validated shape families. Fall back to current SDPA otherwise.

### Computation

For each `(batch, head, query tile)` CTA:

1. Load one Q tile and stream causal K/V tiles through shared memory.
2. Compute a QK tile without ever writing the full `S x S` score matrix.
3. Apply scale and the diagonal causal mask.
4. Update per-row online-softmax maximum and normalization sum in FP32.
5. Rescale the existing output accumulator and accumulate `P @ V` in FP32.
6. Normalize once and store the final attention output.

This preserves FlashAttention's key IO advantage: Q is loaded once per query
tile, K/V are streamed, and scores/probabilities remain on chip.

### SM120-specific schedule

- Start with `BLOCK_M x BLOCK_N` candidates `64x64`, `64x128`, and `128x64`.
- Compile independent head-dimension variants for 8, 32, 64, and 128; optimize
  `head_dim=32` first.
- Use coalesced 16-byte global transactions where alignment permits.
- Double-buffer K/V shared-memory stages and overlap loading with arithmetic.
- Test four versus eight warps, stage count, and shared-memory carveout. SM120 has
  48 resident warps per SM, 128 KB shared memory per SM, and a 99 KB per-block
  limit, so Hopper/SM100 launch configurations must not be copied blindly.
- Use causal tile pruning: do not launch or iterate over future key tiles; only
  the diagonal tile needs element-wise causal masking.
- Try the Tensor Core TF32 path with FP32 accumulation, but retain an FP32-FMA
  variant. The TF32 variant is accepted only if it passes the benchmark's exact
  correctness gate across all accuracy trials.
- Use a persistent tile scheduler only if measurements show that conventional
  CTA scheduling underutilizes this 26-SM GPU. Large batch/head grids may already
  provide enough parallelism.

## Why CUDA C++ plus CUTLASS/CuTe

CUTLASS/CuTe provides the tensor layouts, tiled copies, MMA wrappers, and pipeline
primitives while still allowing an architecture-specific kernel and explicit
schedule. This is safer than hand-writing PTX and gives finer control than the
current high-level SDPA call. NVIDIA explicitly recommends CUTLASS for complex
Tensor Core GEMM work.

Triton remains useful for a fast prototype and tile experiments, and its current
Blackwell tutorial demonstrates TMA, persistent scheduling, warp specialization,
and epilogue subtiling. It is not the recommended final route here because the
ready-made Blackwell attention implementations and FA4 techniques target SM100
B200/GB200, whereas this GPU is consumer SM120. We need control over the actual
SM120 instruction and resource path, plus SASS/resource inspection.

## Expected benefit and go/no-go threshold

Using case 13's 72.8% attention share and Amdahl's law:

| Custom attention speedup | Approx. case-13 end-to-end speedup | Approx. new time from 32.71 ms |
|---:|---:|---:|
| 1.15x | 1.10x | 29.6 ms |
| 1.25x | 1.17x | 28.0 ms |
| 1.40x | 1.26x | 25.9 ms |

The initial success threshold should be **at least 1.15x end-to-end on case 13**
with no correctness regression. Below that, the engineering cost is unlikely to
be worthwhile. A 1.20-1.30x case-13 gain is a credible stretch goal, not a
promise. Case 8 will barely move from this kernel because attention is only 10%
of its time.

## Implementation phases (when authorized)

1. Create a correctness-first tiled online-softmax kernel for case 13.
2. Add an isolated microbenchmark against current SDPA and the reference output.
3. Add causal tile pruning and tune tile/warp/stage configurations.
4. Evaluate TF32 Tensor Cores versus FP32 FMA under the benchmark error limit.
5. Add direct packed-QKV loads and output-layout stores.
6. Integrate through a PyTorch custom op and verify `torch.compile`/CUDA Graphs.
7. Run the full suite, enabling the custom path only where it wins and passes.
8. Only then add head-dimension 8/64/128 or short-sequence variants.

Each phase must log kernel time, end-to-end time, max/mean error, registers per
thread, shared memory per CTA, achieved occupancy, memory throughput, Tensor Core
utilization, and launch count.

## Designs rejected as the first step

- **One persistent kernel for an entire Transformer block:** intermediate tensors
  do not fit on chip and the operations need incompatible parallel decompositions.
- **Custom D=1024 GEMMs first:** those kernels consume most of case 8, but CUTLASS
  GEMMs are already highly optimized. Beating them in float32/TF32 is lower
  probability than specializing the currently generic SM80-named attention path.
- **Only fuse GELU into the FFN GEMM:** safe but GELU is about 1.8-2.3% of profiled
  time, so the ceiling is small.
- **Port FlashAttention-4 unchanged:** FA4 is designed and measured on B200/GB200
  SM100-class hardware, not this SM120 GeForce GPU.
- **Support case 14 with dense attention:** its float32 input alone needs about
  12.21 GiB on an 8 GiB GPU, and dense `S=100000` attention remains quadratic.
  Supporting it requires changing the algorithm/workload contract, not merely a
  faster kernel.

## Primary references

- NVIDIA Blackwell Tuning Guide (CUDA 13.0):
  https://docs.nvidia.com/cuda/archive/13.0.2/pdf/Blackwell_Tuning_Guide.pdf
- NVIDIA CUTLASS Blackwell/SM120 functionality:
  https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html
- NVIDIA CUTLASS Blackwell FMHA example (SM100 reference, not an SM120 drop-in):
  https://github.com/NVIDIA/cutlass/blob/main/examples/77_blackwell_fmha/77_blackwell_fmha.cu
- FlashAttention-3 paper:
  https://arxiv.org/abs/2407.08608
- FlashAttention-4 paper:
  https://arxiv.org/abs/2603.05451
- Triton persistent matmul tutorial:
  https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
- PyTorch custom C++/CUDA operator tutorial:
  https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
