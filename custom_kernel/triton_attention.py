"""Forward-only causal attention kernels written specifically for this benchmark.

The kernel implements tiled online softmax and never materializes the S x S
attention matrix. Unsupported shapes deliberately fall back in the model layer.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _causal_attention_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_os,
    stride_oh,
    stride_od,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    scale_log2,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    # torch.compile represents Python float arguments as fp64; force the
    # arithmetic path back to FP32 so both tl.dot accumulators remain FP32.
    scale_log2 = scale_log2.to(tl.float32)
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head - batch * num_heads

    query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    feature_offsets = tl.arange(0, head_dim)
    query_mask = query_offsets < seq_len
    q_ptrs = (
        q_ptr
        + batch * stride_qb
        + head * stride_qh
        + query_offsets[:, None] * stride_qs
        + feature_offsets[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=query_mask[:, None], other=0.0)

    running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, head_dim), tl.float32)

    # Causal pruning: this CTA never visits key tiles strictly to its right.
    causal_end = tl.minimum((query_block + 1) * BLOCK_M, seq_len)
    for key_start in range(0, causal_end, BLOCK_N):
        key_start = tl.multiple_of(key_start, BLOCK_N)
        key_offsets = key_start + tl.arange(0, BLOCK_N)
        key_mask = key_offsets < seq_len

        k_ptrs = (
            k_ptr
            + batch * stride_kb
            + head * stride_kh
            + key_offsets[:, None] * stride_ks
            + feature_offsets[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=key_mask[:, None], other=0.0)
        scores = tl.dot(q, tl.trans(k), input_precision=DOT_PRECISION)
        causal_mask = query_offsets[:, None] >= key_offsets[None, :]
        scores = scores * scale_log2
        scores = tl.where(causal_mask & key_mask[None, :] & query_mask[:, None], scores, -float("inf"))

        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        probabilities = tl.exp2(scores - new_max[:, None])
        correction = tl.exp2(running_max - new_max)
        tile_sum = tl.sum(probabilities, axis=1)

        accumulator *= correction[:, None]
        v_ptrs = (
            v_ptr
            + batch * stride_vb
            + head * stride_vh
            + key_offsets[:, None] * stride_vs
            + feature_offsets[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=key_mask[:, None], other=0.0)
        accumulator = tl.dot(
            probabilities.to(tl.float32),
            v,
            accumulator,
            input_precision=DOT_PRECISION,
        )
        running_sum = running_sum * correction + tile_sum
        running_max = new_max

    output = accumulator / running_sum[:, None]
    # Store B,S,H,D so the following output projection needs no transpose/copy.
    out_ptrs = (
        out_ptr
        + batch * stride_ob
        + query_offsets[:, None] * stride_os
        + head * stride_oh
        + feature_offsets[None, :] * stride_od
    )
    tl.store(out_ptrs, output, mask=query_mask[:, None])


def is_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> bool:
    """Return whether the specialized kernel supports this exact invocation."""
    return (
        causal
        and q.is_cuda
        and q.dtype == torch.float32
        and q.ndim == 4
        and q.shape == k.shape == v.shape
        and q.shape[-1] == 32
        and q.shape[-2] in (128, 1024)
        and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    )


def launch_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    *,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
    dot_precision: str = "tf32",
) -> torch.Tensor:
    """Launch an explicit configuration, primarily for reproducible tuning."""
    if not is_supported(q, k, v, causal=True):
        raise ValueError(f"unsupported custom-attention input: shape={tuple(q.shape)}, dtype={q.dtype}")

    batch, heads, seq_len, head_dim = q.shape
    output = torch.empty(
        (batch, seq_len, heads, head_dim), device=q.device, dtype=q.dtype
    )

    grid = (triton.cdiv(seq_len, block_m), batch * heads)
    _causal_attention_fwd[grid](
        q,
        k,
        v,
        output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        num_heads=heads,
        seq_len=seq_len,
        head_dim=head_dim,
        scale_log2=scale * (1.0 / math.log(2.0)),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        DOT_PRECISION=dot_precision,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Return causal attention in contiguous B,S,H,D layout.

    Callers must check :func:`is_supported` first.
    """
    if q.shape[-2] >= 1024:
        config = (64, 32, 4, 2)
    else:
        config = (64, 32, 4, 2)
    return launch_causal_attention(
        q,
        k,
        v,
        scale,
        block_m=config[0],
        block_n=config[1],
        num_warps=config[2],
        num_stages=config[3],
    )
