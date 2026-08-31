"""Diagnostic FP16 causal attention kernel; production code is untouched."""

from __future__ import annotations

import argparse
import math
import statistics

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fp16_attention(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_os, stride_oh, stride_od,
    num_heads: tl.constexpr, seq_len: tl.constexpr, head_dim: tl.constexpr,
    scale_log2, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    scale_log2 = scale_log2.to(tl.float32)
    qb = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // num_heads
    head = bh - batch * num_heads
    m = qb * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, BLOCK_D)
    d_mask = d < head_dim
    m_mask = m < seq_len
    q = tl.load(
        q_ptr + batch * stride_qb + head * stride_qh + m[:, None] * stride_qs + d[None, :] * stride_qd,
        mask=m_mask[:, None] & d_mask[None, :], other=0.0,
    )
    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    causal_end = tl.minimum((qb + 1) * BLOCK_M, seq_len)
    for start_n in range(0, causal_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n = start_n + tl.arange(0, BLOCK_N)
        n_mask = n < seq_len
        k = tl.load(
            k_ptr + batch * stride_kb + head * stride_kh + n[:, None] * stride_ks + d[None, :] * stride_kd,
            mask=n_mask[:, None] & d_mask[None, :], other=0.0,
        )
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale_log2
        scores = tl.where((m[:, None] >= n[None, :]) & n_mask[None, :] & m_mask[:, None], scores, -float("inf"))
        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, tile_max)
        p = tl.exp2(scores - new_max[:, None])
        alpha = tl.exp2(row_max - new_max)
        acc *= alpha[:, None]
        v = tl.load(
            v_ptr + batch * stride_vb + head * stride_vh + n[:, None] * stride_vs + d[None, :] * stride_vd,
            mask=n_mask[:, None] & d_mask[None, :], other=0.0,
        )
        acc = tl.dot(p.to(tl.float16), v, acc, out_dtype=tl.float32)
        row_sum = row_sum * alpha + tl.sum(p, axis=1)
        row_max = new_max
    out = acc / row_sum[:, None]
    tl.store(
        out_ptr + batch * stride_ob + m[:, None] * stride_os + head * stride_oh + d[None, :] * stride_od,
        out, mask=m_mask[:, None] & d_mask[None, :],
    )


def launch(q, k, v, scale, bm, bn, warps, stages):
    b, h, s, d = q.shape
    out = torch.empty((b, s, h, d), device=q.device, dtype=torch.float16)
    _fp16_attention[(triton.cdiv(s, bm), b * h)](
        q, k, v, out,
        *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        num_heads=h, seq_len=s, head_dim=d,
        scale_log2=scale / math.log(2.0), BLOCK_M=bm, BLOCK_N=bn,
        BLOCK_D=max(16, triton.next_power_of_2(d)),
        num_warps=warps, num_stages=stages,
    )
    return out


def timed(fn, warmup=5, repeats=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for i in range(repeats): starts[i].record(); fn(); ends[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--seq", type=int, choices=(128, 1024), required=True)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    args = parser.parse_args()
    torch.manual_seed(1234)
    packed = torch.randn(args.batch, args.seq, 3, args.heads, args.head_dim, device="cuda", dtype=torch.float16)
    q, k, v = packed.permute(2, 0, 3, 1, 4).unbind(0)
    scale = args.head_dim**-0.5
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale).transpose(1, 2).contiguous()
    print(f"batch={args.batch} seq={args.seq} vendor={timed(lambda: F.scaled_dot_product_attention(q,k,v,is_causal=True,scale=scale)):.6f}")
    for c in (
        (32, 16, 2, 2), (32, 32, 2, 2), (32, 32, 4, 2),
        (64, 16, 2, 2), (64, 32, 2, 2), (64, 32, 4, 2),
        (64, 32, 4, 3), (64, 64, 4, 2),
        (128, 16, 4, 2), (128, 32, 4, 2), (128, 64, 4, 2),
    ):
        try:
            fn = lambda c=c: launch(q, k, v, scale, *c)
            out = fn(); err = (out.float() - ref.float()).abs()
            failed = ((err > .002) & (err > .02 * ref.float().abs())).sum().item()
            print(f"config={c} ms={timed(fn):.6f} max_abs={err.max().item():.7f} failed={failed}")
        except Exception as exc:
            print(f"config={c} ERROR={exc}")


if __name__ == "__main__": main()
