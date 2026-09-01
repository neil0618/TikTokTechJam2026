"""Reproducible isolated tuner for the custom causal-attention kernel."""

from __future__ import annotations

import argparse
import statistics

import torch
import torch.nn.functional as F

from .triton_attention import launch_causal_attention


CONFIGS = [
    (32, 16, 4, 2),
    (32, 16, 4, 3),
    (32, 32, 4, 2),
    (32, 32, 4, 3),
    (32, 32, 4, 4),
    (32, 64, 4, 2),
    (64, 32, 4, 2),
    (64, 32, 4, 3),
    (64, 32, 4, 4),
    (64, 32, 8, 2),
    (64, 64, 4, 2),
    (64, 64, 4, 3),
    (64, 64, 8, 2),
    (128, 32, 4, 2),
    (128, 32, 8, 2),
    (128, 64, 4, 2),
    (128, 64, 8, 2),
]


def timed_ms(fn, warmup: int = 3, repeats: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=1024, choices=(128, 1024))
    args = parser.parse_args()
    torch.manual_seed(1234)
    batch, heads, seq_len, head_dim = args.batch_size, 4, args.seq_len, 32
    packed = torch.randn(
        batch, seq_len, 3, heads, head_dim, device="cuda", dtype=torch.float32
    )
    q, k, v = packed.permute(2, 0, 3, 1, 4).unbind(0)
    scale = head_dim**-0.5
    reference = F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=True, scale=scale
    )
    sdpa_ms = timed_ms(
        lambda: F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True, scale=scale
        )
    )
    print(f"sdpa median: {sdpa_ms:.6f} ms")
    print("BM BN warps stages | median_ms | max_abs | failed")
    for bm, bn, warps, stages in CONFIGS:
        try:
            fn = lambda: launch_causal_attention(
                q,
                k,
                v,
                scale,
                block_m=bm,
                block_n=bn,
                num_warps=warps,
                num_stages=stages,
            )
            output = fn().permute(0, 2, 1, 3)
            error = (output - reference).abs()
            failed = int(
                ((error > 0.002) & (error > 0.02 * reference.abs())).sum().item()
            )
            median = timed_ms(fn)
            print(
                f"{bm:3d} {bn:3d} {warps:5d} {stages:6d} | "
                f"{median:9.6f} | {error.max().item():7.6f} | {failed}"
            )
        except Exception as error:
            print(f"{bm:3d} {bn:3d} {warps:5d} {stages:6d} | ERROR: {error}")


if __name__ == "__main__":
    main()
