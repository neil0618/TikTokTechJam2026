"""Isolated tuner for the D128 mixed projection shape families."""

from __future__ import annotations

import argparse
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F

from .triton_linear import mixed_linear


CONFIGS = (
    (32, 32, 32, 2, 2, 4),
    (32, 64, 32, 4, 2, 4),
    (64, 32, 32, 2, 2, 4),
    (64, 64, 32, 4, 2, 4),
    (64, 128, 32, 4, 2, 4),
    (128, 32, 32, 4, 2, 4),
    (128, 64, 32, 4, 2, 4),
    (128, 128, 32, 8, 2, 4),
    (32, 64, 64, 4, 2, 8),
    (64, 64, 64, 4, 2, 8),
    (64, 128, 64, 4, 2, 8),
    (128, 64, 64, 4, 2, 8),
)


def timed_ms(fn, warmup: int = 5, repeats: int = 30) -> float:
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


def measure(name, reference_fn, kernel_factory) -> None:
    reference = reference_fn()
    print(f"{name} torch={timed_ms(reference_fn):.6f} ms")
    print("BM BN BK warps stages group | median_ms | max_abs | failed")
    for config in CONFIGS:
        bm, bn, bk, warps, stages, group = config
        try:
            fn = kernel_factory(config)
            output = fn()
            error = (output.float() - reference.float()).abs()
            failed = int(
                ((error > 0.002) & (error > 0.02 * reference.float().abs())).sum().item()
            )
            print(
                f"{bm:3d} {bn:3d} {bk:2d} {warps:5d} {stages:6d} {group:5d} | "
                f"{timed_ms(fn):9.6f} | {error.max().item():7.6f} | {failed}"
            )
        except Exception as error:
            print(f"{bm:3d} {bn:3d} {bk:2d} {warps:5d} {stages:6d} {group:5d} | ERROR: {error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    args = parser.parse_args()
    torch.manual_seed(1234)
    x_fp32 = torch.randn(args.rows, 128, device="cuda", dtype=torch.float32)
    x_fp16 = x_fp32.to(torch.float16)
    regular = nn.Linear(128, 128, device="cuda", dtype=torch.float16).eval()
    qkv = nn.Linear(128, 384, device="cuda", dtype=torch.float16).eval()

    measure(
        "qkv_fp16_out",
        lambda: qkv(x_fp32.to(torch.float16)),
        lambda c: lambda: mixed_linear(
            x_fp32,
            qkv,
            output_dtype=torch.float16,
            block_m=c[0], block_n=c[1], block_k=c[2],
            num_warps=c[3], num_stages=c[4], group_m=c[5],
        ),
    )
    measure(
        "ffn_in_gelu_fp16_out",
        lambda: F.gelu(
            regular(x_fp32.to(torch.float16)).to(torch.float32),
            approximate="none",
        ).to(torch.float16),
        lambda c: lambda: mixed_linear(
            x_fp32,
            regular,
            output_dtype=torch.float16,
            fuse_gelu=True,
            block_m=c[0], block_n=c[1], block_k=c[2],
            num_warps=c[3], num_stages=c[4], group_m=c[5],
        ),
    )
    measure(
        "projection_fp32_out",
        lambda: regular(x_fp16).to(torch.float32),
        lambda c: lambda: mixed_linear(
            x_fp16,
            regular,
            output_dtype=torch.float32,
            block_m=c[0], block_n=c[1], block_k=c[2],
            num_warps=c[3], num_stages=c[4], group_m=c[5],
        ),
    )


if __name__ == "__main__":
    main()
