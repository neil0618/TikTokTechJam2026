#!/usr/bin/env python3
"""Isolated D32 mixed-projection screen for benchmark case 7."""

from __future__ import annotations

import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F

from mixed_precision.triton_linear import mixed_linear


ROWS = 64 * 128
CONFIGS = (
    (64, 32, 16, 2, 4),
    (64, 64, 16, 4, 4),
    (128, 32, 16, 4, 8),
    (128, 64, 16, 4, 8),
    (256, 32, 16, 4, 8),
    (64, 32, 32, 2, 4),
    (128, 32, 32, 4, 8),
)


def timed(fn, repeats: int = 500) -> float:
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / repeats)
    return statistics.median(values)


def measure(name, reference_fn, kernel_factory) -> None:
    reference = reference_fn()
    print(f"operation={name} vendor_ms={timed(reference_fn):.6f}")
    for config in CONFIGS:
        bm, bn, bk, warps, group = config
        try:
            fn = kernel_factory(config)
            output = fn()
            error = (output.float() - reference.float()).abs()
            failed = int(
                ((error > 0.002) & (error > 0.02 * reference.float().abs()))
                .sum()
                .item()
            )
            print(
                f"config=BM{bm}_BN{bn}_BK{bk}_W{warps}_G{group} "
                f"median_ms={timed(fn):.6f} max_abs={error.max().item():.8f} "
                f"failed={failed}"
            )
        except Exception as error:
            print(f"config={config} ERROR={error}")


def main() -> None:
    torch.manual_seed(1234)
    x_fp32 = torch.randn(ROWS, 32, device="cuda", dtype=torch.float32)
    x_fp16 = x_fp32.to(torch.float16)
    regular = nn.Linear(32, 32, device="cuda", dtype=torch.float16).eval()
    qkv = nn.Linear(32, 96, device="cuda", dtype=torch.float16).eval()
    measure(
        "qkv_fp16_out",
        lambda: qkv(x_fp32.to(torch.float16)),
        lambda c: lambda: mixed_linear(
            x_fp32,
            qkv,
            output_dtype=torch.float16,
            block_m=c[0],
            block_n=c[1],
            block_k=c[2],
            num_warps=c[3],
            num_stages=2,
            group_m=c[4],
        ),
    )
    measure(
        "ffn_in_exact_gelu_fp16_out",
        lambda: F.gelu(
            regular(x_fp32.to(torch.float16)).to(torch.float32),
            approximate="none",
        ).to(torch.float16),
        lambda c: lambda: mixed_linear(
            x_fp32,
            regular,
            output_dtype=torch.float16,
            fuse_gelu=True,
            block_m=c[0],
            block_n=c[1],
            block_k=c[2],
            num_warps=c[3],
            num_stages=2,
            group_m=c[4],
        ),
    )
    measure(
        "projection_fp32_out",
        lambda: regular(x_fp16).to(torch.float32),
        lambda c: lambda: mixed_linear(
            x_fp16,
            regular,
            output_dtype=torch.float32,
            block_m=c[0],
            block_n=c[1],
            block_k=c[2],
            num_warps=c[3],
            num_stages=2,
            group_m=c[4],
        ),
    )


if __name__ == "__main__":
    main()
