"""Standalone expanded tile probe for the case-8 mixed GEMMs.

This file is diagnostic only. It imports the production kernel but does not
modify the model or its dispatch policy.
"""

from __future__ import annotations

import statistics

import torch
import torch.nn as nn

from mixed_precision.triton_linear import mixed_linear


CONFIGS = (
    # Current schedules.
    (64, 128, 32, 4, 3, 8),
    (128, 64, 32, 4, 3, 8),
    (64, 128, 32, 4, 3, 1),
    (64, 128, 32, 4, 3, 2),
    (64, 128, 32, 4, 3, 4),
    (64, 128, 32, 4, 3, 16),
    (64, 128, 32, 4, 3, 32),
    (128, 64, 32, 4, 3, 1),
    (128, 64, 32, 4, 3, 2),
    (128, 64, 32, 4, 3, 4),
    (128, 64, 32, 4, 3, 16),
    (128, 64, 32, 4, 3, 32),
    # Same tile sizes, stage/group variants.
    (64, 128, 32, 4, 2, 4),
    (64, 128, 32, 4, 4, 4),
    (64, 128, 64, 4, 2, 4),
    (64, 128, 64, 4, 3, 8),
    (128, 64, 64, 4, 2, 4),
    (128, 64, 64, 4, 3, 8),
    # Wider N tiles reduce repeated activation reads.
    (32, 256, 32, 4, 2, 4),
    (32, 256, 32, 8, 2, 4),
    (32, 256, 64, 8, 2, 4),
    (64, 256, 32, 8, 2, 4),
    # Smaller M can help occupancy when accumulator pressure is high.
    (32, 128, 32, 4, 2, 4),
    (32, 128, 64, 4, 2, 4),
    (32, 64, 64, 4, 2, 4),
    # Larger K blocks test weight/activation pipeline amortization.
    (64, 128, 128, 4, 2, 4),
    (128, 64, 128, 4, 2, 4),
)


def timed_ms(fn, warmup: int = 10, repeats: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for i in range(repeats):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends))


def measure(name, x, linear, output_dtype, fuse_gelu=False):
    print(f"\n{name}")
    print("BM BN BK W S G | median_ms | max_abs | failed")
    reference = linear(x.to(torch.float16)).to(torch.float32)
    if fuse_gelu:
        reference = torch.nn.functional.gelu(reference, approximate="none")
    reference = reference.to(output_dtype)
    for bm, bn, bk, warps, stages, group in CONFIGS:
        try:
            fn = lambda: mixed_linear(
                x,
                linear,
                output_dtype=output_dtype,
                fuse_gelu=fuse_gelu,
                block_m=bm,
                block_n=bn,
                block_k=bk,
                group_m=group,
                num_warps=warps,
                num_stages=stages,
            )
            out = fn()
            err = (out.float() - reference.float()).abs()
            failed = ((err > 0.002) & (err > 0.02 * reference.float().abs())).sum().item()
            print(
                f"{bm:2d} {bn:3d} {bk:3d} {warps:1d} {stages:1d} {group:1d} | "
                f"{timed_ms(fn):9.6f} | {err.max().item():7.6f} | {failed}"
            )
        except Exception as exc:
            print(f"{bm:2d} {bn:3d} {bk:3d} {warps:1d} {stages:1d} {group:1d} | ERROR: {exc}")


def group_tournament(name, x, linear, bm, bn, output_dtype):
    groups = (1, 2, 4, 8, 16, 32)
    samples = {group: [] for group in groups}
    fns = {
        group: lambda group=group: mixed_linear(
            x,
            linear,
            output_dtype=output_dtype,
            block_m=bm,
            block_n=bn,
            block_k=32,
            group_m=group,
            num_warps=4,
            num_stages=3,
        )
        for group in groups
    }
    for fn in fns.values():
        for _ in range(10):
            fn()
    torch.cuda.synchronize()
    for round_index in range(5):
        order = groups if round_index % 2 == 0 else tuple(reversed(groups))
        for group in order:
            samples[group].append(timed_ms(fns[group], warmup=3, repeats=100))
    print(f"\ncontrolled groups: {name}")
    for group in groups:
        print(f"group={group:2d} round_medians={samples[group]} overall={statistics.median(samples[group]):.6f}")


def main():
    torch.manual_seed(1234)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = 64 * 128
    x32 = torch.randn(rows, 1024, device="cuda", dtype=torch.float32)
    x16 = x32.half()
    qkv = nn.Linear(1024, 3072, device="cuda", dtype=torch.float16).eval()
    regular = nn.Linear(1024, 1024, device="cuda", dtype=torch.float16).eval()
    measure("qkv fp32->fp16", x32, qkv, torch.float16)
    measure("ffn-in fp32->gelu->fp16", x32, regular, torch.float16, True)
    measure("output fp16->fp32", x16, regular, torch.float32)
    group_tournament("qkv 64x128", x32, qkv, 64, 128, torch.float16)
    group_tournament("output 128x64", x16, regular, 128, 64, torch.float32)


if __name__ == "__main__":
    main()
