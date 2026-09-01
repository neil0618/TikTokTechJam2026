"""Standalone tile probe for fused D128 projection + residual + LayerNorm."""

from __future__ import annotations

import argparse
import statistics

import torch
import torch.nn as nn

from mixed_precision.triton_linear import mixed_linear_add_layer_norm


CONFIGS = (
    (8, 32, 4, 2),
    (16, 32, 4, 1), (16, 32, 4, 2), (16, 32, 4, 3),
    (16, 64, 4, 1), (16, 64, 4, 2), (16, 64, 4, 3),
    (16, 128, 4, 2),
    (32, 16, 4, 2),
    (32, 32, 4, 1), (32, 32, 4, 2), (32, 32, 4, 3),
    (32, 32, 8, 1), (32, 32, 8, 2), (32, 32, 8, 3),
    (32, 64, 4, 2), (32, 64, 8, 2),
    (32, 128, 8, 2),
    (64, 32, 8, 2), (64, 64, 8, 2),
)


def timed_ms(fn, warmup=5, repeats=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for i in range(repeats):
        starts[i].record(); fn(); ends[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends))


def controlled_tournament(x, linear, residual, norm, rows):
    choices = {
        2048: ((32, 32, 8, 2), (32, 16, 4, 2), (32, 128, 8, 2), (16, 64, 4, 3)),
        8192: ((32, 32, 8, 2), (32, 16, 4, 2), (32, 32, 8, 1), (16, 64, 4, 2)),
        16384: ((32, 32, 4, 2), (32, 32, 4, 1), (32, 32, 4, 3), (32, 64, 4, 2)),
        65536: ((16, 64, 4, 2), (16, 128, 4, 2), (32, 128, 8, 2)),
    }[rows]
    fns = {
        c: lambda c=c: mixed_linear_add_layer_norm(
            x, linear, residual, norm,
            block_m=c[0], block_k=c[1], num_warps=c[2], num_stages=c[3],
        )
        for c in choices
    }
    for fn in fns.values():
        for _ in range(10): fn()
    torch.cuda.synchronize()
    results = {c: [] for c in choices}
    for round_index in range(7):
        order = choices if round_index % 2 == 0 else tuple(reversed(choices))
        for c in order:
            results[c].append(timed_ms(fns[c], warmup=3, repeats=100))
    print("\ncontrolled tournament")
    for c in choices:
        print(f"config={c} rounds={results[c]} overall={statistics.median(results[c]):.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    args = parser.parse_args()
    torch.manual_seed(1234)
    x = torch.randn(args.rows, 128, device="cuda", dtype=torch.float16)
    residual = torch.randn(args.rows, 128, device="cuda", dtype=torch.float32)
    linear = nn.Linear(128, 128, device="cuda", dtype=torch.float16).eval()
    norm = nn.LayerNorm(128, device="cuda", dtype=torch.float32).eval()
    reference_residual = linear(x).float() + residual
    reference_norm = norm(reference_residual)
    print(f"rows={args.rows}")
    print("BM BK W S | median_ms | max_abs_res | max_abs_norm | failed")
    for bm, bk, warps, stages in CONFIGS:
        try:
            fn = lambda: mixed_linear_add_layer_norm(
                x, linear, residual, norm,
                block_m=bm, block_k=bk,
                num_warps=warps, num_stages=stages,
            )
            out_res, out_norm = fn()
            er = (out_res - reference_residual).abs()
            en = (out_norm - reference_norm).abs()
            failed = ((en > 0.002) & (en > 0.02 * reference_norm.abs())).sum().item()
            print(
                f"{bm:2d} {bk:3d} {warps:1d} {stages:1d} | {timed_ms(fn):9.6f} | "
                f"{er.max().item():11.7f} | {en.max().item():11.7f} | {failed}"
            )
        except Exception as exc:
            print(f"{bm:2d} {bk:3d} {warps:1d} {stages:1d} | ERROR: {exc}")
    controlled_tournament(x, linear, residual, norm, args.rows)


if __name__ == "__main__":
    main()
