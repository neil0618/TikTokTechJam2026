#!/usr/bin/env python3
"""Grouped microkernel tournament for the launch-bound case-2 shape."""

from __future__ import annotations

import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F

from mixed_precision.triton_linear import (
    mixed_ffn_add_layer_norm,
    mixed_linear,
    mixed_linear_add_layer_norm,
)


def grouped_ms(fn, repeats=2000):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def tournament(name, fns):
    for fn in fns.values():
        for _ in range(20):
            fn()
    torch.cuda.synchronize()
    samples = {key: [] for key in fns}
    keys = tuple(fns)
    for round_index in range(7):
        order = keys if round_index % 2 == 0 else tuple(reversed(keys))
        for key in order:
            samples[key].append(grouped_ms(fns[key]))
    print(f"\n{name}")
    for key, values in samples.items():
        print(
            f"config={key} median_ms={statistics.median(values):.6f} "
            f"mean_ms={statistics.fmean(values):.6f} min_ms={min(values):.6f}"
        )


def main():
    torch.manual_seed(1234)
    device = torch.device("cuda")
    rows = 128
    x32 = torch.randn(rows, 128, device=device, dtype=torch.float32)
    x16 = x32.half()
    residual = torch.randn_like(x32)
    qkv = nn.Linear(128, 384, device=device, dtype=torch.float16).eval()
    linear = nn.Linear(128, 128, device=device, dtype=torch.float16).eval()
    linear2 = nn.Linear(128, 128, device=device, dtype=torch.float16).eval()
    norm = nn.LayerNorm(128, device=device, dtype=torch.float32).eval()

    qkv_configs = (
        (16, 64, 32, 4, 4),
        (32, 64, 32, 4, 4),
        (32, 128, 32, 4, 4),
        (32, 128, 32, 8, 4),
        (64, 64, 32, 4, 2),
        (64, 128, 32, 4, 2),
        (128, 64, 32, 4, 1),
    )
    qkv_fns = {
        c: lambda c=c: mixed_linear(
            x32, qkv, output_dtype=torch.float16,
            block_m=c[0], block_n=c[1], block_k=c[2],
            num_warps=c[3], group_m=c[4], num_stages=2,
        )
        for c in qkv_configs
    }
    reference_qkv = qkv(x16)
    for key, fn in qkv_fns.items():
        error = (fn().float() - reference_qkv.float()).abs()
        print(f"qkv_accuracy config={key} max_abs={error.max().item():.8f}")
    tournament("packed_qkv", qkv_fns)

    projection_configs = (
        (8, 32, 4, 2),
        (16, 32, 4, 1),
        (16, 32, 4, 2),
        (32, 16, 4, 2),
        (32, 32, 4, 1),
        (32, 32, 4, 2),
        (32, 32, 8, 1),
        (32, 64, 4, 2),
        (64, 32, 4, 2),
    )
    projection_fns = {
        c: lambda c=c: mixed_linear_add_layer_norm(
            x16, linear, residual, norm,
            block_m=c[0], block_k=c[1], num_warps=c[2], num_stages=c[3],
            normalized_dtype=torch.float16,
        )
        for c in projection_configs
    }
    tournament("projection_residual_layernorm", projection_fns)

    ffn_configs = (
        (8, 2, 1), (8, 4, 1), (8, 4, 2),
        (16, 2, 1), (16, 4, 1), (16, 4, 2),
        (32, 4, 1), (32, 4, 2), (32, 8, 1),
        (64, 4, 1),
    )
    ffn_fns = {
        c: lambda c=c: mixed_ffn_add_layer_norm(
            x16, linear, linear2, residual, norm,
            block_m=c[0], num_warps=c[1], num_stages=c[2],
            normalized_dtype=torch.float16,
        )
        for c in ffn_configs
    }
    reference_hidden = F.gelu(linear(x16).float(), approximate="none").half()
    reference_residual = linear2(reference_hidden).float() + residual
    reference_norm = norm(reference_residual)
    for key, fn in ffn_fns.items():
        out_residual, out_norm = fn()
        error = (out_norm.float() - reference_norm).abs()
        failed = ((error > .002) & (error > .02 * reference_norm.abs())).sum()
        print(
            f"ffn_accuracy config={key} max_abs={error.max().item():.8f} "
            f"failed={failed.item()}"
        )
    tournament("full_ffn_residual_layernorm", ffn_fns)


if __name__ == "__main__":
    main()
