"""Isolated case-8 tuner for custom mixed-precision projection kernels."""

from __future__ import annotations

import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F

from .triton_linear import mixed_linear


CONFIGS = (
    (64, 64, 32, 4, 3),
    (64, 128, 32, 4, 3),
    (128, 64, 32, 4, 3),
    (128, 128, 32, 8, 3),
    (64, 64, 64, 4, 3),
    (64, 128, 64, 4, 3),
)


def timed_ms(fn, warmup: int = 5, repeats: int = 20) -> float:
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


def run_projection(name: str, x: torch.Tensor, linear: nn.Linear, output_dtype: torch.dtype, fuse_gelu: bool) -> None:
    if fuse_gelu:
        reference_fn = lambda: F.gelu(linear(x.to(torch.float16)).to(torch.float32), approximate="none").to(output_dtype)
    else:
        reference_fn = lambda: linear(x.to(torch.float16)).to(output_dtype)
    reference = reference_fn()
    print(f"{name} torch={timed_ms(reference_fn):.6f} ms")
    print("BM BN BK warps stages | median_ms | max_abs | failed")
    for bm, bn, bk, warps, stages in CONFIGS:
        try:
            fn = lambda: mixed_linear(
                x,
                linear,
                output_dtype=output_dtype,
                fuse_gelu=fuse_gelu,
                block_m=bm,
                block_n=bn,
                block_k=bk,
                num_warps=warps,
                num_stages=stages,
            )
            output = fn()
            error = (output.float() - reference.float()).abs()
            failed = int(
                ((error > 0.002) & (error > 0.02 * reference.float().abs())).sum().item()
            )
            print(
                f"{bm:2d} {bn:3d} {bk:2d} {warps:5d} {stages:6d} | "
                f"{timed_ms(fn):9.6f} | {error.max().item():7.6f} | {failed}"
            )
        except Exception as error:
            print(f"{bm:2d} {bn:3d} {bk:2d} {warps:5d} {stages:6d} | ERROR: {error}")


def main() -> None:
    torch.manual_seed(1234)
    rows = 64 * 128
    x_fp32 = torch.randn(rows, 1024, device="cuda", dtype=torch.float32)
    x_fp16 = x_fp32.to(torch.float16)
    regular = nn.Linear(1024, 1024, device="cuda", dtype=torch.float16).eval()
    qkv = nn.Linear(1024, 3072, device="cuda", dtype=torch.float16).eval()
    run_projection("qkv_fp16_out", x_fp32, qkv, torch.float16, False)
    run_projection("ffn_in_gelu_fp16_out", x_fp32, regular, torch.float16, True)
    run_projection("projection_fp32_out", x_fp16, regular, torch.float32, False)


if __name__ == "__main__":
    main()
