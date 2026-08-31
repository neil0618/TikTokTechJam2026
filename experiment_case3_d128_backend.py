#!/usr/bin/env python3
"""Controlled case-3 review of D128 projection and SDPA backend dispatch."""

from __future__ import annotations

import statistics

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

import torch_transformer_benchmark as bench
from mixed_precision import MixedPrecisionTransformer, copy_mixed_model_weights


CONFIG = bench.TransformerConfig(4, 128, 128, 4, 128, 4, True)


def force_d128_projection_route(model: MixedPrecisionTransformer) -> None:
    """Enable existing D128 projections while keeping FP16 vendor attention."""
    model.use_d128_triton = True
    model.use_d128_custom_attention = False
    for layer in model.layers:
        layer.use_d128_triton = True
        layer.attention.use_d128_triton = True
        layer.attention.use_d128_custom_attention = False


class CudnnAttentionWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, x: torch.Tensor, valid_token_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return self.model(x, valid_token_mask)


def prepare(device: torch.device):
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    reference = bench.BaselineTransformer(CONFIG).to(device).eval()
    vendor = MixedPrecisionTransformer(CONFIG).to(device).eval()
    efficient = MixedPrecisionTransformer(CONFIG).to(device).eval()
    cudnn = MixedPrecisionTransformer(CONFIG).to(device).eval()
    cudnn_full_ffn = MixedPrecisionTransformer(CONFIG).to(device).eval()
    copy_mixed_model_weights(reference, vendor)
    copy_mixed_model_weights(reference, efficient)
    copy_mixed_model_weights(reference, cudnn)
    copy_mixed_model_weights(reference, cudnn_full_ffn)
    vendor.assume_all_tokens_valid = True
    vendor.convert_projections_to_fp16()
    for model in (efficient, cudnn, cudnn_full_ffn):
        force_d128_projection_route(model)
        model.assume_all_tokens_valid = True
        model.convert_projections_to_fp16()
    cudnn_full_ffn.fuse_full_ffn = True
    for layer in cudnn_full_ffn.layers:
        layer.fuse_full_ffn = True
    vendor = torch.compile(vendor, mode="reduce-overhead")
    efficient = torch.compile(efficient, mode="reduce-overhead")
    cudnn = torch.compile(
        CudnnAttentionWrapper(cudnn), mode="reduce-overhead"
    )
    cudnn_full_ffn = torch.compile(
        CudnnAttentionWrapper(cudnn_full_ffn), mode="reduce-overhead"
    )
    return reference, vendor, efficient, cudnn, cudnn_full_ffn


def grouped_ms(model, x, mask, repeats: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        model(x, mask)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def main() -> int:
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference, vendor, efficient, cudnn, cudnn_full_ffn = prepare(device)
    models = {
        "vendor_efficient": vendor,
        "d128_efficient": efficient,
        "d128_cudnn": cudnn,
        "d128_cudnn_full_ffn": cudnn_full_ffn,
    }
    passed = True
    with torch.inference_mode():
        for trial in range(3):
            x, mask = bench.generate_random_case(
                CONFIG, device, torch.float32, 1234 + trial, 0.0, 1.0
            )
            expected = reference(x, mask)
            for name, model in models.items():
                result = bench.compare_outputs(expected, model(x, mask), 0.02, 0.002)
                passed &= result.passed
                print(
                    f"accuracy trial={trial + 1} route={name} "
                    f"status={'PASS' if result.passed else 'FAIL'} "
                    f"max_abs={result.max_abs_error:.8f} "
                    f"failed={result.failed_elements}/{result.total_elements}"
                )

        x, mask = bench.generate_random_case(
            CONFIG, device, torch.float32, 101234, 0.0, 1.0
        )
        for model in models.values():
            bench.warmup_model(model, x, mask, 50, device)
        grouped = {name: [] for name in models}
        orders = (
            tuple(models),
            tuple(reversed(models)),
            (
                "d128_efficient",
                "vendor_efficient",
                "d128_cudnn_full_ffn",
                "d128_cudnn",
            ),
        )
        for round_index in range(9):
            order = orders[round_index % len(orders)]
            for name in order:
                value = grouped_ms(models[name], x, mask, 1000)
                grouped[name].append(value)
                print(
                    f"round={round_index + 1} route={name} grouped_mean_ms={value:.6f}"
                )
    for name, values in grouped.items():
        print(
            f"result route={name} median_ms={statistics.median(values):.6f} "
            f"mean_ms={statistics.fmean(values):.6f} min_ms={min(values):.6f}"
        )
    print(
        "cudnn_speedup="
        f"{statistics.median(grouped['d128_efficient']) / statistics.median(grouped['d128_cudnn']):.6f}x"
    )
    print(
        "combined_speedup="
        f"{statistics.median(grouped['vendor_efficient']) / statistics.median(grouped['d128_cudnn_full_ffn']):.6f}x"
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
