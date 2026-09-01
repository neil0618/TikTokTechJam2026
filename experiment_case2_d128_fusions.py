#!/usr/bin/env python3
"""Controlled case-2 experiment for launch-reducing D128 fusions."""

from __future__ import annotations

import statistics

import torch

import torch_transformer_benchmark as bench
from mixed_precision import MixedPrecisionTransformer, copy_mixed_model_weights


CONFIG = bench.TransformerConfig(1, 128, 128, 4, 128, 4, True)


def force_route(
    model: MixedPrecisionTransformer,
    *,
    attention_fusion: bool,
    full_ffn_fusion: bool,
) -> None:
    model.use_d128_triton = True
    model.use_d128_custom_attention = False
    model.fuse_attention_output_norm = attention_fusion
    model.fuse_full_ffn = full_ffn_fusion
    for layer in model.layers:
        layer.use_d128_triton = True
        layer.attention.use_d128_triton = True
        layer.attention.use_d128_custom_attention = False
        layer.fuse_attention_output_norm = attention_fusion
        layer.attention.fuse_output_norm = attention_fusion
        layer.fuse_full_ffn = full_ffn_fusion


def grouped_ms(model, x, mask, repeats: int = 1000) -> float:
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
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = bench.BaselineTransformer(CONFIG).to(device).eval()
    vendor = MixedPrecisionTransformer(CONFIG).to(device).eval()
    d128 = MixedPrecisionTransformer(CONFIG).to(device).eval()
    attention_fused = MixedPrecisionTransformer(CONFIG).to(device).eval()
    full_ffn = MixedPrecisionTransformer(CONFIG).to(device).eval()
    all_fused = MixedPrecisionTransformer(CONFIG).to(device).eval()
    for model in (vendor, d128, attention_fused, full_ffn, all_fused):
        copy_mixed_model_weights(reference, model)
        model.assume_all_tokens_valid = True
        model.convert_projections_to_fp16()
    force_route(d128, attention_fusion=False, full_ffn_fusion=False)
    force_route(attention_fused, attention_fusion=True, full_ffn_fusion=False)
    force_route(full_ffn, attention_fusion=False, full_ffn_fusion=True)
    force_route(all_fused, attention_fusion=True, full_ffn_fusion=True)
    models = {
        "vendor": torch.compile(vendor, mode="reduce-overhead"),
        "d128": torch.compile(d128, mode="reduce-overhead"),
        "d128_attention_fused": torch.compile(
            attention_fused, mode="reduce-overhead"
        ),
        "d128_full_ffn": torch.compile(full_ffn, mode="reduce-overhead"),
        "d128_all_fused": torch.compile(all_fused, mode="reduce-overhead"),
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
        samples = {name: [] for name in models}
        orders = (
            tuple(models),
            tuple(reversed(models)),
            (
                "d128",
                "vendor",
                "d128_all_fused",
                "d128_full_ffn",
                "d128_attention_fused",
            ),
        )
        for round_index in range(9):
            for name in orders[round_index % len(orders)]:
                value = grouped_ms(models[name], x, mask)
                samples[name].append(value)
                print(
                    f"round={round_index + 1} route={name} "
                    f"grouped_mean_ms={value:.6f}"
                )
    for name, values in samples.items():
        print(
            f"result route={name} median_ms={statistics.median(values):.6f} "
            f"mean_ms={statistics.fmean(values):.6f} min_ms={min(values):.6f}"
        )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
