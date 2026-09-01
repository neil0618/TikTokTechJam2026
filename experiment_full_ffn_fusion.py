#!/usr/bin/env python3
"""Controlled tournament for a single-launch D128 FFN block."""

from __future__ import annotations

import argparse
import statistics

import torch

import torch_transformer_benchmark as bench
from mixed_precision import MixedPrecisionTransformer, copy_mixed_model_weights


CASES = {
    1: bench.TransformerConfig(64, 128, 128, 4, 128, 4, True),
    4: bench.TransformerConfig(16, 128, 128, 4, 128, 4, True),
    5: bench.TransformerConfig(128, 128, 128, 4, 128, 4, True),
    9: bench.TransformerConfig(64, 128, 128, 1, 128, 4, True),
    10: bench.TransformerConfig(64, 128, 128, 2, 128, 4, True),
    11: bench.TransformerConfig(64, 128, 128, 16, 128, 4, True),
    12: bench.TransformerConfig(64, 32, 128, 4, 128, 4, True),
    13: bench.TransformerConfig(64, 1024, 128, 4, 128, 4, True),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=CASES, required=True)
    parser.add_argument("--accuracy-trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--with-attention-fusion", action="store_true")
    parser.add_argument("--incumbent-attention-fusion", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    config = CASES[args.case]
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    baseline = bench.BaselineTransformer(config).to(device).eval()
    incumbent = MixedPrecisionTransformer(
        config,
        enable_full_ffn_fusion=False,
        enable_attention_output_norm_fusion=args.incumbent_attention_fusion,
    )
    challenger = MixedPrecisionTransformer(
        config,
        enable_full_ffn_fusion=True,
        enable_attention_output_norm_fusion=(
            args.with_attention_fusion or args.incumbent_attention_fusion
        ),
    )
    copy_mixed_model_weights(baseline, incumbent)
    copy_mixed_model_weights(baseline, challenger)
    for model in (incumbent, challenger):
        model.assume_all_tokens_valid = True
        model.to(device).eval()
        model.convert_projections_to_fp16()
    incumbent = torch.compile(incumbent, mode="reduce-overhead")
    challenger = torch.compile(challenger, mode="reduce-overhead")

    challenger_name = (
        "fused_attention_and_full_ffn"
        if args.with_attention_fusion
        else "fused_full_ffn"
    )
    incumbent_name = (
        "fused_attention_only"
        if args.incumbent_attention_fusion
        else "incumbent"
    )
    if args.incumbent_attention_fusion:
        challenger_name = "fused_attention_and_full_ffn"
    names = (incumbent_name, challenger_name)
    models = {names[0]: incumbent, names[1]: challenger}
    passed = True
    print(f"case={args.case} config={config}")
    print(f"gpu={torch.cuda.get_device_name(device)}")
    print(
        f"protocol=accuracy:{args.accuracy_trials}, warmup:{args.warmup}, "
        f"repeats:{args.repeats}, rounds:{args.rounds}"
    )

    with torch.inference_mode():
        for trial in range(args.accuracy_trials):
            x, mask = bench.generate_random_case(
                config, device, torch.float32, 1234 + trial, 0.0, 1.0
            )
            reference = baseline(x, mask)
            for name in names:
                result = bench.compare_outputs(
                    reference, models[name](x, mask), 0.02, 0.002
                )
                passed &= result.passed
                print(
                    f"accuracy trial={trial + 1} route={name} "
                    f"status={'PASS' if result.passed else 'FAIL'} "
                    f"max_abs={result.max_abs_error:.8f} "
                    f"failed={result.failed_elements}/{result.total_elements}"
                )

        x, mask = bench.generate_random_case(
            config, device, torch.float32, 101234, 0.0, 1.0
        )
        for model in models.values():
            bench.warmup_model(model, x, mask, args.warmup, device)
        samples = {name: [] for name in names}
        for round_index in range(args.rounds):
            order = names if round_index % 2 == 0 else tuple(reversed(names))
            for name in order:
                values = bench.benchmark_once(
                    models[name], x, mask, args.repeats, device
                )
                samples[name].extend(values)
                print(
                    f"round={round_index + 1} route={name} "
                    f"median_ms={statistics.median(values):.6f}"
                )

    for name in names:
        values = samples[name]
        print(
            f"result route={name} median_ms={statistics.median(values):.6f} "
            f"mean_ms={statistics.fmean(values):.6f} min_ms={min(values):.6f}"
        )
    gain = statistics.median(samples[names[0]]) / statistics.median(samples[names[1]])
    print(f"challenger_speedup={gain:.6f}x")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
