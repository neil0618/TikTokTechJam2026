#!/usr/bin/env python3
"""Controlled case-8 tournament for FP16 normalized activation storage."""

from __future__ import annotations

import argparse
import statistics

import torch

import torch_transformer_benchmark as bench
from mixed_precision import MixedPrecisionTransformer, copy_mixed_model_weights


CASES = {
    1: bench.TransformerConfig(64, 128, 128, 4, 128, 4, True),
    3: bench.TransformerConfig(4, 128, 128, 4, 128, 4, True),
    4: bench.TransformerConfig(16, 128, 128, 4, 128, 4, True),
    5: bench.TransformerConfig(128, 128, 128, 4, 128, 4, True),
    7: bench.TransformerConfig(64, 128, 32, 4, 32, 4, True),
    8: bench.TransformerConfig(64, 128, 1024, 4, 1024, 4, True),
    9: bench.TransformerConfig(64, 128, 128, 1, 128, 4, True),
    10: bench.TransformerConfig(64, 128, 128, 2, 128, 4, True),
    11: bench.TransformerConfig(64, 128, 128, 16, 128, 4, True),
    12: bench.TransformerConfig(64, 32, 128, 4, 128, 4, True),
    13: bench.TransformerConfig(64, 1024, 128, 4, 128, 4, True),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=CASES, default=8)
    args = parser.parse_args()
    config = CASES[args.case]
    device = torch.device("cuda")
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = bench.BaselineTransformer(config).to(device).eval()
    incumbent = MixedPrecisionTransformer(config)
    challenger = MixedPrecisionTransformer(
        config, enable_fp16_normalized_stream=True
    )
    for model in (incumbent, challenger):
        copy_mixed_model_weights(reference, model)
        model.assume_all_tokens_valid = True
        model.to(device).eval()
        model.convert_projections_to_fp16()
    models = {
        "incumbent": torch.compile(incumbent, mode="reduce-overhead"),
        "fp16_normalized_stream": torch.compile(
            challenger, mode="reduce-overhead"
        ),
    }
    passed = True
    with torch.inference_mode():
        for trial in range(3):
            x, mask = bench.generate_random_case(
                config, device, torch.float32, 1234 + trial, 0.0, 1.0
            )
            expected = reference(x, mask)
            for name, model in models.items():
                result = bench.compare_outputs(
                    expected, model(x, mask), 0.02, 0.002
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
            bench.warmup_model(model, x, mask, 10, device)
        samples = {name: [] for name in models}
        names = tuple(models)
        for round_index in range(5):
            order = names if round_index % 2 == 0 else tuple(reversed(names))
            for name in order:
                values = bench.benchmark_once(
                    models[name], x, mask, 30, device
                )
                samples[name].extend(values)
                print(
                    f"round={round_index + 1} route={name} "
                    f"median_ms={statistics.median(values):.6f}"
                )
    for name, values in samples.items():
        print(
            f"result route={name} median_ms={statistics.median(values):.6f} "
            f"mean_ms={statistics.fmean(values):.6f} min_ms={min(values):.6f}"
        )
    print(
        "speedup="
        f"{statistics.median(samples['incumbent']) / statistics.median(samples['fp16_normalized_stream']):.6f}x"
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
