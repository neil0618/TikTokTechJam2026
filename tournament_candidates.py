#!/usr/bin/env python3
"""Controlled, same-process tournament for shape-specific candidate routes."""

from __future__ import annotations

import argparse
import statistics

import torch

import torch_transformer_benchmark as bench
from custom_kernel import CustomKernelTransformer, copy_custom_model_weights
from mixed_precision import MixedPrecisionTransformer, copy_mixed_model_weights


CASES = {
    3: bench.TransformerConfig(4, 128, 128, 4, 128, 4, True),
    11: bench.TransformerConfig(64, 128, 128, 16, 128, 4, True),
}


def prepare(case: int, device: torch.device):
    config = CASES[case]
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    baseline = bench.BaselineTransformer(config).to(device).eval()

    if case == 3:
        incumbent = MixedPrecisionTransformer(
            config, enable_case11_schedule=False
        )
        challenger = CustomKernelTransformer(config)
        challenger.assume_all_tokens_valid = True
        copy_mixed_model_weights(baseline, incumbent)
        copy_custom_model_weights(baseline, challenger)
        incumbent = incumbent.to(device).eval()
        incumbent.assume_all_tokens_valid = True
        incumbent.convert_projections_to_fp16()
        challenger = challenger.to(device).eval()
        names = ("mixed_vendor", "fp32_custom_attention")
    else:
        incumbent = MixedPrecisionTransformer(config)
        challenger = MixedPrecisionTransformer(
            config, enable_case11_schedule=True
        )
        copy_mixed_model_weights(baseline, incumbent)
        copy_mixed_model_weights(baseline, challenger)
        incumbent = incumbent.to(device).eval()
        challenger = challenger.to(device).eval()
        for model in (incumbent, challenger):
            model.assume_all_tokens_valid = True
            model.convert_projections_to_fp16()
        names = ("current_d128_tiles", "expanded_case11_tiles")

    return config, baseline, incumbent, challenger, names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=CASES, required=True)
    parser.add_argument("--accuracy-trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This tournament requires CUDA")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    config, baseline, incumbent, challenger, names = prepare(args.case, device)
    incumbent = torch.compile(incumbent, mode="reduce-overhead")
    challenger = torch.compile(challenger, mode="reduce-overhead")

    print(f"case={args.case} config={config}")
    print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"protocol=accuracy:{args.accuracy_trials}, warmup:{args.warmup}, repeats:{args.repeats}, rounds:{args.rounds}")
    print(f"incumbent={names[0]} challenger={names[1]}")

    passed = True
    with torch.inference_mode():
        for trial in range(args.accuracy_trials):
            x, mask = bench.generate_random_case(
                config, device, torch.float32, 1234 + trial, 0.0, 1.0
            )
            reference = baseline(x, mask)
            for name, model in ((names[0], incumbent), (names[1], challenger)):
                result = bench.compare_outputs(reference, model(x, mask), 0.02, 0.002)
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
        bench.warmup_model(incumbent, x, mask, args.warmup, device)
        bench.warmup_model(challenger, x, mask, args.warmup, device)
        samples = {names[0]: [], names[1]: []}
        models = {names[0]: incumbent, names[1]: challenger}
        for round_index in range(args.rounds):
            order = names if round_index % 2 == 0 else tuple(reversed(names))
            for name in order:
                batch = bench.benchmark_once(
                    models[name], x, mask, args.repeats, device
                )
                samples[name].extend(batch)
                print(
                    f"round={round_index + 1} route={name} "
                    f"median_ms={statistics.median(batch):.6f}"
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
