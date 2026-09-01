#!/usr/bin/env python3
"""Controlled case-2 efficient-SDPA versus cuDNN-SDPA tournament."""

from __future__ import annotations

import statistics

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

import torch_transformer_benchmark as bench
from mixed_precision import MixedPrecisionTransformer, copy_mixed_model_weights


CONFIG = bench.TransformerConfig(1, 128, 128, 4, 128, 4, True)


class BackendWrapper(nn.Module):
    def __init__(self, model: nn.Module, backend: SDPBackend) -> None:
        super().__init__()
        self.model = model
        self.backend = backend

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        with sdpa_kernel(self.backend):
            return self.model(x, mask)


def prepare(reference, device, backend):
    model = MixedPrecisionTransformer(CONFIG).to(device).eval()
    copy_mixed_model_weights(reference, model)
    model.assume_all_tokens_valid = True
    model.convert_projections_to_fp16()
    return torch.compile(BackendWrapper(model, backend), mode="reduce-overhead")


def grouped_ms(model, x, mask, repeats=1000):
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
    models = {
        "efficient": prepare(reference, device, SDPBackend.EFFICIENT_ATTENTION),
        "cudnn": prepare(reference, device, SDPBackend.CUDNN_ATTENTION),
    }
    passed = True
    with torch.inference_mode():
        for trial in range(3):
            x, mask = bench.generate_random_case(
                CONFIG, device, torch.float32, 1234 + trial, 0.0, 1.0
            )
            expected = reference(x, mask)
            for name, model in models.items():
                result = bench.compare_outputs(expected, model(x, mask), .02, .002)
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
        names = tuple(models)
        for round_index in range(7):
            order = names if round_index % 2 == 0 else tuple(reversed(names))
            for name in order:
                value = grouped_ms(models[name], x, mask)
                samples[name].append(value)
                print(f"round={round_index + 1} route={name} ms={value:.6f}")
    for name, values in samples.items():
        print(
            f"result route={name} median_ms={statistics.median(values):.6f} "
            f"mean_ms={statistics.fmean(values):.6f} min_ms={min(values):.6f}"
        )
    print(
        "cudnn_speedup="
        f"{statistics.median(samples['efficient']) / statistics.median(samples['cudnn']):.6f}x"
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
