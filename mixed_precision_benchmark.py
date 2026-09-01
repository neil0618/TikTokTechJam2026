#!/usr/bin/env python3
"""Compare a strict IEEE-FP32 baseline with the mixed FP16/FP32 candidate."""

from __future__ import annotations

import torch
import torch.nn as nn

import torch_transformer_benchmark as bench
from mixed_precision import MixedPrecisionTransformer, copy_mixed_model_weights


class PreallocatedMicrobatchModel(nn.Module):
    def __init__(self, model: nn.Module, microbatch_size: int) -> None:
        super().__init__()
        self.model = model
        self.microbatch_size = microbatch_size

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.shape[0] <= self.microbatch_size:
            return self.model(x, valid_token_mask)
        output = torch.empty_like(x, dtype=torch.float32)
        for start in range(0, x.shape[0], self.microbatch_size):
            end = min(start + self.microbatch_size, x.shape[0])
            mask_slice = (
                None if valid_token_mask is None else valid_token_mask[start:end]
            )
            output[start:end].copy_(self.model(x[start:end], mask_slice))
        return output


def main() -> int:
    args = bench.parse_args()
    device = bench.resolve_device(args.device)
    config = bench.TransformerConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.layers,
        causal=args.causal,
    )
    config.validate()
    bench.validate_args(args, device, torch.float32)

    # This runner deliberately ignores dtype/TF32 performance shortcuts for
    # the reference: it is always strict IEEE FP32.
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    baseline = bench.BaselineTransformer(config)
    mixed = MixedPrecisionTransformer(config)
    mixed.assume_all_tokens_valid = args.padding_ratio <= 0
    copy_mixed_model_weights(baseline, mixed)
    baseline = baseline.to(device=device, dtype=torch.float32).eval()
    mixed = mixed.to(device=device).eval()
    mixed.convert_projections_to_fp16()
    dispatch = {
        "case8_triton": mixed.use_case8_triton,
        "d128_triton": mixed.use_d128_triton,
        "d128_fp32_attention": mixed.use_d128_custom_attention,
        "fp16_custom_attention": mixed.use_fp16_custom_attention,
        "case11_schedule": mixed.use_case11_schedule,
        "attention_output_norm_fusion": mixed.fuse_attention_output_norm,
        "full_ffn_fusion": mixed.fuse_full_ffn,
        "fp16_normalized_stream": mixed.use_fp16_normalized_stream,
        "d32_triton": mixed.use_d32_triton,
    }

    baseline_microbatch_size = args.microbatch_size
    mixed_microbatch_size = args.microbatch_size
    if args.microbatch_size is None:
        use_automatic_microbatching = device.type == "cuda" and config.batch_size >= 4096
        baseline_microbatch_size = 250 if use_automatic_microbatching else 0
        mixed_microbatch_size = 64 if use_automatic_microbatching else 0

    compile_mixed = args.compile_user
    if compile_mixed is None:
        compile_mixed = device.type == "cuda"
    baseline = bench.maybe_compile(baseline, args.compile_baseline, args.compile_mode)
    mixed = bench.maybe_compile(mixed, compile_mixed, args.compile_mode)
    if baseline_microbatch_size > 0:
        baseline = bench.MicrobatchModel(baseline, baseline_microbatch_size)
    if mixed_microbatch_size > 0:
        mixed = PreallocatedMicrobatchModel(mixed, mixed_microbatch_size)

    print("=== Strict-FP32 baseline vs mixed candidate ===")
    print(config)
    print(f"device={device}, baseline_dtype=torch.float32, baseline_tf32=False")
    print("mixed_policy=FP16 projections; FP32 accumulation/reductions/GELU/residual/LayerNorm/output")
    print(f"case8_custom_triton_projections={dispatch['case8_triton']}")
    print(f"d128_custom_triton_projections={dispatch['d128_triton']}")
    print(f"d32_custom_triton_projections={dispatch['d32_triton']}")
    print(f"d128_custom_triton_attention={dispatch['d128_fp32_attention']}")
    print(f"fp16_custom_triton_attention={dispatch['fp16_custom_attention']}")
    print(f"case11_expanded_d128_schedule={dispatch['case11_schedule']}")
    print(
        "attention_output_norm_fusion="
        f"{dispatch['attention_output_norm_fusion']}"
    )
    print(f"full_ffn_fusion={dispatch['full_ffn_fusion']}")
    print(f"fp16_normalized_stream={dispatch['fp16_normalized_stream']}")
    print(f"compile_baseline={args.compile_baseline}, compile_mixed={compile_mixed}, compile_mode={args.compile_mode}")
    print(f"baseline_microbatch_size={baseline_microbatch_size or 'disabled'}")
    print(f"mixed_microbatch_size={mixed_microbatch_size or 'disabled'}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    passed = bench.run_accuracy_tests(
        baseline,
        mixed,
        config,
        device,
        torch.float32,
        args.accuracy_trials,
        args.seed,
        args.padding_ratio,
        args.input_scale,
        args.rtol,
        args.atol,
    )
    if not passed and not args.benchmark_on_failure:
        print("\nPerformance benchmark skipped because accuracy validation failed.")
        return 2
    bench.benchmark_models(
        baseline,
        mixed,
        config,
        device,
        torch.float32,
        args.seed,
        args.padding_ratio,
        args.input_scale,
        args.warmup,
        args.repeats,
        args.benchmark_rounds,
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
