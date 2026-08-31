#!/usr/bin/env python3
"""Benchmark the separate custom-kernel candidate against the original baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

import torch_transformer_benchmark as bench
from custom_kernel import CustomKernelTransformer, copy_custom_model_weights


class PreallocatedMicrobatchModel(nn.Module):
    """Run fixed-size slices and copy them directly into the final output."""

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

        output = torch.empty_like(x)
        for start in range(0, x.shape[0], self.microbatch_size):
            end = min(start + self.microbatch_size, x.shape[0])
            mask_slice = (
                None if valid_token_mask is None else valid_token_mask[start:end]
            )
            chunk = self.model(x[start:end], mask_slice)
            output[start:end].copy_(chunk)
        return output


def main() -> int:
    args = bench.parse_args()
    device = bench.resolve_device(args.device)
    dtype = bench.resolve_dtype(args.dtype)
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
    bench.validate_args(args, device, dtype)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32

    baseline = bench.BaselineTransformer(config)
    custom = CustomKernelTransformer(config)
    custom_attention_enabled = custom.custom_attention_enabled
    custom.assume_all_tokens_valid = args.padding_ratio <= 0
    copy_custom_model_weights(baseline, custom)
    baseline = baseline.to(device=device, dtype=dtype).eval()
    custom = custom.to(device=device, dtype=dtype).eval()

    microbatch_size = args.microbatch_size
    if microbatch_size is None:
        microbatch_size = 250 if device.type == "cuda" and config.batch_size >= 4096 else 0

    compile_user = args.compile_user
    if compile_user is None:
        compile_user = device.type == "cuda"
    baseline = bench.maybe_compile(baseline, args.compile_baseline, args.compile_mode)
    custom = bench.maybe_compile(custom, compile_user, args.compile_mode)
    if microbatch_size > 0:
        baseline = bench.MicrobatchModel(baseline, microbatch_size)
        custom = (
            PreallocatedMicrobatchModel(custom, microbatch_size)
            if compile_user
            else bench.MicrobatchModel(custom, microbatch_size)
        )

    custom_active = (
        custom_attention_enabled
        and dtype == torch.float32
        and args.padding_ratio <= 0
    )
    print("=== Custom-kernel configuration ===")
    print(config)
    print(f"device={device}, dtype={dtype}, torch={torch.__version__}")
    print(f"compile_baseline={args.compile_baseline}, compile_custom={compile_user}, compile_mode={args.compile_mode}")
    print(f"microbatch_size={microbatch_size or 'disabled'}")
    print(f"custom_attention_active={custom_active}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    passed = bench.run_accuracy_tests(
        baseline, custom, config, device, dtype, args.accuracy_trials, args.seed,
        args.padding_ratio, args.input_scale, args.rtol, args.atol,
    )
    if not passed and not args.benchmark_on_failure:
        print("\nPerformance benchmark skipped because accuracy validation failed.")
        return 2
    bench.benchmark_models(
        baseline, custom, config, device, dtype, args.seed, args.padding_ratio,
        args.input_scale, args.warmup, args.repeats, args.benchmark_rounds,
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
