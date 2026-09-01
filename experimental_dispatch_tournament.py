"""End-to-end tournaments for diagnostic-only kernel dispatch changes."""

from __future__ import annotations

import argparse
import statistics

import torch
import triton

import mixed_precision.transformer as mixed_transformer_module
import torch_transformer_benchmark as bench
from custom_kernel.triton_attention import launch_causal_attention
from experimental_fp16_attention import launch as launch_fp16_attention
from mixed_precision import MixedPrecisionTransformer, copy_mixed_model_weights


CASES = {
    1: bench.TransformerConfig(64, 128, 128, 4, 128, 4, True),
    4: bench.TransformerConfig(16, 128, 128, 4, 128, 4, True),
    5: bench.TransformerConfig(128, 128, 128, 4, 128, 4, True),
    8: bench.TransformerConfig(64, 128, 1024, 4, 1024, 4, True),
    9: bench.TransformerConfig(64, 128, 128, 1, 128, 4, True),
    10: bench.TransformerConfig(64, 128, 128, 2, 128, 4, True),
    11: bench.TransformerConfig(64, 128, 128, 16, 128, 4, True),
    12: bench.TransformerConfig(64, 32, 128, 4, 128, 4, True),
    13: bench.TransformerConfig(64, 1024, 128, 4, 128, 4, True),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=CASES, required=True)
    parser.add_argument("--variant", choices=("attention_stage3", "fused_bk16", "fp16_attention", "fp16_norm_outputs"), required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    config = CASES[args.case]
    device = torch.device("cuda")
    torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    baseline = bench.BaselineTransformer(config).to(device).eval()
    incumbent = MixedPrecisionTransformer(config)
    challenger = MixedPrecisionTransformer(config)
    copy_mixed_model_weights(baseline, incumbent)
    copy_mixed_model_weights(baseline, challenger)
    for model in (incumbent, challenger):
        model.to(device).eval(); model.assume_all_tokens_valid = True
        model.convert_projections_to_fp16()
    if args.variant == "fp16_attention" and args.case in (9, 10, 11, 13):
        for layer in challenger.layers:
            layer.attention.use_d128_custom_attention = True

    x, mask = bench.generate_random_case(config, device, torch.float32, 101234, 0.0, 1.0)
    incumbent = torch.compile(incumbent, mode="reduce-overhead")
    bench.warmup_model(incumbent, x, mask, 10, device)

    if args.variant == "attention_stage3":
        def attention_stage3(q, k, v, scale):
            return launch_causal_attention(
                q, k, v, scale,
                block_m=64, block_n=32, num_warps=4, num_stages=3,
            )
        mixed_transformer_module.causal_attention = attention_stage3
    elif args.variant == "fused_bk16":
        original = mixed_transformer_module.mixed_linear_add_layer_norm
        def fused_bk16(x, linear, residual, norm, **kwargs):
            if x.numel() // 128 == 2048:
                kwargs.update(block_m=32, block_k=16, num_warps=4, num_stages=2)
            return original(x, linear, residual, norm, **kwargs)
        mixed_transformer_module.mixed_linear_add_layer_norm = fused_bk16
    elif args.variant == "fp16_attention":
        original_linear = mixed_transformer_module.mixed_linear
        def fp16_qkv_linear(x, linear, **kwargs):
            if linear.in_features == 128 and linear.out_features == 384:
                kwargs["output_dtype"] = torch.float16
            return original_linear(x, linear, **kwargs)
        attention_configs = {
            1: (128, 32, 4, 2),
            4: (64, 32, 4, 2),
            5: (64, 64, 4, 2),
            9: (64, 32, 4, 2),
            10: (32, 32, 4, 2),
            11: (64, 32, 4, 2),
            13: (128, 32, 4, 2),
        }
        def fp16_attention(q, k, v, scale):
            return launch_fp16_attention(q, k, v, scale, *attention_configs[args.case])
        mixed_transformer_module.mixed_linear = fp16_qkv_linear
        mixed_transformer_module.is_supported = lambda q, k, v, causal: causal and q.dtype == torch.float16
        mixed_transformer_module.causal_attention = fp16_attention
    else:
        final_norm_id = id(challenger.final_norm)
        def fused_add_layer_norm_half(x, update, norm):
            n_cols = x.shape[-1]
            n_rows = x.numel() // n_cols
            block_size = triton.next_power_of_2(n_cols)
            warps = 2 if block_size <= 64 else (1 if block_size <= 256 else 8)
            residual = torch.empty_like(x)
            normalized = torch.empty_like(
                x,
                dtype=torch.float32 if id(norm) == final_norm_id else torch.float16,
            )
            bench._fused_add_layer_norm_kernel[(n_rows,)](
                x, update, norm.weight, norm.bias, residual, normalized,
                N_COLS=n_cols, EPS=norm.eps, BLOCK_SIZE=block_size,
                num_warps=warps,
            )
            return residual, normalized
        mixed_transformer_module.fused_add_layer_norm = fused_add_layer_norm_half

    challenger = torch.compile(challenger, mode="reduce-overhead")
    bench.warmup_model(challenger, x, mask, 10, device)

    with torch.inference_mode():
        for trial in range(3):
            ax, am = bench.generate_random_case(config, device, torch.float32, 1234 + trial, 0.0, 1.0)
            ref = baseline(ax, am)
            for name, model in (("incumbent", incumbent), (args.variant, challenger)):
                result = bench.compare_outputs(ref, model(ax, am), 0.02, 0.002)
                print(f"accuracy trial={trial+1} route={name} pass={result.passed} max_abs={result.max_abs_error:.8f} failed={result.failed_elements}")
        samples = {"incumbent": [], args.variant: []}
        models = {"incumbent": incumbent, args.variant: challenger}
        names = tuple(models)
        for round_index in range(args.rounds):
            order = names if round_index % 2 == 0 else tuple(reversed(names))
            for name in order:
                values = bench.benchmark_once(models[name], x, mask, args.repeats, device)
                samples[name].extend(values)
                print(f"round={round_index+1} route={name} median={statistics.median(values):.6f}")
    for name, values in samples.items():
        print(f"result route={name} median={statistics.median(values):.6f} mean={statistics.fmean(values):.6f} min={min(values):.6f}")
    print(f"speedup={statistics.median(samples['incumbent']) / statistics.median(samples[args.variant]):.6f}x")


if __name__ == "__main__":
    main()
