#!/usr/bin/env python3
"""Controlled end-to-end D32 Triton projection experiment for case 7."""

from __future__ import annotations

import statistics

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

import torch_transformer_benchmark as bench
from mixed_precision import MixedPrecisionTransformer, copy_mixed_model_weights
from mixed_precision.triton_linear import mixed_linear
from torch_transformer_benchmark import fused_add_layer_norm


CONFIG = bench.TransformerConfig(64, 128, 32, 4, 32, 4, True)


@triton.jit
def _d32_linear_add_layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    norm_weight_ptr,
    norm_bias_ptr,
    residual_out_ptr,
    normalized_out_ptr,
    n_rows,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, 32)
    inner = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, 32), tl.float32)
    for k_start in range(0, 32, BLOCK_K):
        k = k_start + inner
        x = tl.load(
            x_ptr + rows[:, None] * 32 + k[None, :],
            mask=(rows[:, None] < n_rows) & (k[None, :] < 32),
            other=0.0,
        ).to(tl.float16)
        weight = tl.load(
            weight_ptr + cols[:, None] * 32 + k[None, :],
            mask=k[None, :] < 32,
            other=0.0,
        )
        accumulator += tl.dot(x, tl.trans(weight), out_dtype=tl.float32)
    projected = accumulator + tl.load(bias_ptr + cols)[None, :].to(tl.float32)
    residual = projected + tl.load(
        residual_ptr + rows[:, None] * 32 + cols[None, :],
        mask=rows[:, None] < n_rows,
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(residual, axis=1) / 32.0
    centered = residual - mean[:, None]
    variance = tl.sum(centered * centered, axis=1) / 32.0
    normalized = centered * tl.rsqrt(variance + EPS)[:, None]
    normalized = (
        normalized * tl.load(norm_weight_ptr + cols)[None, :].to(tl.float32)
        + tl.load(norm_bias_ptr + cols)[None, :].to(tl.float32)
    )
    offsets = rows[:, None] * 32 + cols[None, :]
    mask = rows[:, None] < n_rows
    tl.store(residual_out_ptr + offsets, residual, mask=mask)
    tl.store(normalized_out_ptr + offsets, normalized, mask=mask)


def d32_linear_add_layer_norm(x, linear, residual, norm):
    rows = x.numel() // 32
    residual_out = torch.empty_like(residual)
    normalized_out = torch.empty_like(residual)
    _d32_linear_add_layer_norm_kernel[(triton.cdiv(rows, 64),)](
        x,
        linear.weight,
        linear.bias,
        residual,
        norm.weight,
        norm.bias,
        residual_out,
        normalized_out,
        rows,
        EPS=norm.eps,
        BLOCK_M=64,
        BLOCK_K=16,
        num_warps=2,
        num_stages=2,
    )
    return residual_out, normalized_out


class ExperimentalD32Transformer(MixedPrecisionTransformer):
    def __init__(self, *args, fuse_output_norm: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fuse_output_norm = fuse_output_norm

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dtype != torch.float32:
            raise ValueError("D32 candidate requires FP32 residual input")
        effective_mask = None if self.assume_all_tokens_valid else valid_token_mask
        normalized = self.layers[0].norm1(x)
        for layer_index, layer in enumerate(self.layers):
            batch, seq_len, _ = normalized.shape
            attention = layer.attention
            qkv = mixed_linear(
                normalized,
                attention.qkv_proj,
                output_dtype=torch.float16,
                block_m=64,
                block_n=32,
                block_k=32,
                group_m=4,
                num_warps=2,
                num_stages=2,
            ).view(batch, seq_len, 3, 4, 8)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            attention_mask = (
                None
                if effective_mask is None
                else effective_mask[:, None, None, :]
            )
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=self.config.causal,
                scale=attention.scale,
            )
            context = context.transpose(1, 2).contiguous().view(
                batch, seq_len, 32
            )
            if self.fuse_output_norm:
                x, normalized_ffn = d32_linear_add_layer_norm(
                    context, attention.out_proj, x, layer.norm2
                )
            else:
                attention_output = mixed_linear(
                    context,
                    attention.out_proj,
                    output_dtype=torch.float32,
                    block_m=64,
                    block_n=32,
                    block_k=16,
                    group_m=4,
                    num_warps=2,
                    num_stages=2,
                )
                x, normalized_ffn = fused_add_layer_norm(
                    x, attention_output, layer.norm2
                )
            activated = mixed_linear(
                normalized_ffn,
                layer.ffn_in,
                output_dtype=torch.float16,
                fuse_gelu=True,
                block_m=64,
                block_n=32,
                block_k=16,
                group_m=4,
                num_warps=2,
                num_stages=2,
            )
            next_norm = (
                self.layers[layer_index + 1].norm1
                if layer_index + 1 < len(self.layers)
                else self.final_norm
            )
            if self.fuse_output_norm:
                x, normalized = d32_linear_add_layer_norm(
                    activated, layer.ffn_out, x, next_norm
                )
            else:
                ffn_output = mixed_linear(
                    activated,
                    layer.ffn_out,
                    output_dtype=torch.float32,
                    block_m=64,
                    block_n=32,
                    block_k=16,
                    group_m=4,
                    num_warps=2,
                    num_stages=2,
                )
                x, normalized = fused_add_layer_norm(x, ffn_output, next_norm)
        if effective_mask is not None:
            return normalized.masked_fill(~effective_mask[..., None], 0)
        return normalized


def main() -> int:
    device = torch.device("cuda")
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = bench.BaselineTransformer(CONFIG).to(device).eval()
    vendor = MixedPrecisionTransformer(CONFIG).to(device).eval()
    d32 = ExperimentalD32Transformer(CONFIG).to(device).eval()
    d32_fused = ExperimentalD32Transformer(
        CONFIG, fuse_output_norm=True
    ).to(device).eval()
    copy_mixed_model_weights(reference, vendor)
    copy_mixed_model_weights(reference, d32)
    copy_mixed_model_weights(reference, d32_fused)
    for model in (vendor, d32, d32_fused):
        model.assume_all_tokens_valid = True
        model.convert_projections_to_fp16()
    vendor = torch.compile(vendor, mode="reduce-overhead")
    d32 = torch.compile(d32, mode="reduce-overhead")
    d32_fused = torch.compile(d32_fused, mode="reduce-overhead")
    models = {
        "vendor": vendor,
        "d32_triton": d32,
        "d32_fused_epilogue": d32_fused,
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
            bench.warmup_model(model, x, mask, 20, device)
        samples = {name: [] for name in models}
        orders = (
            ("vendor", "d32_triton", "d32_fused_epilogue"),
            ("d32_fused_epilogue", "d32_triton", "vendor"),
            ("d32_triton", "vendor", "d32_fused_epilogue"),
        )
        for round_index in range(6):
            order = orders[round_index % len(orders)]
            for name in order:
                batch = bench.benchmark_once(models[name], x, mask, 30, device)
                samples[name].extend(batch)
                print(
                    f"round={round_index + 1} route={name} "
                    f"median_ms={statistics.median(batch):.6f}"
                )
    for name, values in samples.items():
        print(
            f"result route={name} median_ms={statistics.median(values):.6f} "
            f"mean_ms={statistics.fmean(values):.6f} min_ms={min(values):.6f}"
        )
    print(
        "d32_speedup="
        f"{statistics.median(samples['vendor']) / statistics.median(samples['d32_triton']):.6f}x"
    )
    print(
        "d32_fused_speedup="
        f"{statistics.median(samples['vendor']) / statistics.median(samples['d32_fused_epilogue']):.6f}x"
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
