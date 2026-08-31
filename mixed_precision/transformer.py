"""Mixed FP16/FP32 inference candidate with an FP32 residual stream.

Projection operands and weights use FP16 Tensor Cores. Numerically sensitive
reductions, nonlinearities, residuals, normalization, and model output remain
FP32. This module is intentionally separate from both existing implementations.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from torch_transformer_benchmark import TransformerConfig, fused_add_layer_norm

from custom_kernel.triton_attention import causal_attention, is_supported
from .triton_linear import mixed_linear, mixed_linear_add_layer_norm


class MixedPrecisionSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_case8_triton: bool,
        use_d128_triton: bool,
        use_d128_custom_attention: bool,
        use_case11_schedule: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.use_case8_triton = use_case8_triton
        self.use_d128_triton = use_d128_triton
        self.use_d128_custom_attention = use_d128_custom_attention
        self.use_case11_schedule = use_case11_schedule
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def convert_projections_to_fp16(self) -> None:
        self.qkv_proj.half()
        self.out_proj.half()

    def forward(
        self,
        normalized_fp32: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = normalized_fp32.shape
        if self.use_case8_triton:
            qkv_fp16 = mixed_linear(
                normalized_fp32,
                self.qkv_proj,
                output_dtype=torch.float16,
                block_m=64,
                block_n=128,
                block_k=32,
                num_warps=4,
                num_stages=3,
            )
        elif self.use_d128_triton:
            n_rows = batch * seq_len
            if n_rows >= 65536:
                qkv_config = (128, 64, 32, 4, 8)
            elif n_rows >= 16384:
                qkv_config = (64, 128, 32, 4, 8)
            elif n_rows >= 8192:
                qkv_config = (
                    (64, 128, 32, 4, 4)
                    if self.use_case11_schedule
                    else (128, 128, 32, 4, 8)
                )
            else:
                qkv_config = (
                    (32, 64, 32, 4, 8)
                    if self.use_d128_custom_attention
                    else (32, 64, 32, 4, 4)
                )
            qkv_fp16 = mixed_linear(
                normalized_fp32,
                self.qkv_proj,
                output_dtype=(
                    torch.float32
                    if self.use_d128_custom_attention
                    else torch.float16
                ),
                block_m=qkv_config[0],
                block_n=qkv_config[1],
                block_k=qkv_config[2],
                num_warps=qkv_config[3],
                num_stages=2,
                group_m=qkv_config[4],
            )
        else:
            qkv_fp16 = self.qkv_proj(normalized_fp32.to(torch.float16))
        qkv_fp16 = qkv_fp16.view(
            batch, seq_len, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv_fp16.permute(2, 0, 3, 1, 4).unbind(0)
        attention_mask = (
            None
            if valid_token_mask is None
            else valid_token_mask[:, None, None, :]
        )
        if (
            self.use_d128_custom_attention
            and attention_mask is None
            and is_supported(q, k, v, causal)
        ):
            context_fp16 = causal_attention(q, k, v, self.scale).view(
                batch, seq_len, self.d_model
            )
        else:
            use_cudnn_attention = (
                normalized_fp32.is_cuda
                and normalized_fp32.dtype == torch.float32
                and batch == 64
                and (
                    (self.d_model == 128 and self.num_heads == 1 and seq_len == 128)
                    or (self.d_model == 128 and self.num_heads == 4 and seq_len == 1024)
                )
            )
            if use_cudnn_attention:
                with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                    context_fp16 = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=attention_mask,
                        dropout_p=0.0,
                        is_causal=causal,
                        scale=self.scale,
                    )
            else:
                context_fp16 = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=causal,
                    scale=self.scale,
                )
            context_fp16 = context_fp16.transpose(1, 2).contiguous().view(
                batch, seq_len, self.d_model
            )
        # The residual stream never becomes FP16. The case-8 kernel retains
        # the Tensor-Core accumulator in FP32 through the final store.
        if self.use_case8_triton:
            return mixed_linear(
                context_fp16,
                self.out_proj,
                output_dtype=torch.float32,
                block_m=128,
                block_n=64,
                block_k=32,
                num_warps=4,
                num_stages=3,
            )
        if self.use_d128_triton:
            n_rows = batch * seq_len
            if n_rows >= 32000:
                output_config = (128, 32, 32, 8)
            elif n_rows >= 8192:
                output_config = (
                    (128, 64, 32, 4)
                    if self.use_case11_schedule
                    else (64, 32, 32, 8)
                )
            else:
                output_config = (
                    (32, 64, 32, 8)
                    if self.use_d128_custom_attention
                    else (64, 64, 64, 8)
                )
            return mixed_linear(
                context_fp16,
                self.out_proj,
                output_dtype=torch.float32,
                block_m=output_config[0],
                block_n=output_config[1],
                block_k=output_config[2],
                num_warps=4,
                num_stages=2,
                group_m=output_config[3],
            )
        return self.out_proj(context_fp16).to(torch.float32)


class MixedPrecisionTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        use_case8_triton: bool,
        use_d128_triton: bool,
        use_d128_custom_attention: bool,
        use_case11_schedule: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MixedPrecisionSelfAttention(
            d_model,
            num_heads,
            use_case8_triton,
            use_d128_triton,
            use_d128_custom_attention,
            use_case11_schedule,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)
        self.use_case8_triton = use_case8_triton
        self.use_d128_triton = use_d128_triton
        self.use_case11_schedule = use_case11_schedule

    def convert_projections_to_fp16(self) -> None:
        self.attention.convert_projections_to_fp16()
        self.ffn_in.half()
        self.ffn_out.half()


class MixedPrecisionTransformer(nn.Module):
    """FP16 projections with FP32 reductions, nonlinearities, and residuals."""

    def __init__(
        self,
        config: TransformerConfig,
        *,
        enable_case11_schedule: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.use_case8_triton = (
            config.batch_size == 64
            and config.seq_len == 128
            and config.d_model == 1024
            and config.num_heads == 4
            and config.ffn_dim == 1024
            and config.num_layers == 4
            and config.causal
        )
        self.use_d128_triton = (
            config.d_model == 128
            and config.ffn_dim == 128
            and config.num_layers == 4
            and config.causal
            and (
                (
                    config.num_heads == 4
                    and (
                        (config.batch_size == 128 and config.seq_len == 128)
                        or (config.batch_size == 64 and config.seq_len in (32, 128, 1024))
                        or (config.batch_size == 16 and config.seq_len == 128)
                    )
                )
                or (
                    config.batch_size == 64
                    and config.seq_len == 128
                    and config.num_heads in (1, 2, 16)
                )
            )
        )
        self.use_d128_custom_attention = (
            self.use_d128_triton
            and config.num_heads == 4
            and config.seq_len == 128
        )
        self.use_case11_schedule = (
            enable_case11_schedule
            and config.batch_size == 64
            and config.seq_len == 128
            and config.d_model == 128
            and config.num_heads == 16
            and config.ffn_dim == 128
            and config.num_layers == 4
            and config.causal
        )
        self.layers = nn.ModuleList(
            [
                MixedPrecisionTransformerBlock(
                    config.d_model,
                    config.num_heads,
                    config.ffn_dim,
                    self.use_case8_triton,
                    self.use_d128_triton,
                    self.use_d128_custom_attention,
                    self.use_case11_schedule,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.assume_all_tokens_valid = False

    def convert_projections_to_fp16(self) -> None:
        for layer in self.layers:
            layer.convert_projections_to_fp16()

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dtype != torch.float32:
            raise ValueError("mixed candidate requires an FP32 input/residual stream")

        effective_mask = None if self.assume_all_tokens_valid else valid_token_mask
        normalized = self.layers[0].norm1(x)
        for layer_index, layer in enumerate(self.layers):
            attention_output = layer.attention(
                normalized, effective_mask, self.config.causal
            )
            x, normalized_ffn = fused_add_layer_norm(
                x, attention_output, layer.norm2
            )
            next_norm = (
                self.layers[layer_index + 1].norm1
                if layer_index + 1 < len(self.layers)
                else self.final_norm
            )
            ffn_norm_fused = False

            if layer.use_case8_triton:
                activated_fp16 = mixed_linear(
                    normalized_ffn,
                    layer.ffn_in,
                    output_dtype=torch.float16,
                    fuse_gelu=True,
                    block_m=64,
                    block_n=128,
                    block_k=32,
                    num_warps=4,
                    num_stages=3,
                )
                ffn_output = mixed_linear(
                    activated_fp16,
                    layer.ffn_out,
                    output_dtype=torch.float32,
                    block_m=128,
                    block_n=64,
                    block_k=32,
                    num_warps=4,
                    num_stages=3,
                )
            elif layer.use_d128_triton:
                n_rows = normalized_ffn.numel() // 128
                if n_rows >= 16384:
                    ffn_in_config = (64, 128, 32, 8)
                elif n_rows >= 8192:
                    ffn_in_config = (
                        (64, 128, 32, 4)
                        if layer.use_case11_schedule
                        else (64, 64, 32, 8)
                    )
                else:
                    ffn_in_config = (32, 64, 64, 8)
                if n_rows >= 65536:
                    fused_out_config = (16, 64, 4)
                elif n_rows >= 16384:
                    fused_out_config = (32, 32, 4)
                elif n_rows >= 8192:
                    fused_out_config = (32, 32, 8)
                else:
                    fused_out_config = (32, 32, 8)
                activated_fp16 = mixed_linear(
                    normalized_ffn,
                    layer.ffn_in,
                    output_dtype=torch.float16,
                    fuse_gelu=True,
                    block_m=ffn_in_config[0],
                    block_n=ffn_in_config[1],
                    block_k=ffn_in_config[2],
                    num_warps=4,
                    num_stages=2,
                    group_m=ffn_in_config[3],
                )
                if n_rows == 8192:
                    ffn_out_config = (
                        (128, 64, 32, 4)
                        if layer.use_case11_schedule
                        else (64, 32, 32, 8)
                    )
                    ffn_output = mixed_linear(
                        activated_fp16,
                        layer.ffn_out,
                        output_dtype=torch.float32,
                        block_m=ffn_out_config[0],
                        block_n=ffn_out_config[1],
                        block_k=ffn_out_config[2],
                        num_warps=4,
                        num_stages=2,
                        group_m=ffn_out_config[3],
                    )
                else:
                    x, normalized = mixed_linear_add_layer_norm(
                        activated_fp16,
                        layer.ffn_out,
                        x,
                        next_norm,
                        block_m=fused_out_config[0],
                        block_k=fused_out_config[1],
                        num_warps=fused_out_config[2],
                        num_stages=2,
                    )
                    ffn_norm_fused = True
            else:
                # GEMM operands are FP16, while exact GELU is evaluated in FP32.
                hidden_fp16 = layer.ffn_in(normalized_ffn.to(torch.float16))
                activated_fp32 = F.gelu(
                    hidden_fp16.to(torch.float32), approximate="none"
                )
                ffn_output = layer.ffn_out(
                    activated_fp32.to(torch.float16)
                ).to(torch.float32)

            if not ffn_norm_fused:
                x, normalized = fused_add_layer_norm(x, ffn_output, next_norm)

        if effective_mask is not None:
            return normalized.masked_fill(~effective_mask[..., None], 0)
        return normalized


def copy_mixed_model_weights(
    baseline: nn.Module, mixed: MixedPrecisionTransformer
) -> None:
    """Copy FP32 reference weights, packing Q/K/V before FP16 conversion."""
    source = copy.deepcopy(baseline.state_dict())
    mapped = {}
    for key in mixed.state_dict():
        if key.endswith("attention.qkv_proj.weight"):
            prefix = key[: -len("qkv_proj.weight")]
            mapped[key] = torch.cat(
                [
                    source[prefix + "q_proj.weight"],
                    source[prefix + "k_proj.weight"],
                    source[prefix + "v_proj.weight"],
                ],
                dim=0,
            )
        elif key.endswith("attention.qkv_proj.bias"):
            prefix = key[: -len("qkv_proj.bias")]
            mapped[key] = torch.cat(
                [
                    source[prefix + "q_proj.bias"],
                    source[prefix + "k_proj.bias"],
                    source[prefix + "v_proj.bias"],
                ],
                dim=0,
            )
        else:
            mapped[key] = source[key]
    mixed.load_state_dict(mapped, strict=True)
