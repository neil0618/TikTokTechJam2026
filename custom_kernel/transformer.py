"""Transformer candidate using the separate custom attention kernel."""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_transformer_benchmark import TransformerConfig, fused_add_layer_norm

from .triton_attention import causal_attention, is_supported


class CustomKernelSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, custom_enabled: bool) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.custom_enabled = custom_enabled
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).view(
            batch, seq_len, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)

        # SDPA has less fixed launch cost at the single-batch short-sequence edge.
        use_custom = (
            self.custom_enabled
            and valid_token_mask is None
            and is_supported(q, k, v, causal)
        )
        if use_custom:
            context_bshd = causal_attention(q, k, v, self.scale)
            context = context_bshd.view(batch, seq_len, self.d_model)
        else:
            attention_mask = (
                None
                if valid_token_mask is None
                else valid_token_mask[:, None, None, :]
            )
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=causal,
                scale=self.scale,
            )
            context = context.transpose(1, 2).contiguous().view(
                batch, seq_len, self.d_model
            )
        return self.out_proj(context)


class CustomKernelTransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, ffn_dim: int, custom_enabled: bool
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CustomKernelSelfAttention(
            d_model, num_heads, custom_enabled
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)


class CustomKernelTransformer(nn.Module):
    """Shape-dispatched candidate; unsupported shapes use the proven SDPA path."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.custom_attention_enabled = (
            config.causal
            and config.d_model // config.num_heads == 32
            and config.seq_len in (128, 1024)
            and not (config.batch_size == 1 and config.seq_len == 128)
        )
        self.layers = nn.ModuleList(
            [
                CustomKernelTransformerBlock(
                    config.d_model,
                    config.num_heads,
                    config.ffn_dim,
                    self.custom_attention_enabled,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.assume_all_tokens_valid = False

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        effective_mask = None if self.assume_all_tokens_valid else valid_token_mask
        normalized = self.layers[0].norm1(x)
        for layer_index, layer in enumerate(self.layers):
            attention_output = layer.attention(
                normalized, effective_mask, self.config.causal
            )
            x, normalized_ffn = fused_add_layer_norm(
                x, attention_output, layer.norm2
            )
            ffn_output = layer.ffn_out(
                F.gelu(layer.ffn_in(normalized_ffn), approximate="none")
            )
            next_norm = (
                self.layers[layer_index + 1].norm1
                if layer_index + 1 < len(self.layers)
                else self.final_norm
            )
            x, normalized = fused_add_layer_norm(x, ffn_output, next_norm)

        if effective_mask is not None:
            return normalized.masked_fill(~effective_mask[..., None], 0)
        return normalized


def copy_custom_model_weights(baseline: nn.Module, custom: nn.Module) -> None:
    """Map the baseline's separate Q/K/V projections into packed QKV weights."""
    source = copy.deepcopy(baseline.state_dict())
    mapped = {}
    for key in custom.state_dict():
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
    custom.load_state_dict(mapped, strict=True)
