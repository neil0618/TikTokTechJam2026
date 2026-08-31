"""Shape-tuned mixed-precision projection kernels.

FP32 activations are converted to FP16 in registers, FP16 weights feed Tensor
Cores, accumulators and bias arithmetic remain FP32, and the caller chooses an
FP16 or FP32 output. Exact GELU can be fused into the accumulator epilogue.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mixed_linear_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_rows,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    FUSE_GELU: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(n_rows, BLOCK_M)
    num_pid_n = tl.cdiv(out_features, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    inner = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k_start in range(0, in_features, BLOCK_K):
        k = k_start + inner
        x = tl.load(
            x_ptr + rows[:, None] * in_features + k[None, :],
            mask=(rows[:, None] < n_rows) & (k[None, :] < in_features),
            other=0.0,
        ).to(tl.float16)
        weight = tl.load(
            weight_ptr + cols[:, None] * in_features + k[None, :],
            mask=(cols[:, None] < out_features) & (k[None, :] < in_features),
            other=0.0,
        )
        accumulator += tl.dot(x, tl.trans(weight), out_dtype=tl.float32)

    value = accumulator + tl.load(
        bias_ptr + cols, mask=cols < out_features, other=0.0
    )[None, :].to(tl.float32)
    if FUSE_GELU:
        value = 0.5 * value * (1.0 + tl.erf(value * 0.7071067811865476))
    tl.store(
        out_ptr + rows[:, None] * out_features + cols[None, :],
        value,
        mask=(rows[:, None] < n_rows) & (cols[None, :] < out_features),
    )


@triton.jit
def _mixed_linear_add_layer_norm_kernel(
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
    N_FEATURES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """D128 projection with residual-add and LayerNorm in one program."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N_FEATURES)
    inner = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, N_FEATURES), tl.float32)

    for k_start in range(0, N_FEATURES, BLOCK_K):
        k = k_start + inner
        x = tl.load(
            x_ptr + rows[:, None] * N_FEATURES + k[None, :],
            mask=(rows[:, None] < n_rows) & (k[None, :] < N_FEATURES),
            other=0.0,
        ).to(tl.float16)
        weight = tl.load(
            weight_ptr + cols[:, None] * N_FEATURES + k[None, :],
            mask=k[None, :] < N_FEATURES,
            other=0.0,
        )
        accumulator += tl.dot(x, tl.trans(weight), out_dtype=tl.float32)

    projected = accumulator + tl.load(bias_ptr + cols)[None, :].to(tl.float32)
    residual = projected + tl.load(
        residual_ptr + rows[:, None] * N_FEATURES + cols[None, :],
        mask=rows[:, None] < n_rows,
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(residual, axis=1) / N_FEATURES
    centered = residual - mean[:, None]
    variance = tl.sum(centered * centered, axis=1) / N_FEATURES
    normalized = centered * tl.rsqrt(variance + EPS)[:, None]
    normalized = (
        normalized * tl.load(norm_weight_ptr + cols)[None, :].to(tl.float32)
        + tl.load(norm_bias_ptr + cols)[None, :].to(tl.float32)
    )
    mask = rows[:, None] < n_rows
    offsets = rows[:, None] * N_FEATURES + cols[None, :]
    tl.store(residual_out_ptr + offsets, residual, mask=mask)
    tl.store(normalized_out_ptr + offsets, normalized, mask=mask)


@triton.jit
def _mixed_ffn_add_layer_norm_kernel(
    x_ptr,
    in_weight_ptr,
    in_bias_ptr,
    out_weight_ptr,
    out_bias_ptr,
    residual_ptr,
    norm_weight_ptr,
    norm_bias_ptr,
    residual_out_ptr,
    normalized_out_ptr,
    n_rows,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Complete D128 FFN, residual-add, and LayerNorm in one program."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, 128)
    row_mask = rows < n_rows

    x = tl.load(
        x_ptr + rows[:, None] * 128 + cols[None, :],
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float16)
    in_weight = tl.load(
        in_weight_ptr + cols[:, None] * 128 + cols[None, :]
    )
    hidden = tl.dot(x, tl.trans(in_weight), out_dtype=tl.float32)
    hidden += tl.load(in_bias_ptr + cols)[None, :].to(tl.float32)
    hidden = 0.5 * hidden * (
        1.0 + tl.erf(hidden * 0.7071067811865476)
    )

    out_weight = tl.load(
        out_weight_ptr + cols[:, None] * 128 + cols[None, :]
    )
    projected = tl.dot(
        hidden.to(tl.float16), tl.trans(out_weight), out_dtype=tl.float32
    )
    projected += tl.load(out_bias_ptr + cols)[None, :].to(tl.float32)
    residual = projected + tl.load(
        residual_ptr + rows[:, None] * 128 + cols[None, :],
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(residual, axis=1) / 128.0
    centered = residual - mean[:, None]
    variance = tl.sum(centered * centered, axis=1) / 128.0
    normalized = centered * tl.rsqrt(variance + EPS)[:, None]
    normalized = (
        normalized * tl.load(norm_weight_ptr + cols)[None, :].to(tl.float32)
        + tl.load(norm_bias_ptr + cols)[None, :].to(tl.float32)
    )
    offsets = rows[:, None] * 128 + cols[None, :]
    tl.store(residual_out_ptr + offsets, residual, mask=row_mask[:, None])
    tl.store(normalized_out_ptr + offsets, normalized, mask=row_mask[:, None])


@triton.jit
def _add_layer_norm_mixed_output_kernel(
    x_ptr,
    update_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    normalized_ptr,
    n_rows,
    N_FEATURES: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_FEATURES
    offsets = row * N_FEATURES + cols
    residual = (
        tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        + tl.load(update_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    )
    mean = tl.sum(residual, axis=0) / N_FEATURES
    centered = residual - mean
    variance = tl.sum(centered * centered, axis=0) / N_FEATURES
    normalized = centered * tl.rsqrt(variance + EPS)
    normalized = (
        normalized * tl.load(weight_ptr + cols, mask=mask, other=0.0)
        + tl.load(bias_ptr + cols, mask=mask, other=0.0)
    )
    tl.store(residual_ptr + offsets, residual, mask=mask)
    tl.store(normalized_ptr + offsets, normalized, mask=mask)


def mixed_linear(
    x: torch.Tensor,
    linear: nn.Linear,
    *,
    output_dtype: torch.dtype,
    fuse_gelu: bool = False,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    group_m: int = 8,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    if not (
        x.is_cuda
        and x.is_contiguous()
        and x.dtype in (torch.float16, torch.float32)
        and linear.weight.is_cuda
        and linear.weight.dtype == torch.float16
        and linear.bias is not None
    ):
        raise ValueError("unsupported mixed projection inputs")

    in_features = linear.in_features
    out_features = linear.out_features
    n_rows = x.numel() // in_features
    output = torch.empty(
        (*x.shape[:-1], out_features), device=x.device, dtype=output_dtype
    )
    grid = (
        triton.cdiv(n_rows, block_m) * triton.cdiv(out_features, block_n),
    )
    _mixed_linear_kernel[grid](
        x,
        linear.weight,
        linear.bias,
        output,
        n_rows,
        in_features=in_features,
        out_features=out_features,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        FUSE_GELU=fuse_gelu,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def mixed_linear_add_layer_norm(
    x: torch.Tensor,
    linear: nn.Linear,
    residual: torch.Tensor,
    norm: nn.LayerNorm,
    *,
    block_m: int,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
    normalized_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse a D128 mixed projection, residual addition, and FP32 LayerNorm."""
    if not (
        x.is_cuda
        and x.is_contiguous()
        and x.dtype in (torch.float16, torch.float32)
        and residual.is_cuda
        and residual.is_contiguous()
        and residual.dtype == torch.float32
        and x.shape[:-1] == residual.shape[:-1]
        and x.shape[-1] == residual.shape[-1]
        and x.shape[-1] in (32, 128)
        and linear.in_features == linear.out_features == x.shape[-1]
        and linear.weight.is_cuda
        and linear.weight.dtype == torch.float16
        and linear.bias is not None
        and norm.normalized_shape == (x.shape[-1],)
    ):
        raise ValueError("unsupported fused projection/residual/LayerNorm inputs")

    n_features = x.shape[-1]
    n_rows = x.numel() // n_features
    residual_out = torch.empty_like(residual)
    normalized_out = torch.empty_like(residual, dtype=normalized_dtype)
    _mixed_linear_add_layer_norm_kernel[(triton.cdiv(n_rows, block_m),)](
        x,
        linear.weight,
        linear.bias,
        residual,
        norm.weight,
        norm.bias,
        residual_out,
        normalized_out,
        n_rows,
        EPS=norm.eps,
        N_FEATURES=n_features,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return residual_out, normalized_out


def mixed_ffn_add_layer_norm(
    x: torch.Tensor,
    linear_in: nn.Linear,
    linear_out: nn.Linear,
    residual: torch.Tensor,
    norm: nn.LayerNorm,
    *,
    block_m: int = 16,
    num_warps: int = 4,
    num_stages: int = 2,
    normalized_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse the complete D128 mixed FFN, residual, and FP32 LayerNorm."""
    if not (
        x.is_cuda
        and x.is_contiguous()
        and x.dtype in (torch.float16, torch.float32)
        and residual.is_cuda
        and residual.is_contiguous()
        and residual.dtype == torch.float32
        and x.shape == residual.shape
        and x.shape[-1] == 128
        and linear_in.in_features == linear_in.out_features == 128
        and linear_out.in_features == linear_out.out_features == 128
        and linear_in.weight.dtype == linear_out.weight.dtype == torch.float16
        and linear_in.bias is not None
        and linear_out.bias is not None
        and norm.normalized_shape == (128,)
    ):
        raise ValueError("unsupported fused D128 FFN inputs")

    n_rows = x.numel() // 128
    residual_out = torch.empty_like(residual)
    normalized_out = torch.empty_like(residual, dtype=normalized_dtype)
    _mixed_ffn_add_layer_norm_kernel[(triton.cdiv(n_rows, block_m),)](
        x,
        linear_in.weight,
        linear_in.bias,
        linear_out.weight,
        linear_out.bias,
        residual,
        norm.weight,
        norm.bias,
        residual_out,
        normalized_out,
        n_rows,
        EPS=norm.eps,
        BLOCK_M=block_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return residual_out, normalized_out


def mixed_add_layer_norm(
    x: torch.Tensor,
    update: torch.Tensor,
    norm: nn.LayerNorm,
    *,
    normalized_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 residual/LayerNorm with a selectable normalized-output dtype."""
    if not (
        x.is_cuda
        and x.is_contiguous()
        and x.dtype == torch.float32
        and update.is_cuda
        and update.is_contiguous()
        and update.dtype == torch.float32
        and x.shape == update.shape
        and normalized_dtype in (torch.float16, torch.float32)
    ):
        raise ValueError("unsupported mixed LayerNorm inputs")
    n_features = x.shape[-1]
    n_rows = x.numel() // n_features
    block_size = triton.next_power_of_2(n_features)
    if block_size <= 64:
        num_warps = 2
    elif block_size <= 256:
        num_warps = 1
    else:
        num_warps = 8
    residual = torch.empty_like(x)
    normalized = torch.empty_like(x, dtype=normalized_dtype)
    _add_layer_norm_mixed_output_kernel[(n_rows,)](
        x,
        update,
        norm.weight,
        norm.bias,
        residual,
        normalized,
        n_rows,
        N_FEATURES=n_features,
        EPS=norm.eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return residual, normalized
