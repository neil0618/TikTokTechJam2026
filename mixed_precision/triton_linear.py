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
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """D128 projection with residual-add and LayerNorm in one program."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, 128)
    inner = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, 128), tl.float32)

    for k_start in range(0, 128, BLOCK_K):
        k = k_start + inner
        x = tl.load(
            x_ptr + rows[:, None] * 128 + k[None, :],
            mask=(rows[:, None] < n_rows) & (k[None, :] < 128),
            other=0.0,
        ).to(tl.float16)
        weight = tl.load(
            weight_ptr + cols[:, None] * 128 + k[None, :],
            mask=k[None, :] < 128,
            other=0.0,
        )
        accumulator += tl.dot(x, tl.trans(weight), out_dtype=tl.float32)

    projected = accumulator + tl.load(bias_ptr + cols)[None, :].to(tl.float32)
    residual = projected + tl.load(
        residual_ptr + rows[:, None] * 128 + cols[None, :],
        mask=rows[:, None] < n_rows,
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
    mask = rows[:, None] < n_rows
    offsets = rows[:, None] * 128 + cols[None, :]
    tl.store(residual_out_ptr + offsets, residual, mask=mask)
    tl.store(normalized_out_ptr + offsets, normalized, mask=mask)


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse a D128 mixed projection, residual addition, and FP32 LayerNorm."""
    if not (
        x.is_cuda
        and x.is_contiguous()
        and x.dtype == torch.float16
        and residual.is_cuda
        and residual.is_contiguous()
        and residual.dtype == torch.float32
        and x.shape[:-1] == residual.shape[:-1]
        and x.shape[-1] == residual.shape[-1] == 128
        and linear.in_features == linear.out_features == 128
        and linear.weight.is_cuda
        and linear.weight.dtype == torch.float16
        and linear.bias is not None
        and norm.normalized_shape == (128,)
    ):
        raise ValueError("unsupported fused projection/residual/LayerNorm inputs")

    n_rows = x.numel() // 128
    residual_out = torch.empty_like(residual)
    normalized_out = torch.empty_like(residual)
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
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return residual_out, normalized_out
