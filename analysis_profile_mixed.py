#!/usr/bin/env python3
"""Read-only profiler driver for the current mixed-precision candidate.

This is an analysis utility. It does not alter benchmark or implementation
behavior and deliberately constructs only the optimized model.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

from mixed_precision import MixedPrecisionTransformer
from mixed_precision_benchmark import PreallocatedMicrobatchModel
from torch_transformer_benchmark import TransformerConfig


CASES = {
    6: TransformerConfig(10000, 128, 128, 4, 128, 4, True),
    8: TransformerConfig(64, 128, 1024, 4, 1024, 4, True),
    10: TransformerConfig(64, 128, 128, 2, 128, 4, True),
    11: TransformerConfig(64, 128, 128, 16, 128, 4, True),
    13: TransformerConfig(64, 1024, 128, 4, 128, 4, True),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, required=True, choices=CASES)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    config = CASES[args.case]
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda")
    model = MixedPrecisionTransformer(config).to(device).eval()
    model.assume_all_tokens_valid = True
    model.convert_projections_to_fp16()
    model = torch.compile(model, mode="reduce-overhead")
    if args.case == 6:
        model = PreallocatedMicrobatchModel(model, 64)

    x = torch.randn(
        config.batch_size,
        config.seq_len,
        config.d_model,
        device=device,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(x, None)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            for _ in range(args.iterations):
                model(x, None)
            torch.cuda.synchronize()

    events = prof.events()
    cuda_events = [event for event in events if event.device_type.name == "CUDA"]
    cuda_by_name: dict[str, list[float]] = defaultdict(list)
    for event in cuda_events:
        cuda_by_name[event.name].append(event.self_device_time_total)
    total_cuda_us = sum(sum(times) for times in cuda_by_name.values())
    print(f"case={args.case}")
    print(f"config={config}")
    print(f"iterations={args.iterations}")
    print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"cuda_kernel_events={len(cuda_events)}")
    print(f"summed_self_cuda_ms={total_cuda_us / 1000.0:.6f}")
    print(f"peak_allocated_mib={torch.cuda.max_memory_allocated() / 2**20:.3f}")
    print(f"peak_reserved_mib={torch.cuda.max_memory_reserved() / 2**20:.3f}")
    print()
    print("CUDA KERNEL EVENTS")
    for name, times in sorted(
        cuda_by_name.items(), key=lambda item: sum(item[1]), reverse=True
    ):
        print(
            f"{sum(times) / 1000.0:10.6f} ms  {len(times):5d} calls  "
            f"{sum(times) / len(times):9.3f} us/call  {name}"
        )
    print()
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=args.top,
            max_name_column_width=100,
            max_shapes_column_width=80,
        )
    )


if __name__ == "__main__":
    main()
