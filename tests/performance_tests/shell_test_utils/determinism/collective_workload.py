# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bounded TP/SP workload for the captured-collective CI pilot.

Run through tools.determinism.capture_recipe so early policy and capture bindings
are installed before Core imports. This synthetic workload is not model acceptance.
"""

from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

MAPPINGS = {
    "copy": "copy_to_tensor_model_parallel_region",
    "reduce": "reduce_from_tensor_model_parallel_region",
    "gather_first": "gather_from_sequence_parallel_region",
    "scatter_first": "reduce_scatter_to_sequence_parallel_region",
    "gather_last": "all_gather_last_dim_from_tensor_parallel_region",
    "scatter_last": "reduce_scatter_last_dim_to_tensor_parallel_region",
}


def _tensor(shape, dtype, seed: int, rank: int, layout: str, device: str = "cuda") -> torch.Tensor:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed + rank)
    values = torch.randn(shape, dtype=torch.float64, generator=generator).to(dtype)
    values.reshape(-1)[0] = rank + 1
    if layout == "offset":
        backing = torch.zeros((shape[0] + 1, *shape[1:]), dtype=dtype, device=device)
        view = backing[1:]
        view.copy_(values)
        assert view.storage_offset() > 0 and view.is_contiguous()
        return view
    if layout != "strided":
        raise ValueError("Unsupported pilot tensor layout")
    value = values.transpose(0, 2).contiguous().transpose(0, 2).to(device)
    assert not value.is_contiguous()
    return value


def main(argv: list[str] | None = None) -> int:
    """Exercise six mappings, both precisions/layouts and pair/world groups."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deterministic-mode", action="store_true", required=True)
    parser.parse_args(argv)
    import torch

    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Run the workload through the early-policy capture launcher")
    if int(os.environ["WORLD_SIZE"]) not in (4, 8):
        raise ValueError("The CI pilot requires four or eight ranks on one node")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    from megatron.core.tensor_parallel import mappings

    torch.distributed.init_process_group("nccl")
    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    try:
        for size in (2, world):
            selected = None
            for start in range(0, world, size):
                members = list(range(start, start + size))
                options = torch.distributed.ProcessGroupNCCL.Options(
                    is_high_priority_stream=size == world
                )
                if size == world:
                    options.config.min_ctas = 2
                    options.config.max_ctas = 4
                group = torch.distributed.new_group(members, backend="nccl", pg_options=options)
                if rank in members:
                    selected = group
            try:
                for case, name in MAPPINGS.items():
                    for dtype in (torch.float32, torch.bfloat16):
                        for layout in ("offset", "strided"):
                            shape = [17, 2, 64]
                            dimension = 0 if case.endswith("first") else 2
                            if case.startswith("scatter"):
                                shape[dimension] *= size
                            output_shape = shape[:]
                            if case.startswith("gather"):
                                output_shape[dimension] *= size
                            elif case.startswith("scatter"):
                                output_shape[dimension] //= size
                            value = _tensor(shape, dtype, 1700, rank, layout).requires_grad_(True)
                            gradient = _tensor(output_shape, dtype, 9100, rank, layout)
                            output = getattr(mappings, name)(value, group=selected)
                            output.backward(gradient)
                            assert value.grad is not None and torch.isfinite(value.grad).all()
                torch.cuda.synchronize()
                torch.distributed.barrier()
            finally:
                if selected is not None:
                    torch.distributed.destroy_process_group(selected)
    finally:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
