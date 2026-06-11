# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dump per-rank optimizer parameter sharding info to JSON.

Useful for replaying optimizer steps in tests/microbenchmarks without
re-running the model. The dump captures everything needed to reconstruct
the parameter layout: global/local shape, per-param mesh, placements,
chunk offsets, and Megatron attributes (TP partition, QKV, M-FSDP name).

Optimizer-agnostic: works with any PyTorch optimizer whose params are
DTensors managed by Megatron-FSDP. Plain (non-DTensor) params are also
handled — they're dumped with `local_shape == global_shape` and the
mesh/placements fields set to None.
"""

import json
import logging
import os
import pathlib
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)


def dump_optimizer_parameters(
    optimizer: torch.optim.Optimizer,
    out_path: str | os.PathLike,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    """Write a per-rank JSON snapshot of `optimizer`'s parameter sharding.

    Each rank writes its own file. If `out_path` ends in `.json`, the
    suffix becomes `.rank<N>.json`; otherwise `.rank<N>.json` is appended.

    Args:
        optimizer: any PyTorch optimizer over DTensor params.
        out_path: output path; rank suffix is added.
        extra_meta: optional dict merged into the top-level JSON under
            the `extra` key. Useful for caller-specific knobs (e.g.,
            optimizer hyperparameters not captured by `param_groups`).
    """
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    # Preserve param-group structure (no hyperparameters; just the partition).
    # Group hyperparameters (lr, betas, weight_decay, ...) are intentionally
    # not recorded — callers that care should pass them via `extra_meta`.
    groups = [
        {"params": [_dump_param(p, optimizer.state.get(p, {})) for p in group["params"]]}
        for group in optimizer.param_groups
    ]
    spec: dict[str, Any] = {"world_size": world_size, "rank": rank, "groups": groups}
    if extra_meta is not None:
        spec["extra"] = extra_meta

    out = pathlib.Path(out_path)
    rank_out = (
        out.with_suffix(f".rank{rank}.json")
        if out.suffix == ".json"
        else out.parent / f"{out.name}.rank{rank}.json"
    )
    rank_out.parent.mkdir(parents=True, exist_ok=True)
    rank_out.write_text(json.dumps(spec, indent=2))
    if rank == 0:
        logger.info("Optimizer inputs dumped to %s (and rank-N siblings)", rank_out)


def _dump_param(p: torch.Tensor, state: dict[str, Any]) -> dict[str, Any]:
    """Convert one optimizer param to a JSON-serializable dict."""
    info: dict[str, Any] = {
        "name": _param_name(p),
        "global_shape": tuple(int(s) for s in p.shape),
        "dtype": str(p.dtype),
        "is_qkv": _is_qkv(p),
        "partition_dim": getattr(p, "partition_dim", None),
        "tensor_model_parallel": getattr(p, "tensor_model_parallel", None),
    }

    if isinstance(p, DTensor):
        local = p.to_local()
        mesh = p.device_mesh
        info["local_shape"] = tuple(int(s) for s in local.shape)
        info["local_offsets"] = _local_offsets(local)
        info["mesh_shape"] = tuple(int(s) for s in mesh.shape)
        info["mesh_dim_names"] = list(mesh.mesh_dim_names) if mesh.mesh_dim_names else None
        info["placements"] = [repr(pl) for pl in p.placements]
    else:
        info["local_shape"] = tuple(int(s) for s in p.shape)
        info["local_offsets"] = None
        info["mesh_shape"] = None
        info["mesh_dim_names"] = None
        info["placements"] = None

    mom = state.get("momentum_buffer")
    if mom is not None:
        info["momentum_dtype"] = str(mom.dtype)
        mom_local = mom.to_local() if isinstance(mom, DTensor) else mom
        info["momentum_local_shape"] = tuple(int(s) for s in mom_local.shape)
    else:
        info["momentum_dtype"] = None
        info["momentum_local_shape"] = None

    return info


def _param_name(p: torch.Tensor) -> str | None:
    """M-FSDP attaches `megatron_fsdp_param_name`; orig_param wraps it for FSDP units."""
    name = getattr(p, "megatron_fsdp_param_name", None)
    if name is None:
        orig = getattr(p, "orig_param", None)
        if orig is not None:
            name = getattr(orig, "megatron_fsdp_param_name", None)
    return name


def _is_qkv(p: torch.Tensor) -> bool:
    return bool(getattr(p, "is_qkv", False)) or bool(
        getattr(getattr(p, "orig_param", None), "is_qkv", False)
    )


def _local_offsets(local: torch.Tensor) -> tuple[int, ...] | None:
    """Read chunk offsets from `local.__create_chunk_list__()` if present."""
    fn = getattr(local, "__create_chunk_list__", None)
    if fn is None:
        return None
    try:
        chunks = fn()
    except Exception:
        return None
    if not chunks:
        return None
    return tuple(int(o) for o in chunks[0].offsets)
