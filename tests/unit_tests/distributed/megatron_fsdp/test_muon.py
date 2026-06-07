# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Muon optimizer correctness + microbenchmark tests.

Replays `FSDPTensorParallelMuon.step()` against fake DTensors
reconstructed from a captured per-rank dump of a real 2-node / 8-rank
training run. The dump JSON files live next to this test file and only
record parameter sharding (global shape, per-rank local shape,
placements, mesh layout). The optimizer is constructed here with
hardcoded hyperparameters typical for Muon training.

Two tests:
  * `test_muon_step` — pytest-benchmark perf measurement of one step.
  * `test_muon_step_numerics` — runs one FSDP+TP step, then for each param
    gathers across the DP group (mirroring FSDP Phase 2), runs a plain
    `emerging_optimizers.Muon` step over that single param (with manual QKV
    split where applicable), slices the reference output back to this rank's
    local shard, and asserts allclose with bf16-friendly tolerance. Each
    param is processed in isolation so peak HBM per rank is bounded by one
    param's DP-gathered footprint.

Launch:
    cd Megatron-LM
    torchrun --nproc_per_node=8 --standalone -m pytest \
        tests/unit_tests/distributed/megatron_fsdp/test_muon.py -v -s
"""

import json
import os
import pathlib
import re
from typing import Any

import pytest
import torch
import torch.distributed as dist
from packaging.version import Version
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from emerging_optimizers.orthogonalized_optimizers import Muon

from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    update_uneven_dtensor_chunk_metadata,
)
from megatron.core.optimizer.emerging_optimizers import (
    HAVE_EMERGING_OPTIMIZERS,
    FSDPTensorParallelMuon,
)


# QKV-fused parameter sub-split along dim 0 of the (num_query_groups, S, in_dim) view.
QKV_SPLIT_SHAPES = (1024, 128, 128)


SPEC_DIR = pathlib.Path(__file__).parent / "muon_inputs_fsdp4_tp2"
EXPECTED_WORLD_SIZE = 8
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))


pytestmark = [
    pytest.mark.skipif(
        Version(os.getenv("NVIDIA_PYTORCH_VERSION", "24.01")) <= Version("25.05"),
        reason="Skip emerging optimizer tests for LTS test",
    ),
    pytest.mark.skipif(
        not HAVE_EMERGING_OPTIMIZERS, reason="Muon tests require the emerging-optimizers package"
    ),
    pytest.mark.skipif(
        WORLD_SIZE != EXPECTED_WORLD_SIZE,
        reason=f"Dump captured at WORLD_SIZE={EXPECTED_WORLD_SIZE}, "
        f"got WORLD_SIZE={WORLD_SIZE}",
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is required for the Muon microbenchmark."
    ),
]


# `distributed_setup` fixture is provided by conftest.py in this directory.


# ---------- Spec loading ----------


def _load_spec(rank: int) -> dict[str, Any]:
    return json.loads((SPEC_DIR / f"rank{rank}.json").read_text())


_DTYPE_MAP = {
    "torch.float32": torch.float32,
    "torch.float": torch.float32,
    "torch.float64": torch.float64,
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
    "torch.half": torch.float16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
}


def _parse_dtype(s: str) -> torch.dtype:
    return _DTYPE_MAP[s]


_PLACEMENT_RE = re.compile(r"Shard\(dim=(-?\d+)\)")


def _parse_placement(s: str) -> Shard:
    m = _PLACEMENT_RE.fullmatch(s)
    if not m:
        raise ValueError(f"cannot parse placement: {s!r}")
    return Shard(int(m.group(1)))


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Row-major contiguous strides for a tensor of ``shape`` — no allocation.

    `DTensor.from_local(..., shape=global_shape)` requires `stride` whenever
    `shape` is given (PyTorch enforces "pass both or neither").
    """
    stride = [1] * len(shape)
    running = 1
    for i in reversed(range(len(shape))):
        stride[i] = running
        running *= shape[i]
    return tuple(stride)


# ---------- DTensor construction ----------


def _make_dtensor_from_spec(
    p_spec: dict[str, Any], mesh: DeviceMesh, device: torch.device
) -> DTensor:
    """Build a DTensor matching the recorded global shape + per-rank local shape.

    The dump captures the M-FSDP-specific layout where `_local_tensor` can
    reflect a TP-shard view while the placements declare additional sharding
    handled at gather time. We honor whatever the spec recorded: allocate the
    recorded `local_shape`, pass the recorded `placements`, and let
    `update_uneven_dtensor_chunk_metadata` install the chunk-list bookkeeping
    M-FSDP relies on.
    """
    dtype = _parse_dtype(p_spec["dtype"])
    local_shape = tuple(p_spec["local_shape"])
    global_shape = tuple(p_spec["global_shape"])
    placements = [_parse_placement(s) for s in p_spec["placements"]]

    # fp32 randn → cast to target dtype so we get sensible values for bf16/fp16 too.
    local = torch.randn(local_shape, dtype=torch.float32, device=device).to(dtype=dtype)
    dt = DTensor.from_local(
        local_tensor=local,
        device_mesh=mesh,
        placements=placements,
        shape=torch.Size(global_shape),
        stride=_contiguous_stride(global_shape),
        run_check=False,
    )
    update_uneven_dtensor_chunk_metadata(dt)
    return dt


# ---------- Mesh + process groups ----------


def _build_mesh(spec: dict[str, Any]) -> DeviceMesh:
    return init_device_mesh(
        "cuda", tuple(spec["mesh_shape"]), mesh_dim_names=tuple(spec["mesh_dim_names"])
    )


# ---------- Param + optimizer construction ----------


def _build_param(p_spec: dict[str, Any], mesh: DeviceMesh, device: torch.device) -> DTensor:
    """Build one DTensor param (with random `.grad` attached) from a slim spec."""
    p = _make_dtensor_from_spec(p_spec, mesh, device)
    if p_spec.get("is_qkv"):
        p.is_qkv = True
    # Random gradient matching the param's DTensor layout (fp32 randn → cast so
    # bf16/fp16 grads still get sensible values).
    grad = torch.randn_like(p, dtype=torch.float32).to(dtype=p.dtype)
    update_uneven_dtensor_chunk_metadata(grad)
    p.grad = grad
    return p


def _build_params(
    spec: dict[str, Any], mesh: DeviceMesh, device: torch.device
) -> list[dict[str, list[DTensor]]]:
    """Build DTensor params + random grads from the slim spec."""
    params = [_build_param(p_spec, mesh, device) for p_spec in spec["params"]]
    # Single param group; bench picks the hyperparameters.
    return [{"params": params}]


def _build_optimizer(
    param_groups: list[dict[str, list[DTensor]]], dp_group: dist.ProcessGroup
) -> FSDPTensorParallelMuon:
    """Construct a `FSDPTensorParallelMuon` with typical Muon hyperparameters.

    The JSON fixtures only describe parameter sharding; the choice of
    optimizer hyperparameters is the bench's concern. These values are
    representative for Muon training.
    """
    return FSDPTensorParallelMuon(
        params=param_groups,
        dp_group=dp_group,
        lr=8e-4,
        momentum=0.9,
        weight_decay=0.1,
        extra_scale_factor=0.2,
        tp_mode="blockwise",
        split_qkv=True,
        qkv_split_shapes=list(QKV_SPLIT_SHAPES),
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
    )


# ---------- Replicated reference Muon (for numerics test) ----------


def _gather_dp(local: torch.Tensor, dp_group: dist.ProcessGroup) -> torch.Tensor:
    """Gather per-rank local shards along dim 0 across the DP process group.

    Each rank passes its local shard; ranks may have different dim-0 sizes.
    Returns the DP-assembled tensor (per TP-rank — plain ``Muon`` on the
    result mirrors FSDP's ``tp_mode="blockwise"`` path, which does no
    TP-collective NS).
    """
    dp_world = dist.get_world_size(dp_group)
    all_shapes = [None] * dp_world
    dist.all_gather_object(all_shapes, tuple(local.shape), group=dp_group)

    # NCCL all_gather only supports uniform-size inputs across ranks, so pad
    # each rank's local along dim 0 to the max dim-0 before gathering; dim 1+
    # is already uniform across DP (DP shards only dim 0).
    max_dim0 = max(s[0] for s in all_shapes)
    other_dims = tuple(local.shape[1:])
    padded = torch.zeros((max_dim0,) + other_dims, dtype=local.dtype, device=local.device)
    if local.shape[0] > 0:
        padded[: local.shape[0]] = local

    output = torch.empty(
        (dp_world * max_dim0,) + other_dims, dtype=local.dtype, device=local.device
    )
    dist.all_gather_into_tensor(output, padded, group=dp_group)

    parts = [
        output[r * max_dim0 : r * max_dim0 + s[0]] for r, s in enumerate(all_shapes) if s[0] > 0
    ]
    return torch.cat(parts, dim=0)


def _build_reference_optimizer(params: list[torch.Tensor]) -> Muon:
    """Plain `emerging_optimizers.Muon` over a tiny list of leaf tensors.

    Mirrors `_build_optimizer`'s hyperparameters. The reference is invoked
    per-FSDP-param (1 tensor for non-QKV, 3 for QKV) so its momentum state is
    freed when the local instance is GC'd, keeping per-iteration HBM bounded.
    Plain Muon defaults `nesterov=False`; `FSDPTensorParallelMuon` inherits
    `nesterov=True` from `TensorParallelMuon`, so we pass it explicitly.
    """
    return Muon(
        params=[{"params": params}],
        lr=8e-4,
        momentum=0.9,
        weight_decay=0.1,
        nesterov=True,
        extra_scale_factor=0.2,
    )


def _reference_step_single(
    is_qkv: bool, full_p: torch.Tensor, full_g: torch.Tensor
) -> torch.Tensor:
    """One plain-Muon step on a single FSDP param's full-tensor view.

    For QKV-fused params, splits along the (num_query_groups, sum(QKV), in_dim)
    view to mirror `TensorParallelMuon.orthogonalize` — each of Q/K/V is
    orthogonalized independently — then re-fuses the result.

    Returns the updated full tensor (same shape as ``full_p``).
    """
    if not is_qkv:
        full_p.grad = full_g
        _build_reference_optimizer([full_p]).step()
        return full_p

    g0, in_dim = full_p.shape
    s = sum(QKV_SPLIT_SHAPES)
    assert g0 % s == 0, f"QKV global dim-0 {g0} not divisible by sum(qkv_split_shapes)={s}"
    num_query_groups = g0 // s
    p_view = full_p.view(num_query_groups, s, in_dim)
    g_view = full_g.view(num_query_groups, s, in_dim)
    p_parts = [
        t.reshape(-1, in_dim).clone().contiguous()
        for t in torch.split(p_view, list(QKV_SPLIT_SHAPES), dim=1)
    ]
    g_parts = [
        t.reshape(-1, in_dim).clone().contiguous()
        for t in torch.split(g_view, list(QKV_SPLIT_SHAPES), dim=1)
    ]
    for pp, gp in zip(p_parts, g_parts):
        pp.grad = gp
    _build_reference_optimizer(p_parts).step()
    # Re-fuse: each updated part back to (num_query_groups, qkv_dim, in_dim), cat, view.
    refused_parts = [pp.view(num_query_groups, -1, in_dim) for pp in p_parts]
    return torch.cat(refused_parts, dim=1).reshape(g0, in_dim).contiguous()


def _slice_dp_full_to_local(
    local_dim0: int, full_tensor: torch.Tensor, dp_group: dist.ProcessGroup
) -> torch.Tensor:
    """Slice the DP-gathered ``full_tensor`` to this rank's local shard.

    Offset = sum of preceding DP ranks' dim-0 sizes (Megatron-FSDP shards
    along dim 0 across DP). Mirrors the reshard step FSDP does after
    Phase 3, scoped to the DP group rather than the full mesh.
    """
    dp_world = dist.get_world_size(dp_group)
    dp_rank = dist.get_rank(dp_group)
    all_dim0 = [None] * dp_world
    dist.all_gather_object(all_dim0, local_dim0, group=dp_group)
    offset = sum(all_dim0[:dp_rank])
    return full_tensor[offset : offset + local_dim0].contiguous()


def _reference_local_for_param(
    ref_p: torch.Tensor, p_spec: dict[str, Any], dp_group: dist.ProcessGroup
) -> torch.Tensor:
    """Compute one param's reference post-step local shard.

    For boundary params (any rank has a non-empty local that differs from the
    global shape), gathers across DP — matching FSDP Phase 2 — runs Muon on
    the full-tensor view, and slices the result back to this rank's local
    shard. For non-boundary params, runs Muon directly on the local shard.

    Empty local shards on non-boundary params are passed through unchanged.
    Requires ``ref_p.grad`` to be attached.
    """
    global_shape = tuple(p_spec["global_shape"])
    is_qkv = p_spec.get("is_qkv", False)

    # Mirror FSDP's `_needs_boundary_gather`: a param is "boundary" if any
    # rank has a non-empty local with shape != global.
    local_is_boundary = ref_p.numel() > 0 and global_shape != tuple(ref_p.shape)
    boundary_votes = [None] * dist.get_world_size()
    dist.all_gather_object(boundary_votes, local_is_boundary)

    if not any(boundary_votes):
        return _reference_step_single(is_qkv, ref_p, ref_p.grad)

    # Every rank participates in the gather collectives, regardless of whether
    # its local shard is empty — skipping asymmetrically misaligns subsequent
    # collective calls. Per-TP-rank NS runs locally because tp_mode="blockwise".
    full_p = _gather_dp(ref_p, dp_group)
    full_g = _gather_dp(ref_p.grad, dp_group)
    full_ref_after = _reference_step_single(is_qkv, full_p, full_g)
    return _slice_dp_full_to_local(ref_p.shape[0], full_ref_after, dp_group)


# ---------- Tests ----------


def test_muon_step(distributed_setup: dict[str, Any], benchmark) -> None:
    """Build the full optimizer state and benchmark `step()`.

    pytest-benchmark's `pedantic` mode runs the step a fixed number of times
    (deterministic across ranks) and reports min/median/max/mean/stddev.
    """
    rank = distributed_setup["rank"]
    world_size = distributed_setup["world_size"]
    device = distributed_setup["device"]

    spec = _load_spec(rank)
    assert (
        spec["world_size"] == world_size
    ), f"spec world_size={spec['world_size']} != launch world_size={world_size}"

    mesh = _build_mesh(spec)
    dp_group = mesh.get_group("dp_cp")

    param_groups = _build_params(spec, mesh, device)
    optimizer = _build_optimizer(param_groups, dp_group=dp_group)

    def _one_step():
        optimizer.step()
        torch.cuda.synchronize()

    benchmark.pedantic(_one_step, rounds=5, warmup_rounds=3, iterations=1)
    dist.barrier()


def test_muon_step_numerics(distributed_setup: dict[str, Any]) -> None:
    """Compare `FSDPTensorParallelMuon` against a fully-replicated `Muon` reference.

    Flow:
        1. Build DTensor params + grads from the spec.
        2. Clone the local shards into `reference_params` (plain tensors).
        3. Run FSDP+TP Muon on the DTensors (mutates in place).
        4. Run plain `emerging_optimizers.Muon` on `reference_params`. For
           boundary params, gather across DP (mirroring FSDP Phase 2), run
           Muon on the full view (manual QKV split where applicable), and
           slice the result back to the local shard. For non-boundary params,
           run Muon directly on the local shard.
        5. Assert each FSDP post-step shard is close to its reference local
           with `atol=rtol=1e-2`.
    """
    rank = distributed_setup["rank"]
    world_size = distributed_setup["world_size"]
    device = distributed_setup["device"]

    spec = _load_spec(rank)
    assert (
        spec["world_size"] == world_size
    ), f"spec world_size={spec['world_size']} != launch world_size={world_size}"

    mesh = _build_mesh(spec)
    dp_group = mesh.get_group("dp_cp")

    param_groups = _build_params(spec, mesh, device)
    fsdp_params = param_groups[0]["params"]

    # Build reference param shards — a clone of each pre-step local, with the
    # grad attached. The reference Muon will mutate these in place (boundary
    # params via a gathered full-tensor view, non-boundary via the local shard
    # directly). Grads are captured as views, not clones, because
    # `FSDPTensorParallelMuon.step()` doesn't mutate them.
    reference_params = [p.to_local().clone() for p in fsdp_params]
    for ref_p, fsdp_p in zip(reference_params, fsdp_params):
        ref_p.grad = fsdp_p.grad.to_local()

    # Run FSDP+TP Muon step on the DTensors.
    fsdp_opt = _build_optimizer(param_groups, dp_group=dp_group)
    fsdp_opt.step()
    torch.cuda.synchronize()

    # Run the replicated reference Muon. This pass only touches the reference
    # shards and the spec — it never reads `fsdp_params`.
    reference_params = [
        _reference_local_for_param(ref_p, p_spec, dp_group)
        for ref_p, p_spec in zip(reference_params, spec["params"])
    ]

    # Compare each FSDP post-step shard against the replicated reference.
    for i, (fsdp_p, ref_p) in enumerate(zip(fsdp_params, reference_params)):
        torch.testing.assert_close(
            fsdp_p.to_local().float(),
            ref_p.float(),
            atol=1e-2,
            rtol=1e-2,
            msg=lambda m, i=i: (
                f"rank{rank} param[{i}] shape={tuple(spec['params'][i]['global_shape'])}: {m}"
            ),
        )
