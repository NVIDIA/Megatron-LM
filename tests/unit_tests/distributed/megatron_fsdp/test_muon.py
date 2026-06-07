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


def _gather_dp(dt: DTensor, dp_group: dist.ProcessGroup) -> torch.Tensor:
    """Gather a DTensor across the DP process group only (matching FSDP Phase 2).

    Concatenates DP-rank shards along dim 0. Each TP rank gets its own
    DP-assembled tensor (TP-shard along the TP-sharded dims, if any).
    Plain ``Muon`` on the result mirrors FSDP's ``tp_mode="blockwise"`` path,
    which does no TP-collective NS — each TP rank's chunk is orthogonalized
    independently.

    This intentionally bypasses the captured DTensor placements (which are
    ambiguous in this fixture for length-1 lists and clobbered when empty
    ``_local_tensor``s alias across DTensors). Instead we trust the per-rank
    local shapes within the DP group.
    """
    dp_world = dist.get_world_size(dp_group)
    local = dt.to_local()
    all_shapes = [None] * dp_world
    dist.all_gather_object(all_shapes, tuple(local.shape), group=dp_group)

    # NCCL all_gather only supports uniform-size inputs across ranks, so pad
    # each rank's local along dim 0 to the max dim-0 before gathering; dim 1+
    # is already uniform across DP (DP shards only dim 0).
    max_dim0 = max(s[0] for s in all_shapes)
    other_dims = tuple(local.shape[1:])
    padded = torch.zeros((max_dim0,) + other_dims, dtype=dt.dtype, device=dt.device)
    if local.shape[0] > 0:
        padded[: local.shape[0]] = local

    output = torch.empty((dp_world * max_dim0,) + other_dims, dtype=dt.dtype, device=dt.device)
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

    Runs the FSDP+TP Muon step once, then per param: restores the DTensor's
    initial local state, all-gathers the full-tensor view on every rank, runs
    a plain `emerging_optimizers.Muon` step on that single param (manual QKV
    split where applicable), slices the reference output back to this rank's
    local shard, and asserts allclose with `atol=rtol=1e-2`. Each param is
    processed in isolation so peak HBM stays at one param's full-tensor size.
    Also asserts every value is finite (no NaN/Inf).
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

    # Snapshot per-rank initial param shards so we can restore them after the
    # FSDP step and feed t=0 values to the reference. Grads aren't snapshotted
    # because `FSDPTensorParallelMuon.step()` reads `p.grad` but doesn't mutate
    # it in place.
    init_local_params = [p.to_local().clone() for p in fsdp_params]

    # Run FSDP+TP Muon step, capture per-rank local results.
    fsdp_opt = _build_optimizer(param_groups, dp_group=dp_group)
    fsdp_opt.step()
    torch.cuda.synchronize()
    fsdp_after_local = [p.to_local().clone() for p in fsdp_params]

    # Restore param shards so subsequent gathers see t=0 values. `to_local()`
    # returns the underlying storage, so in-place `copy_` propagates back.
    for p, p0 in zip(fsdp_params, init_local_params):
        p.to_local().copy_(p0)
    del init_local_params

    nonfinite = []
    mismatches = []
    world = dist.get_world_size()
    for i, (fsdp_p, fsdp_local_after) in enumerate(zip(fsdp_params, fsdp_after_local)):
        # Mirror FSDP's `_needs_boundary_gather` / `_get_boundary_gather_param_indices`:
        # a param is "boundary" if any rank has a non-empty local with shape != global.
        # FSDP gathers boundary params; non-boundary params are processed locally
        # (this rank's local shard is the full slab it owns, or empty).
        fsdp_local = fsdp_p.to_local()
        local_is_boundary = fsdp_local.numel() > 0 and tuple(fsdp_p.shape) != tuple(
            fsdp_local.shape
        )
        boundary_votes = [None] * world
        dist.all_gather_object(boundary_votes, local_is_boundary)
        is_boundary = any(boundary_votes)

        if is_boundary:
            # Every rank participates in the gather collectives, regardless of
            # whether its local shard is empty — skipping asymmetrically misaligns
            # subsequent collective calls. Gather across DP only (matching FSDP
            # Phase 2); per-TP-rank NS runs locally because tp_mode="blockwise".
            full_p = _gather_dp(fsdp_p, dp_group)
            full_g = _gather_dp(fsdp_p.grad, dp_group)
            full_ref_after = _reference_step_single(
                spec["params"][i].get("is_qkv", False), full_p, full_g
            )
            ref_local = _slice_dp_full_to_local(
                fsdp_p.to_local().shape[0], full_ref_after, dp_group
            )
            del full_p, full_g, full_ref_after
        elif fsdp_local_after.numel() > 0:
            # Non-boundary: FSDP runs Muon locally on this rank's local shard.
            # Mirror that with plain Muon on the same local tensor.
            local_p = fsdp_p.to_local().clone()
            local_g = fsdp_p.grad.to_local().clone()
            ref_local = _reference_step_single(
                spec["params"][i].get("is_qkv", False), local_p, local_g
            )
            del local_p, local_g
        else:
            # Empty local on a non-boundary param — nothing to compare here.
            continue

        if not torch.isfinite(fsdp_local_after).all() or not torch.isfinite(ref_local).all():
            nonfinite.append(i)
        elif not torch.allclose(fsdp_local_after.float(), ref_local.float(), atol=1e-2, rtol=1e-2):
            diff = (fsdp_local_after.float() - ref_local.float()).abs()
            mismatches.append(
                (i, diff.max().item(), diff.mean().item(), tuple(spec["params"][i]["global_shape"]))
            )
        del ref_local

    dist.barrier()

    if rank == 0:
        n_checked = sum(1 for p in fsdp_after_local if p.numel() > 0)
        print(
            f"\n[muon-microbench numerics] params_checked={n_checked} "
            f"nonfinite={len(nonfinite)} mismatches={len(mismatches)}"
        )
        for idx, mx, mn, shape in mismatches[:10]:
            print(f"  param[{idx}] shape={shape} max_abs_diff={mx:.3e} mean={mn:.3e}")

    assert not nonfinite, f"rank{rank}: {len(nonfinite)} params contain NaN/Inf after step"
    assert not mismatches, (
        f"rank{rank}: {len(mismatches)} params diverge from replicated Muon reference; "
        f"worst max_abs_diff = {max(m[1] for m in mismatches):.2e}"
    )
