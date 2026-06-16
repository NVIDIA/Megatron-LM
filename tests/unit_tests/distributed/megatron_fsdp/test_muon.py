# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Muon optimizer correctness + microbenchmark tests.

Replays `FSDPTensorParallelMuon.step()` against fake DTensors
reconstructed from a captured per-rank dump of a real 2-node / 8-rank
training run. The dump JSON files live next to this test file and only
record parameter sharding (per-param global/local shape, dtype,
placements, mesh layout including the global rank IDs that form the
mesh). The optimizer is constructed here with hardcoded hyperparameters
typical for Muon training.

The dump groups params by param-group (wd vs no-wd). Muon operates on
2D+ matrices, so this test consumes the wd group (group 0) only — the
no-wd group is all 1D bias/norm params that Megatron routes to AdamW.

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
import math
import os
import pathlib
import re
from typing import Any, Callable

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
from tests.unit_tests.distributed.megatron_fsdp.conftest import DistributedSetup


# QKV-fused parameter sub-split along dim 0 of the (num_query_groups, S, in_dim) view.
QKV_SPLIT_SHAPES = (1024, 128, 128)


# Fixture dirs by WORLD_SIZE — one dump capture per parallelism config.
_SPEC_DIR_BY_WORLD_SIZE = {
    8: "muon_inputs_fsdp4_tp2",  # 2-node: (dp_cp=4, tp=2)
    128: "muon_inputs_outer2_fsdp32_tp2",  # 32-node HSDP: (outer_fsdp_dp=2, dp_cp=32, tp=2)
}
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
SPEC_DIR = (
    pathlib.Path(__file__).parent / _SPEC_DIR_BY_WORLD_SIZE[WORLD_SIZE]
    if WORLD_SIZE in _SPEC_DIR_BY_WORLD_SIZE
    else None
)


pytestmark = [
    pytest.mark.skipif(
        Version(os.getenv("NVIDIA_PYTORCH_VERSION", "24.01")) <= Version("25.05"),
        reason="Skip emerging optimizer tests for LTS test",
    ),
    pytest.mark.skipif(
        not HAVE_EMERGING_OPTIMIZERS, reason="Muon tests require the emerging-optimizers package"
    ),
    pytest.mark.skipif(
        WORLD_SIZE not in _SPEC_DIR_BY_WORLD_SIZE,
        reason=f"No fixture for WORLD_SIZE={WORLD_SIZE}; "
        f"supported: {sorted(_SPEC_DIR_BY_WORLD_SIZE)}",
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is required for the Muon microbenchmark."
    ),
]


# `distributed_setup` fixture is provided by conftest.py in this directory; it
# sets the per-rank device but does not initialize the process group. We rely
# on `init_device_mesh`'s documented side effect of initializing the default
# process group on first call, triggered from `_build_mesh_cache`. Teardown
# is handled by the session `cleanup` fixture in tests/unit_tests/conftest.py.


# ---------- Spec loading ----------


def _load_dump(rank: int) -> dict[str, Any]:
    """Load the per-rank dump emitted by `dump_optimizer_parameters`.

    Schema: {world_size, rank, groups: [{params: [{global_shape, dtype,
    is_qkv, local_shape, mesh_shape, mesh_dim_names, mesh_ranks, placements}]}]}
    """
    return json.loads((SPEC_DIR / f"rank{rank}.json").read_text())


def _muon_param_specs(dump: dict[str, Any]) -> list[dict[str, Any]]:
    """Return wd-group 2D param specs — what the replicated-Muon reference can handle.

    The wd group also contains a handful of 3D Mamba state-space matrices that
    ``FSDPTensorParallelMuon`` reshapes internally, but plain
    ``emerging_optimizers.Muon`` (used as the numerics reference) rejects them
    with "Only 2D parameters are supported." Filtering to ndim==2 keeps the
    reference comparable; 3D coverage is a follow-up.
    """
    return [p for p in dump["groups"][0]["params"] if len(p["global_shape"]) == 2]


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


# ---------- Mesh + DTensor construction ----------
#
# Each param records its own mesh, and sub-meshes vary per rank — e.g. the
# (2,)-rank EP sub-mesh is [0,4] on rank 0, [1,5] on rank 1, etc. PyTorch's
# `DeviceMesh.__init__` calls `dist.new_group(ranks=...)` (or `split_group`)
# for every sub-group it forms; both are collective across the world group
# and require all ranks to call them with matching ranks lists in matching
# order. So we can't have rank 0 construct [0,4] while rank 1 constructs
# [1,5] independently — the calls diverge and NCCL eventually times out.
#
# Solution: every rank locally enumerates the unique meshes referenced by
# its own param specs, all_gather_object the world-wide union, then all
# ranks construct every unique mesh in sorted order. Each rank caches only
# the meshes it actually participates in.


MeshKey = tuple[tuple[int, ...], tuple[str, ...], tuple[int, ...]]


def _mesh_key(param_spec: dict[str, Any]) -> MeshKey:
    """Hashable identity of one param's mesh: (shape, dim_names, flat_ranks).

    `mesh_ranks` is recorded flat (row-major) by `dump_optimizer_parameters`
    only when it isn't `range(mesh.numel())`; canonical ranks are omitted from
    the dump and reconstructed here. `mesh_shape` gives the reshape used to
    build the DeviceMesh.
    """
    shape = tuple(param_spec["mesh_shape"])
    mesh_ranks = param_spec.get("mesh_ranks")
    if mesh_ranks is None:
        mesh_ranks = range(math.prod(shape))
    return (shape, tuple(param_spec["mesh_dim_names"]), tuple(mesh_ranks))


def _build_mesh_cache(
    param_specs: list[dict[str, Any]], device_type: str
) -> dict[MeshKey, DeviceMesh]:
    """Collectively construct every DeviceMesh referenced by any rank's param specs.

    Returns a cache containing only the meshes this rank participates in —
    DeviceMesh objects for non-member meshes are constructed (to keep the
    new_group collectives in lockstep across ranks) but immediately discarded.

    Builds a 1D world mesh first to bootstrap the default process group on
    first call (documented `init_device_mesh` side effect) and to make the
    `all_gather_object` group explicit.
    """
    world_mesh = init_device_mesh(device_type, (WORLD_SIZE,))
    world_group = world_mesh.get_group()
    my_rank = dist.get_rank()
    local_keys = sorted({_mesh_key(p) for p in param_specs})
    gathered: list[list[MeshKey] | None] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, local_keys, group=world_group)
    all_keys = sorted({k for ks in gathered if ks is not None for k in ks})

    cache: dict[MeshKey, DeviceMesh] = {}
    for key in all_keys:
        shape, dim_names, flat_ranks = key
        mesh_tensor = torch.tensor(flat_ranks, dtype=torch.int).reshape(shape)
        mesh = DeviceMesh(device_type, mesh_tensor, mesh_dim_names=dim_names)
        if my_rank in flat_ranks:
            cache[key] = mesh
    return cache


def _make_dtensor_from_spec(
    param_spec: dict[str, Any], mesh: DeviceMesh, device: torch.device
) -> DTensor:
    """Build a DTensor matching the recorded global shape + per-rank local shape.

    The dump captures the M-FSDP-specific layout where `_local_tensor` can
    reflect a TP-shard view while the placements declare additional sharding
    handled at gather time. We honor whatever the param spec recorded:
    allocate the recorded `local_shape`, pass the recorded `placements`, and
    let `update_uneven_dtensor_chunk_metadata` install the chunk-list
    bookkeeping M-FSDP relies on.
    """
    dtype = _parse_dtype(param_spec["dtype"])
    local_shape = tuple(param_spec["local_shape"])
    global_shape = tuple(param_spec["global_shape"])
    placements = [_parse_placement(s) for s in param_spec["placements"]]

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


# ---------- Param + optimizer construction ----------


def _build_param(
    param_spec: dict[str, Any], device: torch.device, mesh_cache: dict[MeshKey, DeviceMesh]
) -> DTensor:
    """Build one DTensor param (with random `.grad` attached).

    The param's DeviceMesh comes from ``mesh_cache``; the caller can look up
    the same mesh via ``mesh_cache[_mesh_key(param_spec)]`` when it needs the
    per-param ``dp_cp`` sub-group (e.g. for the numerics reference gather).
    """
    mesh = mesh_cache[_mesh_key(param_spec)]
    p = _make_dtensor_from_spec(param_spec, mesh, device)
    if param_spec.get("is_qkv"):
        p.is_qkv = True
    # Random gradient matching the param's DTensor layout (fp32 randn → cast so
    # bf16/fp16 grads still get sensible values).
    grad = torch.randn_like(p, dtype=torch.float32).to(dtype=p.dtype)
    update_uneven_dtensor_chunk_metadata(grad)
    p.grad = grad
    return p


def _build_optimizer(params: list[DTensor]) -> FSDPTensorParallelMuon:
    """Construct a `FSDPTensorParallelMuon` with typical Muon hyperparameters.

    The JSON fixtures only describe parameter sharding; the choice of
    optimizer hyperparameters is the bench's concern. These values are
    representative for Muon training.
    """
    return FSDPTensorParallelMuon(
        params=params,
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


class ReplicatedMuon:
    """Plain (non-DTensor) Muon, used as the numerics reference.

    Mirrors `FSDPTensorParallelMuon`'s behavior per-param: optionally DP-gather
    to assemble the full tensor (FSDP Phase 2), run Muon's NS orthogonalization
    on the full view (with manual QKV split where applicable), then re-slice to
    this rank's local shard. Hyperparameters match `_build_optimizer`'s — plain
    `Muon` defaults `nesterov=False`, but `FSDPTensorParallelMuon` inherits
    `nesterov=True` from `TensorParallelMuon`, so we pass it explicitly.

    Stateless across params: a fresh inner `Muon` is built per call so its
    momentum state is freed when the local instance is GC'd, keeping per-step
    HBM bounded by one param's DP-gathered footprint.
    """

    def __init__(self, qkv_split_shapes: tuple[int, ...]) -> None:
        self._qkv_split_shapes = tuple(qkv_split_shapes)

    def step(
        self,
        local_shards: list[torch.Tensor],
        param_specs: list[dict[str, Any]],
        dp_group_of: Callable[[dict[str, Any]], dist.ProcessGroup],
    ) -> list[torch.Tensor]:
        """Run one Muon step on each param's local shard; return new local shards.

        For each param: boundary-vote across its own DP group, optionally
        DP-gather + run NS on the full view + slice back, mirroring FSDP
        Phase 2/3. ``dp_group_of(param_spec)`` returns the DP group for that
        param's mesh — separates the test's mesh-cache wiring from this
        class's per-param logic.
        """
        return [
            self._step_one(ref_p, p, dp_group_of(p)) for ref_p, p in zip(local_shards, param_specs)
        ]

    def _step_one(
        self, ref_p: torch.Tensor, param_spec: dict[str, Any], dp_group: dist.ProcessGroup
    ) -> torch.Tensor:
        """One param's step. Requires ``ref_p.grad`` to be attached."""
        global_shape = tuple(param_spec["global_shape"])
        is_qkv = param_spec.get("is_qkv", False)

        # Mirror FSDP's `_needs_boundary_gather`: a param is "boundary" if any
        # rank in the DP group has a non-empty local with shape != global.
        local_is_boundary = ref_p.numel() > 0 and global_shape != tuple(ref_p.shape)
        boundary_votes = [None] * dist.get_world_size(dp_group)
        dist.all_gather_object(boundary_votes, local_is_boundary, group=dp_group)

        if not any(boundary_votes):
            return self._step_single(is_qkv, ref_p, ref_p.grad)

        full_p = self._gather_dp(ref_p, dp_group)
        full_g = self._gather_dp(ref_p.grad, dp_group)
        full_after = self._step_single(is_qkv, full_p, full_g)
        return self._slice_dp_full_to_local(ref_p.shape[0], full_after, dp_group)

    def _build(self, params: list[torch.Tensor]) -> Muon:
        return Muon(
            params=[{"params": params}],
            lr=8e-4,
            momentum=0.9,
            weight_decay=0.1,
            nesterov=True,
            extra_scale_factor=0.2,
        )

    def _step_single(
        self, is_qkv: bool, full_p: torch.Tensor, full_g: torch.Tensor
    ) -> torch.Tensor:
        """One plain-Muon step on a single FSDP param's full-tensor view.

        For QKV-fused params, splits along the (num_query_groups, sum(QKV),
        in_dim) view to mirror `TensorParallelMuon.orthogonalize` — each of
        Q/K/V is orthogonalized independently — then re-fuses the result.

        Returns the updated full tensor (same shape as ``full_p``).
        """
        if not is_qkv:
            full_p.grad = full_g
            self._build([full_p]).step()
            return full_p

        qkv = list(self._qkv_split_shapes)
        out_dim, in_dim = full_p.shape
        s = sum(qkv)
        assert out_dim % s == 0, f"QKV out_dim {out_dim} not divisible by sum(qkv_split_shapes)={s}"
        num_query_groups = out_dim // s
        p_view = full_p.view(num_query_groups, s, in_dim)
        g_view = full_g.view(num_query_groups, s, in_dim)
        p_parts = [t.reshape(-1, in_dim) for t in torch.split(p_view, qkv, dim=1)]
        g_parts = [t.reshape(-1, in_dim) for t in torch.split(g_view, qkv, dim=1)]
        for pp, gp in zip(p_parts, g_parts):
            pp.grad = gp
        self._build(p_parts).step()
        # Re-fuse: each updated part back to (num_query_groups, qkv_dim, in_dim), cat, view.
        refused_parts = [pp.view(num_query_groups, -1, in_dim) for pp in p_parts]
        return torch.cat(refused_parts, dim=1).reshape(out_dim, in_dim)

    @staticmethod
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

    @staticmethod
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
        return full_tensor[offset : offset + local_dim0]


# ---------- Tests ----------


def test_muon_step(distributed_setup: DistributedSetup, benchmark) -> None:
    """Build the full optimizer state and benchmark `step()`.

    pytest-benchmark's `pedantic` mode runs the step a fixed number of times
    (deterministic across ranks) and reports min/median/max/mean/stddev.
    """
    rank = distributed_setup.rank
    world_size = distributed_setup.world_size
    device_type = distributed_setup.device.type
    device = distributed_setup.device

    dump = _load_dump(rank)
    assert (
        dump["world_size"] == world_size
    ), f"dump world_size={dump['world_size']} != launch world_size={world_size}"

    param_specs: list[dict[str, Any]] = _muon_param_specs(dump)
    mesh_cache: dict[MeshKey, DeviceMesh] = _build_mesh_cache(param_specs, device_type)
    params: list[DTensor] = [_build_param(p, device, mesh_cache) for p in param_specs]
    optimizer: FSDPTensorParallelMuon = _build_optimizer(params)

    def _one_step():
        optimizer.step()
        torch.cuda.synchronize()

    benchmark.pedantic(_one_step, rounds=5, warmup_rounds=3, iterations=1)
    dist.barrier()


def test_muon_step_numerics(distributed_setup: DistributedSetup) -> None:
    """Compare `FSDPTensorParallelMuon` against a fully-replicated `Muon` reference.

    Flow:
        1. Build DTensor params + grads from the dump (per-param sub-meshes).
        2. Clone the local shards into `reference_params` (plain tensors).
        3. Run FSDP+TP Muon on the DTensors (mutates in place).
        4. Run plain `emerging_optimizers.Muon` on `reference_params`. For
           boundary params, gather across the param's own DP group (mirroring
           FSDP Phase 2), run Muon on the full view (manual QKV split where
           applicable), and slice the result back to the local shard. For
           non-boundary params, run Muon directly on the local shard.
        5. Assert each FSDP post-step shard is close to its reference local
           with `atol=rtol=1e-2`.
    """
    rank = distributed_setup.rank
    world_size = distributed_setup.world_size
    device_type = distributed_setup.device.type
    device = distributed_setup.device

    dump = _load_dump(rank)
    assert (
        dump["world_size"] == world_size
    ), f"dump world_size={dump['world_size']} != launch world_size={world_size}"

    param_specs: list[dict[str, Any]] = _muon_param_specs(dump)
    mesh_cache: dict[MeshKey, DeviceMesh] = _build_mesh_cache(param_specs, device_type)
    fsdp_params: list[DTensor] = [_build_param(p, device, mesh_cache) for p in param_specs]

    # Build reference param shards — a clone of each pre-step local, with the
    # grad attached. The reference Muon will mutate these in place (boundary
    # params via a gathered full-tensor view, non-boundary via the local shard
    # directly). Grads are captured as views, not clones, because
    # `FSDPTensorParallelMuon.step()` doesn't mutate them.
    reference_params = [p.to_local().clone() for p in fsdp_params]
    for ref_p, fsdp_p in zip(reference_params, fsdp_params):
        ref_p.grad = fsdp_p.grad.to_local()

    # Run FSDP+TP Muon step on the DTensors.
    fsdp_opt: FSDPTensorParallelMuon = _build_optimizer(fsdp_params)
    fsdp_opt.step()

    # Run the replicated reference Muon. This pass only touches the reference
    # shards and the param specs — it never reads `fsdp_params`.
    reference = ReplicatedMuon(QKV_SPLIT_SHAPES)
    reference_params = reference.step(
        reference_params,
        param_specs,
        dp_group_of=lambda p: mesh_cache[_mesh_key(p)].get_group("dp_cp"),
    )

    # Compare each FSDP post-step shard against the replicated reference.
    for i, (fsdp_p, ref_p) in enumerate(zip(fsdp_params, reference_params)):
        torch.testing.assert_close(
            fsdp_p.to_local().float(),
            ref_p.float(),
            atol=1e-2,
            rtol=1e-2,
            msg=lambda m, i=i: (
                f"rank{rank} param[{i}] shape={tuple(param_specs[i]['global_shape'])}: {m}"
            ),
        )
