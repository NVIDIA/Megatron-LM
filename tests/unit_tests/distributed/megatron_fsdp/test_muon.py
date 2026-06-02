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
  * `test_muon_step_numerics` — runs the step twice from the same state
    with fresh optimizer instances, asserts bit-deterministic outputs
    and no NaN/Inf in any local tensor.

Launch:
    cd Megatron-LM
    torchrun --nproc_per_node=8 --standalone -m pytest \
        tests/unit_tests/distributed/megatron_fsdp/test_muon.py -v -s
"""

import json
import os
import pathlib
import re
import pytest
import torch
import torch.distributed as dist
from packaging.version import Version
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    update_uneven_dtensor_chunk_metadata,
)
from megatron.core.optimizer.emerging_optimizers import (
    HAVE_EMERGING_OPTIMIZERS,
    FSDPTensorParallelMuon,
)


SPEC_DIR = pathlib.Path(__file__).parent / "muon_inputs_fsdp4_tp2"
EXPECTED_WORLD_SIZE = 8
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))


pytestmark = [
    pytest.mark.skipif(
        Version(os.getenv("NVIDIA_PYTORCH_VERSION", "99.99")) <= Version("25.05"),
        reason="Skip Muon FSDP tests on LTS containers",
    ),
    pytest.mark.skipif(
        not HAVE_EMERGING_OPTIMIZERS,
        reason="Muon tests require the emerging-optimizers package",
    ),
    pytest.mark.skipif(
        WORLD_SIZE != EXPECTED_WORLD_SIZE,
        reason=f"Dump captured at WORLD_SIZE={EXPECTED_WORLD_SIZE}, "
        f"got WORLD_SIZE={WORLD_SIZE}",
    ),
]


# ---------- Distributed setup ----------


@pytest.fixture(scope="module")
def distributed_setup():
    """Pin the CUDA device. `init_device_mesh` lazy-inits the default
    process group on first use from torchrun's RANK/WORLD_SIZE env vars."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Not running under torchrun. Use torchrun to run this test file.")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Muon microbenchmark.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    yield {"rank": rank, "world_size": world_size, "device": device}

    if dist.is_initialized():
        dist.destroy_process_group()


# ---------- Spec loading ----------


def _load_spec(rank: int) -> dict:
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


def _parse_placement(s: str):
    m = _PLACEMENT_RE.fullmatch(s)
    if not m:
        raise ValueError(f"cannot parse placement: {s!r}")
    return Shard(int(m.group(1)))


def _contiguous_stride(shape):
    """Row-major contiguous strides for a tensor of ``shape`` — no allocation.

    `DTensor.from_local(..., shape=global_shape)` requires `stride` whenever
    `shape` is given (PyTorch enforces "pass both or neither").
    """
    stride = []
    running = 1
    for s in reversed(shape):
        stride.append(running)
        running *= s
    return tuple(reversed(stride))


# ---------- DTensor construction ----------


def _make_dtensor_from_spec(p_spec: dict, mesh, device):
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


def _build_mesh(spec: dict):
    return init_device_mesh(
        "cuda",
        tuple(spec["mesh_shape"]),
        mesh_dim_names=tuple(spec["mesh_dim_names"]),
    )


# ---------- Param + optimizer construction ----------


def _build_params_and_grads(spec: dict, mesh, device):
    """Build DTensor params + random grads from the slim spec."""
    params = []
    for p_spec in spec["params"]:
        p = _make_dtensor_from_spec(p_spec, mesh, device)
        if p_spec.get("is_qkv"):
            p.is_qkv = True
        # Random gradient matching the param's DTensor layout.
        grad_local = torch.randn_like(p._local_tensor, dtype=torch.float32).to(dtype=p.dtype)
        placements = [_parse_placement(s) for s in p_spec["placements"]]
        global_shape = tuple(p_spec["global_shape"])
        grad = DTensor.from_local(
            local_tensor=grad_local,
            device_mesh=mesh,
            placements=placements,
            shape=torch.Size(global_shape),
            stride=_contiguous_stride(global_shape),
            run_check=False,
        )
        update_uneven_dtensor_chunk_metadata(grad)
        p.grad = grad
        params.append(p)
    # Single param group; bench picks the hyperparameters.
    return [{"params": params}]


def _build_optimizer(param_groups, dp_group):
    """Construct a `FSDPTensorParallelMuon` with typical Muon hyperparameters.

    The JSON fixtures only describe parameter sharding; the choice of
    optimizer hyperparameters is the bench's concern. These values are
    representative for Muon training.
    """
    return FSDPTensorParallelMuon(
        params=param_groups,
        dp_group=dp_group,
        pg_collection=None,
        lr=8e-4,
        momentum=0.9,
        weight_decay=0.1,
        num_ns_steps=5,
        coefficient_type="quintic",
        scale_mode="spectral",
        extra_scale_factor=0.2,
        tp_mode="blockwise",
        split_qkv=True,
        qkv_split_shapes=[1024, 128, 128],
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
    )


# ---------- Tests ----------


def test_muon_step(distributed_setup, benchmark):
    """Build the full optimizer state and benchmark `step()`.

    pytest-benchmark's `pedantic` mode runs the step a fixed number of times
    (deterministic across ranks) and reports min/median/max/mean/stddev.
    """
    rank = distributed_setup["rank"]
    world_size = distributed_setup["world_size"]
    device = distributed_setup["device"]

    spec = _load_spec(rank)
    assert spec["world_size"] == world_size, (
        f"spec world_size={spec['world_size']} != launch world_size={world_size}"
    )

    mesh = _build_mesh(spec)
    dp_group = mesh.get_group("dp_cp")

    param_groups = _build_params_and_grads(spec, mesh, device)
    optimizer = _build_optimizer(param_groups, dp_group=dp_group)

    # Spot-check: first param's local shape on this rank matches the spec.
    first_param = param_groups[0]["params"][0]
    first_spec = spec["params"][0]
    assert tuple(first_param._local_tensor.shape) == tuple(first_spec["local_shape"])

    def _one_step():
        optimizer.step()
        torch.cuda.synchronize()

    benchmark.pedantic(_one_step, rounds=5, warmup_rounds=3, iterations=1)
    dist.barrier()


def test_muon_step_numerics(distributed_setup):
    """Verify `FSDPTensorParallelMuon.step()` is bit-deterministic.

    Runs one step from the same initial params + grads twice with two
    fresh optimizer instances and compares the resulting `_local_tensor`
    on every rank. Catches non-determinism from random kernel launches,
    autotuner re-selections, NCCL collective non-associativity, and
    similar bugs that any correctness-affecting regression would touch.
    Asserts every value is finite (no NaN/Inf) as a basic sanity check.
    """
    rank = distributed_setup["rank"]
    world_size = distributed_setup["world_size"]
    device = distributed_setup["device"]

    spec = _load_spec(rank)
    assert spec["world_size"] == world_size, (
        f"spec world_size={spec['world_size']} != launch world_size={world_size}"
    )

    mesh = _build_mesh(spec)
    dp_group = mesh.get_group("dp_cp")

    param_groups = _build_params_and_grads(spec, mesh, device)
    init = [p._local_tensor.clone() for p in param_groups[0]["params"]]
    grad_snapshots = [p.grad._local_tensor.clone() for p in param_groups[0]["params"]]

    def _run_one_step():
        optimizer = _build_optimizer(param_groups, dp_group=dp_group)
        optimizer.step()
        torch.cuda.synchronize()
        return [p._local_tensor.clone() for p in param_groups[0]["params"]]

    after_a = _run_one_step()

    # Reset to the initial state and replay.
    for p, p0, g0 in zip(param_groups[0]["params"], init, grad_snapshots):
        p._local_tensor.copy_(p0)
        p.grad._local_tensor.copy_(g0)

    after_b = _run_one_step()
    dist.barrier()

    # Determinism + finiteness.
    nonfinite = []
    drifts = []
    for i, (a, b) in enumerate(zip(after_a, after_b)):
        if a.numel() == 0:
            continue
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            nonfinite.append(i)
            continue
        if not torch.equal(a, b):
            drifts.append((i, (a.float() - b.float()).abs().max().item()))

    if rank == 0:
        n_checked = sum(1 for p in param_groups[0]["params"] if p._local_tensor.numel() > 0)
        print(
            f"\n[muon-microbench numerics] params_checked={n_checked} "
            f"nonfinite={len(nonfinite)} drifts={len(drifts)}"
        )

    assert not nonfinite, f"rank{rank}: {len(nonfinite)} params contain NaN/Inf after step"
    assert not drifts, (
        f"rank{rank}: optimizer.step() is not bit-deterministic on "
        f"{len(drifts)} params; worst max_abs_diff = {max(d[1] for d in drifts):.2e}"
    )
