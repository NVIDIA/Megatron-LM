# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Muon optimizer microbenchmark.

Replays just `FSDPTensorParallelMuon.step()` against fake DTensors
reconstructed from a captured per-rank dump of a real 2-node / 8-rank
training run. The dump JSON files live next to this test file; the
dump's `world_size` must match the launch world size.

Launch:
    cd Megatron-LM
    torchrun --nproc_per_node=8 --standalone -m pytest \
        tests/unit_tests/distributed/megatron_fsdp/test_muon_microbench.py -v -s
"""

import inspect
import json
import os
import pathlib
import re
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from packaging.version import Version
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard

from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    update_uneven_dtensor_chunk_metadata,
)
from megatron.core.optimizer.emerging_optimizers import (
    HAVE_EMERGING_OPTIMIZERS,
    FSDPTensorParallelMuon,
)


# When MUON_BENCH_DISABLE_USE_SYRK=1, simulate an unpatched EO `main` where
# `newton_schulz_tp` does not accept `use_syrk`: strip the kwarg before
# dispatching. Lets us validate the bench against EO origin/main without
# requiring the (still-unmerged) EO patch that adds `use_syrk` to the TP path.
if os.environ.get("MUON_BENCH_DISABLE_USE_SYRK", "0") == "1":
    import emerging_optimizers.orthogonalized_optimizers.muon_utils as _eo_mu
    import megatron.core.optimizer.emerging_optimizers as _mcore_eo

    _eo_orig_ns_tp = _eo_mu.newton_schulz_tp

    # Kwargs added by the (still-unmerged) EO `newton_schulz_tp` patch.
    _PATCH_ONLY_NS_TP_KWARGS = (
        "use_syrk",
        "distributed_gram_recurrence",
        "distributed_gram_refresh_interval",
    )

    def _eo_ns_tp_no_syrk(*args, **kwargs):
        for k in _PATCH_ONLY_NS_TP_KWARGS:
            kwargs.pop(k, None)
        return _eo_orig_ns_tp(*args, **kwargs)

    # Patch both the EO module and Megatron's rebound name (`from ... import`).
    _eo_mu.newton_schulz_tp = _eo_ns_tp_no_syrk
    _mcore_eo.newton_schulz_tp = _eo_ns_tp_no_syrk


SPEC_DIR = pathlib.Path(__file__).parent
SPEC_PREFIX = "muon_inputs_2nodes"  # 2 nodes × 4 GPUs, 2D mesh (dp_cp=4, tp=2)
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
    """Set up torch.distributed and CUDA device for torchrun + pytest."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Not running under torchrun. Use torchrun to run this test file.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Muon microbenchmark.")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    yield {"rank": rank, "world_size": world_size, "device": device}

    if dist.is_initialized():
        dist.destroy_process_group()


# ---------- Spec loading ----------


def _load_spec(rank: int) -> dict:
    return json.loads((SPEC_DIR / f"{SPEC_PREFIX}.rank{rank}.json").read_text())


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


_PLACEMENT_RE = re.compile(
    r"(Shard|_StridedShard|Replicate)\((?:dim=)?(-?\d+)?(?:,\s*split_factor=(\d+))?\)?"
)


def _parse_placement(s: str):
    m = _PLACEMENT_RE.search(s)
    if not m:
        raise ValueError(f"cannot parse placement: {s!r}")
    kind, dim, split = m.group(1), m.group(2), m.group(3)
    if kind == "Replicate":
        return Replicate()
    if kind == "_StridedShard":
        return _StridedShard(int(dim), split_factor=int(split))
    return Shard(int(dim))


# ---------- DTensor construction ----------


def _make_dtensor_from_spec(p_spec: dict, mesh, device):
    """Build a DTensor matching the recorded global shape + per-rank local shape.

    The dump captures the M-FSDP-specific layout where `_local_tensor` can
    reflect a TP-shard view while the placements declare additional
    sharding handled at gather time. We honor whatever the spec recorded:
    allocate the recorded `local_shape`, pass the recorded `placements`,
    and let `update_uneven_dtensor_chunk_metadata` install the chunk-list
    bookkeeping M-FSDP relies on.
    """
    dtype = _parse_dtype(p_spec["dtype"])
    local_shape = tuple(p_spec["local_shape"])
    global_shape = tuple(p_spec["global_shape"])
    placements = [_parse_placement(s) for s in p_spec["placements"]]

    # fp32 randn → cast to target dtype so we get sensible values for bf16/fp16 too.
    local = torch.randn(local_shape, dtype=torch.float32, device=device).to(dtype=dtype)
    global_stride = torch.empty(global_shape, dtype=dtype, device=device).stride()
    dt = DTensor.from_local(
        local_tensor=local,
        device_mesh=mesh,
        placements=placements,
        shape=torch.Size(global_shape),
        stride=global_stride,
        run_check=False,
    )
    update_uneven_dtensor_chunk_metadata(dt)
    return dt


def _attach_metadata(dt: torch.Tensor, p_spec: dict) -> None:
    """Copy the per-param attrs the optimizer reads."""
    if p_spec.get("partition_dim") is not None:
        dt.partition_dim = p_spec["partition_dim"]
    if p_spec.get("tensor_model_parallel") is not None:
        dt.tensor_model_parallel = p_spec["tensor_model_parallel"]
    if p_spec.get("is_qkv"):
        dt.is_qkv = True
    if p_spec.get("name"):
        dt.megatron_fsdp_param_name = p_spec["name"]


# ---------- Mesh + process groups ----------


def _build_mesh(spec: dict):
    p0 = spec["groups"][0]["params"][0]
    return init_device_mesh(
        "cuda", tuple(p0["mesh_shape"]), mesh_dim_names=tuple(p0["mesh_dim_names"])
    )


def _build_pg_collection(mesh):
    """Minimal stand-in for `ProcessGroupCollection` — exposes dim-named groups."""
    pgc = SimpleNamespace()
    for name in mesh.mesh_dim_names or ():
        try:
            setattr(pgc, name, mesh.get_group(name))
        except Exception:  # pragma: no cover — defensive
            pass
    return pgc


# ---------- Optimizer construction ----------


def _build_optimizer(spec: dict, param_groups, pg_collection, dp_group, override_lr):
    opt_cfg = dict(spec["optimizer"])
    # Reduce reliance on patches: if requested, disable use_syrk so we don't
    # depend on Emerging-Optimizers patches that add `use_syrk` to
    # `newton_schulz_tp`. Set MUON_BENCH_DISABLE_USE_SYRK=1 to test EO origin/main.
    if os.environ.get("MUON_BENCH_DISABLE_USE_SYRK", "0") == "1":
        opt_cfg["use_syrk"] = False
    sig = inspect.signature(FSDPTensorParallelMuon.__init__)
    accepted = set(sig.parameters)
    kwargs = {k: v for k, v in opt_cfg.items() if k in accepted}
    # Per-group hyperparameters; override lr (dump captures warmup-step-0 lr=0).
    for g in param_groups:
        g["lr"] = override_lr
    return FSDPTensorParallelMuon(
        params=param_groups, dp_group=dp_group, pg_collection=pg_collection, **kwargs
    )


def _build_params_and_grads(spec: dict, mesh, device):
    """Build DTensor params + grads from the loaded spec, grouped as in the dump."""
    param_groups = []
    for g_spec in spec["groups"]:
        ps = []
        for p_spec in g_spec["params"]:
            p = _make_dtensor_from_spec(p_spec, mesh, device)
            _attach_metadata(p, p_spec)
            # Random gradient matching the param's DTensor layout.
            grad_local = torch.randn_like(p._local_tensor, dtype=torch.float32).to(
                dtype=p.dtype
            )
            placements = [_parse_placement(s) for s in p_spec["placements"]]
            global_stride = torch.empty(
                tuple(p_spec["global_shape"]), dtype=p.dtype, device=device
            ).stride()
            grad = DTensor.from_local(
                local_tensor=grad_local,
                device_mesh=mesh,
                placements=placements,
                shape=torch.Size(tuple(p_spec["global_shape"])),
                stride=global_stride,
                run_check=False,
            )
            update_uneven_dtensor_chunk_metadata(grad)
            p.grad = grad
            ps.append(p)
        param_groups.append(
            {
                "params": ps,
                "lr": float(g_spec.get("lr", 0.0)),
                "momentum": float(g_spec.get("momentum", 0.9)),
                "weight_decay": float(g_spec.get("weight_decay", 0.0)),
            }
        )
    return param_groups


# ---------- Tests ----------


def _check_spec_world_size(spec, world_size):
    assert spec["world_size"] == world_size, (
        f"spec world_size={spec['world_size']} != launch world_size={world_size}"
    )


def test_muon_step_smoke(distributed_setup):
    """Build the full optimizer state and run a few `step()` calls.

    Confirms the dump can be replayed end-to-end (parameter reconstruction,
    optimizer construction, optimizer step) at world_size=8 without errors.
    """
    rank = distributed_setup["rank"]
    world_size = distributed_setup["world_size"]
    device = distributed_setup["device"]

    spec = _load_spec(rank)
    _check_spec_world_size(spec, world_size)

    mesh = _build_mesh(spec)
    pg = _build_pg_collection(mesh)
    # For both 2-node ZeRO-3 and 32-node HSDP, the inner FSDP all-gather group
    # is `dp_cp`. Outer HSDP (`outer_fsdp_dp`) replication is handled by the
    # optimizer internally via the mesh.
    dp_group = mesh.get_group("dp_cp")

    param_groups = _build_params_and_grads(spec, mesh, device)
    optimizer = _build_optimizer(
        spec, param_groups, pg_collection=pg, dp_group=dp_group, override_lr=8e-4
    )

    # Spot-check: first param's local shape on this rank matches the spec.
    first_param = param_groups[0]["params"][0]
    first_spec = spec["groups"][0]["params"][0]
    assert tuple(first_param._local_tensor.shape) == tuple(first_spec["local_shape"])

    warmup = 60
    iters = 20
    for _ in range(warmup):
        optimizer.step()
    torch.cuda.synchronize()
    dist.barrier()

    nsys_on = os.environ.get("MUON_BENCH_NSYS", "0") == "1"
    times_ms = []
    if nsys_on:
        torch.cuda.cudart().cudaProfilerStart()
        with torch.autograd.profiler.emit_nvtx(record_shapes=False):
            for _ in range(iters):
                t0 = time.perf_counter()
                optimizer.step()
                torch.cuda.synchronize()
                times_ms.append((time.perf_counter() - t0) * 1000.0)
        torch.cuda.cudart().cudaProfilerStop()
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)
    dist.barrier()

    if rank == 0:
        times_sorted = sorted(times_ms)
        p50 = times_sorted[len(times_sorted) // 2]
        print(
            f"\n[muon-microbench] params={sum(len(g['params']) for g in param_groups)} "
            f"iters={iters} "
            f"min={min(times_ms):.2f}ms p50={p50:.2f}ms max={max(times_ms):.2f}ms"
        )
