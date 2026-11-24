# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared helpers for the GTP unit tests (fixtures live in conftest.py)."""

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch import is_mxfp8_available, is_nvfp4_available

_GROUP_CACHE = {}


def cached_group(ranks):
    """Process group over ``ranks``, created once per process.

    ``dist.new_group`` is a world collective whose NCCL communicator lives until the process
    exits, and the GTP bucket runs every module in one process: a group per test exhausts NCCL
    on large-memory systems. The whole world returns ``group.WORLD``. Every rank must call this
    for every rank set, in the same order, as for ``dist.new_group``.
    """
    ranks = tuple(ranks)
    if ranks == tuple(range(torch.distributed.get_world_size())):
        return torch.distributed.group.WORLD
    if ranks not in _GROUP_CACHE:
        _GROUP_CACHE[ranks] = torch.distributed.new_group(list(ranks))
    return _GROUP_CACHE[ranks]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_distributed(fn, required_world_size: int, *args) -> None:
    """Run ``fn(rank, world_size, port, *args)`` on every torchrun rank.

    ``port`` is unused (dist already initialized by torchrun) but kept so
    worker signatures don't need editing.
    """
    actual_world_size = torch.distributed.get_world_size()
    if actual_world_size != required_world_size:
        pytest.skip(
            f"Requires world_size={required_world_size}, "
            f"got {actual_world_size} (launch with torchrun --nproc-per-node={required_world_size})"
        )
    fn(torch.distributed.get_rank(), actual_world_size, None, *args)


def _requires_multi_gpu(n: int = 4):
    if torch.cuda.device_count() < n:
        pytest.skip(f"Requires at least {n} CUDA devices")


def _requires_mxfp8():
    available, reason = is_mxfp8_available(return_reason=True)
    if not available:
        pytest.skip(f"MXFP8 not available: {reason}")


def _requires_nvfp4():
    if not is_nvfp4_available():
        pytest.skip("NVFP4 not available (requires compute capability >= 10.0)")


def _make_gtp_linear(in_f, out_f, gtp_remat_group, dtype=torch.bfloat16, **kwargs):
    """Construct a bias-free GTP-sharded te.Linear on CUDA.

    Mirrors the production integration (extensions/transformer_engine.py): TE has no GTP
    construction hooks, so the module is built stock, ``module.gtp_remat_size`` (the forward-gather
    gate) is stamped post-init, and the BF16 weight is sliced by ``wrap_module_params_gtp``.
    """
    from megatron.core.tensor_parallel.gtp_api import wrap_module_params_gtp

    layer = te.Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        **kwargs,
    )
    layer.gtp_remat_size = gtp_remat_group.size()
    wrap_module_params_gtp(layer, layer.weight_names, gtp_remat_group)
    return layer


def _make_gtp_remat_grouped_linear(
    num_gemms, in_f, out_f, gtp_remat_group, dtype=torch.bfloat16, **kwargs
):
    """Construct a bias-free GTP-sharded te.GroupedLinear on CUDA (post-init slice, see
    _make_gtp_linear)."""
    from megatron.core.tensor_parallel.gtp_api import wrap_module_params_gtp

    layer = te.GroupedLinear(
        num_gemms=num_gemms,
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        **kwargs,
    )
    layer.gtp_remat_size = gtp_remat_group.size()
    # GroupedLinear exposes per-expert weight0..weight{num_gemms-1} (it no longer declares
    # weight_names); build the names here to match attach_gtp_to_presharded_module.
    weight_names = [f"weight{idx}" for idx in range(num_gemms)]
    wrap_module_params_gtp(layer, weight_names, gtp_remat_group, is_grouped=True)
    return layer
