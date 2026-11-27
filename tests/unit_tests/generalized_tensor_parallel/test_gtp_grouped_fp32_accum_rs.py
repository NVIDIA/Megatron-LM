# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Batched FP32-accumulating reduce-scatter and reduce-scatter output pooling."""

import pytest
import torch

from megatron.core.distributed.reduce_scatter_with_fp32_accumulation import (
    _ReduceScatterWithFP32AccumulationWorkHandle,
)
from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _torchrun_dist_init,
    reset_gtp_globals,
)

NUM_WEIGHTS = 8


def _grouped_weights(group, shape):
    """Routed-expert style weights that reduce-scatter as one batch."""
    module = torch.nn.Module()
    names = [f"weight{i}" for i in range(NUM_WEIGHTS)]
    for name in names:
        module.register_parameter(
            name, torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16, device="cuda"))
        )
    gtp.wrap_module_params_gtp(module, names, group, is_grouped=True)
    return module.weight0


def _configure_fp32_accumulation(monkeypatch, group):
    if group.size() <= 2:
        pytest.skip("FP32-accumulation reduce-scatter requires more than two ranks")
    assert not gtp.uses_gtp_symm_pool(group, collective="rs")
    monkeypatch.setattr(gtp, "_wgrad_buf_pool", {})
    monkeypatch.setattr(gtp.GTP_CONFIG, "reduce_scatter_with_fp32_accumulation", True)
    # Per-token loss skips the 1/world pre-scale, so the expected sum is exact in BF16.
    monkeypatch.setattr(gtp.GTP_CONFIG, "calculate_per_token_loss", True)
    monkeypatch.setattr(gtp.GTP_CONFIG, "check_param_states", False)


@pytest.mark.parametrize("async_op", [False, True])
def test_batched_fp32_accum_rs_waits_the_backend_completion_handle(monkeypatch, async_op):
    """The FP32 sums wait the handle that completes a coalesced all-to-all on this backend.

    ProcessGroupNCCL2 completes each collective on its own handle and not on the coalescing
    manager's grouped work; ProcessGroupNCCL does the reverse, and its per-op handles must not
    be waited. The all-to-alls stay in one coalesced group either way.
    """
    group = torch.distributed.group.WORLD
    _configure_fp32_accumulation(monkeypatch, group)
    weight = _grouped_weights(group, (64 * group.size(), 64))
    composites = []
    composite_cls = gtp._GTPCompositeWorkHandle

    class _Recording(composite_cls):
        def __init__(self, handles):
            super().__init__(handles)
            composites.append(self)

    monkeypatch.setattr(gtp, "_GTPCompositeWorkHandle", _Recording)
    coalesced_regions = []
    coalescing_manager = torch.distributed._coalescing_manager

    def recording_coalescing_manager(*args, **kwargs):
        coalesced_regions.append(kwargs)
        return coalescing_manager(*args, **kwargs)

    monkeypatch.setattr(torch.distributed, "_coalescing_manager", recording_coalescing_manager)
    wgrads = [
        torch.full((64 * group.size(), 64), group.rank() + 1, dtype=torch.bfloat16, device="cuda")
        for _ in range(NUM_WEIGHTS)
    ]
    _, handle, _ = weight._reduce_scatter(wgrads, async_op=async_op)
    if handle is not None:
        handle.wait()
    torch.cuda.synchronize()
    assert len(composites) == 1, "the batched FP32-accumulation path was not taken"
    sums = [
        h
        for h in composites[0].handles
        if isinstance(h, _ReduceScatterWithFP32AccumulationWorkHandle)
    ]
    assert len(sums) == NUM_WEIGHTS
    assert len(coalesced_regions) == 1, "the all-to-alls must stay in one coalesced group"
    # The grouped work is waited first on every backend.
    assert len(composites[0].handles) == NUM_WEIGHTS + 1 and composites[0].handles[0] not in sums
    per_op = gtp._coalesced_per_op_handles_complete(group)
    backend = group._get_backend(torch.device("cuda", torch.cuda.current_device()))
    assert per_op == (type(backend).__name__ == "ProcessGroupNCCL2")
    assert all((h.all_to_all_handle is not None) == per_op for h in sums)


@pytest.mark.parametrize("async_op", [False, True])
def test_batched_fp32_accum_rs_never_sums_unwritten_scratch(monkeypatch, async_op):
    """The FP32 sum must read the all-to-all result, whatever the scratch held before.

    The scratch is poisoned with NaN before the collective is issued. A sum that runs before
    the all-to-all has written the scratch turns that NaN into the reduced gradient.
    """
    group = torch.distributed.group.WORLD
    _configure_fp32_accumulation(monkeypatch, group)
    world_size = group.size()
    # Large enough that the all-to-alls are still in flight when the wait returns.
    shape = (2048 * world_size, 4096)
    weight = _grouped_weights(group, shape)
    pool_get = gtp._wgrad_pool_get

    def poisoned_pool_get(*args, **kwargs):
        scratch = pool_get(*args, **kwargs)
        scratch.fill_(float("nan"))
        return scratch

    monkeypatch.setattr(gtp, "_wgrad_pool_get", poisoned_pool_get)
    expected = world_size * (world_size + 1) / 2
    for _ in range(5):
        wgrads = [
            torch.full(shape, group.rank() + 1, dtype=torch.bfloat16, device="cuda")
            for _ in range(NUM_WEIGHTS)
        ]
        outputs, handle, release_bufs = weight._reduce_scatter(wgrads, async_op=async_op)
        if handle is not None:
            handle.wait()
        # Read on the stream that waited, before any device-wide synchronization.
        wrong = [int((output != expected).sum().item()) for output in outputs]
        assert wrong == [0] * NUM_WEIGHTS, f"reduce-scatter outputs with wrong elements: {wrong}"
        assert all(output.dtype == torch.float32 for output in outputs)
        weight._release_wgrad_scratch(("_rs_a2a_bufs",))
        del outputs, release_bufs, wgrads
        cache = gtp.get_global_GTP_cache()
        for w in weight._weights:
            if w._rs_ticket is not None:
                cache.release(w._rs_ticket)
        torch.cuda.synchronize()


@pytest.mark.parametrize(
    "graphed, full_iteration, pooled",
    [
        (False, False, True),
        (False, True, True),
        # Independently replayed per-layer graphs may overlap: keep the buffer.
        (True, False, False),
        # One graph with a fixed order: pool the buffer as eager does.
        (True, True, True),
    ],
)
def test_reduce_scatter_output_pooling(monkeypatch, graphed, full_iteration, pooled):
    """A graphed reduce-scatter output is exclusive only under per-layer graphs."""
    monkeypatch.setattr(gtp, "_FULL_ITERATION", full_iteration)
    monkeypatch.setattr(gtp, "_chain_is_graphed", lambda chain_id: graphed)
    cache = gtp.GTPWeightCache()
    key = ("rs", "key")
    buf = torch.empty(4)
    cache._slots[0] = gtp._TicketSlot(
        key=key, param=None, dtype=torch.float32, reduce_scatter=True, fwd=False, buf=buf
    )
    cache.release(0)
    if pooled:
        assert cache._slots[0].buf is None
        assert len(cache._pool[key]) == 1 and cache._pool[key][0] is buf
    else:
        assert cache._slots[0].buf is buf
        assert not cache._pool[key]
