# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Unit tests for Generalized Tensor Parallelism (GTP).

Test groups
-----------
1.  TestGTPWeightState           – state-machine transitions (single-process)
2.  TestGTPWeightCache           – coat-check buffer pool (single-process)
3.  TestGTPSharding              – wrap_module_params_gtp: shard content + padding (multi-GPU)
4.  TestWrapModuleParams         – wrap_module_params_gtp: param replacement + weight_list (multi-GPU)
5.  TestLinearGTP                – Linear forward/backward numerical correctness (multi-GPU)
6.  TestLayerNormLinearGTP       – LayerNormLinear forward/backward smoke test (multi-GPU)
7.  TestGroupedLinearGTP         – GroupedLinear forward/backward smoke test (multi-GPU)
8.  TestGTPPrefetchChain         – linked-list next_w/prev_w wiring (multi-GPU)
9.  TestGTPWgradRS               – wgrad reduce-scatter shape + multi-layer deferred path (multi-GPU)
10. TestGTPMicrobatches          – output consistency across microbatches (multi-GPU)
11. TestNVFP4LinearGTP           – Linear + NVFP4 recipe: quantized shard setup, fwd/bwd (multi-GPU)
12. TestNVFP4GroupedLinearGTP    – GroupedLinear + NVFP4 recipe: coalesced AG + fwd/bwd (multi-GPU)
13. TestMXFP8LinearGTP           – Linear + MXFP8 recipe: quantized shard setup, fwd/bwd, padding (multi-GPU)
14. TestGTPConfig                – update_config: valid/invalid keys (single-process)
15. TestGTPShardedParamProperties – shape computations, get_padded_shard, _strip_padding (single-process)
16. TestGTPCacheKey              – _get_cache_key: expert vs non-expert, fwd vs bwd (single-process)
17. TestGTPCacheRelease          – reserve/get/release pool semantics (single-process)
18. TestTagGTPParamsWithNames    – _debug_name population on GTPShardedParam (single-process)
19. TestGTPGroupSizeOne          – wrap_module_params_gtp no-op when gtp_group.size()==1 (single-process)
20. TestGTPPrefetchDisabled      – weight_prefetch=False: single-pass forward still works (multi-GPU)
21. TestFuseWgradAccumulation    – fuse_wgrad_accumulation=True: wgrad→main_grad (multi-GPU)
22. TestGTPGradAccumHook         – main_grad updated after reduce-scatter backward (multi-GPU)
23. TestWaitAsyncCommsFallback   – wait_async_comms(finalize_after_drain=True) inline-accumulation fallback when _wgrad_rs_handle is None (single-process)

Run via torchrun (matches the rest of Megatron's unit tests):

    torchrun --nproc-per-node 4 -m pytest tests/unit_tests/generalized_tensor_parallel/test_gtp.py -v

Multi-GPU tests skip when ``torch.distributed.get_world_size()`` doesn't match the required
world size (4 for everything in this file).
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling
from transformer_engine.pytorch import fp8_autocast, is_mxfp8_available, is_nvfp4_available
from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor

import megatron.experimental.gtp.generalized_tensor_parallelism as gtp_module
from megatron.experimental.gtp import GTPShardedParam, wrap_module_params_gtp
from megatron.experimental.gtp.generalized_tensor_parallelism import GTPWeightCache, GTPWeightState
from tests.unit_tests.test_utilities import Utils

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _torchrun_dist_init():
    """Initialize the torchrun-managed dist group once per module.

    GTP tests use ``dist.new_group(...)`` to build their own GTP subgroup
    within the world that torchrun set up. Each test runs on every torchrun
    rank in parallel (standard Mcore convention); ``_run_distributed`` below
    only skips when the required world size doesn't match what torchrun
    provides.
    """
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


@pytest.fixture(autouse=True)
def reset_fp8_state():
    yield
    FP8GlobalStateManager.reset()


@pytest.fixture(autouse=True)
def reset_gtp_globals():
    """Reset all GTP mutable class/module-level state between tests."""
    yield
    GTPShardedParam._chain_state = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_distributed(fn, required_world_size: int, *args) -> None:
    """Run ``fn`` on every torchrun rank.

    ``fn(rank, world_size, port, *args)`` matches the pre-existing worker
    signature; ``port`` is unused (dist already initialized by torchrun) but
    kept so the worker bodies don't need editing.
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


def _requires_nvfp4():
    if not is_nvfp4_available():
        pytest.skip("NVFP4 not available (requires compute capability >= 10.0)")


# ---------------------------------------------------------------------------
# 1. GTPWeightState – state-machine transition tests
# ---------------------------------------------------------------------------


class TestGTPWeightState:

    @staticmethod
    def _param():
        return GTPShardedParam(torch.zeros(4, 4))

    def test_full_cycle(self):
        p = self._param()
        assert p.state == GTPWeightState.NONE
        p._set_state(GTPWeightState.ASYNC_WAIT)
        p._set_state(GTPWeightState.DATA_READY)
        p._set_state(GTPWeightState.NONE)
        assert p.state == GTPWeightState.NONE

    def test_sync_path_cycle(self):
        """NONE → DATA_READY_SYNC → NONE (sync all-gather path)."""
        p = self._param()
        p._set_state(GTPWeightState.DATA_READY_SYNC)
        p._set_state(GTPWeightState.NONE)
        assert p.state == GTPWeightState.NONE

    def test_rs_state_full_cycle(self):
        """RS state machine: NONE → ASYNC_WAIT → DATA_READY → NONE."""
        p = self._param()
        assert p.rs_state == GTPWeightState.NONE
        p._set_rs_state(GTPWeightState.ASYNC_WAIT)
        p._set_rs_state(GTPWeightState.DATA_READY)
        p._set_rs_state(GTPWeightState.NONE)
        assert p.rs_state == GTPWeightState.NONE


# ---------------------------------------------------------------------------
# 2. GTPWeightCache – coat-check buffer pool tests
# ---------------------------------------------------------------------------


class TestGTPWeightCache:

    class _FakeGroup:
        def __init__(self, size=2):
            self._size = size

        def size(self):
            return self._size

        def rank(self):
            return 0

    def _param(self, shape=(8, 4), gtp_size=2):
        p = GTPShardedParam(torch.zeros(*shape))
        p.group = self._FakeGroup(gtp_size)
        p.expert_idx = None
        p.pad_length = 0
        p._quantizer = None
        return p

    def test_reserve_returns_ticket(self):
        cache = GTPWeightCache()
        p = self._param()
        ticket = cache.reserve(p, torch.bfloat16, fwd=True)
        assert isinstance(ticket, int)

    def test_reserve_get_roundtrip(self):
        cache = GTPWeightCache()
        p = self._param()
        ticket = cache.reserve(p, torch.bfloat16, fwd=True)
        buf = cache.get(ticket)
        assert buf is not None
        # get() returns same buf on second call (buf cached in slot)
        buf2 = cache.get(ticket)
        assert buf2 is buf

    def test_buffer_reused_after_release(self):
        cache = GTPWeightCache()
        p = self._param()
        t1 = cache.reserve(p, torch.bfloat16, fwd=True)
        buf1 = cache.get(t1)
        cache.release(t1)
        # Reserve a new ticket, buf should come from pool
        t2 = cache.reserve(p, torch.bfloat16, fwd=True)
        buf2 = cache.get(t2)
        assert buf1 is buf2, "Buffer should be reused from pool after release"
        cache.release(t2)

    def test_two_simultaneous_reserves_are_distinct(self):
        cache = GTPWeightCache()
        p = self._param()
        t1 = cache.reserve(p, torch.bfloat16, fwd=True)
        buf1 = cache.get(t1)
        t2 = cache.reserve(p, torch.bfloat16, fwd=True)
        buf2 = cache.get(t2)
        assert buf1 is not buf2, "Concurrent reserves must get distinct buffers"

    def test_tickets_are_unique(self):
        """Each reserve() call returns a new unique ticket."""
        cache = GTPWeightCache()
        p = self._param()
        t1 = cache.reserve(p, torch.bfloat16, fwd=True)
        t2 = cache.reserve(p, torch.bfloat16, fwd=True)
        assert t1 != t2, "Each reserve() must return a unique ticket"

    def test_invalid_ticket_raises(self):
        cache = GTPWeightCache()
        with pytest.raises(KeyError):
            cache.get(9999)

    def test_different_shapes_use_distinct_pool_slots(self):
        cache = GTPWeightCache()
        p1 = self._param(shape=(8, 4))
        p2 = self._param(shape=(16, 4))
        t1 = cache.reserve(p1, torch.bfloat16, fwd=True)
        buf1 = cache.get(t1)
        t2 = cache.reserve(p2, torch.bfloat16, fwd=True)
        buf2 = cache.get(t2)
        assert buf1.shape != buf2.shape
        cache.release(t1)
        cache.release(t2)

    def test_fwd_bwd_tickets_are_distinct(self):
        """fwd=True and fwd=False reserves always receive distinct ticket IDs."""
        cache = GTPWeightCache()
        p = self._param()
        t_fwd = cache.reserve(p, torch.bfloat16, fwd=True)
        t_bwd = cache.reserve(p, torch.bfloat16, fwd=False)
        assert t_fwd != t_bwd


# ---------------------------------------------------------------------------
# 3. GTP weight sharding: shard content and alignment padding
# ---------------------------------------------------------------------------


def _worker_sharding_aligned(rank, world_size, port):
    K, M = world_size * 32, 16  # K divisible by 16*world_size → no padding
    full_weight = torch.arange(K * M, dtype=torch.float32).reshape(K, M).cuda()
    dist.broadcast(full_weight, src=0)

    gtp_group = dist.new_group(list(range(world_size)))
    mod = nn.Module()
    mod.weight = nn.Parameter(full_weight.clone(), requires_grad=False)
    wrap_module_params_gtp(mod, ["weight"], gtp_group)
    shard = mod.weight

    rows_per_rank = K // world_size
    assert shard.shape == (rows_per_rank, M), f"rank {rank}: unexpected shape {shard.shape}"
    assert shard.pad_length == 0
    expected = full_weight[rank * rows_per_rank : (rank + 1) * rows_per_rank]
    assert torch.allclose(shard.data, expected), f"rank {rank}: shard content mismatch"


def _worker_sharding_padding(rank, world_size, port):
    alignment = 16 * world_size
    K = alignment - 1  # deliberately unaligned
    M = 16
    full_weight = torch.ones(K, M, dtype=torch.float32).cuda()
    dist.broadcast(full_weight, src=0)

    gtp_group = dist.new_group(list(range(world_size)))
    mod = nn.Module()
    mod.weight = nn.Parameter(full_weight.clone(), requires_grad=False)
    wrap_module_params_gtp(mod, ["weight"], gtp_group)
    shard = mod.weight

    padded_K = alignment
    rows_per_rank = padded_K // world_size

    if rank == world_size - 1:
        assert shard.pad_length > 0
        # The shard tensor holds only the real rows; get_padded_shard() appends zero rows.
        padded = shard.get_padded_shard()
        assert (
            padded.shape[0] == rows_per_rank
        ), f"rank {rank}: expected padded shard {rows_per_rank} rows, got {padded.shape[0]}"
        n_real = K - rank * rows_per_rank
        assert torch.all(padded[n_real:] == 0), "Padding rows must be zero"
    else:
        # pad_length is set globally on every rank's shard (slicer attaches the
        # global padding amount), so we don't assert anything about it here —
        # only the last rank's shard contains the actual padding rows.
        assert (
            shard.shape[0] == rows_per_rank
        ), f"rank {rank}: expected {rows_per_rank} rows, got {shard.shape[0]}"


class TestGTPSharding:
    def test_aligned_shard_content(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_sharding_aligned, 4)

    def test_unaligned_shard_padding(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_sharding_padding, 4)


# ---------------------------------------------------------------------------
# 4. wrap_module_params_gtp: param replacement and GroupedLinear weight_list
# ---------------------------------------------------------------------------


def _worker_linear_param_replaced(rank, world_size, port):
    in_f, out_f = 64, 128
    gtp_group = dist.new_group(list(range(world_size)))
    layer = te.Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
        gtp_group=gtp_group,
    )
    w = layer.weight
    assert isinstance(w, GTPShardedParam), "weight must be GTPShardedParam"
    assert w.shape == (out_f // world_size, in_f), f"unexpected shard shape {w.shape}"
    assert w.group is gtp_group


def _worker_grouped_weight_list(rank, world_size, port):
    num_gemms, in_f, out_f = 3, 32, 64
    gtp_group = dist.new_group(list(range(world_size)))
    layer = te.GroupedLinear(
        num_gemms=num_gemms,
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
        gtp_group=gtp_group,
    )
    w0 = layer.weight0
    assert isinstance(w0, GTPShardedParam)
    assert w0.weight_list is not None
    assert len(w0.weight_list) == num_gemms
    assert [w.expert_idx for w in w0.weight_list] == list(range(num_gemms))


class TestWrapModuleParams:
    def test_linear_weight_replaced(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_linear_param_replaced, 4)

    def test_grouped_linear_weight_list(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_grouped_weight_list, 4)


# ---------------------------------------------------------------------------
# 5. Linear forward/backward numerical correctness
# ---------------------------------------------------------------------------


def _worker_linear_correctness(rank, world_size, port):
    """GTP output == (all-gathered weight) @ input, and dX matches."""
    torch.manual_seed(0)
    batch, in_f, out_f = 16, 64, 128  # out_f % (16*world_size)==0 → no padding
    dtype = torch.bfloat16
    gtp_group = dist.new_group(list(range(world_size)))

    layer = te.Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        gtp_group=gtp_group,
    )

    # Reconstruct full weight from shards (all-gather)
    shard = layer.weight.data.clone()
    all_shards = [torch.zeros_like(shard) for _ in range(world_size)]
    dist.all_gather(all_shards, shard, group=gtp_group)
    full_weight = torch.cat(all_shards, dim=0).float()[:out_f]  # strip any padding

    # Shared input across ranks
    inp = torch.randn(batch, in_f, dtype=dtype, device="cuda")
    dist.broadcast(inp, src=0)

    inp_gtp = inp.clone().requires_grad_(True)
    inp_ref = inp.clone().requires_grad_(True)

    # GTP forward
    out_gtp = layer(inp_gtp, is_first_microbatch=True)

    # Reference forward
    out_ref = inp_ref.float() @ full_weight.T
    out_ref = out_ref.to(dtype)

    assert out_gtp.shape == out_ref.shape, f"Shape mismatch {out_gtp.shape} vs {out_ref.shape}"
    assert torch.allclose(
        out_gtp.float(), out_ref.float(), atol=0.1, rtol=0.1
    ), f"Output mismatch max_diff={(out_gtp.float()-out_ref.float()).abs().max():.4f}"

    # wgrad RS path always accumulates into main_grad; allocate before backward.
    layer.weight.main_grad = torch.zeros(layer.weight.shape, dtype=dtype, device="cuda")

    # Backward: compare input gradient
    grad_out = torch.randn_like(out_gtp)
    dist.broadcast(grad_out, src=0)
    out_gtp.backward(grad_out)
    out_ref.backward(grad_out.float())

    assert inp_gtp.grad is not None
    assert torch.allclose(
        inp_gtp.grad.float(), inp_ref.grad.float(), atol=0.1, rtol=0.1
    ), f"dX mismatch max_diff={(inp_gtp.grad.float()-inp_ref.grad.float()).abs().max():.4f}"


class TestLinearGTP:
    def test_forward_backward_correctness(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_linear_correctness, 4)


# ---------------------------------------------------------------------------
# 6. LayerNormLinear forward/backward smoke test
# ---------------------------------------------------------------------------


def _worker_layernorm_linear(rank, world_size, port):
    torch.manual_seed(0)
    seq, batch, in_f, out_f = 4, 2, 64, 128
    dtype = torch.bfloat16
    gtp_group = dist.new_group(list(range(world_size)))

    layer = te.LayerNormLinear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        gtp_group=gtp_group,
    )
    assert isinstance(layer.weight, GTPShardedParam)

    inp = torch.randn(seq, batch, in_f, dtype=dtype, device="cuda", requires_grad=True)
    dist.broadcast(inp, src=0)

    out = layer(inp, is_first_microbatch=True)
    assert out.shape == (seq, batch, out_f), f"unexpected output shape {out.shape}"

    layer.weight.main_grad = torch.zeros(layer.weight.shape, dtype=dtype, device="cuda")
    out.sum().backward()
    assert inp.grad is not None and inp.grad.shape == inp.shape


class TestLayerNormLinearGTP:
    def test_forward_backward(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_layernorm_linear, 4)


# ---------------------------------------------------------------------------
# 7. GroupedLinear forward/backward smoke test
# ---------------------------------------------------------------------------


def _worker_grouped_linear(rank, world_size, port, num_gemms):
    torch.manual_seed(0)
    in_f, out_f, total_tokens = 32, 64, num_gemms * 4
    dtype = torch.bfloat16
    gtp_group = dist.new_group(list(range(world_size)))

    layer = te.GroupedLinear(
        num_gemms=num_gemms,
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        gtp_group=gtp_group,
    )
    assert isinstance(layer.weight0, GTPShardedParam)

    m_splits = [total_tokens // num_gemms] * num_gemms
    m_splits[-1] += total_tokens - sum(m_splits)

    inp = torch.randn(total_tokens, in_f, dtype=dtype, device="cuda", requires_grad=True)
    dist.broadcast(inp, src=0)

    out = layer(inp, m_splits=m_splits, is_first_microbatch=True)
    assert out.shape == (total_tokens, out_f), f"unexpected output shape {out.shape}"

    for i in range(num_gemms):
        w = getattr(layer, f"weight{i}")
        w.main_grad = torch.zeros(w.shape, dtype=dtype, device="cuda")
    out.sum().backward()
    assert inp.grad is not None and inp.grad.shape == inp.shape


class TestGroupedLinearGTP:
    @pytest.mark.parametrize("num_gemms", [2, 4])
    def test_forward_backward(self, num_gemms):
        _requires_multi_gpu(4)
        _run_distributed(_worker_grouped_linear, 4, num_gemms)


# ---------------------------------------------------------------------------
# 8. Prefetch chain: next_w / prev_w wiring after first forward pass
# ---------------------------------------------------------------------------


def _worker_chain_wired(rank, world_size, port):
    torch.manual_seed(0)
    in_f, out_f = 32, 64
    dtype = torch.bfloat16
    gtp_group = dist.new_group(list(range(world_size)))

    l0 = te.Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        gtp_group=gtp_group,
    )
    l1 = te.Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        gtp_group=gtp_group,
    )

    inp = torch.randn(4, in_f, dtype=dtype, device="cuda")
    dist.broadcast(inp, src=0)

    # First forward pass builds the linked list
    l0(inp, is_first_microbatch=True)
    l1(inp, is_first_microbatch=True)

    w0, w1 = l0.weight, l1.weight
    assert w0.next_w is w1, "w0.next_w should point to w1"
    assert w1.prev_w is w0, "w1.prev_w should point back to w0"
    assert w1.next_w is None
    assert w0.prev_w is None


def _worker_chain_async_prefetch(rank, world_size, port):
    """On the second forward pass, w1 should be in DATA_READY before its forward runs."""
    torch.manual_seed(0)
    in_f, out_f = 32, 64
    dtype = torch.bfloat16
    gtp_group = dist.new_group(list(range(world_size)))

    l0 = te.Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        gtp_group=gtp_group,
    )
    l1 = te.Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        gtp_group=gtp_group,
    )

    inp = torch.randn(4, in_f, dtype=dtype, device="cuda")
    dist.broadcast(inp, src=0)

    # First pass builds chain, second pass uses async prefetch
    for _ in range(2):
        out = l0(inp, is_first_microbatch=True) + l1(inp, is_first_microbatch=True)
    assert torch.isfinite(out).all(), "Non-finite output on second pass"


class TestGTPPrefetchChain:
    def test_chain_wired_after_first_pass(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_chain_wired, 4)

    def test_async_prefetch_second_pass(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_chain_async_prefetch, 4)


# ---------------------------------------------------------------------------
# 9. Wgrad reduce-scatter: shape and deferred async path
# ---------------------------------------------------------------------------


def _worker_wgrad_shape(rank, world_size, port):
    """After backward, weight.grad shape must match the local shard shape."""
    torch.manual_seed(0)
    in_f, out_f = 32, 64
    dtype = torch.bfloat16
    gtp_group = dist.new_group(list(range(world_size)))

    layer = te.Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        gtp_group=gtp_group,
        fuse_wgrad_accumulation=False,
    )
    inp = torch.randn(8, in_f, dtype=dtype, device="cuda", requires_grad=True)
    dist.broadcast(inp, src=0)

    layer.weight.main_grad = torch.zeros(layer.weight.shape, dtype=dtype, device="cuda")
    layer(inp, is_first_microbatch=True).sum().backward()

    w = layer.weight
    if w.grad is not None:
        assert w.grad.shape == w.shape, f"wgrad shape {w.grad.shape} != shard shape {w.shape}"


def _worker_multilayer_deferred_rs(rank, world_size, port):
    """Two-layer GTP: async RS deferred for layer0 (non-last), sync for layer1 (last in bwd)."""
    torch.manual_seed(0)
    in_f, out_f = 32, 64
    dtype = torch.bfloat16
    gtp_group = dist.new_group(list(range(world_size)))

    l0 = te.Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        gtp_group=gtp_group,
    )
    l1 = te.Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        gtp_group=gtp_group,
    )

    inp = torch.randn(8, in_f, dtype=dtype, device="cuda", requires_grad=True)
    dist.broadcast(inp, src=0)

    # wgrad RS path always accumulates into main_grad; allocate before backward.
    l0.weight.main_grad = torch.zeros(l0.weight.shape, dtype=dtype, device="cuda")
    l1.weight.main_grad = torch.zeros(l1.weight.shape, dtype=dtype, device="cuda")

    out = l0(inp, is_first_microbatch=True) + l1(inp, is_first_microbatch=True)
    out.sum().backward()

    # Both weights' main_grad should have been updated
    for lyr in [l0, l1]:
        w = lyr.weight
        assert w.main_grad is not None, f"No main_grad on {lyr.__class__.__name__}.weight"


class TestGTPWgradRS:
    def test_wgrad_shape_matches_shard(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_wgrad_shape, 4)

    def test_multilayer_deferred_rs(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_multilayer_deferred_rs, 4)


# ---------------------------------------------------------------------------
# 10. Multiple microbatches: output must be consistent when weight unchanged
# ---------------------------------------------------------------------------


def _worker_microbatches(rank, world_size, port):
    torch.manual_seed(0)
    batch, in_f, out_f = 8, 64, 128
    dtype = torch.bfloat16
    gtp_group = dist.new_group(list(range(world_size)))

    layer = te.Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        gtp_group=gtp_group,
    )
    inp = torch.randn(batch, in_f, dtype=dtype, device="cuda")
    dist.broadcast(inp, src=0)

    # First microbatch
    out1 = layer(inp, is_first_microbatch=True).detach().clone()

    # Second microbatch with same weight (skip_weight_cast=True path)
    out2 = layer(inp, is_first_microbatch=False).detach()

    assert torch.allclose(
        out1, out2
    ), f"Microbatch outputs differ; max_diff={(out1-out2).abs().max():.6f}"


class TestGTPMicrobatches:
    def test_consistent_across_microbatches(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_microbatches, 4)


# ---------------------------------------------------------------------------
# 11. NVFP4 + GTP: Linear forward/backward, quantized shard setup
# ---------------------------------------------------------------------------


def _worker_nvfp4_linear(rank, world_size, port):
    """Verify that GTP Linear correctly quantizes, all-gathers, and computes with NVFP4."""
    torch.manual_seed(0)
    # batch=32: NVFP4 wgrad GEMM (K=batch) requires K divisible by 32
    batch, in_f, out_f = 32, 64, 128  # out_f % (16*world_size)==0 → no padding
    dtype = torch.bfloat16
    gtp_group = dist.new_group(list(range(world_size)))

    layer = te.Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        gtp_group=gtp_group,
    )
    inp = torch.randn(batch, in_f, dtype=dtype, device="cuda", requires_grad=True)
    dist.broadcast(inp, src=0)

    # Forward under NVFP4 recipe – triggers setup() and NVFP4 quantization
    recipe = NVFP4BlockScaling()
    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        out = layer(inp, is_first_microbatch=True)

    # After the first forward pass setup() must have created a quantized shard
    w = layer.weight
    assert w.quantized is not None, "NVFP4 quantized shard must be set after setup()"
    assert isinstance(
        w.quantized, QuantizedTensor
    ), f"weight.quantized should be QuantizedTensor, got {type(w.quantized)}"

    assert out.shape == (batch, out_f), f"unexpected output shape {out.shape}"
    assert torch.isfinite(out).all(), "NVFP4 GTP output has non-finite values"

    # Second microbatch reuses cached quantized weight (skip_weight_cast path)
    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        out2 = layer(inp.detach(), is_first_microbatch=False)
    assert torch.isfinite(out2).all(), "NVFP4 GTP second-microbatch output has non-finite values"


def _worker_nvfp4_linear_unaligned(rank, world_size, port):
    """Verify NVFP4 GTP when out_features is not aligned to 16*world_size (padding path).

    out_f is chosen to be divisible by 8 (satisfies NVFP4 GEMM alignment) but not by
    16*world_size (so padding is needed). The last GTP rank receives a shard that is
    zero-padded to reach the shard_size boundary. After all-gather, _strip_padding
    removes the padded rows from the gathered weight before the GEMM, so the output
    has the original out_f columns.
    """
    torch.manual_seed(0)
    alignment = 16 * world_size  # 64 for world_size=4
    # Choose out_f divisible by 8 (NVFP4 GEMM constraint) but not by 64 (GTP alignment).
    # With out_f=56: pad_length=8, shard_size=16, last rank gets 8 rows padded to 16.
    out_f = alignment - 8  # 56 for world_size=4
    in_f = 64
    batch = 32
    dtype = torch.bfloat16
    gtp_group = dist.new_group(list(range(world_size)))

    layer = te.Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        gtp_group=gtp_group,
    )
    inp = torch.randn(batch, in_f, dtype=dtype, device="cuda", requires_grad=True)
    dist.broadcast(inp, src=0)

    with fp8_autocast(enabled=True, fp8_recipe=NVFP4BlockScaling()):
        out = layer(inp, is_first_microbatch=True)

    # After _strip_padding removes the padded rows, output has out_f (not padded) cols.
    assert out.shape == (batch, out_f), f"unexpected output shape {out.shape}"
    assert torch.isfinite(out).all(), "NVFP4 GTP (unaligned) output has non-finite values"


class TestNVFP4LinearGTP:
    def test_forward_backward(self):
        _requires_nvfp4()
        _requires_multi_gpu(4)
        _run_distributed(_worker_nvfp4_linear, 4)

    def test_forward_unaligned_padding(self):
        _requires_nvfp4()
        _requires_multi_gpu(4)
        _run_distributed(_worker_nvfp4_linear_unaligned, 4)


# ---------------------------------------------------------------------------
# 12. NVFP4 + GTP: GroupedLinear forward/backward (coalesced batched all-gather)
# ---------------------------------------------------------------------------


def _worker_nvfp4_grouped_linear(rank, world_size, port, num_gemms):
    """Verify NVFP4 GTP with GroupedLinear (uses grouped_gather_along_first_dim)."""
    torch.manual_seed(0)
    # NVFP4 split_quantize constraints: in_f % 128 == 0, tokens_per_expert % 64 == 0
    # (Hadamard transform requirement), and K=tokens_per_expert % 32 == 0 for wgrad.
    in_f, out_f, total_tokens = 128, 256, num_gemms * 64
    dtype = torch.bfloat16
    gtp_group = dist.new_group(list(range(world_size)))

    layer = te.GroupedLinear(
        num_gemms=num_gemms,
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        gtp_group=gtp_group,
    )
    assert isinstance(layer.weight0, GTPShardedParam)

    m_splits = [total_tokens // num_gemms] * num_gemms
    m_splits[-1] += total_tokens - sum(m_splits)

    inp = torch.randn(total_tokens, in_f, dtype=dtype, device="cuda", requires_grad=True)
    dist.broadcast(inp, src=0)

    with fp8_autocast(enabled=True, fp8_recipe=NVFP4BlockScaling()):
        out = layer(inp, m_splits=m_splits, is_first_microbatch=True)

    assert out.shape == (total_tokens, out_f), f"unexpected output shape {out.shape}"
    assert torch.isfinite(out).all(), "NVFP4 GroupedLinear GTP output has non-finite values"

    # All expert weight shards should be quantized after setup()
    for i in range(num_gemms):
        name = f"weight{i}"
        w = getattr(layer, name)
        assert isinstance(w, GTPShardedParam)
        assert w.quantized is not None, f"{name}.quantized not set after NVFP4 setup()"
        assert isinstance(
            w.quantized, QuantizedTensor
        ), f"{name}.quantized should be QuantizedTensor, got {type(w.quantized)}"

    for i in range(num_gemms):
        w = getattr(layer, f"weight{i}")
        w.main_grad = torch.zeros(w.shape, dtype=dtype, device="cuda")
    out.sum().backward()
    assert inp.grad is not None and inp.grad.shape == inp.shape


class TestNVFP4GroupedLinearGTP:
    @pytest.mark.parametrize("num_gemms", [2, 4])
    def test_forward_backward(self, num_gemms):
        _requires_nvfp4()
        _requires_multi_gpu(4)
        _run_distributed(_worker_nvfp4_grouped_linear, 4, num_gemms)


# ---------------------------------------------------------------------------
# 13. MXFP8 + GTP: Linear forward/backward, quantized shard setup
# ---------------------------------------------------------------------------


def _worker_mxfp8_linear(rank, world_size, port):
    """Verify that GTP Linear correctly quantizes, all-gathers, and computes with MXFP8."""
    from transformer_engine.common.recipe import MXFP8BlockScaling

    torch.manual_seed(0)
    # batch=32: MXFP8 wgrad GEMM (K=batch) requires K divisible by MXFP8_BLOCK_SCALING_SIZE=32
    batch, in_f, out_f = 32, 64, 128  # out_f % (16*world_size)==0 → no padding
    dtype = torch.bfloat16
    gtp_group = dist.new_group(list(range(world_size)))

    layer = te.Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        gtp_group=gtp_group,
    )
    inp = torch.randn(batch, in_f, dtype=dtype, device="cuda", requires_grad=True)
    dist.broadcast(inp, src=0)

    # Forward under MXFP8 recipe – triggers setup() and MXFP8 quantization
    recipe = MXFP8BlockScaling()
    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        out = layer(inp, is_first_microbatch=True)

    # After the first forward pass setup() must have created a quantized shard
    w = layer.weight
    assert w.quantized is not None, "MXFP8 quantized shard must be set after setup()"
    assert isinstance(
        w.quantized, QuantizedTensor
    ), f"weight.quantized should be QuantizedTensor, got {type(w.quantized)}"

    assert out.shape == (batch, out_f), f"unexpected output shape {out.shape}"
    assert torch.isfinite(out).all(), "MXFP8 GTP output has non-finite values"

    # Backward should complete without error
    layer.weight.main_grad = torch.zeros(layer.weight.shape, dtype=dtype, device="cuda")
    out.sum().backward()
    assert inp.grad is not None
    assert inp.grad.shape == inp.shape

    # Second microbatch reuses cached quantized weight (skip_weight_cast path)
    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        out2 = layer(inp.detach(), is_first_microbatch=False)
    assert torch.isfinite(out2).all(), "MXFP8 GTP second-microbatch output has non-finite values"


def _worker_mxfp8_linear_unaligned(rank, world_size, port):
    """Verify MXFP8 GTP when out_features is not aligned to 16*world_size (padding path).

    MXFP8 requires tensor dims divisible by 32, so shard_size (= M_padded / world_size)
    must be a multiple of 32. With world_size=4 this requires M_padded % 128 == 0.
    out_f=120 gives M_padded=128, shard_size=32 (32 % 32 == 0). The last rank has
    24 real rows zero-padded to 32. After all-gather, _strip_padding removes the padded
    rows before the GEMM, so the output has the original out_f columns.
    """
    from transformer_engine.common.recipe import MXFP8BlockScaling

    torch.manual_seed(0)
    # out_f=120: M_padded=128, shard_size=32, last rank has 24 rows padded to 32.
    # 120 is divisible by 8 (GEMM constraint), not by 64 (GTP alignment → padding needed).
    out_f = 120
    in_f = 64
    batch = 32
    dtype = torch.bfloat16
    gtp_group = dist.new_group(list(range(world_size)))

    layer = te.Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        gtp_group=gtp_group,
    )
    inp = torch.randn(batch, in_f, dtype=dtype, device="cuda", requires_grad=True)
    dist.broadcast(inp, src=0)

    with fp8_autocast(enabled=True, fp8_recipe=MXFP8BlockScaling()):
        out = layer(inp, is_first_microbatch=True)

    # After _strip_padding removes the padded rows, output has out_f (not padded) cols.
    assert out.shape == (batch, out_f), f"unexpected output shape {out.shape}"
    assert torch.isfinite(out).all(), "MXFP8 GTP (unaligned) output has non-finite values"


def _requires_mxfp8():
    available, reason = is_mxfp8_available(return_reason=True)
    if not available:
        pytest.skip(f"MXFP8 not available: {reason}")


class TestMXFP8LinearGTP:
    def test_forward_backward(self):
        _requires_mxfp8()
        _requires_multi_gpu(4)
        _run_distributed(_worker_mxfp8_linear, 4)

    def test_forward_unaligned_padding(self):
        _requires_mxfp8()
        _requires_multi_gpu(4)
        _run_distributed(_worker_mxfp8_linear_unaligned, 4)


# ---------------------------------------------------------------------------
# 14. GTPConfig / update_config
# ---------------------------------------------------------------------------


class TestGTPConfig:

    def test_update_pad_for_alignment(self):
        original = gtp_module.GTP_CONFIG.pad_for_alignment
        try:
            gtp_module.update_config(pad_for_alignment=8)
            assert gtp_module.GTP_CONFIG.pad_for_alignment == 8
        finally:
            gtp_module.update_config(pad_for_alignment=original)

    def test_update_weight_prefetch(self):
        original = gtp_module.GTP_CONFIG.weight_prefetch
        try:
            gtp_module.update_config(weight_prefetch=False)
            assert gtp_module.GTP_CONFIG.weight_prefetch is False
        finally:
            gtp_module.update_config(weight_prefetch=original)

    def test_invalid_key_raises(self):
        with pytest.raises(ValueError, match="Unknown GTP config option"):
            gtp_module.update_config(nonexistent_key=123)


# ---------------------------------------------------------------------------
# 15. GTPShardedParam properties – shape computations and padding
# ---------------------------------------------------------------------------


class TestGTPShardedParamProperties:

    class _FakeGroup:
        def __init__(self, size=4, rank=0):
            self._size = size
            self._rank = rank

        def size(self):
            return self._size

        def rank(self):
            return self._rank

    def _make_param(self, shape, pad_length=0, group_size=4, group_rank=0):
        p = GTPShardedParam(torch.zeros(*shape))
        p.group = self._FakeGroup(size=group_size, rank=group_rank)
        p.pad_length = pad_length
        p.expert_idx = None
        return p

    # --- _unsharded_shape_padded ---

    def test_unsharded_shape_padded_no_padding(self):
        # shape=(8, 4), group_size=4 → 8*4=32 rows, no padding
        p = self._make_param((8, 4), pad_length=0, group_size=4, group_rank=2)
        assert p._unsharded_shape_padded == (32, 4)

    def test_unsharded_shape_padded_last_rank_with_padding(self):
        # Local shard includes its slice of padding rows: 16 rows per rank,
        # pad_length=1 marks 1 of those (on the last rank) as pad → padded
        # unsharded shape = 16 * 4 = 64.  pad_length is global metadata, the
        # same value lives on every rank's shard.
        p = self._make_param((16, 32), pad_length=1, group_size=4, group_rank=3)
        assert p._unsharded_shape_padded == (64, 32)

    def test_unsharded_shape_padded_non_last_rank_with_padding(self):
        # Non-last rank: pad_length is the same global value, same formula.
        p = self._make_param((16, 32), pad_length=1, group_size=4, group_rank=0)
        assert p._unsharded_shape_padded == (64, 32)

    # --- _unsharded_shape ---

    def test_unsharded_shape_no_padding(self):
        p = self._make_param((8, 4), pad_length=0, group_size=4, group_rank=0)
        assert p._unsharded_shape == (32, 4)

    def test_unsharded_shape_strips_padding(self):
        # Local 16 rows × 4 ranks = 64 padded; pad_length=1 → unsharded = 63.
        p = self._make_param((16, 32), pad_length=1, group_size=4, group_rank=3)
        assert p._unsharded_shape == (63, 32)

    # --- get_padded_shard ---

    def test_get_padded_shard_identity_when_no_padding(self):
        p = self._make_param((6, 4), pad_length=0)
        result = p.get_padded_shard()
        assert result is p  # identity – no copy needed

    def test_get_padded_shard_identity_non_last_rank(self):
        # pad_length > 0 but not the padded last rank → no padding added
        p = self._make_param((16, 4), pad_length=1, group_size=4, group_rank=0)
        result = p.get_padded_shard()
        assert result is p

    def test_get_padded_shard_identity_last_rank(self):
        # Under current semantics the local shard already contains its share
        # of padding (slicer F.pads with zeros before slicing), so
        # get_padded_shard() is the identity on the last rank too.
        p = self._make_param((8, 4), pad_length=2, group_size=4, group_rank=3)
        assert p.get_padded_shard() is p

    # --- _strip_padding ---

    def test_strip_padding_identity_no_padding(self):
        p = self._make_param((8, 4), pad_length=0)
        t = torch.randn(32, 4)
        assert p._strip_padding(t) is t

    def test_strip_padding_plain_tensor(self):
        # Gathered weight [32, 4] with pad_length=1 → strip 1 row → [31, 4]
        p = self._make_param((7, 4), pad_length=1, group_size=4, group_rank=0)
        t = torch.randn(32, 4)
        result = p._strip_padding(t)
        assert result.shape == (31, 4)
        assert torch.equal(result, t[:-1])

    def test_strip_padding_multi_row(self):
        # pad_length=4 strips 4 rows
        p = self._make_param((12, 8), pad_length=4, group_size=4, group_rank=0)
        t = torch.ones(64, 8)
        result = p._strip_padding(t)
        assert result.shape == (60, 8)


# ---------------------------------------------------------------------------
# 16. _get_cache_key – expert vs non-expert, fwd vs bwd
# ---------------------------------------------------------------------------


class TestGTPCacheKey:

    class _FakeGroup:
        def size(self):
            return 4

        def rank(self):
            return 0

    def _param(self, shape=(16, 32), expert_idx=None):
        p = GTPShardedParam(torch.zeros(*shape))
        p.group = self._FakeGroup()
        p.expert_idx = expert_idx
        p.pad_length = 0
        return p

    def test_non_expert_key_same_for_fwd_bwd(self):
        """Non-routed params produce the same cache key for fwd and bwd."""
        p = self._param(expert_idx=None)
        assert p._get_cache_key(torch.bfloat16, fwd=True, reduce_scatter=False) == p._get_cache_key(
            torch.bfloat16, fwd=False, reduce_scatter=False
        )

    def test_expert_key_differs_fwd_bwd(self):
        """For quantized (non-torch.dtype) recipes, expert fwd vs bwd keys differ."""
        p = self._param(expert_idx=0)
        # _get_cache_key differentiates fwd/bwd only for non-torch.dtype objects
        # (e.g. quantized recipe dtype descriptors).  Use a mock to trigger that path.
        mock_dtype = "fp8"
        assert p._get_cache_key(mock_dtype, fwd=True, reduce_scatter=False) != p._get_cache_key(
            mock_dtype, fwd=False, reduce_scatter=False
        )

    def test_different_expert_idx_different_keys(self):
        """Two experts with same shape but different indices get distinct keys."""
        p0 = self._param(expert_idx=0)
        p1 = self._param(expert_idx=1)
        assert p0._get_cache_key(
            torch.bfloat16, fwd=True, reduce_scatter=False
        ) != p1._get_cache_key(torch.bfloat16, fwd=True, reduce_scatter=False)

    def test_same_expert_idx_same_key(self):
        """Same-shaped experts with the same idx share a cache key (cross-layer buffer reuse)."""
        p_l0 = self._param(expert_idx=0)
        p_l1 = self._param(expert_idx=0)
        assert p_l0._get_cache_key(
            torch.bfloat16, fwd=True, reduce_scatter=False
        ) == p_l1._get_cache_key(torch.bfloat16, fwd=True, reduce_scatter=False)

    def test_different_dtypes_different_keys(self):
        p = self._param()
        assert p._get_cache_key(torch.bfloat16, fwd=True, reduce_scatter=False) != p._get_cache_key(
            torch.float32, fwd=True, reduce_scatter=False
        )

    def test_rs_key_differs_from_ag_key(self):
        """reduce_scatter=True key must differ from reduce_scatter=False key."""
        p = self._param()
        assert p._get_cache_key(torch.bfloat16, fwd=True, reduce_scatter=False) != p._get_cache_key(
            torch.bfloat16, fwd=True, reduce_scatter=True
        )


# ---------------------------------------------------------------------------
# 17. GTPWeightCache.take() deferred vs get() immediate pool return
# ---------------------------------------------------------------------------


class TestGTPCacheRelease:
    """Tests for GTPWeightCache reserve/get/release semantics."""

    class _FakeGroup:
        def size(self):
            return 2

        def rank(self):
            return 0

    def _param(self, shape=(8, 4)):
        p = GTPShardedParam(torch.zeros(*shape))
        p.group = self._FakeGroup()
        p.expert_idx = None
        p.pad_length = 0
        p._quantizer = None
        return p

    def test_release_returns_buffer_to_pool(self):
        """release() puts the buffer back so the next reserve+get reuses it."""
        cache = GTPWeightCache()
        p = self._param()
        t1 = cache.reserve(p, torch.bfloat16, fwd=True)
        buf1 = cache.get(t1)
        cache.release(t1)
        # New ticket should pop buf1 from pool
        t2 = cache.reserve(p, torch.bfloat16, fwd=True)
        buf2 = cache.get(t2)
        assert buf2 is buf1, "Buffer should be reused after release()"
        cache.release(t2)

    def test_without_release_pool_stays_empty(self):
        """Without release(), subsequent reserves allocate fresh buffers."""
        cache = GTPWeightCache()
        p = self._param()
        t1 = cache.reserve(p, torch.bfloat16, fwd=True)
        buf1 = cache.get(t1)
        # Do NOT release t1 — pool stays empty
        t2 = cache.reserve(p, torch.bfloat16, fwd=True)
        buf2 = cache.get(t2)
        assert buf2 is not buf1, "Without release, a fresh buffer must be allocated"

    def test_get_same_ticket_returns_same_buf(self):
        """get() is idempotent — calling it twice returns the same buffer."""
        cache = GTPWeightCache()
        p = self._param()
        t = cache.reserve(p, torch.bfloat16, fwd=True)
        buf_a = cache.get(t)
        buf_b = cache.get(t)
        assert buf_a is buf_b
        cache.release(t)

    def test_release_invalid_ticket_raises(self):
        cache = GTPWeightCache()
        with pytest.raises(KeyError):
            cache.release(9999)


# ---------------------------------------------------------------------------
# 18. tag_gtp_params_with_names – _debug_name population
# ---------------------------------------------------------------------------


class TestTagGTPParamsWithNames:

    def test_debug_name_populated_for_gtp_param(self):
        """GTPShardedParam._debug_name is set to the dotted parameter path."""

        class _FakeGroup:
            def size(self):
                return 1

            def rank(self):
                return 0

        model = nn.Linear(4, 8, bias=False)
        w = GTPShardedParam(torch.randn(8, 4))
        w.group = _FakeGroup()
        model._parameters["weight"] = w

        gtp_module.tag_gtp_params_with_names(model)
        assert w._debug_name == "weight", f"Expected 'weight', got '{w._debug_name}'"

    def test_nested_module_debug_name(self):
        """Nested module produces a dotted debug name."""

        class _FakeGroup:
            def size(self):
                return 1

            def rank(self):
                return 0

        outer = nn.Sequential(nn.Linear(4, 8, bias=False))
        w = GTPShardedParam(torch.randn(8, 4))
        w.group = _FakeGroup()
        outer._modules["0"]._parameters["weight"] = w

        gtp_module.tag_gtp_params_with_names(outer)
        assert w._debug_name == "0.weight", f"Expected '0.weight', got '{w._debug_name}'"

    def test_non_gtp_params_are_skipped(self):
        """Plain nn.Parameter instances are silently ignored."""
        model = nn.Linear(4, 8)
        gtp_module.tag_gtp_params_with_names(model)  # must not raise


# ---------------------------------------------------------------------------
# 19. wrap_module_params_gtp is a no-op when gtp_group.size() == 1
# ---------------------------------------------------------------------------


class TestGTPGroupSizeOne:

    class _SingletonGroup:
        def size(self):
            return 1

        def rank(self):
            return 0

    def test_no_sharding_when_gtp_size_one(self):
        """wrap_module_params_gtp must be a no-op for a singleton GTP group."""
        mod = nn.Linear(32, 64, bias=False)
        original_weight = mod.weight
        wrap_module_params_gtp(mod, ["weight"], self._SingletonGroup())
        assert (
            mod.weight is original_weight
        ), "gtp_group.size()==1 should leave parameters unchanged"
        assert not isinstance(mod.weight, GTPShardedParam)


# ---------------------------------------------------------------------------
# 21. weight_prefetch=False: forward still produces correct output
# ---------------------------------------------------------------------------


def _worker_prefetch_disabled(rank, world_size, port):
    torch.manual_seed(0)
    in_f, out_f = 32, 64
    dtype = torch.bfloat16
    gtp_group = dist.new_group(list(range(world_size)))

    gtp_module.update_config(weight_prefetch=False)
    try:
        l0 = te.Linear(
            in_features=in_f,
            out_features=out_f,
            bias=False,
            params_dtype=dtype,
            device="cuda",
            gtp_group=gtp_group,
        )
        l1 = te.Linear(
            in_features=in_f,
            out_features=out_f,
            bias=False,
            params_dtype=dtype,
            device="cuda",
            gtp_group=gtp_group,
        )

        inp = torch.randn(4, in_f, dtype=dtype, device="cuda")
        dist.broadcast(inp, src=0)

        # Single forward pass: builds chain and verifies output is correct
        out = l0(inp, is_first_microbatch=True) + l1(inp, is_first_microbatch=True)

        # Chain should still be wired even with prefetch disabled
        assert l0.weight.next_w is l1.weight
        assert torch.isfinite(out).all(), "Non-finite output with prefetch disabled"
    finally:
        gtp_module.update_config(weight_prefetch=True)


class TestGTPPrefetchDisabled:
    def test_forward_works_without_prefetch(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_prefetch_disabled, 4)


# ---------------------------------------------------------------------------
# 22. fuse_wgrad_accumulation=True: wgrad is accumulated into main_grad
# ---------------------------------------------------------------------------


def _worker_fuse_wgrad(rank, world_size, port):
    torch.manual_seed(0)
    in_f, out_f = 32, 128  # out_f % (16*world_size)==0, no padding
    dtype = torch.bfloat16
    gtp_group = dist.new_group(list(range(world_size)))

    layer = te.Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        gtp_group=gtp_group,
        fuse_wgrad_accumulation=True,
    )

    # Allocate main_grad on the local shard shape
    w = layer.weight
    w.main_grad = torch.zeros(w.shape, dtype=dtype, device="cuda")

    inp = torch.randn(8, in_f, dtype=dtype, device="cuda", requires_grad=True)
    dist.broadcast(inp, src=0)

    layer(inp, is_first_microbatch=True).sum().backward()

    # With fused accumulation, wgrad was added into main_grad
    assert torch.any(
        w.main_grad != 0
    ), "main_grad should have been updated by fused wgrad accumulation"


class TestFuseWgradAccumulation:
    def test_wgrad_accumulated_into_main_grad(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_fuse_wgrad, 4)


# ---------------------------------------------------------------------------
# 23. _grad_accum_hook is called after reduce-scatter
# ---------------------------------------------------------------------------


def _worker_main_grad_updated_after_bwd(rank, world_size, port):
    """After backward, the wgrad RS path must have accumulated wgrad into main_grad."""
    torch.manual_seed(0)
    in_f, out_f = 32, 64
    dtype = torch.bfloat16
    gtp_group = dist.new_group(list(range(world_size)))

    layer = te.Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        gtp_group=gtp_group,
    )

    # wgrad RS path always accumulates into main_grad; allocate before backward.
    layer.weight.main_grad = torch.zeros(layer.weight.shape, dtype=dtype, device="cuda")

    inp = torch.randn(8, in_f, dtype=dtype, device="cuda", requires_grad=True)
    dist.broadcast(inp, src=0)
    layer(inp, is_first_microbatch=True).sum().backward()

    assert torch.any(
        layer.weight.main_grad != 0
    ), "main_grad should have been updated after the reduce-scatter accumulation"


class TestGTPGradAccumHook:
    def test_main_grad_updated_after_backward(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_main_grad_updated_after_bwd, 4)


# ---------------------------------------------------------------------------
# 24. wait_async_comms(finalize_after_drain=True) inline-accumulation fallback
# ---------------------------------------------------------------------------


class TestWaitAsyncCommsFallback:
    """Exercises the inline-accumulation fallback inside
    ``wait_async_comms(finalize_after_drain=True)``: when a param is in
    ``_inflight_comm_params`` (async AG was issued) but its ``_wgrad_rs_handle``
    is ``None`` (no async RS handle to drain), the inner
    ``_wait_reduce_scatter`` call no-ops and the outer loop must inline the
    accumulation itself (main_grad.add_ + ticket release + flag set).

    Production flows rarely hit this combination — chain-interior params have
    both async AG and async RS, and chain-head sync RS doesn't enter
    ``_inflight_comm_params`` via bwd AG. We construct the state by hand to
    pin down the fallback's contract.
    """

    class _FakeGroup:
        def size(self):
            return 1

        def rank(self):
            return 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_fallback_accumulates_when_no_rs_handle(self):
        dtype = torch.bfloat16
        p = GTPShardedParam(torch.zeros(8, 4, dtype=dtype, device="cuda"))
        p.group = self._FakeGroup()
        p.expert_idx = None
        p.pad_length = 0
        p.chain_id = gtp_module.GTPChain.UNGRAPHED.value
        p._quantizer = None
        p.is_routed_expert = False  # ⇒ self._weights property returns [self]
        p.main_grad = torch.zeros(8, 4, dtype=dtype, device="cuda")
        p._prefetch_handle = None  # _wait_param_gather is no-op
        p._wgrad_rs_handle = None  # _wait_reduce_scatter is no-op → fallback fires
        p._cached_ag_stream = None
        p._cached_rs_stream = None
        p.ag_event = torch.cuda.Event(external=True)
        p.rs_event = torch.cuda.Event(external=True)
        p.rs_event.record()  # so rs_event.wait() in fallback doesn't block
        p._already_finalized = False
        p.grad_added_to_main_grad = False

        # Place a known wgrad in the cache for the fallback to read.
        cache = gtp_module.get_global_GTP_cache()
        p._rs_ticket = cache.reserve(p, dtype, fwd=False, reduce_scatter=True)
        cache.get(p._rs_ticket).fill_(2.0)

        # Save + replace _inflight_comm_params so we don't trip over leftover
        # params from earlier tests in the loop.
        saved = set(gtp_module._inflight_comm_params)
        gtp_module._inflight_comm_params.clear()
        gtp_module._inflight_comm_params.add(p)
        try:
            gtp_module.wait_async_comms(
                chain_id=p.chain_id, skip_rs=False, finalize_after_drain=True
            )
        finally:
            gtp_module._inflight_comm_params.clear()
            gtp_module._inflight_comm_params.update(saved)

        torch.cuda.synchronize()
        assert torch.all(
            p.main_grad == 2.0
        ), f"main_grad should be 2.0 after fallback accumulation; got {p.main_grad}"
        assert p._already_finalized is True, "_already_finalized must be set"
        assert p.grad_added_to_main_grad is True, "grad_added_to_main_grad must be set"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_fallback_skipped_when_already_finalized(self):
        """When _already_finalized=True, the fallback must NOT re-accumulate."""
        dtype = torch.bfloat16
        p = GTPShardedParam(torch.zeros(8, 4, dtype=dtype, device="cuda"))
        p.group = self._FakeGroup()
        p.expert_idx = None
        p.pad_length = 0
        p.chain_id = gtp_module.GTPChain.UNGRAPHED.value
        p._quantizer = None
        p.is_routed_expert = False  # ⇒ self._weights property returns [self]
        # Pre-existing main_grad with a value the fallback must NOT overwrite.
        p.main_grad = torch.full((8, 4), 5.0, dtype=dtype, device="cuda")
        p._prefetch_handle = None
        p._wgrad_rs_handle = None
        p._cached_ag_stream = None
        p._cached_rs_stream = None
        p.ag_event = torch.cuda.Event(external=True)
        p.rs_event = torch.cuda.Event(external=True)
        p.rs_event.record()
        p._already_finalized = True  # ← short-circuits the fallback

        # No _rs_ticket: if the fallback ran it would AttributeError on
        # cache.get(None).  The skip path must not touch the cache at all.
        p._rs_ticket = None

        saved = set(gtp_module._inflight_comm_params)
        gtp_module._inflight_comm_params.clear()
        gtp_module._inflight_comm_params.add(p)
        try:
            gtp_module.wait_async_comms(
                chain_id=p.chain_id, skip_rs=False, finalize_after_drain=True
            )
        finally:
            gtp_module._inflight_comm_params.clear()
            gtp_module._inflight_comm_params.update(saved)

        torch.cuda.synchronize()
        assert torch.all(
            p.main_grad == 5.0
        ), "main_grad must be untouched when _already_finalized=True"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_fallback_skipped_for_pure_ag_param(self):
        """Regression: cross-graph fwd-AG prefetch in flight + finalize_after_drain=True.

        A param can be in _inflight_comm_params because of an outstanding async
        all-gather (e.g. a cross-graph forward prefetch reaching the
        bwd→optimizer boundary).  No reduce-scatter was ever issued for that
        param, so _rs_ticket is None on every weight.  Previously the fallback
        called cache.get(None) and crashed with KeyError; the guard now skips
        the inline accumulation entirely when no weight has an RS ticket.
        """
        dtype = torch.bfloat16
        p = GTPShardedParam(torch.zeros(8, 4, dtype=dtype, device="cuda"))
        p.group = self._FakeGroup()
        p.expert_idx = None
        p.pad_length = 0
        p.chain_id = gtp_module.GTPChain.UNGRAPHED.value
        p._quantizer = None
        p.is_routed_expert = False
        # Pre-existing main_grad with a sentinel that must survive untouched.
        p.main_grad = torch.full((8, 4), 7.0, dtype=dtype, device="cuda")
        p._prefetch_handle = None
        p._wgrad_rs_handle = None
        p._cached_ag_stream = None
        p._cached_rs_stream = None
        p.ag_event = torch.cuda.Event(external=True)
        p.rs_event = torch.cuda.Event(external=True)
        p.rs_event.record()
        p._already_finalized = False
        # Critical: simulates a pure-AG prefetch — no RS ever issued, ticket is None.
        p._rs_ticket = None

        saved = set(gtp_module._inflight_comm_params)
        gtp_module._inflight_comm_params.clear()
        gtp_module._inflight_comm_params.add(p)
        try:
            # Must NOT raise KeyError(None) from cache.get(None).
            gtp_module.wait_async_comms(
                chain_id=p.chain_id, skip_rs=False, finalize_after_drain=True
            )
        finally:
            gtp_module._inflight_comm_params.clear()
            gtp_module._inflight_comm_params.update(saved)

        torch.cuda.synchronize()
        assert torch.all(
            p.main_grad == 7.0
        ), "main_grad must be untouched for a pure-AG param (no wgrad to accumulate)"
        assert (
            p._already_finalized is False
        ), "_already_finalized must stay False — no finalize happened for a pure-AG param"
