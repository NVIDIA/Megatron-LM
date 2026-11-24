# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for Generalized Tensor Parallelism (GTP).

Scope: sharding, single-module numerics against a dense reference, and behavior/regression guards
for the GTP runtime. Model-level loss/grad, checkpoint and CUDA-graph replay correctness live in
the integration tests (test_gtp_grad_correctness, test_gtp_loss_correctness, test_gtp_dcp,
test_gtp_grouped_cg, ...); chain registration and classification live in
test_gtp_prefetch_registration.

Multi-GPU tests skip unless ``torch.distributed.get_world_size()`` is 4.
"""

import contextlib

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

import transformer_engine.pytorch as te
from transformer_engine.pytorch import fp8_autocast
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor

import megatron.core.tensor_parallel.generalized_tensor_parallelism as gtp_module
import megatron.core.tensor_parallel.gtp_cuda_graphs as gtp_cuda_graphs
import megatron.core.tensor_parallel.gtp_eager_wgrad as gtp_eager_wgrad
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
    GTPChain,
    GTPShardedParam,
    GTPWeightCache,
    wrap_module_params_gtp,
)
from megatron.core.tensor_parallel.gtp_cuda_graphs import preserve_gtp_prefetch_state
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _make_gtp_linear,
    _make_gtp_remat_grouped_linear,
    _requires_multi_gpu,
    _requires_mxfp8,
    _run_distributed,
    cached_group,
)
from tests.unit_tests.test_utilities import Utils


class _FakeGroup:
    """Minimal mock for a dist process group — used in single-process unit tests."""

    def __init__(self, size=1, rank=0):
        self._size = size
        self._rank = rank
        self.group_name = f"fake_group_{id(self)}"

    def size(self):
        return self._size

    def rank(self):
        return self._rank


@contextlib.contextmanager
def _model_parallel_with_gtp(gtp_remat_size):
    """Re-initialize MPU with a GTP_remat axis, then restore the module's default MPU state."""
    Utils.initialize_model_parallel(gtp_remat_size=gtp_remat_size)
    try:
        yield parallel_state.get_gtp_weight_remat_group()
    finally:
        Utils.initialize_model_parallel()


def _padded_row_shard(full, rows_per_rank, rank):
    """This rank's dim-0 rows of ``full`` zero-padded to ``rows_per_rank * world_size`` rows."""
    pad = rows_per_rank * dist.get_world_size() - full.shape[0]
    padded = torch.nn.functional.pad(full, (0, 0, 0, pad))
    return padded[rank * rows_per_rank : (rank + 1) * rows_per_rank]


def _all_gather_rows(tensor):
    """Concatenate every rank's ``tensor`` along dim 0."""
    parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, tensor.contiguous())
    return torch.cat(parts, dim=0)


class TestGTPWeightCacheSchedulingDomain:
    @staticmethod
    def _make_param(group, chain_id):
        param = GTPShardedParam(torch.zeros(4, 4, device="cuda"))
        param.group = group
        param.chain_id = chain_id
        return param

    def test_cache_reuse_isolated_by_chain_and_process_group(self):
        cache = GTPWeightCache()
        group = _FakeGroup(size=2)
        other_group = _FakeGroup(size=2)

        first = self._make_param(group, GTPChain.UNGRAPHED.value)
        second = self._make_param(group, GTPChain.UNGRAPHED.value)
        first_ticket = cache.reserve(first, torch.bfloat16, fwd=True)
        first_buffer = cache.get(first_ticket)
        cache.release(first_ticket)
        second_ticket = cache.reserve(second, torch.bfloat16, fwd=True)
        assert cache.get(second_ticket) is first_buffer
        cache.release(second_ticket)

        graphed = self._make_param(group, GTPChain.GRAPHED.value)
        graphed_ticket = cache.reserve(graphed, torch.bfloat16, fwd=True)
        assert cache.get(graphed_ticket) is not first_buffer

        other = self._make_param(other_group, GTPChain.UNGRAPHED.value)
        other_ticket = cache.reserve(other, torch.bfloat16, fwd=True)
        assert cache.get(other_ticket) is not first_buffer

    def test_released_rs_output_is_not_reused_by_its_previous_owner(self):
        """Releasing an eager RS output gives up the ticket's ownership: once another ticket
        has acquired the buffer, the first ticket's next get() returns a different buffer.
        """
        cache = GTPWeightCache()
        group = _FakeGroup(size=2)
        a = self._make_param(group, GTPChain.UNGRAPHED.value)
        b = self._make_param(group, GTPChain.UNGRAPHED.value)
        a_ticket = cache.reserve(a, torch.float32, fwd=False, reduce_scatter=True)
        b_ticket = cache.reserve(b, torch.float32, fwd=False, reduce_scatter=True)
        a_buffer = cache.get(a_ticket)
        cache.release(a_ticket)
        assert cache.get(b_ticket) is a_buffer, "B should acquire the buffer A released"
        assert cache.get(a_ticket) is not a_buffer, "A reused the buffer B now owns"
        assert cache.get(b_ticket) is a_buffer, "B lost the buffer it owns"


# ---------------------------------------------------------------------------
# wrap_module_params_gtp: shard content and padding
# ---------------------------------------------------------------------------


def _worker_shard_content(rank, world_size, port, rows, pad_length):
    """Every rank holds its rows of the zero-padded full weight; padding sits at the tail."""
    full_weight = torch.arange(rows * 16, dtype=torch.float32, device="cuda").reshape(rows, 16) + 1
    mod = nn.Module()
    mod.weight = nn.Parameter(full_weight.clone(), requires_grad=False)
    wrap_module_params_gtp(mod, ["weight"], cached_group(range(world_size)))

    shard = mod.weight
    assert isinstance(shard, GTPShardedParam)
    assert shard.pad_length == pad_length
    rows_per_rank = (rows + pad_length) // world_size
    torch.testing.assert_close(
        shard.data, _padded_row_shard(full_weight, rows_per_rank, rank), rtol=0, atol=0
    )


class TestGTPSharding:
    # 128 rows: aligned. 63 rows: 16 * 4 alignment needs one padding row on the last rank.
    @pytest.mark.parametrize("rows,pad_length", [(128, 0), (63, 1)], ids=["aligned", "padded"])
    def test_shard_content(self, rows, pad_length):
        _requires_multi_gpu(4)
        _run_distributed(_worker_shard_content, 4, rows, pad_length)


class TestGTPGroupSizeOne:
    def test_no_sharding_when_gtp_remat_size_one(self):
        """wrap_module_params_gtp must be a no-op for a singleton GTP group."""
        mod = nn.Linear(32, 64, bias=False)
        original_weight = mod.weight
        wrap_module_params_gtp(mod, ["weight"], _FakeGroup())
        assert (
            mod.weight is original_weight
        ), "gtp_remat_group.size()==1 should leave parameters unchanged"
        assert not isinstance(mod.weight, GTPShardedParam)


class TestGTPRematPgCollectionWithoutParallelState:
    """Resolving the GTP shard group with GTP off must return None, not assert.

    TE linear ``__init__`` calls ``use_mpu_process_groups(["gtp_remat", "expt_gtp_remat"])``; both
    getters use ``check_initialized=False``, so an uninitialized GTP axis must yield None groups
    rather than break construction of every non-GTP module.
    """

    def test_gtp_remat_pgs_are_none_and_do_not_raise(self, mocker):
        # Force the uninitialized GTP state deterministically (independent of suite ordering).
        mocker.patch.object(parallel_state, "_GTP_WEIGHT_REMAT_GROUP", None)
        mocker.patch.object(parallel_state, "_EXPERT_GTP_WEIGHT_REMAT_GROUP", None)

        pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["gtp_remat", "expt_gtp_remat"]
        )

        assert pg_collection.gtp_remat is None
        assert pg_collection.expt_gtp_remat is None


# ---------------------------------------------------------------------------
# Linear chain: output, dX and wgrad against a dense reference
# ---------------------------------------------------------------------------


def _worker_linear_chain_matches_dense(rank, world_size, port, out_f, pad_length, fuse, prefetch):
    """Two chained GTP linears match a dense FP32 reference for two iterations.

    Backward walks the chain in reverse: the tail (l1) reduce-scatters asynchronously and the head
    (l0) reduce-scatters synchronously, then finalizes l1's deferred result (the cascade). Each
    rank feeds its own input, so the expected wgrad is the mean of the per-rank wgrads (the RS
    pre-scales by 1/gtp) and main_grad holds this rank's rows of it, zero in the padding. The
    second iteration runs on prefetched weights.
    """
    gtp_module.update_gtp_config(weight_prefetch=prefetch)
    group = cached_group(range(world_size))
    batch, in_f, hidden = 8, 64, 128
    layers = [
        _make_gtp_linear(in_f, hidden, group, fuse_wgrad_accumulation=fuse),
        _make_gtp_linear(hidden, out_f, group, fuse_wgrad_accumulation=fuse),
    ]
    torch.manual_seed(0)  # Same full weights on every rank.
    full = [
        torch.randn(hidden, in_f, device="cuda") / in_f**0.5,
        torch.randn(out_f, hidden, device="cuda") / hidden**0.5,
    ]
    expected_rows = [hidden // world_size, (out_f + pad_length) // world_size]
    for layer, weight, rows in zip(layers, full, expected_rows):
        assert layer.weight.shape == (rows, weight.shape[1])
        with torch.no_grad():
            layer.weight.copy_(_padded_row_shard(weight, rows, rank))
        layer.weight.main_grad = torch.zeros(layer.weight.shape, device="cuda")
    # The reference uses the weights exactly as GTP stores them.
    full = [w.bfloat16().float() for w in full]

    for iteration in range(2):
        for layer in layers:
            layer.weight.main_grad.zero_()
        torch.manual_seed(1 + iteration * world_size + rank)
        x = torch.randn(batch, in_f, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        dy = torch.randn(batch, out_f, dtype=torch.bfloat16, device="cuda")
        first = iteration == 0
        y = layers[1](layers[0](x, is_first_microbatch=first), is_first_microbatch=first)
        y.backward(dy)
        assert layers[1].weight.prev_w is layers[0].weight, "the layers must form one chain"
        torch.cuda.synchronize()
        after_backward = [layer.weight.main_grad.clone() for layer in layers]
        gtp_module.wait_for_gtp_grad_reduction_on_current_stream()
        torch.cuda.synchronize()

        x_all, dy_all = _all_gather_rows(x.detach()).float(), _all_gather_rows(dy).float()
        h_all = x_all @ full[0].T
        dh_all = dy_all @ full[1]
        expected_wgrads = [dh_all.T @ x_all / world_size, dy_all.T @ h_all / world_size]
        tol = dict(rtol=2e-2, atol=3e-2)
        torch.testing.assert_close(y.float(), x.detach().float() @ full[0].T @ full[1].T, **tol)
        torch.testing.assert_close(x.grad.float(), dy.float() @ full[1] @ full[0], **tol)
        for i, (layer, wgrad, rows) in enumerate(zip(layers, expected_wgrads, expected_rows)):
            expected = _padded_row_shard(wgrad, rows, rank)
            # Before the fence: only the cascade can have finalized l1.
            torch.testing.assert_close(after_backward[i], expected, **tol)
            # The fence must not add a gradient twice.
            torch.testing.assert_close(layer.weight.main_grad, expected, **tol)


class TestLinearGTP:
    @pytest.mark.parametrize(
        "out_f,pad_length,fuse,prefetch",
        [(128, 0, False, True), (120, 8, True, True), (128, 0, False, False)],
        ids=["aligned", "padded-fused-wgrad", "prefetch-disabled"],
    )
    def test_chain_matches_dense_reference(self, out_f, pad_length, fuse, prefetch):
        _requires_multi_gpu(4)
        _run_distributed(_worker_linear_chain_matches_dense, 4, out_f, pad_length, fuse, prefetch)


# ---------------------------------------------------------------------------
# Megatron linear: the wgrad handed to the reduce-scatter carries main_grad's dtype
# ---------------------------------------------------------------------------


def _worker_wgrad_rs_dtype(rank, world_size, port):
    """With an FP32 main_grad, the gtp_remat reduce-scatter must run in FP32, not the compute dtype.

    Guards a SILENT failure: a bf16 wgrad makes the RS round at every rank, which leaves gradients
    finite and merely less precise, so the value-based tests here pass either way.

    Covers Megatron's linear (layers.py); TE types its wgrad from main_grad via grad_buffer().
    """
    from megatron.core.tensor_parallel.layers import (
        linear_with_grad_accumulation_and_async_allreduce,
    )

    torch.manual_seed(0)
    batch, in_f, out_f = 16, 64, 128  # out_f % (16*world_size)==0 -> no padding
    layer = nn.Linear(in_f, out_f, bias=False, dtype=torch.bfloat16, device="cuda")
    wrap_module_params_gtp(layer, ["weight"], cached_group(range(world_size)))
    layer.weight.main_grad = torch.zeros(layer.weight.shape, dtype=torch.float32, device="cuda")

    seen = []
    captured = []
    original = GTPShardedParam.wgrad_reduce_scatter

    def recording_rs(self, wgrad, nvtx_label=None):
        seen.append(wgrad.dtype)
        captured.append(wgrad.detach().float().clone())
        return original(self, wgrad, nvtx_label=nvtx_label)

    try:
        GTPShardedParam.wgrad_reduce_scatter = recording_rs
        inp = torch.randn(batch, in_f, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        linear_with_grad_accumulation_and_async_allreduce(
            input=inp,
            weight=layer.weight,
            bias=None,
            gradient_accumulation_fusion=True,
            allreduce_dgrad=False,
            sequence_parallel=False,
            gtp_remat_size=world_size,
        ).sum().backward()
    finally:
        GTPShardedParam.wgrad_reduce_scatter = original

    # Every rank asserts: the tally is rank-local.
    assert seen, "wgrad_reduce_scatter was never reached -- the weight is not GTP-sharded"
    assert set(seen) == {torch.float32}, (
        f"gtp_remat reduce-scatter ran in {set(seen)} with an FP32 main_grad. The wgrad must carry "
        f"main_grad's dtype, or the cross-rank sum rounds before main_grad ever sees it."
    )
    # The widened branch is the only caller of _wgrad_gemm here, so pin the VALUE too: a wrong
    # layout/operand order would still be FP32 and still reduce-scatter cleanly. grad_output is
    # all ones (loss = .sum()), so every wgrad row is just the input column sum -- computed in
    # FP32 without a GEMM, the reference carries no BF16 rounding and the tolerance stays tight.
    expected = inp.detach().float().sum(dim=0).expand(out_f, in_f)
    torch.testing.assert_close(captured[0], expected, rtol=1e-5, atol=1e-4)


class TestGTPWgradRSDtype:
    def test_wgrad_rs_follows_main_grad_dtype(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_wgrad_rs_dtype, 4)


# ---------------------------------------------------------------------------
# GroupedLinear forward/backward smoke tests
# ---------------------------------------------------------------------------


def _worker_grouped_linear(rank, world_size, port):
    torch.manual_seed(0)
    num_gemms, in_f, out_f, total_tokens = 2, 32, 64, 8
    dtype = torch.bfloat16
    gtp_remat_group = cached_group(range(world_size))

    layer = _make_gtp_remat_grouped_linear(num_gemms, in_f, out_f, gtp_remat_group, dtype)
    w0 = layer.weight0
    assert isinstance(w0, GTPShardedParam)
    assert [w.expert_idx for w in w0.weight_list] == list(range(num_gemms))

    m_splits = [total_tokens // num_gemms] * num_gemms
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
    def test_forward_backward(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_grouped_linear, 4)


def _worker_ops_grouped_linear(rank, world_size, port):
    """GTP on the fusible-op ``te.ops.GroupedLinear`` -- the unfused fallback path (run standalone,
    so no op fusion). Exercises materialize (fwd/bwd all-gather) + wgrad reduce-scatter wiring in
    transformer_engine/pytorch/ops/basic/grouped_linear.py."""
    torch.manual_seed(0)
    num_gemms, in_f, out_f, total_tokens = 2, 32, 64, 8
    dtype = torch.bfloat16
    gtp_remat_group = cached_group(range(world_size))

    op = te.ops.GroupedLinear(num_gemms, in_f, out_f, bias=False, device="cuda", dtype=dtype)
    op.gtp_remat_size = gtp_remat_group.size()
    wrap_module_params_gtp(
        op, [f"weight{i}" for i in range(num_gemms)], gtp_remat_group, is_grouped=True
    )
    assert isinstance(op.weight0, GTPShardedParam)

    m_splits = [total_tokens // num_gemms] * num_gemms
    split_sizes = torch.tensor(m_splits, dtype=torch.int64, device="cuda")

    inp = torch.randn(total_tokens, in_f, dtype=dtype, device="cuda", requires_grad=True)
    dist.broadcast(inp, src=0)

    out = op(inp, split_sizes)
    assert out.shape == (total_tokens, out_f), f"unexpected output shape {out.shape}"

    for i in range(num_gemms):
        w = getattr(op, f"weight{i}")
        w.main_grad = torch.zeros(w.shape, dtype=dtype, device="cuda")
        # DDP initializes this on every param; the backward wgrad-fusion path sets it True and
        # returns a throwaway dummy .grad (real grad is reduce-scattered into main_grad).
        w.grad_added_to_main_grad = False
    out.sum().backward()
    assert inp.grad is not None and inp.grad.shape == inp.shape
    # The wgrad reduce-scatter wrote each per-expert shard's gradient into main_grad (the gradient
    # of record for GTP), and flagged it so DDP won't double-add the dummy .grad.
    for i in range(num_gemms):
        w = getattr(op, f"weight{i}")
        assert w.grad_added_to_main_grad is True
        assert torch.count_nonzero(w.main_grad) > 0, f"weight{i} main_grad not populated by RS"


class TestOpsGroupedLinearGTP:
    """GTP on the fusible-op ``te.ops.GroupedLinear`` (the unfused fallback for grouped-MLP)."""

    def test_forward_backward(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_ops_grouped_linear, 4)


# ---------------------------------------------------------------------------
# Chain classification and gather-buffer keys
# ---------------------------------------------------------------------------


class TestLatentProjectionChainClassification:
    """Classify opted-in latent projections through the public GTP chain API."""

    FC1 = "decoder.layers.3.mlp.fc1_latent_proj.weight"
    FC2 = "decoder.layers.3.mlp.fc2_latent_proj.weight"

    @pytest.mark.parametrize(
        "impl,scope,graphed",
        [
            ("none", None, False),
            ("local", {"moe_router"}, True),
            ("local", {"mamba", "attn"}, False),
        ],
        ids=["eager", "router-scope", "unrelated-scope"],
    )
    def test_latent_projections_follow_router_scope(self, impl, scope, graphed):
        params = tuple(GTPShardedParam(torch.zeros(1, device="cuda")) for _ in range(2))

        class _Model:
            def named_parameters(_self):
                return iter(zip((self.FC1, self.FC2), params))

        gtp_module.classify_gtp_remat_chains(
            _Model(), cuda_graph_modules=scope, cuda_graph_impl=impl
        )
        assert params[0].chain_id == params[1].chain_id
        assert [gtp_module._chain_is_graphed(p.chain_id) for p in params] == [graphed, graphed]


class TestGroupedDoubleBuffer:
    """One-block-ahead grouped chains must double-buffer: consecutive MoE layers get distinct
    gather buffers (else prefetching layer N+1 clobbers layer N's in-use weight). Pure cache-key
    logic, no GPU/distributed needed."""

    class _Fake:
        _unsharded_shape_padded = (128, 256)
        expert_idx = 0

        def __init__(self, chain_id):
            self.chain_id = chain_id
            self.group = None

        _double_buffer_parity = gtp_module.GTPShardedParam._double_buffer_parity
        _get_cache_key = gtp_module.GTPShardedParam._get_cache_key

    def _key(self, chain_id):
        return self._Fake(chain_id)._get_cache_key(torch.bfloat16, fwd=True, reduce_scatter=False)

    def test_consecutive_layers_use_two_alternating_buffers(self):
        keys = [self._key("GTP_remat_grouped_fc1_ungraphed") for _ in range(4)]
        # Consecutive layers differ (no clobber); alternating layers share; exactly two buffers.
        assert keys[0] != keys[1]
        assert keys[1] != keys[2]
        assert keys[0] == keys[2]
        assert keys[1] == keys[3]
        assert len(set(keys)) == 2

    def test_fc1_fc2_never_share_a_buffer(self):
        # Both can be in-flight at once on the shared IB stream; role folded into key keeps
        # them distinct even when gathered shapes match (as in this fake).
        assert self._key("GTP_remat_grouped_fc1_ungraphed") != self._key(
            "GTP_remat_grouped_fc2_ungraphed"
        )


def _worker_same_key_neighbours_dont_share_a_buffer(rank, world_size, port):
    """Two same-shaped weights adjacent in a plain (non-grouped) chain need two buffers.

    This is the GTP_ungraphed chain under CUDA graphs: every layer weight is captured, leaving
    only embedding + output_layer behind — same shape, same dtype, same cache key. One-step-ahead
    prefetch keeps both live (w0's consume issues w1's gather), so one buffer means w1's gather
    lands on the weight w0's GEMM is still reading.
    """
    torch.manual_seed(0)
    in_f, out_f = 32, 64
    dtype = torch.bfloat16
    gtp_remat_group = cached_group(range(world_size))

    l0 = _make_gtp_linear(in_f, out_f, gtp_remat_group, dtype)
    l1 = _make_gtp_linear(in_f, out_f, gtp_remat_group, dtype)
    # Distinct values so a clobber shows up in the gathered data, not just in the address.
    with torch.no_grad():
        l0.weight.fill_(1.0)
        l1.weight.fill_(-1.0)

    inp = torch.randn(4, in_f, dtype=dtype, device="cuda")
    dist.broadcast(inp, src=0)

    # First pass wires the chain; the second is the one that prefetches.
    for _ in range(2):
        l0(inp, is_first_microbatch=True)
        l1(inp, is_first_microbatch=True)

    w0, w1 = l0.weight, l1.weight
    assert w0.next_w is w1 and w1.prev_w is w0, "w0 and w1 must be adjacent in one chain"
    assert w0._gather_buffer_identity(dtype) == w1._gather_buffer_identity(
        dtype
    ), "nothing is being tested unless both weights want the same buffer"

    cache = gtp_module.get_global_GTP_cache()
    b0, b1 = cache.get(w0._ag_ticket_fwd), cache.get(w1._ag_ticket_fwd)
    torch.cuda.synchronize()
    assert b0.data_ptr() != b1.data_ptr(), (
        "same-key chain neighbours share one gather buffer — w1's prefetch can clobber "
        "the weight w0's GEMM is still reading"
    )
    assert (b0 == 1).all(), "w0's gathered weight was clobbered by w1's prefetch"
    assert (b1 == -1).all(), "w1's buffer does not hold w1's weight"


class TestSameKeyNeighbourBuffers:
    """A non-grouped chain buys a second buffer where two adjacent weights would share one."""

    def test_adjacent_same_shape_weights_get_distinct_buffers(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_same_key_neighbours_dont_share_a_buffer, 4)


# ---------------------------------------------------------------------------
# MXFP8 + GTP_remat: native quantized shard, forward/backward against a dense reference
# ---------------------------------------------------------------------------


def _make_native_fp8_gtp_linear(in_f, out_f, gtp_remat_group, dtype, recipe):
    """Build a native-FP8 GTP te.Linear the gtp-agnostic way.

    Mirrors megatron/core/extensions/transformer_engine.py: pass the pre-sharded
    out_features to a STOCK te.Linear under fp8_model_init (TE inits+quantizes a native
    MXFP8 shard with no GTP awareness), then attach the GTP wiring post-init.
    """
    from transformer_engine.pytorch import fp8_model_init

    from megatron.core.tensor_parallel.gtp_api import (
        attach_gtp_to_presharded_module,
        gtp_remat_shard_dim0,
    )

    shard_out, pad_length = gtp_remat_shard_dim0(out_f, gtp_remat_group)
    with fp8_model_init(enabled=True, recipe=recipe):
        layer = te.Linear(
            in_features=in_f, out_features=shard_out, bias=False, params_dtype=dtype, device="cuda"
        )
    layer.gtp_remat_size = gtp_remat_group.size()
    attach_gtp_to_presharded_module(layer, gtp_remat_group, pad_length)
    return layer


def _assert_close_rel(actual, expected, label, max_rel=0.1):
    """MXFP8 quantizes both GEMM operands (a few % error); a wrong or misaligned row is O(1)."""
    rel = (actual.float() - expected).norm() / expected.norm()
    assert rel < max_rel, f"{label}: relative error {rel:.3f} >= {max_rel}"


def _worker_mxfp8_linear(rank, world_size, port, out_f, pad_length):
    """Native-MXFP8 GTP Linear: all-gather (+ padding strip), GEMM and backward.

    mxfp8 always implies --fp8-param-gather: the weight is built as a native FP8 shard at
    construction (no BF16 source, no per-forward cast). MXFP8 needs every shard dim divisible by
    32: out_f=120 pads to 128, so the last rank holds 24 real rows plus 8 padding rows.
    """
    from transformer_engine.common.recipe import MXFP8BlockScaling

    torch.manual_seed(0)
    # batch=32: MXFP8 wgrad GEMM (K=batch) requires K divisible by MXFP8_BLOCK_SCALING_SIZE=32
    batch, in_f = 32, 64
    dtype = torch.bfloat16
    recipe = MXFP8BlockScaling()
    layer = _make_native_fp8_gtp_linear(in_f, out_f, cached_group(range(world_size)), dtype, recipe)

    # The weight IS the native FP8 shard: a QuantizedTensor with the GTP surface attached.
    w = layer.weight
    assert isinstance(w, QuantizedTensor), f"weight should be QuantizedTensor, got {type(w)}"
    assert w.quantized is w, "native-FP8 GTP: self.quantized must be the param itself"
    assert getattr(w, "is_gtp_weight_remat", False), "GTP surface missing on native param"
    assert w.pad_length == pad_length
    assert w.shape[0] * world_size == out_f + pad_length, "weight must be dim-0 sharded"
    full_weight = _all_gather_rows(w.dequantize(dtype=dtype))[:out_f].float()

    inp = torch.randn(batch, in_f, dtype=dtype, device="cuda", requires_grad=True)
    dist.broadcast(inp, src=0)
    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        out = layer(inp, is_first_microbatch=True)
    _assert_close_rel(out, inp.detach().float() @ full_weight.T, "output")

    # The second microbatch reuses the same native FP8 weight.
    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        out2 = layer(inp.detach(), is_first_microbatch=False)
    torch.testing.assert_close(out2, out.detach())

    if out_f % 32:
        return  # MXFP8 cannot quantize a grad_output whose width is not a multiple of 32.
    layer.weight.main_grad = torch.zeros(layer.weight.shape, dtype=dtype, device="cuda")
    out.sum().backward()
    _assert_close_rel(inp.grad, full_weight.sum(dim=0).expand(batch, in_f), "dX")


@pytest.mark.launch_on_gb200
class TestMXFP8LinearGTP:
    @pytest.mark.parametrize("out_f,pad_length", [(128, 0), (120, 8)], ids=["aligned", "padded"])
    def test_forward_backward(self, out_f, pad_length):
        _requires_mxfp8()
        _requires_multi_gpu(4)
        _run_distributed(_worker_mxfp8_linear, 4, out_f, pad_length)


# ---------------------------------------------------------------------------
# wait_async_comms(finalize_after_drain=True) inline-accumulation fallback
# ---------------------------------------------------------------------------


class TestWaitAsyncCommsFallback:
    """Exercises the inline-accumulation fallback inside
    ``wait_async_comms(finalize_after_drain=True)``: when a param is in
    ``_inflight_comm_params`` (async AG was issued) but its ``_wgrad_rs_handle``
    is ``None`` (no async RS handle to drain), the inner
    ``_wait_reduce_scatter`` call no-ops and the outer loop must inline the
    accumulation itself (main_grad.add_ + ticket release + flag set) -- unless the
    param was already finalized, or no reduce-scatter ever ran for it (a pure-AG
    cross-graph prefetch: its ``_rs_ticket`` is None and the fallback must not
    read the cache).

    Production flows rarely hit this combination, so the state is built by hand.
    """

    @pytest.mark.parametrize(
        "already_finalized,has_rs_result",
        [(False, True), (True, True), (False, False)],
        ids=["accumulates", "already-finalized", "pure-ag"],
    )
    def test_fallback(self, already_finalized, has_rs_result):
        dtype = torch.bfloat16
        p = GTPShardedParam(torch.zeros(8, 4, dtype=dtype, device="cuda"))
        p.group = _FakeGroup()
        p.expert_idx = None
        p.pad_length = 0
        p.chain_id = gtp_module.GTPChain.UNGRAPHED.value
        p._quantizer = None
        p.is_routed_expert = False  # ⇒ self._weights property returns [self]
        p.main_grad = torch.full((8, 4), 5.0, dtype=dtype, device="cuda")
        p._prefetch_handle = None  # _wait_param_gather is no-op
        p._wgrad_rs_handle = None  # _wait_reduce_scatter is no-op → fallback decides
        p._cached_rs_stream = None
        p.ag_event = torch.cuda.Event(external=True)
        p.rs_event = torch.cuda.Event(external=True)
        p.rs_event.record()  # so rs_event.wait() in fallback doesn't block
        p._already_finalized = already_finalized
        p.grad_added_to_main_grad = False
        p._rs_ticket = None
        if has_rs_result:
            cache = gtp_module.get_global_GTP_cache()
            p._rs_ticket = cache.reserve(p, dtype, fwd=False, reduce_scatter=True)
            cache.get(p._rs_ticket).fill_(2.0)
        gtp_module._inflight_comm_params.add(p)

        gtp_module.wait_async_comms(chain_id=p.chain_id, skip_rs=False, finalize_after_drain=True)

        torch.cuda.synchronize()
        accumulates = has_rs_result and not already_finalized
        torch.testing.assert_close(
            p.main_grad, torch.full_like(p.main_grad, 7.0 if accumulates else 5.0)
        )
        assert p._already_finalized is (already_finalized or accumulates)
        assert p.grad_added_to_main_grad is accumulates


# ---------------------------------------------------------------------------
# GTP_remat DDP integration: bucket alignment and grad-ready wiring
# ---------------------------------------------------------------------------


def _worker_gtp_ddp_bucket_alignment(rank, world_size, port):
    """With GTP params, both DDP buffers must pad bucket ends for the distributed optimizer.

    The distributed optimizer requires every bucket end to be divisible by the buffer's DP size
    (asserted in param_and_grad_buffer.py). Two past bugs left a buffer on the unpadded default
    layout:
      - GTP buffer: GTP_remat=2, DP=4 -> its DP size is 2. With pad_for_alignment=0, a [2, 3]
        weight gives a 3-element shard; bucket_size=2 closes a bucket after each, ending at 3.
      - Regular buffer (DP size 4): each 2-element bias fills a bucket, ending at 2.
    """
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.transformer.transformer_config import TransformerConfig

    gtp_module.GTP_CONFIG.pad_for_alignment = 0
    with _model_parallel_with_gtp(2) as gtp_remat_group:
        model = nn.Sequential(
            te.Linear(3, 2, bias=True, device="cuda"), te.Linear(3, 2, bias=True, device="cuda")
        )
        for layer in model:
            wrap_module_params_gtp(layer, ["weight"], gtp_remat_group)
        config = TransformerConfig(
            num_attention_heads=1, num_layers=1, hidden_size=4, tensor_model_parallel_size=1
        )
        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=True, overlap_grad_reduce=True, bucket_size=2
        )
        # Raises AssertionError in param_and_grad_buffer.py if a bucket end is unpadded.
        DistributedDataParallel(config, ddp_config, model)


class TestGTPDDPBucketAlignment:
    def test_gtp_and_regular_buffers_use_padded_layout(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_gtp_ddp_bucket_alignment, 4)


def _worker_gtp_ddp_grad_ready_wiring(rank, world_size, port):
    """GTP params must drive DDP grad-ready from GTP's manual hook, not autograd.

    GTP defers the main_grad accumulation to a later backward node, so autograd's AccumulateGrad can
    fire register_grad_ready before the grad lands and dispatch the bucket reduce-scatter on stale
    grad_data (corrupts reduce_scatter_with_fp32_accumulation). The fix routes grad-ready through
    register_grad_accum_hook (fired after the add) and skips the autograd hook. This pins that
    wiring: every GTP weight has _grad_accum_hook set and none falls through to the autograd list.

    It also checks that the AccumulateGrad node is materialized and stored on the param. Holding
    that reference is what keeps the leaf on the capture stream for full-iteration CUDA-graph
    capture.
    """
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.transformer.transformer_config import TransformerConfig

    with _model_parallel_with_gtp(2) as gtp_remat_group:
        # bias=False -> all params are GTP_remat weights, so grad_accs must end up empty.
        model = nn.Sequential(
            te.Linear(64, 128, bias=False, device="cuda"),
            te.Linear(64, 128, bias=False, device="cuda"),
        )
        for layer in model:
            wrap_module_params_gtp(layer, ["weight"], gtp_remat_group)

        config = TransformerConfig(
            num_attention_heads=1, num_layers=1, hidden_size=4, tensor_model_parallel_size=1
        )
        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=True, overlap_grad_reduce=True
        )
        ddp_model = DistributedDataParallel(config, ddp_config, model)

        for i, layer in enumerate(model):
            w = layer.weight
            assert isinstance(w, GTPShardedParam), f"layer {i} weight should be a GTP param"
            # Manual hook set -> grad-ready fires after the add; None -> early autograd path (bug).
            assert (
                getattr(w, "_grad_accum_hook", None) is not None
            ), f"layer {i} weight must have _grad_accum_hook set (manual grad-ready, not autograd)"
            # The node must also be RETAINED: a live strong reference across the
            # warmup->capture boundary is what keeps the leaf on the capture stream.
            # Identity (not just non-None): a dropped node would be recreated by expand_as here.
            assert (
                getattr(w, "_grad_accum_node", None) is w.expand_as(w).grad_fn.next_functions[0][0]
            ), f"layer {i} weight must retain its AccumulateGrad node (full-iteration CG capture)"

        assert len(ddp_model.grad_accs) == 0, (
            "GTP params must not register an autograd AccumulateGrad hook "
            f"(grad_accs has {len(ddp_model.grad_accs)} entries)"
        )


class TestGTPDDPGradReadyWiring:
    def test_gtp_params_use_manual_grad_ready_hook(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_gtp_ddp_grad_ready_wiring, 4)


# ---------------------------------------------------------------------------
# Weight readiness and persistent wgrad ring reuse
# ---------------------------------------------------------------------------


@pytest.fixture
def four_rank_gtp_group():
    """The buffer-reuse oracles below assume exactly four shards."""
    if dist.get_world_size() != 4:
        pytest.skip("buffer-reuse regressions require four ranks")
    return cached_group(range(4))


@pytest.mark.internal
class TestGTPBufferReuse:
    """Weight readiness and persistent wgrad reuse across eager and graph scheduling."""

    @pytest.mark.parametrize(
        "readiness,fwd",
        [("missing", True), ("in_flight", False), ("ready", True)],
        ids=["missing-forward", "in_flight-backward", "ready-forward"],
    )
    def test_peek_returns_ready_current_data(
        self, monkeypatch, four_rank_gtp_group, fwd, readiness
    ):
        """Every readiness path exposes current bytes before consume, with exactly one gather.

        Two same-key cache buffers prevent a key-only lookup from identifying the actual AG output.
        The first consume can replace its ticket; subsequent peeks must follow the actual gather.
        """
        cache = gtp_module.GTPWeightCache()
        monkeypatch.setattr(gtp_module, "_GTP_CACHE", cache)
        monkeypatch.setattr(gtp_module, "_GTP_GROUPED_BUF_PARITY_COUNTER", {})
        monkeypatch.setattr(gtp_module.GTP_CONFIG, "check_param_states", True)
        monkeypatch.setattr(gtp_module.GTP_CONFIG, "weight_prefetch", True)
        p = GTPShardedParam(torch.zeros(8, 16, dtype=torch.bfloat16, device="cuda"))
        p.group = four_rank_gtp_group
        p.chain_id = "GTP_remat_grouped_fc1_ungraphed"
        held = [cache.reserve(p, p.dtype, fwd=fwd) for _ in range(2)]
        buffers = [cache.get(ticket) for ticket in held]
        for ticket in held:
            cache.release(ticket)
        assert buffers[0].data_ptr() != buffers[1].data_ptr()

        calls = []
        gather = p._all_gather_weight

        def counted_gather(*args, **kwargs):
            calls.append(kwargs["fwd"])
            return gather(*args, **kwargs)

        monkeypatch.setattr(p, "_all_gather_weight", counted_gather)
        peek = p.peek_group_for_forward if fwd else p.peek_group_for_backward
        consume = p.materialize_group_for_forward if fwd else p.materialize_group_for_backward
        pointers = []
        for step in range(3):
            p.data.fill_(16 * step + dist.get_rank())
            expected = _all_gather_rows(p.data)
            if readiness != "missing":
                _, p._prefetch_handle = p._all_gather_weight(async_op=True, fwd=fwd)
                assert p._prefetch_handle is not None
                assert p.state == gtp_module.GTPWeightState.ASYNC_WAIT
            if readiness == "ready":
                # An external drain (or another peek) has finished the gather's host-side work.
                p._wait_param_gather()
                p._already_ag_drained = True
            initialized, previous, following = p.prefetch_initialized, p.prev_w, p.next_w
            peeked = peek()
            # Read immediately: a later consume must not supply the missing completion wait.
            torch.testing.assert_close(peeked, expected, rtol=0, atol=0)
            assert p._prefetch_handle is None and p._already_ag_drained
            assert p.state == gtp_module.GTPWeightState.DATA_READY
            assert (p.prefetch_initialized, p.prev_w, p.next_w) == (
                initialized,
                previous,
                following,
            )
            # A second peek on a different stream still observes completion and issues no gather.
            with torch.cuda.stream(torch.cuda.Stream()):
                again = peek()
                torch.testing.assert_close(again, expected, rtol=0, atol=0)
            assert again.data_ptr() == peeked.data_ptr()
            gathered = consume()
            assert gathered.data_ptr() == peeked.data_ptr()
            torch.testing.assert_close(gathered, expected, rtol=0, atol=0)
            assert len(calls) == step + 1
            assert p._prefetch_handle is None
            assert not getattr(p, "_already_ag_drained", False)
            pointers.append(peeked.data_ptr())
        assert (
            pointers[1] == pointers[2]
        ), "the established deterministic schedule must reuse storage"

    @pytest.mark.parametrize("eager_first", [True, False], ids=["eager-first", "graph-first"])
    def test_eager_wgrad_preserves_graph_rings(self, monkeypatch, four_rank_gtp_group, eager_first):
        """Eager allocation cannot suppress or rebuild graph rings, even after a reset."""
        monkeypatch.setattr(gtp_eager_wgrad, "_EAGER_WGRAD_RINGS", {})
        monkeypatch.setattr(gtp_cuda_graphs, "_GRAPH_WGRAD_RINGS", {})
        monkeypatch.setattr(gtp_module, "_FULL_ITERATION", False)
        monkeypatch.setattr(gtp_module.GTP_CONFIG, "async_reduction", True)
        monkeypatch.setattr(gtp_module.GTP_CONFIG, "graph_wgrad_ring_size", 2)
        monkeypatch.setattr(gtp_module.GTP_CONFIG, "eager_wgrad_ring_size", 4)
        weights = [gtp_module.GTPShardedParam(torch.zeros(3, 4, device="cuda")) for _ in range(5)]
        for weight in weights:
            weight.group = four_rank_gtp_group
            weight.chain_id = gtp_module.GTPChain.GRAPHED.value
            weight.pad_length = 2
            weight.main_grad = torch.zeros_like(weight)
        graphed, eager = weights[:4], weights[-1]
        eager.chain_id = gtp_module.GTPChain.UNGRAPHED.value
        # Frozen weights are registered too, but DDP does not create their main_grad.
        frozen = gtp_module.GTPShardedParam(torch.zeros_like(eager), requires_grad=False)
        frozen.group = eager.group
        frozen.chain_id = eager.chain_id
        for previous, current in zip(graphed, graphed[1:]):
            previous.next_w, current.prev_w = current, previous
        monkeypatch.setattr(gtp_module, "_GTP_PARAMS", [*weights, frozen])

        old_slots = []
        for _ in range(2):
            if eager_first:
                eager_view = eager.get_wgrad_tensor(persistent=True)
            gtp_module.initialize_graph_wgrad_rings()
            if not eager_first:
                eager_view = eager.get_wgrad_tensor(persistent=True)
            assert len(gtp_cuda_graphs._GRAPH_WGRAD_RINGS) == 1
            key, slots = next(iter(gtp_cuda_graphs._GRAPH_WGRAD_RINGS.items()))
            assert len(slots) == 2
            assert [slot.index for slot in slots] == [0, 1]
            assert all(slot.key == key for slot in slots)
            assert graphed[1]._gtp_graph_wgrad_ring_slot is slots[0]
            assert graphed[2]._gtp_graph_wgrad_ring_slot is slots[1]
            assert graphed[3]._gtp_graph_wgrad_ring_slot is slots[0]
            assert len(gtp_eager_wgrad._EAGER_WGRAD_RINGS) == 1
            assert not hasattr(frozen, "_gtp_eager_wgrad_ring_slot")
            assert eager_view.shape == (10, 4)
            eager_view.fill_(7)
            for slot in slots:
                slot.tensor.fill_(3)
            torch.testing.assert_close(eager_view, torch.full_like(eager_view, 7))
            assert len({slot.tensor.data_ptr() for slot in slots} | {eager_view.data_ptr()}) == 3
            assert all(slot is not old for slot in slots for old in old_slots)
            gtp_module.initialize_graph_wgrad_rings()
            assert next(iter(gtp_cuda_graphs._GRAPH_WGRAD_RINGS.values())) is slots
            assert eager.get_wgrad_tensor().data_ptr() == eager_view.data_ptr()
            old_slots = [*slots, eager._gtp_eager_wgrad_ring_slot]
            torch.cuda.synchronize()
            gtp_module.reset_gtp_state()
            assert not gtp_cuda_graphs._GRAPH_WGRAD_RINGS
            assert not gtp_eager_wgrad._EAGER_WGRAD_RINGS
            for weight in weights:
                assert not hasattr(weight, "_gtp_graph_wgrad_ring_slot")
                assert not hasattr(weight, "_gtp_eager_wgrad_ring_slot")

    @pytest.mark.parametrize(
        "async_reduction,ring_size",
        [(False, 2), (True, 1), (True, 8)],
        ids=["sync", "async-single-slot", "async-excess-capacity"],
    )
    def test_persistent_wgrad_reuse_preserves_every_reduction(
        self, monkeypatch, four_rank_gtp_group, async_reduction, ring_size
    ):
        """Full gradients, padding and exactly-once completion survive shared/repeated writers."""
        group = four_rank_gtp_group
        rank = dist.get_rank(group)
        monkeypatch.setattr(gtp_eager_wgrad, "_EAGER_WGRAD_RINGS", {})
        monkeypatch.setattr(gtp_module, "_GTP_GROUPED_BUF_PARITY_COUNTER", {})
        monkeypatch.setattr(gtp_module.GTP_CONFIG, "async_reduction", async_reduction)
        monkeypatch.setattr(gtp_module.GTP_CONFIG, "reduce_scatter_with_fp32_accumulation", False)
        monkeypatch.setattr(gtp_module.GTP_CONFIG, "calculate_per_token_loss", False)
        # Both FC roles deliberately have the same shape; each has two independent experts.
        layers = []
        expected, completions = {}, {}
        for layer in range(5):
            roles = []
            for role in ("fc1", "fc2"):
                weights = [
                    gtp_module.GTPShardedParam(
                        torch.zeros(33, 16, dtype=torch.bfloat16, device="cuda")
                    )
                    for _ in range(2)
                ]
                for expert, weight in enumerate(weights):
                    weight.group, weight.pad_length, weight.expert_idx = group, 2, expert
                    weight.chain_id = f"GTP_remat_grouped_{role}_ungraphed"
                    weight.is_routed_expert = True
                    weight._debug_name = f"layers.{layer}.{role}.weight{expert}"
                    weight.main_grad = torch.full_like(weight, 0.5)
                    weight._double_buffer_parity()  # Normally assigned by the forward gathers.
                    expected[id(weight)] = weight.main_grad.clone()
                    completions[id(weight)] = 0

                    def completed(weight=weight):
                        completions[id(weight)] += 1

                    weight.register_grad_accum_hook(None, completed)
                weights[0].weight_list = weights
                roles.append(weights)
            layers.append(roles)
        for previous, current in zip(layers, layers[1:]):
            for prev_weights, weights in zip(previous, current):
                prev_weights[0].next_w, weights[0].prev_w = weights[0], prev_weights[0]
        monkeypatch.setattr(
            gtp_module,
            "_GTP_PARAMS",
            [weight for layer in layers for weights in layer for weight in weights],
        )

        if not async_reduction:
            # Validation does not depend on the schedule; check it once. Invalid depths must fail
            # before binding storage, and the same writers then recover.
            for invalid_size in (0, -1):
                monkeypatch.setattr(gtp_module.GTP_CONFIG, "eager_wgrad_ring_size", invalid_size)
                with pytest.raises(ValueError, match="eager_wgrad_ring_size must be at least 1"):
                    layers[0][0][0].get_wgrad_tensor(persistent=True)
        monkeypatch.setattr(gtp_module.GTP_CONFIG, "eager_wgrad_ring_size", ring_size)

        pointers = {}
        try:
            # Repeating the tail before the cascade drains its first RS exercises early reuse.
            for step in range(3):
                for layer_index in (4, 4, 3, 2, 1, 0):
                    for role_index in (1, 0):
                        weights = layers[layer_index][role_index]
                        if async_reduction:
                            with torch.cuda.stream(
                                gtp_module.get_rs_stream(weights[0].chain_id, group)
                            ):
                                torch.cuda._sleep(1_000_000)
                        grads = []
                        for weight in weights:
                            if step == 1:
                                # Skip acquisition so input preparation must finalize the old
                                # reader before copying a foreign gradient into the shared slot.
                                grad = torch.empty(
                                    weight._unsharded_shape,
                                    dtype=weight.main_grad.dtype,
                                    device=weight.device,
                                )
                            else:
                                grad = weight.get_wgrad_tensor(persistent=True)
                                prior = pointers.setdefault(id(weight), grad.data_ptr())
                                assert grad.data_ptr() == prior
                            base = (
                                torch.arange(130 * 16, device="cuda").view(130, 16) % 11 - 5
                            ).float() / 4 + (step + layer_index + weight.expert_idx) / 8
                            grad.copy_(base + rank / 4)
                            mean = torch.nn.functional.pad(base + 3 / 8, (0, 0, 0, 2))
                            expected[id(weight)].add_(mean.chunk(4)[rank])
                            grads.append(grad)
                        weights[0].finalize_group_grads(grads)
                torch.cuda.synchronize()
                for layer in layers:
                    for weights in layer:
                        for weight in weights:
                            torch.testing.assert_close(
                                weight.main_grad, expected[id(weight)], rtol=0, atol=0
                            )
                            assert not torch.count_nonzero(
                                weight._gtp_eager_wgrad_ring_slot.tensor[130:]
                            )
                            calls_per_step = 2 if layer is layers[-1] else 1
                            assert completions[id(weight)] == (step + 1) * calls_per_step
                # Allocate only the configured depth (or the number of writers if smaller).
                expected_slots = min(ring_size, len(layers)) * 2 * 2
                assert len(set(pointers.values())) == expected_slots
        finally:
            # A failed assertion may leave reduce-scatters in flight; finish them before teardown.
            torch.cuda.synchronize()
            for layer in layers:
                for weights in layer:
                    weights[0]._wait_reduce_scatter(finalize_grad=True)


class TestGTPGraphWgradRing:
    @staticmethod
    def _make_padded_chain(count=4):
        group = _FakeGroup(size=2)
        weights = [GTPShardedParam(torch.randn(3, 4, device="cuda")) for _ in range(count)]
        for weight in weights:
            weight.group = group
            weight.chain_id = GTPChain.GRAPHED.value
            weight.pad_length = 2
            weight.main_grad = torch.empty_like(weight)
        for previous, current in zip(weights, weights[1:]):
            previous.next_w = current
            current.prev_w = previous
        return weights

    @staticmethod
    def _configure(monkeypatch):
        monkeypatch.setattr(gtp_module, "_FULL_ITERATION", False)
        monkeypatch.setattr(gtp_module.GTP_CONFIG, "async_reduction", True)
        monkeypatch.setattr(gtp_module.GTP_CONFIG, "graph_wgrad_ring_size", 2)
        monkeypatch.setattr(gtp_cuda_graphs, "_GRAPH_WGRAD_RINGS", {})

    def test_partial_cg_wgrad_ring_ownership(self, monkeypatch):
        self._configure(monkeypatch)
        weights = self._make_padded_chain()
        monkeypatch.setattr(gtp_module, "_GTP_PARAMS", weights)
        gtp_module.initialize_graph_wgrad_rings()

        slot_1 = weights[1]._gtp_graph_wgrad_ring_slot
        slot_2 = weights[2]._gtp_graph_wgrad_ring_slot
        slot_3 = weights[3]._gtp_graph_wgrad_ring_slot
        assert slot_1 is not slot_2
        assert slot_1 is slot_3
        assert len(gtp_cuda_graphs._GRAPH_WGRAD_RINGS) == 1
        assert slot_1.ready_event.query()

        capture_state = gtp_cuda_graphs.GTPCaptureCommState()
        monkeypatch.setattr(gtp_cuda_graphs, "_ACTIVE_CAPTURE_COMM_STATE", capture_state)
        logical_view = weights[1].get_wgrad_tensor()
        assert not capture_state.wgrad_ring_slots
        assert slot_1.tensor.shape == (6, 4)
        assert logical_view.shape == (4, 4)
        assert logical_view.data_ptr() == slot_1.tensor.data_ptr()

        logical_view.fill_(7)
        send_bufs, release_bufs = weights[1]._prepare_wgrad_reduce_scatter_inputs([logical_view])
        assert capture_state.wgrad_ring_slots == [slot_1]
        assert send_bufs[0] is slot_1.tensor
        # The ring slot is sent but never released; the feeding wgrad is released instead.
        assert release_bufs[0] is logical_view
        assert torch.count_nonzero(slot_1.tensor[4:]) == 0

        with pytest.raises(RuntimeError, match="increase GTP_CONFIG.graph_wgrad_ring_size"):
            weights[3]._prepare_wgrad_reduce_scatter_inputs([logical_view])

    def test_native_fp8_style_param_uses_wgrad_ring(self, monkeypatch):
        """GTP-tagged native-FP8 params are not GTPShardedParam instances."""

        class NativeFP8StyleParam:
            def __init__(self, group):
                self.is_gtp_weight_remat = True
                self.group = group
                self.chain_id = GTPChain.GRAPHED.value
                self.pad_length = 2
                self.expert_idx = 0
                self.device = torch.device("cuda")
                self.main_grad = torch.empty(3, 4, device=self.device)
                self._unsharded_shape = (4, 4)
                self._unsharded_shape_padded = (6, 4)
                self.prev_w = None
                self.next_w = None
                self._weights = [self]

        self._configure(monkeypatch)
        group = _FakeGroup(size=2)
        first = NativeFP8StyleParam(group)
        second = NativeFP8StyleParam(group)
        first.next_w = second
        second.prev_w = first
        monkeypatch.setattr(gtp_module, "_GTP_PARAMS", [first, second])

        gtp_module.initialize_graph_wgrad_rings()

        assert second._gtp_graph_wgrad_ring_slot.tensor.shape == (6, 4)
        assert len(gtp_cuda_graphs._GRAPH_WGRAD_RINGS) == 1

    def test_rs_input_copies_into_ring(self, monkeypatch):
        self._configure(monkeypatch)
        weights = self._make_padded_chain(count=2)
        monkeypatch.setattr(gtp_module, "_GTP_PARAMS", weights)
        gtp_module.initialize_graph_wgrad_rings()

        wgrad = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
        send_bufs, release_bufs = weights[1]._prepare_wgrad_reduce_scatter_inputs([wgrad])
        rs_input = send_bufs[0]

        assert rs_input is weights[1]._gtp_graph_wgrad_ring_slot.tensor
        assert release_bufs[0] is wgrad
        torch.testing.assert_close(rs_input[:4], wgrad)
        assert torch.count_nonzero(rs_input[4:]) == 0


class TestActivationRecomputePhaseFlag:
    """GTP's forward-only readiness gate rests on TE's recompute flag being dtype-agnostic."""

    def test_recompute_phase_flag_is_set_under_bf16(self):
        """``in_activation_recompute_phase`` must be True for BF16 recompute, not just FP8.

        GTP aliases TE's ``in_fp8_activation_recompute_phase`` under a name without ``fp8``
        because TE assigns the underlying global unconditionally. ``_all_gather_weight`` uses it
        to skip publishing during recompute (see
        ``test_gtp_ddp_param_sync_race.py::test_recompute_forward_does_not_request_publication``).
        If TE ever made it FP8-only, BF16 recompute would start asking DDP to publish inside
        backward -- silently, since the failure is a wrong-buffer gather rather than an error.
        """
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

        assert not FP8GlobalStateManager.is_fp8_enabled(), "must run outside fp8_autocast"

        hidden, dtype = 128, torch.bfloat16
        observed = []

        class _Probe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = te.Linear(hidden, hidden, bias=False, params_dtype=dtype, device="cuda")

            def forward(self, x):
                observed.append(gtp_module.in_activation_recompute_phase())
                return self.lin(x)

        x = torch.randn(8, hidden, dtype=dtype, device="cuda", requires_grad=True)
        out = te.checkpoint(
            _Probe(),
            x,
            distribute_saved_activations=False,
            get_rng_state_tracker=None,
            tp_group=None,
        )
        out.sum().backward()  # replays the forward -> second observation

        assert observed == [False, True], (
            f"expected [original fwd=False, recompute fwd=True], got {observed}: "
            "in_activation_recompute_phase no longer tracks BF16 activation recompute"
        )


class TestGTPCaptureParamReadiness:
    def test_wgrad_finalization_registration_matches_hook_calls(self, monkeypatch):
        hook_calls = []

        class Param:
            main_grad = torch.empty(2, 3)
            dtype = torch.float32
            zero_out_wgrad = False

            def __init__(self):
                self.chain_id = GTPChain.GRAPHED.value
                self._grad_accum_hook = lambda: hook_calls.append(self)
                self.rs_states = []

            def _set_rs_state(self, state):
                self.rs_states.append(state)

        param = Param()
        dummy_wgrad = object()
        monkeypatch.setattr(gtp_module, "get_dummy_wgrad", lambda *args, **kwargs: dummy_wgrad)

        with gtp_cuda_graphs.track_gtp_capture_comms() as capture:
            assert GTPShardedParam._handle_megatron_grad_accum(param) is dummy_wgrad
            assert GTPShardedParam._handle_megatron_grad_accum(param) is dummy_wgrad
            param.chain_id = GTPChain.UNGRAPHED.value
            assert GTPShardedParam._handle_megatron_grad_accum(param) is dummy_wgrad
            param._grad_accum_hook = None
            assert GTPShardedParam._handle_megatron_grad_accum(param) is dummy_wgrad

        assert capture.finalized_params == [param, param]
        assert hook_calls == [param, param, param]
        assert param.rs_states == [gtp_module.GTPWeightState.NONE] * 4

    def test_warmup_preserves_cross_graph_prefetch_state(self):
        class FakeParam:
            is_gtp_weight_remat = True
            _prefetch_handle = None
            _recompute_prefetch_handle = None

            def __init__(self, already_drained, recompute_already_drained):
                self._already_ag_drained = already_drained
                self._recompute_already_drained = recompute_already_drained

        incoming = FakeParam(already_drained=True, recompute_already_drained=True)
        ordinary = FakeParam(already_drained=False, recompute_already_drained=False)

        observed = []
        for _ in range(2):
            with preserve_gtp_prefetch_state(iter((incoming, ordinary))):
                observed.append((incoming._already_ag_drained, incoming._recompute_already_drained))
                # Each warmup consumes the incoming handoff and creates outgoing readiness.
                incoming._already_ag_drained = False
                incoming._recompute_already_drained = False
                ordinary._already_ag_drained = True
                ordinary._recompute_already_drained = True

        assert observed == [(True, True), (True, True)]
        assert incoming._already_ag_drained is True
        assert incoming._recompute_already_drained is True
        assert ordinary._already_ag_drained is False
        assert ordinary._recompute_already_drained is False
        assert incoming._prefetch_handle is None

    @pytest.mark.parametrize("recompute", [False, True], ids=["cross-graph", "recompute"])
    def test_drained_prefetch_still_waits_on_ag_event(self, monkeypatch, recompute):
        """A drained hand-off skips the Work wait but must still order on the AG event."""

        class ExpectedEventWait(Exception):
            pass

        calls = []

        class FakeEvent:
            def wait(self):
                calls.append("event_wait")
                raise ExpectedEventWait

        if recompute:
            drained, event, handle_wait = (
                "_recompute_already_drained",
                "_recompute_ag_event",
                "_wait_recompute_param_gather",
            )
        else:
            drained, event, handle_wait = "_already_ag_drained", "ag_event", "_wait_param_gather"
        param = type(
            "Param",
            (),
            {
                drained: True,
                "_weights": (),
                event: FakeEvent(),
                handle_wait: lambda self: calls.append("handle_wait"),
            },
        )()
        monkeypatch.setattr(gtp_module.GTP_CONFIG, "check_param_states", False)

        with pytest.raises(ExpectedEventWait):
            if recompute:
                GTPShardedParam._get_recompute_prefetched_weight(param)
            else:
                GTPShardedParam._get_prefetched_weight(param, fwd=True)

        assert calls == ["event_wait"]
        assert getattr(param, drained) is False

    def test_warmup_exception_is_not_masked_by_leaked_handle(self):
        class WarmupError(Exception):
            pass

        param = type(
            "Param",
            (),
            {
                "is_gtp_weight_remat": True,
                "_already_ag_drained": True,
                "_recompute_already_drained": True,
                "_prefetch_handle": None,
                "_recompute_prefetch_handle": None,
            },
        )()

        with pytest.raises(WarmupError):
            with preserve_gtp_prefetch_state((param,)):
                param._already_ag_drained = False
                param._prefetch_handle = object()
                raise WarmupError

        assert param._already_ag_drained is True
        assert param._recompute_already_drained is True

    def test_successful_warmup_rejects_leaked_handle_after_restoring_state(self):
        param = type(
            "Param",
            (),
            {
                "is_gtp_weight_remat": True,
                "_already_ag_drained": True,
                "_recompute_already_drained": False,
                "_prefetch_handle": None,
                "_recompute_prefetch_handle": None,
            },
        )()

        with pytest.raises(RuntimeError, match="undrained AG work"):
            with preserve_gtp_prefetch_state((param,)):
                param._already_ag_drained = False
                param._prefetch_handle = object()

        assert param._already_ag_drained is True
        assert param._recompute_already_drained is False

    def test_capture_comm_selection_is_scoped_by_chain(self, monkeypatch):
        """Selection keeps chain ownership, de-duplicates params and shared streams, and a
        predicate also matches chains registered after the GTP built-ins."""
        state = gtp_cuda_graphs.GTPCaptureCommState()
        for chain_id, graphed in (("future_captured", True), ("future_eager", False)):
            monkeypatch.setitem(
                gtp_module._PREFETCH_CHAIN_METADATA,
                chain_id,
                gtp_module._PrefetchChainMetadata(grouped=True, graphed=graphed),
            )

        def param(chain_id):
            return type("Param", (), {"chain_id": chain_id})()

        graphed = param(GTPChain.GRAPHED.value)
        future = param("future_captured")
        ag_stream, rs_stream, eager_stream = object(), object(), object()
        for p in (graphed, graphed, future):
            state.register_comm(p, ag_stream, reduce_scatter=False)
            state.register_comm(p, rs_stream, reduce_scatter=True)
        # Eager chains and objects without a chain are never selected.
        for p in (param(GTPChain.UNGRAPHED.value), param("future_eager"), object()):
            state.register_comm(p, eager_stream, reduce_scatter=False)
            state.register_comm(p, eager_stream, reduce_scatter=True)

        assert state.get_comms_for_chain(GTPChain.GRAPHED.value) == (
            [graphed],
            [ag_stream],
            [rs_stream],
        )
        assert state.get_comms_for_chains(gtp_module._chain_is_graphed) == (
            [graphed, future],
            [ag_stream],
            [rs_stream],
        )

    def test_graphed_drain_ignores_params_without_chain_id(self):
        # object() has none of the attributes a drain reads, so this raises unless it is skipped.
        gtp_module.wait_async_comms(graphed_only=True, params=[object()])


# ---------------------------------------------------------------------------
# count_zeros_fp32 must exclude GTP alignment padding, end-to-end through the
# real distributed optimizer (build_model_and_main_param_groups stamps
# .gtp_pad_zeros; count_zeros_fp32 subtracts it).
# ---------------------------------------------------------------------------


def _worker_count_zeros_excludes_gtp_padding(rank, world_size, port):
    """GTP alignment padding is a permanent structural zero (never written by the wgrad GEMM),
    not a real zero gradient. Left uncorrected, count_zeros_fp32 over-counts by exactly the
    padding-element count on whichever (GTP-rank, DP-sub-shard) fragment physically owns it.

    OUT_F=48 with gtp_remat_size=2 forces real padding (48 is not a multiple of
    pad_for_alignment(16) * gtp_remat_size(2) = 32). With real random bf16 gradients, no element
    is coincidentally exactly zero, so:
      - the FIXED count_zeros() must report 0 (padding correctly excluded)
      - artificially zeroing the .gtp_pad_zeros correction (simulating the pre-fix code) must
        make count_zeros() jump by exactly pad_length * IN_F -- the total padding this weight
        carries, summed over however the distributed optimizer's DP-bucket slicing split it
        across ranks.
    """
    from megatron.core import parallel_state as ps
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import (
        _allreduce_replicated_grads_over_gtp_remat_group,
    )
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    IN_F, OUT_F = 64, 48  # OUT_F not a multiple of pad_for_alignment * gtp_remat_size(2) -> pads.
    PAD_LENGTH = 16

    with _model_parallel_with_gtp(2) as gtp_remat_group:
        model_parallel_cuda_manual_seed(42)
        layer = _make_gtp_linear(IN_F, OUT_F, gtp_remat_group)
        assert layer.weight.pad_length == PAD_LENGTH, "the test needs real padding"
        pad_elems = PAD_LENGTH * IN_F

        tconfig = TransformerConfig(
            num_attention_heads=1, num_layers=1, hidden_size=IN_F, tensor_model_parallel_size=1
        )
        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=True, overlap_grad_reduce=False
        )
        ddp_model = DistributedDataParallel(tconfig, ddp_config, torch.nn.Sequential(layer))

        opt_config = OptimizerConfig(
            optimizer='adam',
            lr=0.01,
            bf16=True,
            use_distributed_optimizer=True,
            use_precision_aware_optimizer=False,
            main_params_dtype=torch.float32,
            main_grads_dtype=torch.float32,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.float32,
            log_num_zeros_in_grad=True,
        )
        optim = get_megatron_optimizer(opt_config, [ddp_model])

        optim.zero_grad()
        ddp_model.zero_grad_buffer()
        torch.manual_seed(1000 + rank)
        x = torch.randn(8, IN_F, dtype=torch.bfloat16, device='cuda')
        ddp_model.module(x).float().mean().backward()
        ddp_model.finish_grad_sync()
        _allreduce_replicated_grads_over_gtp_remat_group(
            [ddp_model],
            ps.get_gtp_weight_remat_group(check_initialized=False),
            ps.get_expert_gtp_weight_remat_group(check_initialized=False),
        )
        # count_zeros() reads the master-shard .grad, which optim.step() normally populates as
        # its first internal step. Do that copy explicitly so count_zeros() can be exercised
        # standalone, without running (and being coupled to) the rest of step()'s update logic.
        for sub_optimizer in getattr(optim, 'chained_optimizers', [optim]):
            sub_optimizer._copy_model_grads_to_main_grads()

        num_zeros_fixed = optim.count_zeros()
        assert num_zeros_fixed == 0, (
            f"expected 0 zeros with real random gradients (padding correctly excluded), "
            f"got {num_zeros_fixed}"
        )

        # Simulate the pre-fix code: zero out this rank's own .gtp_pad_zeros correction(s) --
        # whatever fraction of the padding the distributed optimizer's DP-bucket slicing handed
        # to this rank.
        for p in optim.get_parameters():
            if getattr(p, "gtp_pad_zeros", 0):
                p.gtp_pad_zeros = 0
        num_zeros_unfixed = optim.count_zeros()
        assert num_zeros_unfixed - num_zeros_fixed == pad_elems, (
            f"expected the pre-fix simulation to over-count by exactly pad_elems={pad_elems}, "
            f"got delta={num_zeros_unfixed - num_zeros_fixed} "
            f"(fixed={num_zeros_fixed}, unfixed={num_zeros_unfixed})"
        )


class TestGTPCountZerosExcludesPadding:
    def test_count_zeros_excludes_gtp_padding_end_to_end(self):
        """Real DistributedOptimizer + count_zeros_fp32: GTP alignment padding must not be
        counted as zero gradient. Regression test for the count_zeros_fp32 + distrib_optimizer
        .gtp_pad_zeros wiring."""
        _requires_multi_gpu(4)
        _run_distributed(_worker_count_zeros_excludes_gtp_padding, 4)


# ---------------------------------------------------------------------------
# TestGTPReplicatedBias - the linear bias must stay replicated, not follow the weight
# ---------------------------------------------------------------------------


def _worker_bias_is_replicated(rank, world_size, port):
    """``_gtp_pre_init`` gives TE a pre-sharded ``out_features`` and TE sizes the bias from it, so
    without ``_gtp_restore_replicated_bias`` the bias is sharded too, with no error. out_features
    500 also pads the weight (to 512), so the bias must not follow the padded size either."""
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    hidden, out_features, bias_value = 256, 500, 100.0
    with _model_parallel_with_gtp(world_size) as gtp_remat_group:
        model_parallel_cuda_manual_seed(1234, force_reset_rng=True)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp', 'cp', 'pp', 'ep', 'expt_tp', 'gtp_remat', 'expt_gtp_remat']
        )
        layer = TEColumnParallelLinear(
            hidden,
            out_features,
            config=TransformerConfig(
                num_layers=1,
                hidden_size=hidden,
                num_attention_heads=8,
                add_bias_linear=True,
                params_dtype=torch.bfloat16,
                bf16=True,
            ),
            init_method=lambda w: nn.init.normal_(w, std=0.02),
            gather_output=False,
            bias=True,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name=None,
            pg_collection=pg_collection,
        ).cuda()
        assert layer.weight.pad_length > 0, "the test needs a padded weight"
        assert layer.bias.shape == (
            out_features,
        ), f"bias is {tuple(layer.bias.shape)}; expected the replicated size ({out_features},)"

        # The shape alone is not enough: a sharded bias fails silently, still returning the full
        # out_features but biasing only the first out_features/gtp_remat_size columns.
        with torch.no_grad():
            layer.bias.fill_(bias_value)
        shards = [torch.empty_like(layer.weight.data) for _ in range(world_size)]
        dist.all_gather(shards, layer.weight.data.contiguous(), group=gtp_remat_group)
        full_weight = torch.cat(shards, dim=0)[:out_features].float()

        inputs = torch.randn(8, hidden, dtype=torch.bfloat16, device="cuda")
        dist.broadcast(inputs, src=0)
        applied = (layer(inputs)[0].float() - inputs.float() @ full_weight.T).mean(dim=0)
        missing = (applied - bias_value).abs() > 1.0
        assert not missing.any(), (
            f"{int(missing.sum())} of {out_features} outputs did not receive the bias: "
            "the bias is still sharded"
        )


class TestGTPReplicatedBias:
    def test_bias_is_replicated(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_bias_is_replicated, 4)


class TestWgradAccumEligibility:
    """Only output_layer accumulates its wgrad across backwards, and only without pipelining."""

    @pytest.mark.parametrize("pipeline_size", [1, 2])
    def test_accumulation_enabled(self, monkeypatch, pipeline_size):
        from types import SimpleNamespace

        monkeypatch.setattr(gtp_module, "is_gtp_param", lambda p: True)
        output_layer, decoder = SimpleNamespace(), SimpleNamespace()
        gtp_module.tag_gtp_params_with_names(
            SimpleNamespace(
                config=SimpleNamespace(pipeline_model_parallel_size=pipeline_size),
                named_parameters=lambda: [
                    ("module.output_layer.weight", output_layer),
                    ("module.decoder.layers.0.mlp.linear_fc1.weight", decoder),
                ],
            )
        )
        assert output_layer._wgrad_accum_enabled is (pipeline_size == 1)
        assert decoder._wgrad_accum_enabled is False
