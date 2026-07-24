# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for Generalized Tensor Parallelism (GTP).

Scope: sharding math, module wiring, and behavior/regression guards. End-to-end fwd/bwd/loss/grad,
fp8, and checkpoint correctness live in the integration tests (test_gtp_loss_correctness,
test_gtp_grad_correctness, test_attention_gtp, test_mamba_gtp, test_moe_egtp,
test_gtp_fp8_param_gather, test_gtp_dcp), so low-level plumbing smoke tests are not duplicated here.

Test groups
-----------
- TestGTPSharding            - wrap_module_params_gtp: shard content + padding
- TestWrapModuleParams       - wrap_module_params_gtp: param replacement + weight_list
- TestLinearGTP / TestLayerNormLinearGTP / TestGroupedLinearGTP - single-layer fwd/bwd
- TestGTPPrefetchChain       - linked-list next_w/prev_w wiring
- TestGTPWgradRS             - wgrad reduce-scatter shape + multi-layer deferred path
- TestGTPMicrobatches        - output consistency across microbatches
- TestMXFP8LinearGTP         - Linear + MXFP8 recipe: quantized shard setup, fwd/bwd, padding
- TestGTPGroupSizeOne        - wrap_module_params_gtp no-op when gtp_remat_group.size()==1
- TestGTPPrefetchDisabled    - weight_prefetch=False single-pass forward
- TestFuseWgradAccumulation  - fuse_wgrad_accumulation=True: wgrad -> main_grad
- TestGTPGradAccumHook       - main_grad updated after reduce-scatter backward
- TestWaitAsyncCommsFallback - inline-accumulation fallback when _wgrad_rs_handle is None
- TestGTPDDPBucketAlignment  - GTP/regular DDP bucket ends padded for dist-opt alignment
- TestGTPDDPGradReadyWiring  - GTP params drive DDP grad-ready via the manual hook, not autograd

Multi-GPU tests skip when ``torch.distributed.get_world_size()`` != the required world size (4).
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.core.tensor_parallel.gtp import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

import transformer_engine.pytorch as te
from transformer_engine.pytorch import fp8_autocast
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor

import megatron.core.tensor_parallel.generalized_tensor_parallelism as gtp_module
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
    GTPShardedParam,
    wrap_module_params_gtp,
)
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _make_gtp_linear,
    _make_gtp_remat_grouped_linear,
    _requires_multi_gpu,
    _requires_mxfp8,
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)


class _FakeGroup:
    """Minimal mock for a dist process group — used in single-process unit tests."""

    def __init__(self, size=1, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _worker_sharding_aligned(rank, world_size, port):
    K, M = world_size * 32, 16  # K divisible by 16*world_size → no padding
    full_weight = torch.arange(K * M, dtype=torch.float32).reshape(K, M).cuda()
    dist.broadcast(full_weight, src=0)

    gtp_remat_group = dist.new_group(list(range(world_size)))
    mod = nn.Module()
    mod.weight = nn.Parameter(full_weight.clone(), requires_grad=False)
    wrap_module_params_gtp(mod, ["weight"], gtp_remat_group)
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

    gtp_remat_group = dist.new_group(list(range(world_size)))
    mod = nn.Module()
    mod.weight = nn.Parameter(full_weight.clone(), requires_grad=False)
    wrap_module_params_gtp(mod, ["weight"], gtp_remat_group)
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
# wrap_module_params_gtp: param replacement and GroupedLinear weight_list
# ---------------------------------------------------------------------------


def _worker_linear_param_replaced(rank, world_size, port):
    in_f, out_f = 64, 128
    gtp_remat_group = dist.new_group(list(range(world_size)))
    layer = _make_gtp_linear(in_f, out_f, gtp_remat_group)
    w = layer.weight
    assert isinstance(w, GTPShardedParam), "weight must be GTPShardedParam"
    assert w.shape == (out_f // world_size, in_f), f"unexpected shard shape {w.shape}"
    assert w.group is gtp_remat_group


def _worker_grouped_weight_list(rank, world_size, port):
    num_gemms, in_f, out_f = 3, 32, 64
    gtp_remat_group = dist.new_group(list(range(world_size)))
    layer = _make_gtp_remat_grouped_linear(num_gemms, in_f, out_f, gtp_remat_group)
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
# Linear forward/backward numerical correctness
# ---------------------------------------------------------------------------


def _worker_linear_correctness(rank, world_size, port):
    """GTP output == (all-gathered weight) @ input, and dX matches."""
    torch.manual_seed(0)
    batch, in_f, out_f = 16, 64, 128  # out_f % (16*world_size)==0 → no padding
    dtype = torch.bfloat16
    gtp_remat_group = dist.new_group(list(range(world_size)))

    layer = _make_gtp_linear(in_f, out_f, gtp_remat_group, dtype)

    # Reconstruct full weight from shards (all-gather)
    shard = layer.weight.data.clone()
    all_shards = [torch.zeros_like(shard) for _ in range(world_size)]
    dist.all_gather(all_shards, shard, group=gtp_remat_group)
    full_weight = torch.cat(all_shards, dim=0).float()[:out_f]  # strip any padding

    # Shared input across ranks
    inp = torch.randn(batch, in_f, dtype=dtype, device="cuda")
    dist.broadcast(inp, src=0)

    inp_gtp = inp.clone().requires_grad_(True)
    inp_ref = inp.clone().requires_grad_(True)

    # GTP_remat forward
    out_gtp = layer(inp_gtp, is_first_microbatch=True)

    # Reference forward
    out_ref = inp_ref.float() @ full_weight.T
    out_ref = out_ref.to(dtype)

    assert out_gtp.shape == out_ref.shape, f"Shape mismatch {out_gtp.shape} vs {out_ref.shape}"
    assert torch.allclose(
        out_gtp.float(), out_ref.float(), atol=1e-5, rtol=1e-5
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
        inp_gtp.grad.float(), inp_ref.grad.float(), atol=1e-5, rtol=1e-5
    ), f"dX mismatch max_diff={(inp_gtp.grad.float()-inp_ref.grad.float()).abs().max():.4f}"


class TestLinearGTP:
    def test_forward_backward_correctness(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_linear_correctness, 4)


# ---------------------------------------------------------------------------
# LayerNormLinear forward/backward smoke test
# ---------------------------------------------------------------------------


def _worker_layernorm_linear(rank, world_size, port):
    torch.manual_seed(0)
    seq, batch, in_f, out_f = 4, 2, 64, 128
    dtype = torch.bfloat16
    gtp_remat_group = dist.new_group(list(range(world_size)))

    layer = te.LayerNormLinear(
        in_features=in_f, out_features=out_f, bias=False, params_dtype=dtype, device="cuda"
    )
    # TE construction is GTP-agnostic: gtp_remat_size (forward-gather gate) is stamped and the
    # BF16 weight is sliced post-init (Megatron side).
    layer.gtp_remat_size = gtp_remat_group.size()
    wrap_module_params_gtp(layer, layer.weight_names, gtp_remat_group)
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
# GroupedLinear forward/backward smoke test
# ---------------------------------------------------------------------------


def _worker_grouped_linear(rank, world_size, port, num_gemms):
    torch.manual_seed(0)
    in_f, out_f, total_tokens = 32, 64, num_gemms * 4
    dtype = torch.bfloat16
    gtp_remat_group = dist.new_group(list(range(world_size)))

    layer = _make_gtp_remat_grouped_linear(num_gemms, in_f, out_f, gtp_remat_group, dtype)
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


def _worker_ops_grouped_linear(rank, world_size, port, num_gemms):
    """GTP on the fusible-op ``te.ops.GroupedLinear`` -- the unfused fallback path (run standalone,
    so no op fusion). Exercises materialize (fwd/bwd all-gather) + wgrad reduce-scatter wiring in
    transformer_engine/pytorch/ops/basic/grouped_linear.py."""
    torch.manual_seed(0)
    in_f, out_f, total_tokens = 32, 64, num_gemms * 4
    dtype = torch.bfloat16
    gtp_remat_group = dist.new_group(list(range(world_size)))

    op = te.ops.GroupedLinear(num_gemms, in_f, out_f, bias=False, device="cuda", dtype=dtype)
    op.gtp_remat_size = gtp_remat_group.size()
    wrap_module_params_gtp(
        op, [f"weight{i}" for i in range(num_gemms)], gtp_remat_group, is_grouped=True
    )
    assert isinstance(op.weight0, GTPShardedParam)

    m_splits = [total_tokens // num_gemms] * num_gemms
    m_splits[-1] += total_tokens - sum(m_splits)
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
        assert w.main_grad.shape == w.shape
        assert torch.count_nonzero(w.main_grad) > 0, f"weight{i} main_grad not populated by RS"


class TestOpsGroupedLinearGTP:
    """GTP on the fusible-op ``te.ops.GroupedLinear`` (the unfused fallback for grouped-MLP)."""

    @pytest.mark.parametrize("num_gemms", [2, 4])
    def test_forward_backward(self, num_gemms):
        _requires_multi_gpu(4)
        _run_distributed(_worker_ops_grouped_linear, 4, num_gemms)


# ---------------------------------------------------------------------------
# Prefetch chain: next_w / prev_w wiring after first forward pass
# ---------------------------------------------------------------------------


def _worker_chain_wired(rank, world_size, port):
    torch.manual_seed(0)
    in_f, out_f = 32, 64
    dtype = torch.bfloat16
    gtp_remat_group = dist.new_group(list(range(world_size)))

    l0 = _make_gtp_linear(in_f, out_f, gtp_remat_group, dtype)
    l1 = _make_gtp_linear(in_f, out_f, gtp_remat_group, dtype)

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
    gtp_remat_group = dist.new_group(list(range(world_size)))

    l0 = _make_gtp_linear(in_f, out_f, gtp_remat_group, dtype)
    l1 = _make_gtp_linear(in_f, out_f, gtp_remat_group, dtype)

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
# Wgrad reduce-scatter: shape and deferred async path
# ---------------------------------------------------------------------------


def _worker_wgrad_shape(rank, world_size, port):
    """After backward, weight.grad shape must match the local shard shape."""
    torch.manual_seed(0)
    in_f, out_f = 32, 64
    dtype = torch.bfloat16
    gtp_remat_group = dist.new_group(list(range(world_size)))

    layer = _make_gtp_linear(in_f, out_f, gtp_remat_group, dtype, fuse_wgrad_accumulation=False)
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
    gtp_remat_group = dist.new_group(list(range(world_size)))

    l0 = _make_gtp_linear(in_f, out_f, gtp_remat_group, dtype)
    l1 = _make_gtp_linear(in_f, out_f, gtp_remat_group, dtype)

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
# Multiple microbatches: output must be consistent when weight unchanged
# ---------------------------------------------------------------------------


def _worker_microbatches(rank, world_size, port):
    torch.manual_seed(0)
    batch, in_f, out_f = 8, 64, 128
    dtype = torch.bfloat16
    gtp_remat_group = dist.new_group(list(range(world_size)))

    layer = _make_gtp_linear(in_f, out_f, gtp_remat_group, dtype)
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
# MXFP8 + GTP_remat: Linear forward/backward, quantized shard setup
# ---------------------------------------------------------------------------


def _make_native_fp8_gtp_linear(in_f, out_f, gtp_remat_group, dtype, recipe):
    """Build a native-FP8 GTP te.Linear the gtp-agnostic way.

    Mirrors megatron/core/extensions/transformer_engine.py: pass the pre-sharded
    out_features to a STOCK te.Linear under fp8_model_init (TE inits+quantizes a native
    MXFP8 shard with no GTP awareness), then attach the GTP wiring post-init.
    """
    from transformer_engine.pytorch import fp8_model_init

    from megatron.core.tensor_parallel.gtp import (
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


def _worker_mxfp8_linear(rank, world_size, port):
    """Verify GTP Linear with a native MXFP8 param: all-gather + GEMM + backward.

    mxfp8 always implies --fp8-param-gather: the weight is built as a native FP8 shard at
    construction (no BF16 source, no per-forward cast).
    """
    from transformer_engine.common.recipe import MXFP8BlockScaling

    from megatron.core.tensor_parallel.generalized_tensor_parallelism import update_gtp_config

    torch.manual_seed(0)
    # batch=32: MXFP8 wgrad GEMM (K=batch) requires K divisible by MXFP8_BLOCK_SCALING_SIZE=32
    batch, in_f, out_f = 32, 64, 128  # out_f % (16*world_size)==0 → no padding
    dtype = torch.bfloat16
    gtp_remat_group = dist.new_group(list(range(world_size)))
    recipe = MXFP8BlockScaling()
    layer = _make_native_fp8_gtp_linear(in_f, out_f, gtp_remat_group, dtype, recipe)

    # The weight IS the native FP8 shard: a QuantizedTensor with the GTP surface attached.
    w = layer.weight
    assert isinstance(w, QuantizedTensor), f"weight should be QuantizedTensor, got {type(w)}"
    assert w.quantized is w, "native-FP8 GTP: self.quantized must be the param itself"
    assert getattr(w, "is_gtp_weight_remat", False), "GTP surface missing on native param"
    assert w.shape[0] * world_size == out_f, "weight must be dim-0 sharded"

    inp = torch.randn(batch, in_f, dtype=dtype, device="cuda", requires_grad=True)
    dist.broadcast(inp, src=0)

    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        out = layer(inp, is_first_microbatch=True)

    assert out.shape == (batch, out_f), f"unexpected output shape {out.shape}"
    assert torch.isfinite(out).all(), "MXFP8 GTP output has non-finite values"

    # Backward should complete without error
    layer.weight.main_grad = torch.zeros(layer.weight.shape, dtype=dtype, device="cuda")
    out.sum().backward()
    assert inp.grad is not None
    assert inp.grad.shape == inp.shape

    # Second microbatch reuses the same native FP8 weight
    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        out2 = layer(inp.detach(), is_first_microbatch=False)
    assert torch.isfinite(out2).all(), "MXFP8 GTP second-microbatch output has non-finite"


def _worker_mxfp8_linear_unaligned(rank, world_size, port):
    """Verify native-FP8 MXFP8 GTP when out_features needs padding.

    MXFP8 requires tensor dims divisible by 32, so shard_size (= M_padded / world_size)
    must be a multiple of 32. With world_size=4 this requires M_padded % 128 == 0.
    out_f=120 gives M_padded=128, shard_size=32 (32 % 32 == 0). The last rank's shard
    holds 24 real rows zero-padded to 32. After all-gather, _strip_padding removes the
    padded rows before the GEMM, so the output has the original out_f columns.
    """
    from transformer_engine.common.recipe import MXFP8BlockScaling

    from megatron.core.tensor_parallel.generalized_tensor_parallelism import update_gtp_config

    torch.manual_seed(0)
    # out_f=120: M_padded=128, shard_size=32, last rank has 24 rows padded to 32.
    out_f = 120
    in_f = 64
    batch = 32
    dtype = torch.bfloat16
    gtp_remat_group = dist.new_group(list(range(world_size)))
    recipe = MXFP8BlockScaling()
    layer = _make_native_fp8_gtp_linear(in_f, out_f, gtp_remat_group, dtype, recipe)
    assert layer.weight.pad_length == 8, f"expected pad 8, got {layer.weight.pad_length}"

    inp = torch.randn(batch, in_f, dtype=dtype, device="cuda", requires_grad=True)
    dist.broadcast(inp, src=0)

    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        out = layer(inp, is_first_microbatch=True)

    # After _strip_padding removes the padded rows, output has out_f (not padded) cols.
    assert out.shape == (batch, out_f), f"unexpected output shape {out.shape}"
    assert torch.isfinite(out).all(), "MXFP8 GTP (unaligned) output has non-finite values"


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
# wrap_module_params_gtp is a no-op when gtp_remat_group.size() == 1
# ---------------------------------------------------------------------------


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
        """The exact call in the TE extension returns None groups, no assert, when GTP is off."""
        # Force the uninitialized GTP state deterministically (independent of suite ordering).
        mocker.patch.object(parallel_state, "_GTP_WEIGHT_REMAT_GROUP", None)
        mocker.patch.object(parallel_state, "_EXPERT_GTP_WEIGHT_REMAT_GROUP", None)

        pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["gtp_remat", "expt_gtp_remat"]
        )

        # Mirror the downstream selection in the TE extension; both branches are None, so
        # _init_gtp_remat_context takes the no-op (GTP-inactive) path.
        assert pg_collection.gtp_remat is None
        assert pg_collection.expt_gtp_remat is None
        for is_expert in (False, True):
            gtp_remat_group = (
                pg_collection.expt_gtp_remat if is_expert else pg_collection.gtp_remat
            )
            assert gtp_remat_group is None


# ---------------------------------------------------------------------------
# weight_prefetch=False: forward still produces correct output
# ---------------------------------------------------------------------------


def _worker_prefetch_disabled(rank, world_size, port):
    torch.manual_seed(0)
    in_f, out_f = 32, 64
    dtype = torch.bfloat16
    gtp_remat_group = dist.new_group(list(range(world_size)))

    gtp_module.update_gtp_config(weight_prefetch=False)
    try:
        l0 = _make_gtp_linear(in_f, out_f, gtp_remat_group, dtype)
        l1 = _make_gtp_linear(in_f, out_f, gtp_remat_group, dtype)

        inp = torch.randn(4, in_f, dtype=dtype, device="cuda")
        dist.broadcast(inp, src=0)

        # Single forward pass: builds chain and verifies output is correct
        out = l0(inp, is_first_microbatch=True) + l1(inp, is_first_microbatch=True)

        # Chain should still be wired even with prefetch disabled
        assert l0.weight.next_w is l1.weight
        assert torch.isfinite(out).all(), "Non-finite output with prefetch disabled"
    finally:
        gtp_module.update_gtp_config(weight_prefetch=True)


class TestGTPPrefetchDisabled:
    def test_forward_works_without_prefetch(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_prefetch_disabled, 4)


# ---------------------------------------------------------------------------
# fuse_wgrad_accumulation=True: wgrad is accumulated into main_grad
# ---------------------------------------------------------------------------


def _worker_fuse_wgrad(rank, world_size, port):
    torch.manual_seed(0)
    in_f, out_f = 32, 128  # out_f % (16*world_size)==0, no padding
    dtype = torch.bfloat16
    gtp_remat_group = dist.new_group(list(range(world_size)))

    layer = _make_gtp_linear(in_f, out_f, gtp_remat_group, dtype, fuse_wgrad_accumulation=True)

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
# _grad_accum_hook is called after reduce-scatter
# ---------------------------------------------------------------------------


def _worker_main_grad_updated_after_bwd(rank, world_size, port):
    """After backward, the wgrad RS path must have accumulated wgrad into main_grad."""
    torch.manual_seed(0)
    in_f, out_f = 32, 64
    dtype = torch.bfloat16
    gtp_remat_group = dist.new_group(list(range(world_size)))

    layer = _make_gtp_linear(in_f, out_f, gtp_remat_group, dtype)

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
# wait_async_comms(finalize_after_drain=True) inline-accumulation fallback
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

    @staticmethod
    def _make_inflight_param(main_grad_fill=0.0, already_finalized=False):
        """Build a minimal GTPShardedParam wired for wait_async_comms testing."""
        dtype = torch.bfloat16
        p = GTPShardedParam(torch.zeros(8, 4, dtype=dtype, device="cuda"))
        p.group = _FakeGroup()
        p.expert_idx = None
        p.pad_length = 0
        p.chain_id = gtp_module.GTPChain.UNGRAPHED.value
        p._quantizer = None
        p.is_routed_expert = False  # ⇒ self._weights property returns [self]
        p.main_grad = torch.full((8, 4), main_grad_fill, dtype=dtype, device="cuda")
        p._prefetch_handle = None  # _wait_param_gather is no-op
        p._wgrad_rs_handle = None  # _wait_reduce_scatter is no-op → fallback fires
        p._cached_ag_stream = None
        p._cached_rs_stream = None
        p.ag_event = torch.cuda.Event(external=True)
        p.rs_event = torch.cuda.Event(external=True)
        p.rs_event.record()  # so rs_event.wait() in fallback doesn't block
        p._already_finalized = already_finalized
        p.grad_added_to_main_grad = False
        return p

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_fallback_accumulates_when_no_rs_handle(self):
        dtype = torch.bfloat16
        p = self._make_inflight_param(main_grad_fill=0.0)

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
        p = self._make_inflight_param(main_grad_fill=5.0, already_finalized=True)
        # No _rs_ticket: if the fallback ran it would AttributeError on cache.get(None).
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
        p = self._make_inflight_param(main_grad_fill=7.0)
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


# ---------------------------------------------------------------------------
# GTP_remat DDP bucket alignment: distributed optimizer bucket-end assertion
# ---------------------------------------------------------------------------


def _worker_gtp_ddp_bucket_alignment(rank, world_size, port):
    """GTP param buffers in DDP must use padded bucket layout with use_distributed_optimizer=True.

    Bug: DDP used param_layout=None for GTP buffers, falling through to
    _compute_default_per_buffer_param_layout, which packs params without padding bucket ends.
    The distributed optimizer requires every bucket end to be divisible by
    intra_dp_cp_group.size() (asserted at param_and_grad_buffer.py:1427).

    Trigger:
      GTP_remat_size=2, DP=4  →  intra_dp_cp_group.size()=2
      pad_for_alignment=0, weight [out=2,in=3]  →  GTP shard=[1,3]=3 elements (odd)
      Two GTP params: total=6, 6%2==0 (total check passes); bucket_size=3 forces
      bucket-0 to contain only the first param, end=3, 3%2≠0  →  AssertionError
    """
    from megatron.core import parallel_state as ps
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.transformer.transformer_config import TransformerConfig

    # The module fixture initialized model_parallel without GTP_remat; re-init with GTP_remat=2.
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=2
    )

    orig_pad = gtp_module.GTP_CONFIG.pad_for_alignment
    gtp_module.GTP_CONFIG.pad_for_alignment = 0
    try:
        gtp_remat_group = ps.get_gtp_weight_remat_group()

        class _TwoLayerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc0 = te.Linear(3, 2, bias=False, device="cuda")
                self.fc1 = te.Linear(3, 2, bias=False, device="cuda")

        model = _TwoLayerModel()
        wrap_module_params_gtp(model.fc0, ["weight"], gtp_remat_group)
        wrap_module_params_gtp(model.fc1, ["weight"], gtp_remat_group)

        config = TransformerConfig(
            num_attention_heads=1, num_layers=1, hidden_size=4, tensor_model_parallel_size=1
        )
        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=True, overlap_grad_reduce=True, bucket_size=3
        )

        # Without the fix this raises AssertionError at param_and_grad_buffer.py:1427:
        #   assert end_index % self.data_parallel_world_size == 0
        DistributedDataParallel(config, ddp_config, model)
    finally:
        gtp_module.GTP_CONFIG.pad_for_alignment = orig_pad
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()  # restore default for remaining tests


def _worker_regular_buffer_padded_when_gtp_params_present(rank, world_size, port):
    """Regular (non-GTP) param buffers in DDP must also use padded layout when GTP is active.

    Bug: when gtp_params is non-empty, full_param_layout.layouts contains stale GTP entries
    that don't belong to the regular buffer, causing KeyErrors in DistOpt's param map.
    DDP avoided this by forcing param_layout=None for regular buffers, but that falls through
    to _compute_default_per_buffer_param_layout, which produces unpadded bucket ends, again
    violating param_and_grad_buffer.py:1427 (end_index % data_parallel_world_size == 0).

    Trigger:
      GTP_remat_size=2, DP=4  →  intra_dp_cp_group.size()=4
      (regular params reduce over the full DP group)
      bias=True  →  each bias has 2 elements (not divisible by 4)
      Two layers: total regular numel=4, 4%4==0 (total check passes); bucket_size=2 forces
      bucket-0 to contain only the first bias, end=2, 2%4≠0  →  AssertionError
    """
    from megatron.core import parallel_state as ps
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.transformer.transformer_config import TransformerConfig

    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=2
    )

    orig_pad = gtp_module.GTP_CONFIG.pad_for_alignment
    gtp_module.GTP_CONFIG.pad_for_alignment = 0
    try:
        gtp_remat_group = ps.get_gtp_weight_remat_group()

        class _TwoLayerModelWithBias(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # bias=True: weight → GTPShardedParam (gtp_buffer), bias → regular param
                self.fc0 = te.Linear(3, 2, bias=True, device="cuda")
                self.fc1 = te.Linear(3, 2, bias=True, device="cuda")

        model = _TwoLayerModelWithBias()
        wrap_module_params_gtp(model.fc0, ["weight"], gtp_remat_group)
        wrap_module_params_gtp(model.fc1, ["weight"], gtp_remat_group)

        config = TransformerConfig(
            num_attention_heads=1, num_layers=1, hidden_size=4, tensor_model_parallel_size=1
        )
        # bucket_size=2: each 2-element bias fills one bucket in the regular buffer.
        # Without the fix: regular buffer uses param_layout=None → bucket-0 ends at 2,
        # 2 % intra_dp_cp_group.size()(=4) != 0 → AssertionError at line 1427.
        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=True, overlap_grad_reduce=True, bucket_size=2
        )

        DistributedDataParallel(config, ddp_config, model)
    finally:
        gtp_module.GTP_CONFIG.pad_for_alignment = orig_pad
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


class TestGTPDDPBucketAlignment:
    def test_gtp_buffers_use_padded_layout_with_distributed_optimizer(self):
        """GTP buffer bucket ends must be padded to intra_dp_cp_group.size()."""
        _requires_multi_gpu(4)
        _run_distributed(_worker_gtp_ddp_bucket_alignment, 4)

    def test_regular_buffers_use_padded_layout_when_gtp_params_present(self):
        """Regular buf bucket ends must be padded even when gtp_params forces layoutrecompute."""
        _requires_multi_gpu(4)
        _run_distributed(_worker_regular_buffer_padded_when_gtp_params_present, 4)


# ---------------------------------------------------------------------------
# GTP_remat DDP grad-ready wiring: register_grad_ready must fire AFTER the wgrad add
# ---------------------------------------------------------------------------


def _worker_gtp_ddp_grad_ready_wiring(rank, world_size, port):
    """GTP params must drive DDP grad-ready from GTP's manual hook, not autograd.

    GTP defers the main_grad accumulation to a later backward node, so autograd's AccumulateGrad can
    fire register_grad_ready before the grad lands and dispatch the bucket reduce-scatter on stale
    grad_data (corrupts reduce_scatter_with_fp32_accumulation). The fix routes grad-ready through
    register_grad_accum_hook (fired after the add) and skips the autograd hook. This pins that
    wiring: every GTP weight has _grad_accum_hook set and none falls through to the autograd list.
    """
    from megatron.core import parallel_state as ps
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.transformer.transformer_config import TransformerConfig

    # The module fixture initialized model_parallel without GTP_remat; re-init with GTP_remat=2.
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=2
    )
    try:
        gtp_remat_group = ps.get_gtp_weight_remat_group()

        class _TwoLayerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # bias=False -> all params are GTP_remat weights, so grad_accs must end up empty.
                self.fc0 = te.Linear(64, 128, bias=False, device="cuda")
                self.fc1 = te.Linear(64, 128, bias=False, device="cuda")

        model = _TwoLayerModel()
        wrap_module_params_gtp(model.fc0, ["weight"], gtp_remat_group)
        wrap_module_params_gtp(model.fc1, ["weight"], gtp_remat_group)

        config = TransformerConfig(
            num_attention_heads=1, num_layers=1, hidden_size=4, tensor_model_parallel_size=1
        )
        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=True, overlap_grad_reduce=True
        )
        ddp_model = DistributedDataParallel(config, ddp_config, model)

        for name, w in [("fc0", model.fc0.weight), ("fc1", model.fc1.weight)]:
            assert isinstance(w, GTPShardedParam), f"{name}.weight should be a GTP param"
            # Manual hook set -> grad-ready fires after the add; None -> early autograd path (bug).
            assert (
                getattr(w, "_grad_accum_hook", None) is not None
            ), f"{name}.weight must have _grad_accum_hook set (manual grad-ready, not autograd)"

        # bias=False -> all params are GTP_remat -> none took the autograd path.
        assert len(ddp_model.grad_accs) == 0, (
            "GTP params must not register an autograd AccumulateGrad hook "
            f"(grad_accs has {len(ddp_model.grad_accs)} entries)"
        )
    finally:
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()  # restore default for remaining tests


class TestGTPDDPGradReadyWiring:
    def test_gtp_params_use_manual_grad_ready_hook(self):
        """GTP params route DDP grad-ready through register_grad_accum_hook, not autograd."""
        _requires_multi_gpu(4)
        _run_distributed(_worker_gtp_ddp_grad_ready_wiring, 4)
