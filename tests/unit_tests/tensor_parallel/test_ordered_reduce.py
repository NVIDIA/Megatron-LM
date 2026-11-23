# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Batch-invariant ordered TP reductions (megatron/core/ops/collectives/ordered_reduce.py)."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import TERowParallelLinear, _BITPForward
from megatron.core.fp8_utils import get_fp8_context, prepare_model_for_fp8_inference
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.ops.collectives import ordered_reduce as orr
from megatron.core.ops.linear import ordered as orl
from megatron.core.tensor_parallel import inference_layers, mappings
from megatron.core.tensor_parallel.inference_layers import InferenceRowParallelLinear
from megatron.core.tensor_parallel.layers import RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP, SharedExpertState
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
)
from megatron.core.transformer.moe.token_dispatcher_inference import NCCLAllGatherDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_tensor_model_parallel_group_if_none, init_method_normal
from tests.unit_tests.test_utilities import Utils

TP = 4
requires_tp4 = pytest.mark.skipif(
    Utils.world_size < TP or Utils.world_size % TP != 0, reason="needs a multiple of 4 GPUs"
)


def _bits(t):
    return t.contiguous().view(torch.uint8)


def _all_partials(local, group):
    """[TP, ...] stack of every rank's partial (every rank gets all of them)."""
    parts = [torch.empty_like(local) for _ in range(group.size())]
    dist.all_gather(parts, local.contiguous(), group=group)
    return torch.stack(parts)


def _left_fold(parts):
    """Reference: ((p_0 + p_1) + p_2) + ... in fp32, one rounding to the input dtype."""
    acc = parts[0].float()
    for r in range(1, parts.shape[0]):
        acc = acc + parts[r].float()
    return acc.to(parts.dtype)


def _partials(shape, dtype, seed):
    """Per-rank partials with mixed magnitudes, so summation order changes the bits."""
    gen = torch.Generator(device="cuda").manual_seed(seed + 1000 * Utils.rank)
    x = torch.randn(shape, device="cuda", generator=gen, dtype=torch.float32)
    scale = torch.logspace(-4, 4, shape[-1], device="cuda")
    return (x * scale).to(dtype)


def _symm_buffer(group, nbytes):
    """A raw symmetric-memory byte buffer and its handle for group, or (None, None) where
    symmetric memory is unavailable. Owned by SymmetricMemoryManager, so teardown frees it
    on every rank at the same point."""
    buf = SymmetricMemoryManager.get_buffer(
        f"ordered_reduce_test:{nbytes}", process_group=group, size_mb=nbytes >> 20
    )
    return buf.symm_buffer, buf.symm_mem_hdl


def _unexpected_ordered_reduce(*args, **kwargs):
    pytest.fail("non-linear caller entered an ordered TP reduction")


@requires_tp4
class TestOrderedReduce:
    def setup_method(self, method):
        Utils.initialize_model_parallel(TP, 1)
        self.group = get_tensor_model_parallel_group_if_none(tp_group=None)

    def teardown_method(self, method):
        torch.cuda.synchronize()
        # Peers may still read this rank's buffer until the group syncs.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("rows,hidden", [(4, 2688), (1024, 8192)])
    def test_reduce_scatter_matches_left_fold(self, dtype, rows, hidden):
        x = _partials((rows, hidden), dtype, seed=rows)
        ref = _left_fold(_all_partials(x, self.group))
        out = orr.ordered_reduce_scatter(x, self.group)
        k = rows // TP
        rank = self.group.rank()
        assert torch.equal(_bits(out), _bits(ref[rank * k : (rank + 1) * k]))

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("shape", [(1, 2688), (3, 1, 4096), (7, 100), (128, 8192)])
    def test_all_reduce_matches_left_fold(self, dtype, shape):
        x = _partials(shape, dtype, seed=len(shape))
        ref = _left_fold(_all_partials(x, self.group))
        x_before = x.clone()
        out = orr.ordered_all_reduce(x, self.group)
        assert out.shape == x.shape
        assert torch.equal(_bits(out), _bits(ref))
        assert torch.equal(x, x_before)

    @pytest.mark.parametrize("batch_invariant", [False, True])
    def test_mappings_forward_is_ordered_and_position_invariant(self, batch_invariant):
        """A row gets the same bits wherever it sits and whatever the batch size."""
        hidden = 4096
        row = _partials((1, hidden), torch.bfloat16, seed=11)
        expected = _left_fold(_all_partials(row, self.group))[0]
        with set_batch_invariant_mode(batch_invariant):
            assert orr.use_ordered_tp_reduce(self.group, torch.bfloat16) == batch_invariant
            for rows in (4, 8, 64, 512):
                for pos in (0, rows // 2 + 1, rows - 1):
                    x = _partials((rows, hidden), torch.bfloat16, seed=rows + pos)
                    x[pos] = row[0]
                    # All-reduce (non-sequence-parallel row-parallel linear).
                    ar = mappings.reduce_from_tensor_model_parallel_region(
                        x.clone(), self.group, ordered=True
                    )
                    assert torch.equal(_bits(ar[pos]), _bits(expected))
                    # Reduce-scatter (sequence parallel); gather to find the row again.
                    rs = mappings.reduce_scatter_to_sequence_parallel_region(
                        x, self.group, ordered=True
                    )
                    full = mappings._gather_along_first_dim(rs, self.group)
                    assert torch.equal(_bits(full[pos]), _bits(expected))

    def test_backward_uses_stock_collectives(self):
        """Backward is unchanged: RS backward is an all-gather, AR backward the identity."""
        with set_batch_invariant_mode(True):
            x = _partials((8, 256), torch.float32, seed=3).requires_grad_()
            mappings.reduce_scatter_to_sequence_parallel_region(
                x, self.group, ordered=True
            ).sum().backward()
            assert torch.equal(x.grad, torch.ones_like(x))
            y = _partials((8, 256), torch.float32, seed=4).requires_grad_()
            mappings.reduce_from_tensor_model_parallel_region(
                y, self.group, ordered=True
            ).sum().backward()
            assert torch.equal(y.grad, torch.ones_like(y))

    def test_shared_mappings_keep_stock_reductions(self, monkeypatch):
        """Batch-invariant mode alone must not change a non-linear caller's collective."""
        monkeypatch.setattr(mappings, "ordered_all_reduce", _unexpected_ordered_reduce)
        monkeypatch.setattr(mappings, "ordered_reduce_scatter", _unexpected_ordered_reduce)
        x = _partials((8, 256), torch.bfloat16, seed=81)
        expected_ar = x.clone()
        dist.all_reduce(expected_ar, group=self.group)
        expected_rs = torch.empty_like(x[:2])
        dist.reduce_scatter_tensor(expected_rs, x, group=self.group)
        with set_batch_invariant_mode(True, backend="te_native"):
            ar = mappings.reduce_from_tensor_model_parallel_region(x.clone(), self.group)
            rs_input = x.detach().requires_grad_()
            rs = mappings.reduce_scatter_to_sequence_parallel_region(rs_input, self.group)
            rs.sum().backward()
        assert torch.equal(_bits(ar), _bits(expected_ar))
        assert torch.equal(_bits(rs), _bits(expected_rs))
        assert torch.equal(rs_input.grad, torch.ones_like(x))

    def test_moe_alltoall_uneven_expert_tp_combine(self, monkeypatch):
        """Exercise the dispatcher's uneven TP4 combine, including its backward gather."""
        monkeypatch.setattr(mappings, "ordered_reduce_scatter", _unexpected_ordered_reduce)
        splits = [1, 2, 3, 2]
        # Isolate communication from routing/permutation: one local expert needs no sort.
        dispatcher = SimpleNamespace(
            num_local_experts=1,
            tp_size=TP,
            tp_group=self.group,
            output_splits_tp=torch.tensor(splits),
            probs=torch.empty(0, dtype=torch.bfloat16, device="cuda"),
        )
        x = (torch.arange(64, device="cuda").reshape(8, 8) + self.group.rank()).bfloat16()
        expected = x.float() * TP + sum(range(TP)) - self.group.rank() * TP
        start = sum(splits[: self.group.rank()])
        expected = expected[start : start + splits[self.group.rank()]].bfloat16()
        x.requires_grad_()
        with set_batch_invariant_mode(True, backend="te_native"):
            out = MoEAlltoAllTokenDispatcher.combine_preprocess(dispatcher, x)
            out.backward(torch.ones_like(out))
        assert torch.equal(out, expected)
        assert torch.equal(x.grad, torch.ones_like(x))

    @pytest.mark.parametrize("inference", [False, True])
    def test_moe_allgather_combine_keeps_stock_reduction(self, inference, monkeypatch):
        """Training TPxEP and inference CUDA-graph combines never opt into TP ordering."""
        monkeypatch.setattr(mappings, "ordered_reduce_scatter", _unexpected_ordered_reduce)
        x = _partials((8, 256), torch.bfloat16, seed=82)
        expected = torch.empty_like(x[:2])
        dist.reduce_scatter_tensor(expected, x, group=self.group)
        if inference:
            dispatcher = object.__new__(NCCLAllGatherDispatcher)
            dispatcher.ep_size = TP
            dispatcher.tp_ep_group = self.group
            monkeypatch.setattr(NCCLAllGatherDispatcher, "_use_allgather_v", False)
            combine = dispatcher.token_combine
        else:
            dispatcher = SimpleNamespace(
                tp_size=1, ep_size=TP, tp_ep_group=self.group, local_probs=x
            )
            combine = lambda x: MoEAllGatherTokenDispatcher.token_combine(dispatcher, x)
        with set_batch_invariant_mode(True, backend="te_native"):
            out = combine(x)
            if inference:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    combine(x)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = combine(x)
                graph.replay()
            else:
                x.requires_grad_()
                out = combine(x)
                out.sum().backward()
                assert torch.equal(x.grad, torch.ones_like(x))
        assert torch.equal(_bits(out), _bits(expected))

    @pytest.mark.parametrize("sequence_parallel", [False, True])
    @pytest.mark.parametrize("caller_reduced", [False, True])
    @pytest.mark.parametrize("batch_invariant", [False, True])
    def test_row_linear_call_sites_opt_into_ordered_reduction(
        self, sequence_parallel, caller_reduced, batch_invariant, monkeypatch
    ):
        if not batch_invariant:
            monkeypatch.setattr(mappings, "ordered_all_reduce", _unexpected_ordered_reduce)
            monkeypatch.setattr(mappings, "ordered_reduce_scatter", _unexpected_ordered_reduce)
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=512,
            num_attention_heads=8,
            tensor_model_parallel_size=TP,
            sequence_parallel=sequence_parallel,
            params_dtype=torch.bfloat16,
            batch_invariant_mode=batch_invariant,
            batch_invariant_backend="te_native",
            attention_backend=AttnBackend.flash,
            flash_attention_version=4,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        layer = RowParallelLinear(
            512,
            256,
            config=config,
            init_method=init_method_normal(0.02),
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            tp_group=self.group,
        )
        x = _partials((8, 1, 128), torch.bfloat16, seed=83)
        with set_batch_invariant_mode(batch_invariant, backend="te_native"):
            partial = torch.nn.functional.linear(x, layer.weight)
            if batch_invariant:
                expected = _left_fold(_all_partials(partial, self.group))
            else:
                expected = partial.clone()
                dist.all_reduce(expected, group=self.group)
            if caller_reduced:
                # Shared-expert overlap defers its FC2 row reduction to this caller.
                shared = SimpleNamespace(
                    _overlap_state=SharedExpertState.FC2_FORWARD_DONE,
                    stream=torch.cuda.current_stream(),
                    config=SimpleNamespace(
                        sequence_parallel=sequence_parallel, moe_shared_expert_overlap=True
                    ),
                    tp_group=self.group,
                    cached_fc2_output=partial,
                )
                SharedExpertMLP.post_forward_comm(shared)
                out = shared.cached_output
            else:
                out, _ = layer(x)
        if sequence_parallel:
            expected = expected.chunk(TP)[self.group.rank()]
        assert torch.equal(_bits(out), _bits(expected))

    def test_symmetric_memory_paths_match_all_to_all(self, monkeypatch):
        """The inference (symmetric-memory pull) path gives the all-to-all path's bits."""
        buf, hdl = _symm_buffer(self.group, 64 << 20)
        if hdl is None:
            pytest.skip("symmetric memory is not available")
        for dtype in (torch.bfloat16, torch.float32):
            # (rows, hidden, byte offset in the buffer). The last case is not 16-byte aligned
            # (odd slice size, tensor not at the buffer start): the unhinted path.
            for rows, hidden, off in ((4, 2688, 0), (64, 4096, 0), (512, 8192, 0), (4, 1001, 4)):
                x = _partials((rows, hidden), dtype, seed=rows + hidden)
                nbytes = x.numel() * x.element_size()
                sin = buf[off : off + nbytes].view(dtype).view(x.shape)
                sin.copy_(x)
                rs = torch.empty(rows // TP, hidden, dtype=dtype, device="cuda")
                orr.symm_ordered_reduce_scatter(sin, hdl, rs)
                ref_rs = orr.ordered_reduce_scatter(x, self.group)
                assert torch.equal(_bits(rs), _bits(ref_rs))

    def test_inference_embedding_reduce_scatter_takes_a_view(self):
        """The engine's embedding reduce-scatter gets the [b, s, h] -> [s, b, h] transpose as a
        view and copies it into the symmetric buffer itself: same bits as contiguous input."""
        from types import SimpleNamespace

        from megatron.core.tensor_parallel.inference_layers import (
            inference_reduce_scatter_to_sequence_parallel_region,
        )

        if torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("symmetric memory needs sm90+")
        SymmetricMemoryManager.get_buffer("tp", process_group=self.group)
        config = SimpleNamespace(inference_disable_triton_nvls_kernels=False)
        x = _partials((2, 32, 1024), torch.bfloat16, seed=9).transpose(0, 1)  # [s, b, h] view
        with set_batch_invariant_mode(True):
            got = inference_reduce_scatter_to_sequence_parallel_region(x, self.group, config)
            want = orr.ordered_reduce_scatter(x.contiguous(), self.group)
        assert torch.equal(_bits(got), _bits(want))

    def test_cuda_graph_capture(self):
        x = _partials((256, 4096), torch.bfloat16, seed=5)
        expected_rs = orr.ordered_reduce_scatter(x, self.group)
        expected_ar = orr.ordered_all_reduce(x, self.group)
        buf, hdl = _symm_buffer(self.group, 16 << 20)
        sin = None
        if hdl is not None:
            sin = buf[: x.numel() * 2].view(torch.bfloat16).view(x.shape)
            sin.copy_(x)
            symm_rs = torch.empty_like(expected_rs)
        static_in = x.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):  # warm up outside capture
            orr.ordered_reduce_scatter(static_in, self.group)
            orr.ordered_all_reduce(static_in, self.group)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            rs = orr.ordered_reduce_scatter(static_in, self.group)
            ar = orr.ordered_all_reduce(static_in, self.group)
            if sin is not None:
                orr.symm_ordered_reduce_scatter(sin, hdl, symm_rs)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_bits(rs), _bits(expected_rs))
        assert torch.equal(_bits(ar), _bits(expected_ar))
        if sin is not None:
            assert torch.equal(_bits(symm_rs), _bits(expected_rs))


@pytest.mark.skipif(Utils.world_size < 2, reason="needs 2 GPUs")
class TestOrderedReduceTP2Unchanged:
    def setup_method(self, method):
        Utils.initialize_model_parallel(2, 1)
        self.group = get_tensor_model_parallel_group_if_none(tp_group=None)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_tp2_uses_stock_collectives(self, monkeypatch):
        monkeypatch.setattr(mappings, "ordered_all_reduce", _unexpected_ordered_reduce)
        monkeypatch.setattr(mappings, "ordered_reduce_scatter", _unexpected_ordered_reduce)
        x = _partials((64, 4096), torch.bfloat16, seed=17)
        stock_ar = x.clone()
        dist.all_reduce(stock_ar, group=self.group)
        stock_rs = torch.empty(32, 4096, dtype=torch.bfloat16, device="cuda")
        dist.reduce_scatter_tensor(stock_rs, x, group=self.group)
        with set_batch_invariant_mode(True):
            ordered = orr.use_ordered_tp_reduce(self.group, x.dtype)
            assert not ordered
            ar = mappings.reduce_from_tensor_model_parallel_region(
                x.clone(), self.group, ordered=ordered
            )
            rs = mappings.reduce_scatter_to_sequence_parallel_region(x, self.group, ordered=ordered)
        assert torch.equal(_bits(ar), _bits(stock_ar))
        assert torch.equal(_bits(rs), _bits(stock_rs))


@requires_tp4
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_te_row_parallel_linear_ordered_reduction(sequence_parallel):
    """TERowParallelLinear at TP=4 under batch-invariant mode reduces with the ordered sum."""
    Utils.initialize_model_parallel(TP, 1)
    try:
        _check_te_row_parallel_linear(sequence_parallel)
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()


def _check_te_row_parallel_linear(sequence_parallel):
    import transformer_engine as te

    model_parallel_cuda_manual_seed(123)
    group = get_tensor_model_parallel_group_if_none(tp_group=None)
    cfg = TransformerConfig(
        num_layers=1,
        hidden_size=1024,
        num_attention_heads=8,
        tensor_model_parallel_size=TP,
        sequence_parallel=sequence_parallel,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        batch_invariant_mode=True,
        flash_attention_version=3,
        params_dtype=torch.bfloat16,
        normalization="RMSNorm",
        attention_backend=AttnBackend.flash,
    )
    layer = TERowParallelLinear(
        input_size=cfg.hidden_size,
        output_size=768,
        config=cfg,
        init_method=init_method_normal(0.02),
        bias=True,
        input_is_parallel=True,
        skip_bias_add=False,
        is_expert=False,
    ).cuda()
    assert layer._bi_tp_forward is _BITPForward.ORDERED_REDUCE
    assert layer.weight.shape == (768, cfg.hidden_size // TP)
    assert layer.weight.tensor_model_parallel and layer.weight.partition_dim == 1
    assert layer.bias.sequence_parallel == sequence_parallel

    with set_batch_invariant_mode(True, backend="te_native"):
        x = _partials((64, 2, cfg.hidden_size // TP), torch.bfloat16, seed=21)
        key, _ = orl._training_buffer_spec(group, 64 * 2 * 768 * x.element_size())
        assert not SymmetricMemoryManager.is_initialized(key)
        layer(x)
        assert SymmetricMemoryManager.is_initialized(key)
        scratch = SymmetricMemoryManager.get_buffer(key).maybe_get_tensor(
            (1, *x.shape[:-1], 768), dtype=x.dtype
        )
        if scratch["handle"] is not None:
            scratch["tensor"].fill_(float("nan"))
        out, out_bias = layer(x)
        assert out_bias is None
        with torch.no_grad():
            partial, bias = te.pytorch.Linear.forward(layer, x)
        if scratch["handle"] is not None:
            assert torch.equal(_bits(scratch["tensor"][0]), _bits(partial))
        expected = _left_fold(_all_partials(partial, group)) + bias
        if sequence_parallel:
            out = mappings._gather_along_first_dim(out.detach(), group)
        assert torch.equal(_bits(out.detach()), _bits(expected))

        # Same tokens in a smaller batch at a different position: same bits.
        x_small = x[8:16].contiguous()
        out_small, _ = layer(x_small)
        if sequence_parallel:
            out_small = mappings._gather_along_first_dim(out_small.detach(), group)
        assert torch.equal(_bits(out_small.detach()), _bits(expected[8:16]))

        # Scratch reuse must not change outstanding outputs or saved backward inputs.
        x.requires_grad_()
        actual, _ = layer(x)
        snapshot = actual.detach().clone()
        layer(x.detach() * 2)
        assert torch.equal(_bits(actual.detach()), _bits(snapshot))
        dy = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, (x, layer.weight, layer.bias), dy)
        partial, bias = te.pytorch.Linear.forward(layer, x)
        reference = (
            mappings.reduce_scatter_to_sequence_parallel_region(partial, group=group, ordered=True)
            if sequence_parallel
            else mappings.reduce_from_tensor_model_parallel_region(partial, group, ordered=True)
        ) + bias
        reference_grads = torch.autograd.grad(reference, (x, layer.weight, layer.bias), dy)
        for got, expected_grad in zip(actual_grads, reference_grads):
            torch.testing.assert_close(got, expected_grad, rtol=0, atol=0)

        with torch.no_grad():
            static_x = x.detach().clone()
            eager, _ = layer(static_x)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                layer(static_x)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured, _ = layer(static_x)
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(_bits(captured), _bits(eager))

        # A caller that reduces itself (shared-expert overlap) gets this rank's partial.
        no_bias = TERowParallelLinear(
            input_size=cfg.hidden_size,
            output_size=768,
            config=cfg,
            init_method=init_method_normal(0.02),
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
        ).cuda()
        no_bias.bi_tp_reduce_by_caller = True
        with torch.no_grad():
            local, _ = no_bias(x)
            expected_local = te.pytorch.Linear.forward(no_bias, x)
        assert torch.equal(_bits(local), _bits(expected_local))


@requires_tp4
def test_inference_row_linear_uses_only_inference_scratch():
    """An inference layer must not reserve the separate training pool."""
    Utils.initialize_model_parallel(TP, 1)
    try:
        group = get_tensor_model_parallel_group_if_none(tp_group=None)
        cfg = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=8,
            tensor_model_parallel_size=TP,
            sequence_parallel=True,
            params_dtype=torch.bfloat16,
            batch_invariant_mode=True,
            attention_backend=AttnBackend.flash,
            flash_attention_version=4,
            attention_dropout=0.0,
        )
        layer = (
            InferenceRowParallelLinear(
                256,
                256,
                config=cfg,
                init_method=init_method_normal(0.02),
                bias=False,
                input_is_parallel=True,
                skip_bias_add=True,
                is_expert=False,
                tp_group=group,
            )
            .cuda()
            .eval()
        )
        x = _partials((32, 1, 64), torch.bfloat16, seed=17)
        with set_batch_invariant_mode(True, backend="te_native"), torch.no_grad():
            out, _ = layer(x)
            expected = _left_fold(_all_partials(x @ layer.weight.t(), group))
        expected = expected[group.rank() * 8 : (group.rank() + 1) * 8]
        assert torch.equal(_bits(out), _bits(expected))
        assert SymmetricMemoryManager.is_initialized("tp")
        assert not any(
            key.startswith("tp_ordered_training:") for key in SymmetricMemoryManager._buffers
        )
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()


@requires_tp4
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_training_scratch_grows_an_undersized_buffer(dtype):
    """An undersized existing pool grows instead of routing the GEMM through all-to-all."""
    Utils.initialize_model_parallel(TP, 1)
    try:
        group = get_tensor_model_parallel_group_if_none(tp_group=None)
        x = _partials((2048, 128), dtype, seed=51)
        weight = _partials((512, 128), dtype, seed=52)
        key, _ = orl._training_buffer_spec(group, 2048 * 512 * x.element_size())
        SymmetricMemoryManager.get_buffer(key, process_group=group, size_mb=1)
        with mock.patch.object(
            orl, "ordered_reduce_scatter_shards", side_effect=AssertionError("unexpected fallback")
        ):
            out = orl.tp_shard_row_linear(x, weight, group, 1, True, use_te_gemm=True)
        assert (
            SymmetricMemoryManager.get_buffer(key).symm_buffer.numel()
            >= 2048 * 512 * x.element_size()
        )
        partial = orl.te_cpp_extensions.general_gemm(weight, x, out_dtype=x.dtype)[0]
        expected = orr.ordered_reduce_scatter(partial, group)
        assert torch.equal(_bits(out), _bits(expected))
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()


@requires_tp4
@pytest.mark.parametrize(
    "tp,peer_tp,hidden,mtp_layers,batch_invariant,fuse_tp",
    [
        pytest.param(4, 2, 4096, 1, True, False, id="bi-mtp"),
        pytest.param(2, 4, 2048, None, True, False, id="bi-multiple-shards"),
        pytest.param(4, None, 4096, None, True, False, id="bi-default-peer"),
        pytest.param(4, None, 4096, None, False, False, id="regular"),
        pytest.param(4, None, 4096, 1, False, True, id="regular-fused-mtp"),
    ],
)
def test_inference_scratch_covers_max_tokens_and_preserves_existing_graphs(
    tp, peer_tp, hidden, mtp_layers, batch_invariant, fuse_tp
):
    """A small captured step and >512MiB eager prefill use valid inference-owned pools."""
    Utils.initialize_model_parallel(tp, 1)
    try:
        group = get_tensor_model_parallel_group_if_none(tp_group=None)
        cfg = TransformerConfig(
            num_layers=1,
            hidden_size=hidden,
            num_attention_heads=8,
            tensor_model_parallel_size=tp,
            sequence_parallel=True,
            params_dtype=torch.bfloat16,
            batch_invariant_mode=batch_invariant,
            batch_invariant_peer_tp_size=peer_tp,
            transformer_impl="inference_optimized",
            normalization="RMSNorm",
            add_bias_linear=False,
            mtp_num_layers=mtp_layers,
            inference_fuse_tp_communication=fuse_tp,
            attention_backend=AttnBackend.flash,
            flash_attention_version=4,
            attention_dropout=0.0,
        )
        layer = (
            InferenceRowParallelLinear(
                256,
                hidden,
                config=cfg,
                init_method=init_method_normal(0.02),
                bias=False,
                input_is_parallel=True,
                skip_bias_add=True,
                is_expert=False,
                tp_group=group,
            )
            .cuda()
            .eval()
        )
        layer.weight.data.fill_(1)
        if fuse_tp:
            layer._set_next_layer_norm_weights(
                torch.ones(hidden, device="cuda", dtype=cfg.params_dtype)
            )

        def forward(x):
            if fuse_tp:
                layer._set_residual(
                    torch.zeros(
                        (x.shape[0] // tp, 1, hidden), device="cuda", dtype=cfg.params_dtype
                    )
                )
            return layer(x)

        small = torch.full(
            (256, 1, 256 // tp), group.rank() + 1, device="cuda", dtype=cfg.params_dtype
        )
        SymmetricMemoryManager.get_buffer("tp", process_group=group, size_mb=8)
        with set_batch_invariant_mode(batch_invariant, backend="te_native"), torch.no_grad():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                forward(small)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured, _ = forward(small)
            graph.replay()
            expected = (256 // tp) * sum(range(1, tp + 1))
            assert (captured == expected).all()
            old = SymmetricMemoryManager.get_buffer("tp")
            # The actual context must reserve its eager ceiling, not its small graph budget.
            context = DynamicInferenceContext(
                model_config=cfg,
                inference_config=InferenceConfig(
                    max_tokens=65792,
                    max_requests=16,
                    max_sequence_length=131072,
                    cuda_graph_max_tokens=256,
                    num_cuda_graphs=2,
                    buffer_size_gb=0.5,
                    block_size_tokens=128,
                    unified_memory_level=0,
                ),
            )
            buf = SymmetricMemoryManager.get_buffer("tp")
            required = (cfg.batch_invariant_tp_shards or tp) // tp * context.max_tokens * hidden * 2
            assert required > (512 << 20)
            assert buf.symm_buffer.numel() >= required
            if mtp_layers:
                assert buf.symm_buffer.numel() >= context.max_tokens * 2 * hidden * 2
            assert buf is not old
            address = buf.symm_buffer.data_ptr()
            large = torch.full(
                (context.max_tokens, 1, 256 // tp),
                2 * (group.rank() + 1),
                device="cuda",
                dtype=cfg.params_dtype,
            )
            with (
                mock.patch.object(
                    inference_layers,
                    "ordered_reduce_scatter_shards",
                    side_effect=AssertionError("unexpected ordered inference fallback"),
                ),
                mock.patch.object(
                    inference_layers,
                    "reduce_scatter_along_first_dim",
                    side_effect=AssertionError("unexpected inference NCCL fallback"),
                ),
            ):
                result, _ = forward(large)
            assert (result == 2 * expected).all()
            if fuse_tp:
                normalized = buf.maybe_get_tensor(
                    (context.max_tokens, 1, hidden), cfg.params_dtype
                )["tensor"]
                assert (normalized == 1).all()
            if mtp_layers:
                projection = (
                    inference_layers.InferenceColumnParallelLinear(
                        2 * hidden,
                        hidden,
                        config=cfg,
                        init_method=init_method_normal(0.02),
                        gather_output=False,
                        bias=False,
                        skip_bias_add=True,
                        is_expert=False,
                        tp_group=group,
                    )
                    .cuda()
                    .eval()
                )
                projection.set_barrier_before_all_gather(True)
                projection.weight.data.fill_(1)
                joined = torch.full(
                    (context.max_tokens // tp, 1, 2 * hidden),
                    0.25,
                    device="cuda",
                    dtype=cfg.params_dtype,
                )
                with mock.patch.object(
                    inference_layers,
                    "gather_along_first_dim",
                    side_effect=AssertionError("unexpected MTP all-gather fallback"),
                ):
                    projected, _ = projection(joined)
                assert (projected == hidden / 2).all()
            assert SymmetricMemoryManager.get_buffer("tp").symm_buffer.data_ptr() == address
            graph.replay()
            assert (captured == expected).all()
            assert not any(
                key.startswith("tp_ordered_training:") for key in SymmetricMemoryManager._buffers
            )
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()


@requires_tp4
def test_inference_scratch_not_reserved_with_disabled_tp_kernels():
    """Disabling symmetric TP kernels must not reserve their unused scratch pool."""
    Utils.initialize_model_parallel(TP, 1)
    try:
        cfg = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=8,
            tensor_model_parallel_size=TP,
            sequence_parallel=True,
            params_dtype=torch.bfloat16,
            transformer_impl="inference_optimized",
            normalization="RMSNorm",
            add_bias_linear=False,
            inference_disable_triton_nvls_kernels=True,
        )
        DynamicInferenceContext(
            model_config=cfg,
            inference_config=InferenceConfig(
                max_tokens=256,
                max_requests=4,
                max_sequence_length=256,
                num_cuda_graphs=2,
                buffer_size_gb=0.01,
                block_size_tokens=128,
                unified_memory_level=0,
            ),
        )
        assert not SymmetricMemoryManager.is_initialized("tp")
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()


@requires_tp4
@pytest.mark.parametrize("tp,local_shards,hidden", [(4, 1, 4096), (2, 2, 2048)])
def test_training_scratch_grows_past_512_mib_without_invalidating_graphs(tp, local_shards, hidden):
    """Size all shard outputs, preserving an earlier graph's smaller allocation."""
    Utils.initialize_model_parallel(tp, 1)
    try:
        group = get_tensor_model_parallel_group_if_none(tp_group=None)
        weight = torch.ones(hidden, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        small = torch.full((32, 64), group.rank() + 1, device="cuda", dtype=weight.dtype)
        use_te_gemm = local_shards == 1
        with set_batch_invariant_mode(True, backend="te_native"):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                orl.tp_shard_row_linear(
                    small, weight, group, local_shards, True, use_te_gemm=use_te_gemm
                )
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = orl.tp_shard_row_linear(
                    small, weight, group, local_shards, True, use_te_gemm=use_te_gemm
                )
            graph.replay()
            expected = 64 * sum(range(1, tp + 1))
            assert (captured == expected).all()
            # 512 MiB plus 32 KiB of local shard outputs, before reduce-scatter.
            x = torch.full(
                (65540, 64), group.rank() + 1, device="cuda", dtype=weight.dtype, requires_grad=True
            )
            out = orl.tp_shard_row_linear(
                x, weight, group, local_shards, True, use_te_gemm=use_te_gemm
            )
            nbytes = local_shards * 65540 * hidden * x.element_size()
            key, _ = orl._training_buffer_spec(group, nbytes)
            buf = SymmetricMemoryManager.get_buffer(key)
            assert buf.symm_mem_hdl is not None, buf.init_failure_reason
            assert buf.symm_buffer.numel() >= nbytes
            assert (out == expected).all()
            out.sum().backward()
            assert (x.grad == hidden).all()
            expected_wgrad = torch.tensor(
                65540 * (group.rank() + 1), device="cuda", dtype=weight.dtype
            )
            assert (weight.grad == expected_wgrad).all()
            graph.replay()
            assert (captured == expected).all()
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()


@requires_tp4
def test_tp3_symmetric_memory_paths():
    """TP = 3 (not a power of two): all-to-all and pull give the left fold."""

    Utils.initialize_model_parallel(1, 1)
    world = dist.get_world_size()
    group = dist.new_group([0, 1, 2])  # every rank must create both groups
    rest = dist.new_group(list(range(3, world)))
    try:
        if dist.get_rank() >= 3:
            # Symmetric-memory rendezvous counters are per process, not per group: the other
            # ranks make the same allocation so every rank's counter stays in step and a
            # later rendezvous does not hang.
            SymmetricMemoryManager.get_buffer("tp3_test", process_group=rest)
            return
        x = _partials((96, 1024), torch.bfloat16, seed=31)
        rank = group.rank()
        ref = _left_fold(_all_partials(x, group))[rank * 32 : (rank + 1) * 32]
        assert torch.equal(_bits(orr.ordered_reduce_scatter(x, group)), _bits(ref))

        symm = SymmetricMemoryManager.get_buffer("tp3_test", process_group=group)
        buf = symm.maybe_get_tensor(list(x.shape), dtype=x.dtype)
        assert buf["handle"] is not None
        buf["tensor"].copy_(x)
        out = orr.symm_ordered_reduce_scatter(buf["tensor"], buf["handle"])
        assert torch.equal(_bits(out), _bits(ref))
    finally:
        torch.cuda.synchronize()
        # Peers may still read this rank's buffer until the group syncs.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()


def test_fold_offsets_past_int32():
    """Partials more than 2**31 elements apart (large all-to-all receive buffers)."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    n = 4096
    stride = 1 << 31
    storage = torch.zeros(stride + n, dtype=torch.bfloat16, device="cuda")
    parts = storage.as_strided((2, n), (stride, 1))
    parts[0].copy_(_partials((n,), torch.bfloat16, seed=41))
    parts[1].copy_(_partials((n,), torch.bfloat16, seed=42))
    out = orr.ordered_fold(parts)
    assert torch.equal(_bits(out), _bits(_left_fold(parts.clone())))


@requires_tp4
@pytest.mark.parametrize("backend", ["te_native", "triton"])
def test_direct_gemm_fused_main_grad_matches_te(backend):
    """Direct GEMMs accumulate two microbatches into an existing main_grad like TE."""
    import transformer_engine as te

    Utils.initialize_model_parallel(TP, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        group = get_tensor_model_parallel_group_if_none(tp_group=None)
        cfg = TransformerConfig(
            num_layers=1,
            hidden_size=512,
            num_attention_heads=8,
            tensor_model_parallel_size=TP,
            sequence_parallel=True,
            batch_invariant_mode=True,
            params_dtype=torch.bfloat16,
            gradient_accumulation_fusion=True,
            attention_backend=AttnBackend.flash,
            flash_attention_version=4,
            hidden_dropout=0,
            attention_dropout=0,
        )
        layer = TERowParallelLinear(
            512,
            512,
            config=cfg,
            init_method=init_method_normal(0.02),
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
        ).cuda()
        reference = te.pytorch.Linear(
            128,
            512,
            bias=False,
            params_dtype=torch.bfloat16,
            parallel_mode=None,
            fuse_wgrad_accumulation=True,
        ).cuda()
        with torch.no_grad():
            reference.weight.copy_(layer.weight)
        for weight in (layer.weight, reference.weight):
            weight.main_grad = torch.full_like(weight, 0.75, dtype=torch.float32)
            weight.grad_added_to_main_grad = False
            weight.zero_out_wgrad = True
        with set_batch_invariant_mode(True, backend=backend):
            for seed in (71, 72):
                x = _partials((32, 1, 128), torch.bfloat16, seed).requires_grad_()
                xr = x.detach().clone().requires_grad_()
                got = layer(x)[0]
                want = mappings.reduce_scatter_to_sequence_parallel_region(
                    reference(xr), group=group, ordered=True
                )
                dy = _partials(got.shape, got.dtype, seed + 1)
                got.backward(dy)
                want.backward(dy)
                assert torch.equal(_bits(got), _bits(want))
                assert torch.equal(_bits(x.grad), _bits(xr.grad))
                assert torch.equal(_bits(layer.weight.main_grad), _bits(reference.weight.main_grad))
                assert layer.weight.grad_added_to_main_grad
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()


@requires_tp4
@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("sequence_parallel", [False, True], ids=["all_reduce", "reduce_scatter"])
def test_mxfp8_ordered_linear_matches_te_and_batch_layout(sequence_parallel):
    """MXFP8 keeps TE's quantized gradients and a row's bits across batch layouts."""
    import transformer_engine as te
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    available, reason = FP8GlobalStateManager.is_mxfp8_available()
    if not available:
        pytest.skip(reason)
    Utils.initialize_model_parallel(TP, 1)
    try:
        group = get_tensor_model_parallel_group_if_none(tp_group=None)
        model_parallel_cuda_manual_seed(123)
        torch.manual_seed(100 + group.rank())
        config = TransformerConfig(
            num_layers=1,
            hidden_size=1024,
            num_attention_heads=8,
            tensor_model_parallel_size=TP,
            sequence_parallel=sequence_parallel,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            batch_invariant_mode=True,
            batch_invariant_backend="te_native",
            flash_attention_version=4,
            bf16=True,
            params_dtype=torch.bfloat16,
            fp8="e4m3",
            fp8_recipe=Fp8Recipe.mxfp8,
            attention_backend=AttnBackend.flash,
        )
        module = TERowParallelLinear(
            1024,
            512,
            config=config,
            init_method=init_method_normal(0.02),
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
            tp_group=group,
        ).cuda()
        reference = te.pytorch.Linear(256, 512, bias=False, params_dtype=torch.bfloat16).cuda()
        with torch.no_grad():
            reference.weight.copy_(module.weight)
        x = torch.randn(128, 1, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_(True)
        with set_batch_invariant_mode(True, backend="te_native"), get_fp8_context(config):
            output, _ = module(x)
            assert module.fp8
            partial = reference(reference_x)
            reduce = orr.ordered_reduce_scatter if sequence_parallel else orr.ordered_all_reduce
            assert torch.equal(_bits(output), _bits(reduce(partial.detach(), group)))
            grad = torch.randn_like(output)
            reference_grad = grad
            if sequence_parallel:
                reference_grad = torch.empty_like(partial)
                dist.all_gather_into_tensor(reference_grad, grad, group=group)
            output.backward(grad)
            partial.backward(reference_grad)
            assert torch.equal(_bits(x.grad), _bits(reference_x.grad))
            assert torch.equal(_bits(module.weight.grad), _bits(reference.weight.grad))
        with (
            torch.no_grad(),
            set_batch_invariant_mode(True, backend="te_native"),
            get_fp8_context(config),
        ):
            batch = x.detach().clone()
            batch[32:] *= 64
            outputs = []
            for inputs in (batch[:32].contiguous(), batch):
                output, _ = module(inputs)
                if sequence_parallel:
                    full = torch.empty((inputs.shape[0], 1, 512), device="cuda", dtype=output.dtype)
                    dist.all_gather_into_tensor(full, output, group=group)
                    output = full
                outputs.append(output)
            assert torch.equal(_bits(outputs[0]), _bits(outputs[1][:32]))
        module.eval()
        reference.eval()
        prepare_model_for_fp8_inference(module)
        with (
            torch.no_grad(),
            set_batch_invariant_mode(True, backend="te_native"),
            get_fp8_context(config),
        ):
            for length in (12, 32):
                inputs = x[:length].detach().contiguous()
                padded = torch.nn.functional.pad(inputs, (0, 0, 0, 0, 0, 32 - length))
                partial = reference(padded)[:length].contiguous()
                reduce = orr.ordered_reduce_scatter if sequence_parallel else orr.ordered_all_reduce
                expected = reduce(partial, group)
                actual, _ = module(inputs)
                assert actual.shape == expected.shape
                assert torch.equal(_bits(actual), _bits(expected))
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()
