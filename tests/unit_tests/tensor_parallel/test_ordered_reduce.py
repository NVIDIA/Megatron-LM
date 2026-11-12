# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Batch-invariant ordered TP reductions (megatron/core/ops/collectives/ordered_reduce.py)."""

import pytest
import torch
import torch.distributed as dist

from megatron.core.extensions.transformer_engine import TERowParallelLinear, _BITPForward
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.ops.collectives import ordered_reduce as orr
from megatron.core.ops.linear import ordered as orl
from megatron.core.tensor_parallel import mappings
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode
from megatron.core.transformer.enums import AttnBackend
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

    def test_mappings_forward_is_ordered_and_position_invariant(self):
        """A row gets the same bits wherever it sits and whatever the batch size."""
        hidden = 4096
        row = _partials((1, hidden), torch.bfloat16, seed=11)
        expected = _left_fold(_all_partials(row, self.group))[0]
        with set_batch_invariant_mode(True):
            assert orr.use_ordered_tp_reduce(self.group, torch.bfloat16)
            for rows in (4, 8, 64, 512):
                for pos in (0, rows // 2 + 1, rows - 1):
                    x = _partials((rows, hidden), torch.bfloat16, seed=rows + pos)
                    x[pos] = row[0]
                    # All-reduce (non-sequence-parallel row-parallel linear).
                    ar = mappings.reduce_from_tensor_model_parallel_region(x.clone(), self.group)
                    assert torch.equal(_bits(ar[pos]), _bits(expected))
                    # Reduce-scatter (sequence parallel); gather to find the row again.
                    rs = mappings.reduce_scatter_to_sequence_parallel_region(x, self.group)
                    full = mappings._gather_along_first_dim(rs, self.group)
                    assert torch.equal(_bits(full[pos]), _bits(expected))

    def test_backward_uses_stock_collectives(self):
        """Backward is unchanged: RS backward is an all-gather, AR backward the identity."""
        with set_batch_invariant_mode(True):
            x = _partials((8, 256), torch.float32, seed=3).requires_grad_()
            mappings.reduce_scatter_to_sequence_parallel_region(x, self.group).sum().backward()
            assert torch.equal(x.grad, torch.ones_like(x))
            y = _partials((8, 256), torch.float32, seed=4).requires_grad_()
            mappings.reduce_from_tensor_model_parallel_region(y, self.group).sum().backward()
            assert torch.equal(y.grad, torch.ones_like(y))

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

    def test_tp2_uses_stock_collectives(self):
        x = _partials((64, 4096), torch.bfloat16, seed=17)
        stock_ar = x.clone()
        dist.all_reduce(stock_ar, group=self.group)
        stock_rs = torch.empty(32, 4096, dtype=torch.bfloat16, device="cuda")
        dist.reduce_scatter_tensor(stock_rs, x, group=self.group)
        with set_batch_invariant_mode(True):
            assert not orr.use_ordered_tp_reduce(self.group, torch.bfloat16)
            ar = mappings.reduce_from_tensor_model_parallel_region(x.clone(), self.group)
            rs = mappings.reduce_scatter_to_sequence_parallel_region(x, self.group)
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
    # The shared GEMM scratch is allocated at construction, outside any forward or graph capture.
    assert SymmetricMemoryManager.is_initialized(orl._training_buffer_key(layer._tp_group))
    assert layer.weight.shape == (768, cfg.hidden_size // TP)
    assert layer.weight.tensor_model_parallel and layer.weight.partition_dim == 1
    assert layer.bias.sequence_parallel == sequence_parallel

    with set_batch_invariant_mode(True, backend="te_native"):
        x = _partials((64, 2, cfg.hidden_size // TP), torch.bfloat16, seed=21)
        scratch = SymmetricMemoryManager.get_buffer(
            orl._training_buffer_key(group)
        ).maybe_get_tensor((1, *x.shape[:-1], 768), dtype=x.dtype)
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
            mappings.reduce_scatter_to_sequence_parallel_region(partial, group=group)
            if sequence_parallel
            else mappings.reduce_from_tensor_model_parallel_region(partial, group)
        ) + bias
        reference_grads = torch.autograd.grad(reference, (x, layer.weight, layer.bias), dy)
        for got, expected_grad in zip(actual_grads, reference_grads):
            torch.testing.assert_close(got, expected_grad, rtol=0, atol=0)

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
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_direct_gemm_falls_back_when_scratch_does_not_fit(dtype):
    """A larger eager/captured input keeps the same bits through all-to-all."""
    from transformer_engine.pytorch.cpp_extensions import general_gemm

    Utils.initialize_model_parallel(TP, 1)
    try:
        group = get_tensor_model_parallel_group_if_none(tp_group=None)
        SymmetricMemoryManager.get_buffer(
            orl._training_buffer_key(group), process_group=group, size_mb=1
        )
        x = _partials((2048, 128), dtype, seed=51)
        weight = _partials((512, 128), dtype, seed=52)
        out = orl.tp_shard_row_linear(x, weight, group, 1, True, use_te_gemm=True)
        partial = general_gemm(weight, x, out_dtype=x.dtype)[0]
        expected = orr.ordered_reduce_scatter(partial, group)
        assert torch.equal(_bits(out), _bits(expected))
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
                    reference(xr), group=group
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
