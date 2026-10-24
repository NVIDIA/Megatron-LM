# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Batch-invariant ordered TP reductions (megatron/core/tensor_parallel/ordered_reduce.py)."""

import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel import mappings
from megatron.core.tensor_parallel import ordered_reduce as orr
from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode
from megatron.core.utils import get_tensor_model_parallel_group_if_none
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


def _left_fold(parts, exact=False):
    """Reference: ((p_0 + p_1) + p_2) + ... in fp32, one rounding to the input dtype.

    exact=True ("multimem" collective): fp64 accumulation, one rounding to fp32, then the cast.
    """
    acc = parts[0].double() if exact else parts[0].float()
    for r in range(1, parts.shape[0]):
        acc = acc + (parts[r].double() if exact else parts[r].float())
    return acc.float().to(parts.dtype)


def _partials(shape, dtype, seed):
    """Per-rank partials with mixed magnitudes, so summation order changes the bits."""
    gen = torch.Generator(device="cuda").manual_seed(seed + 1000 * Utils.rank)
    x = torch.randn(shape, device="cuda", generator=gen, dtype=torch.float32)
    scale = torch.logspace(-4, 4, shape[-1], device="cuda")
    return (x * scale).to(dtype)


def _symm_buffer(group, nbytes):
    """A symmetric-memory buffer for group, or None when unsupported here."""
    try:
        import torch.distributed._symmetric_memory as symm_mem

        from megatron.core.inference.symmetric_memory import _ensure_signal_pad_size

        if torch.cuda.get_device_properties(0).major < 9:
            return None, None
        symm_mem.enable_symm_mem_for_group(group.group_name)
        _ensure_signal_pad_size(group.size())
        buf = symm_mem.empty(nbytes, dtype=torch.uint8, device="cuda")
        return buf, symm_mem.rendezvous(buf, group)
    except Exception:  # pragma: no cover - platform without symmetric memory
        return None, None


@requires_tp4
class TestOrderedReduce:
    def setup_method(self, method):
        Utils.initialize_model_parallel(TP, 1)
        self.group = get_tensor_model_parallel_group_if_none(tp_group=None)

    def teardown_method(self, method):
        from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

        torch.cuda.synchronize()
        SymmetricMemoryManager.destroy()
        orr._PEER_VIEWS.clear()
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("exact", [False, True])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("rows,hidden", [(4, 2688), (64, 4096), (1024, 8192)])
    def test_reduce_scatter_matches_left_fold(self, dtype, rows, hidden, exact):
        x = _partials((rows, hidden), dtype, seed=rows)
        ref = _left_fold(_all_partials(x, self.group), exact)
        out = orr.ordered_reduce_scatter(x, self.group, exact=exact)
        k = rows // TP
        rank = self.group.rank()
        assert torch.equal(_bits(out), _bits(ref[rank * k : (rank + 1) * k]))

    def test_reduce_scatter_uneven_splits(self):
        splits = [3, 5, 1, 7]
        x = _partials((sum(splits), 512), torch.bfloat16, seed=7)
        ref = _left_fold(_all_partials(x, self.group))
        out = orr.ordered_reduce_scatter(x, self.group, input_split_sizes=splits)
        rank = self.group.rank()
        start = sum(splits[:rank])
        assert torch.equal(_bits(out), _bits(ref[start : start + splits[rank]]))

    @pytest.mark.parametrize("exact", [False, True])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("shape", [(1, 2688), (3, 1, 4096), (7, 100), (128, 8192)])
    def test_all_reduce_matches_left_fold(self, dtype, shape, exact):
        x = _partials(shape, dtype, seed=len(shape))
        ref = _left_fold(_all_partials(x, self.group), exact)
        x_before = x.clone()
        out = orr.ordered_all_reduce(x, self.group, exact=exact)
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

    @pytest.mark.parametrize("exact", [False, True])
    def test_symmetric_memory_paths_match_all_to_all(self, exact):
        """The inference (symmetric-memory pull) paths give the all-to-all path's bits."""
        buf, hdl = _symm_buffer(self.group, 64 << 20)
        if hdl is None:
            pytest.skip("symmetric memory is not available")
        for dtype in (torch.bfloat16, torch.float32):
            for rows, hidden in ((4, 2688), (64, 4096), (512, 8192)):
                x = _partials((rows, hidden), dtype, seed=rows + hidden)
                sin = buf[: x.numel() * x.element_size()].view(dtype).view(x.shape)
                sin.copy_(x)
                rs = torch.empty(rows // TP, hidden, dtype=dtype, device="cuda")
                orr.symm_ordered_reduce_scatter(rs, sin, hdl, exact=exact)
                ref_rs = orr.ordered_reduce_scatter(x, self.group, exact=exact)
                assert torch.equal(_bits(rs), _bits(ref_rs))
                ar = torch.empty_like(x)
                orr.symm_ordered_all_reduce(ar, sin, hdl, exact=exact)
                ref_ar = orr.ordered_all_reduce(x, self.group, exact=exact)
                assert torch.equal(_bits(ar), _bits(ref_ar))

    @pytest.mark.parametrize("exact", [False, True])
    def test_symmetric_push_matches_all_to_all(self, exact, monkeypatch):
        """The learner's symmetric-memory push RS gives the all-to-all path's bits.

        Sizes span one to several staging iterations per block and a ragged tail.
        """
        if orr._push_staging(self.group, torch.empty(1, device="cuda")) is None:
            pytest.skip("symmetric memory is not available")
        for dtype in (torch.bfloat16, torch.float32):
            for rows, hidden in ((4, 2688), (12, 1000), (2048, 8192), (8192, 4096)):
                x = _partials((rows, hidden), dtype, seed=rows + hidden + 1)
                push = orr.ordered_reduce_scatter(x, self.group, exact=exact)
                monkeypatch.setattr(orr, "USE_SYMM_PUSH", False)
                a2a = orr.ordered_reduce_scatter(x, self.group, exact=exact)
                monkeypatch.setattr(orr, "USE_SYMM_PUSH", True)
                assert torch.equal(_bits(push), _bits(a2a))
                ref = _left_fold(_all_partials(x, self.group), exact)
                k = rows // TP
                rank = self.group.rank()
                assert torch.equal(_bits(push), _bits(ref[rank * k : (rank + 1) * k]))

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
            symm_ar = torch.empty_like(expected_ar)
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
                orr.symm_ordered_reduce_scatter(symm_rs, sin, hdl)
                orr.symm_ordered_all_reduce(symm_ar, sin, hdl)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_bits(rs), _bits(expected_rs))
        assert torch.equal(_bits(ar), _bits(expected_ar))
        if sin is not None:
            assert torch.equal(_bits(symm_rs), _bits(expected_rs))
            assert torch.equal(_bits(symm_ar), _bits(expected_ar))

    def test_stock_nccl_is_not_position_invariant(self):
        """Informational: shows the invariance tests above are not vacuous at TP=4."""
        hidden = 4096
        row = _partials((1, hidden), torch.bfloat16, seed=13)
        results = []
        for rows, pos in ((4, 0), (4, 3), (4096, 0), (4096, 1500), (4096, 4095)):
            x = _partials((rows, hidden), torch.bfloat16, seed=rows + pos)
            x[pos] = row[0]
            dist.all_reduce(x, group=self.group)
            results.append(x[pos].clone())
        differs = any(not torch.equal(_bits(r), _bits(results[0])) for r in results[1:])
        if not differs:
            pytest.skip("informational: NCCL happened to be position-invariant on this input")


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
    import transformer_engine as te

    from megatron.core.extensions.transformer_engine import TERowParallelLinear
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.utils import init_method_normal

    Utils.initialize_model_parallel(TP, 1)
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
    assert layer._explicit_bi_tp_reduce
    assert layer.weight.shape == (768, cfg.hidden_size // TP)
    assert layer.weight.tensor_model_parallel and layer.weight.partition_dim == 1
    assert layer.bias.sequence_parallel == sequence_parallel

    with set_batch_invariant_mode(True):
        x = _partials((64, 2, cfg.hidden_size // TP), torch.bfloat16, seed=21)
        out, out_bias = layer(x)
        assert out_bias is None
        with torch.no_grad():
            partial, bias = te.pytorch.Linear.forward(layer, x)
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

        # Backward runs on the stock collectives.
        x.requires_grad_()
        layer(x)[0].float().sum().backward()
        assert x.grad is not None and layer.weight.grad is not None
    Utils.destroy_model_parallel()
