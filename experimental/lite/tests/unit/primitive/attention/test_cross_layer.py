# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Generic owner/reader delegation and pinned Core interfaces; no model config."""
import ast
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa import Compressor, CSAIndexer
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.core.utils import _validate_dsa_kernel_backend_dependencies
from megatron.lite.primitive.modules.attention import ratio_one
from megatron.lite.primitive.modules.attention.cross_layer import (
    FrozenIndexerCSA,
    IndexProjection,
    SharedCompressor,
)
from megatron.lite.primitive.parallel.source_state import KVSourceState


def _compressor(state, layer, owner, group):
    result = SharedCompressor.__new__(SharedCompressor)
    torch.nn.Module.__init__(result)
    result.compress_ratio, result.head_dim = 4, 4
    result.pg_collection = SimpleNamespace(cp=group)
    result.bind_source(state, layer, owner)
    return result


@pytest.mark.parametrize("name", ["forward", "_forward_thd_cp"])
def test_core_override_signature(name):
    def signature(fn):
        return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]

    assert signature(getattr(SharedCompressor, name)) == signature(getattr(Compressor, name))


@pytest.mark.parametrize("cp", [False, True])
def test_owner_reader_dispatch_graph_and_lifetime(cp):
    state, group = KVSourceState(), object()
    owner = _compressor(state, 0, 0, group)
    reader = _compressor(state, 1, 0, group)
    x = torch.arange(12.0).reshape(3, 1, 4).requires_grad_()
    prefix = torch.tensor([0, 3], dtype=torch.int32)
    params = SimpleNamespace(cu_seqlens_q=prefix, cu_seqlens_q_padded=prefix, max_seqlen_q=3)
    name = "_forward_thd_cp" if cp else "forward"

    def produce(self, tensor, *args, **kwargs):
        assert self is owner
        return tensor.square(), prefix

    def run(module):
        return (
            module._forward_thd_cp(x, x, prefix, SimpleNamespace(global_start=0), 3)
            if cp
            else module(x, params)
        )

    with patch.object(Compressor, name, produce):
        with state.microbatch():
            value = run(owner)
            assert run(reader) is value
            value[0].sum().backward()
            assert torch.equal(x.grad, 2 * x.detach())
        with state.microbatch(), pytest.raises(RuntimeError, match="owner"):
            run(reader)
    assert not state.values and not state.active
    with pytest.raises(RuntimeError, match="active microbatch"):
        run(reader)


def test_owner_mode_mismatch_and_failure_cleanup():
    state = KVSourceState()
    with pytest.raises(ValueError, match="abort"):
        with state.microbatch():
            state.resolve(0, 0, ("no-grad",), lambda: torch.ones(1))
            with pytest.raises(RuntimeError, match="owner"):
                state.resolve(1, 0, ("grad",), lambda: None)
            with pytest.raises(RuntimeError, match="already active"):
                with state.microbatch():
                    pass
            raise ValueError("abort")
    assert not state.active and state.values == {}


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize("loss_coeff", [0.0, 0.1])
def test_index_relay_native_topk_order_and_ties(tied, grad, loss_coeff):
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils

    class Projection(torch.nn.Linear):
        def forward(self, x):
            return super().forward(x), None

    state = KVSourceState()
    q_owner = IndexProjection(Projection(4, 4, bias=False), state, 0, 0, "index_q")
    w_owner = IndexProjection(Projection(4, 1, bias=False), state, 0, 0, "index_weights")
    q_reader = IndexProjection(None, state, 1, 0, "index_q")
    w_reader = IndexProjection(None, state, 1, 0, "index_weights")
    x = torch.arange(32.0).reshape(8, 1, 4) / 32
    cu = torch.tensor([0, 4, 8, 16], dtype=torch.int32)
    ck = torch.tensor([0, 2, 4, 8], dtype=torch.int32)
    k = torch.zeros(8, 4) if tied else torch.arange(32.0).reshape(8, 4) / 31
    with state.microbatch(), torch.set_grad_enabled(grad):
        q = q_owner(x)
        weights = w_owner(x)
        assert q_reader(-x) is q and w_reader(-x) is weights

        def select(q_value, w_value):
            args = [q_value[0].reshape(8, 1, 4), w_value[0].squeeze(1), k, cu, ck]
            return cp_utils.compute_cp_indexer_topk(
                *args, 8, 2, 4, 0.5, max_seqlen_q=8, use_fused=False
            )[0]

        selected = select(q, weights)
        assert torch.equal(selected, select(q_reader(-x), w_reader(-x)))
        from megatron.core.transformer.experimental_attention_variant.csa import (
            _unfused_indexer_sparse_attn_from_topk,
        )

        single = torch.tensor([0, 8], dtype=torch.int32)
        offsets = torch.tensor([0], dtype=torch.int32)
        ids = torch.arange(8, dtype=torch.int32)[:, None]
        query = x.reshape(8, 1, 4)

        def auxiliary(q_value, w_value):
            args = [query, k, torch.zeros(1), ids, q_value[0].reshape(8, 1, 4)]
            args += [k, w_value[0].squeeze(1), ids, k, 0.5, 0.5, loss_coeff]
            args += [8, False, 1, 8, (single, single, offsets)]
            return _unfused_indexer_sparse_attn_from_topk(*args)[1]

        owner_loss = auxiliary(q, weights)
        reader_loss = auxiliary(q_reader(-x), w_reader(-x))
        assert torch.equal(owner_loss, reader_loss)
        if grad:
            params = (q_owner.linear.weight, w_owner.linear.weight)
            once = torch.autograd.grad(owner_loss, params, retain_graph=True)
            (owner_loss + reader_loss).backward()
            for parameter, expected in zip(params, once):
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                torch.testing.assert_close(parameter.grad, 2 * expected)
                if loss_coeff == 0:
                    assert torch.count_nonzero(parameter.grad) == 0
        else:
            assert not owner_loss.requires_grad and not reader_loss.requires_grad
    assert list(q_reader.parameters()) == [] and list(w_reader.parameters()) == []


def test_native_indexer_interface_contract():
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils

    parameters = inspect.signature(cp_utils.compute_cp_indexer_topk).parameters
    assert (
        list(parameters)
        == (
            'q_indexer_local weights_indexer_local k_indexer_seq_major cu_seqlens_q '
            'cu_seqlens_compressed global_start ratio topk_width indexer_softmax_scale '
            'max_seqlen_q use_fused deterministic precision compact_workspace return_softmax '
            'indexer_layout logical_indexer_layout max_seqlen_kv prebuilt_layout '
            'synthetic_layout'
        ).split()
    )
    assert parameters["use_fused"].default is inspect.Parameter.empty
    assert parameters["deterministic"].default is False
    assert parameters["precision"].default == "bf16"
    import ast

    for cls, attributes in [
        (CSAIndexer, {"linear_wq_b", "linear_weights_proj", "compressor"}),
        (TopKRouter, {"router_replay"}),
    ]:
        tree = ast.parse(inspect.getsource(inspect.getmodule(cls)))
        native = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls.__name__
        )
        constructor = next(
            n for n in native.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        assigned = {
            target.attr
            for n in ast.walk(constructor)
            if isinstance(n, ast.Assign)
            for target in n.targets
            if isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        }
        assert attributes <= assigned


@pytest.mark.parametrize("segments", [1, 2, 3])
def test_cross_segment_sources_native_core_checkpoint(segments, cpu_core_checkpoint):
    """CPU adapts only RNG snapshots; native Core checkpoint/autograd execute."""

    class Projection(torch.nn.Linear):
        def forward(self, x):
            return super().forward(x), None

    torch.manual_seed(7)
    state = KVSourceState()
    owner = IndexProjection(Projection(4, 4, bias=False), state, 0, 0, "index_q")
    readers = [IndexProjection(None, state, i + 1, 0, "index_q") for i in range(segments)]
    x = torch.randn(3, 1, 4, requires_grad=True)
    initial_weight = owner.linear.weight.detach().clone()
    calls = []

    def owner_segment(value):
        calls.append(("owner", torch.is_grad_enabled()))
        return owner(value)[0].sin()

    def reader_segment(reader, value):
        calls.append(("reader", torch.is_grad_enabled()))
        return value.cos() + reader(value)[0].square()

    with state.microbatch():
        actual = state.checkpoint(owner_segment, x, checkpoint_fn=cpu_core_checkpoint)
        for reader in readers:
            actual = state.checkpoint(
                lambda value, reader=reader: reader_segment(reader, value),
                actual,
                checkpoint_fn=cpu_core_checkpoint,
            )
    assert not state.active and not state.values
    actual.sum().backward()
    reference_x = x.detach().clone().requires_grad_()
    reference_weight = initial_weight.requires_grad_()
    shared = torch.nn.functional.linear(reference_x, reference_weight)
    reference = shared.sin()
    for _ in readers:
        reference = reference.cos() + shared.square()
    reference.sum().backward()
    assert torch.equal(actual, reference)
    assert torch.equal(x.grad, reference_x.grad)
    assert torch.equal(owner.linear.weight.grad, reference_weight.grad)
    assert torch.isfinite(x.grad).all() and torch.isfinite(owner.linear.weight.grad).all()
    reverse = [name for name, grad in calls if grad]
    assert reverse == ["reader"] * segments + ["owner"]
    assert not state.active and not state.values


def test_direct_core_signatures_are_pinned_to_dev_7b150acf():
    contracts = {
        ratio_one.async_gather_from_sequence_parallel_region: '(input_, tensor_parallel_output_grad=True, group=None)',
        ratio_one.cp_utils.thd_cp_position_ids: '(cu_seqlens_padded, global_start, local_rows)',
        ratio_one.fused_qk_topk_naive: '(q, k, weights, index_topk, mask=None, varlen_starts=None, varlen_ends=None, key_positions=None, use_relu=True)',
        ratio_one.get_window_topk_idxs_thd: '(window_size, cu_seqlens_q, total_q=None)',
        ratio_one.csa_sparse_attn: '(query, kv, attn_sink, topk_idxs, softmax_scale, topk_length=None, indexer_topk=0, is_thd=False, kv_reconstruction_parts=None, out_rope=None, deterministic=False, cuda_graph=False)',
        ratio_one.batch_of_row: '(cu_seqlens_q, total_q=None)',
        ratio_one.unfused_compressed_sparse_attn: '(query, kv_full, attn_sink, topk_indices, softmax_scale)',
        TopKRouter.__init__: '(self, config, pg_collection=None, is_mtp_layer=False, layer_number=None, hash_moe_layer_threshold=None)',
        WrappedTorchNorm.__new__: "(cls, config, hidden_size, eps=1e-05, persist_layer_norm=False, zero_centered_gamma=False, normalization='LayerNorm')",
        _validate_dsa_kernel_backend_dependencies: '(dsa_kernel_backend)',
    }
    for function, expected in contracts.items():
        signature = inspect.signature(function)
        actual = signature.replace(
            parameters=[
                p.replace(annotation=inspect.Parameter.empty) for p in signature.parameters.values()
            ],
            return_annotation=inspect.Signature.empty,
        )
        assert str(actual) == expected, function.__qualname__


def test_zero_aux_loss_keeps_forward_layout_without_disabling_autograd():

    module = FrozenIndexerCSA.__new__(FrozenIndexerCSA)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(dsa_indexer_loss_coeff=0.0)
    module.projection = torch.nn.Linear(2, 2, bias=False)
    module.train()
    assert not module.training and module.projection.training
    module.projection(torch.ones(1, 2)).sum().backward()
    assert module.projection.weight.grad is not None


def test_core_forward_training_dispatch_contract():
    from megatron.core.transformer.experimental_attention_variant import csa
    from megatron.core.transformer.experimental_attention_variant import (
        deepseek_v4_hybrid_attention as hybrid,
    )

    def conditions(module):
        return {
            n.lineno: ast.unparse(n)
            for n in ast.walk(ast.parse(inspect.getsource(module)))
            if isinstance(n, ast.BoolOp)
            and any(isinstance(a, ast.Attribute) and a.attr == "training" for a in ast.walk(n))
        }

    assert conditions(csa) == {
        line: ("has_indexer_compressed and " if line == 2829 else "")
        + "self.training and torch.is_grad_enabled()"
        for line in (2504, 2829, 2906, 3369, 3964)
    }
    assert conditions(hybrid) == {309: "self.offload_core_attention and self.training"}
