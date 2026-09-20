# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU contracts for binding QSA selection to selective checkpoint invocations."""

from types import MethodType, SimpleNamespace

import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.qsa import (
    QSACoreAttention,
    QSASelection,
    QwenSparseSelfAttention,
)


def _selection(rows):
    return QSASelection(
        doc_ids=torch.zeros((1, rows), dtype=torch.int32),
        positions=torch.arange(rows, dtype=torch.int32).unsqueeze(0),
        selected_bits=None,
        bits_per_row=0,
        bits_per_row_t=torch.tensor(0),
        compress_ratio=4,
        all_selected=False,
        selected_ids=torch.zeros((rows, 1), dtype=torch.int32),
        index_query=torch.randn(rows, 2, 3, requires_grad=True),
        compressed_key=torch.randn(2, 3, requires_grad=True),
    )


def test_qsa_checkpoint_captures_each_microbatch_and_explicit_gradient_tensors(monkeypatch):
    checkpoints, replayed = [], []

    def capture_checkpoint(function, distribute_saved_activations, *args):
        assert distribute_saved_activations is False
        checkpoints.append((function, args))
        return torch.zeros(1)

    monkeypatch.setattr(tensor_parallel, "checkpoint", capture_checkpoint)
    attention = object.__new__(QwenSparseSelfAttention)
    torch.nn.Module.__init__(attention)
    attention.attn_mask_type = AttnMaskType.causal
    attention.core_attention = SimpleNamespace(_selection=_selection(8))

    def capture_core(self, query, key, value, attention_mask, **kwargs):
        replayed.append(kwargs)
        return torch.zeros(1)

    attention._run_core_attention = MethodType(capture_core, attention)
    first = attention.core_attention._selection
    attention._checkpointed_attention_forward(None, None, None, None)
    attention.core_attention._selection = _selection(12)
    second = attention.core_attention._selection
    attention._checkpointed_attention_forward(None, None, None, None)

    assert checkpoints[0][1][-2] is first.index_query
    assert checkpoints[0][1][-1] is first.compressed_key
    assert checkpoints[1][1][-2] is second.index_query
    assert checkpoints[1][1][-1] is second.compressed_key
    checkpoints[0][0](*checkpoints[0][1])
    checkpoints[1][0](*checkpoints[1][1])
    assert replayed[0]["qsa_selection"].positions.shape == (1, 8)
    assert replayed[1]["qsa_selection"].positions.shape == (1, 12)
    for kwargs in replayed:
        assert kwargs["qsa_selection"].index_query is None
        assert kwargs["qsa_selection"].compressed_key is None
    assert replayed[0]["qsa_index_query"] is first.index_query
    assert replayed[0]["qsa_compressed_key"] is first.compressed_key
    assert replayed[1]["qsa_index_query"] is second.index_query
    assert replayed[1]["qsa_compressed_key"] is second.compressed_key


def test_qsa_core_uses_bound_selection_even_after_module_field_is_overwritten():
    core = object.__new__(QSACoreAttention)
    torch.nn.Module.__init__(core)
    core.config = SimpleNamespace(qsa_force_sparse=True, qsa_indexer_loss_coeff=0.7)
    core.pg_collection = SimpleNamespace(cp=None)
    core.sparse_backend = "id_sparse"
    first, second = _selection(8), _selection(12)
    core.set_selection(second)
    captured = []

    def sparse(self, query, key, value, selection, is_thd):
        captured.append(selection)
        return query.reshape(query.size(0), 1, -1)

    def attach(self, output, query, key, selection, packed_seq_params, cp_size):
        captured.append(selection)
        return output

    core._id_sparse_forward = MethodType(sparse, core)
    core._attach_indexer_loss = MethodType(attach, core)
    detached_query = first.index_query.detach().requires_grad_()
    detached_key = first.compressed_key.detach().requires_grad_()
    query = torch.randn(8, 1, 2, 3)
    key = torch.randn(8, 1, 1, 3)
    output = core(
        query,
        key,
        key,
        None,
        qsa_selection=first,
        qsa_index_query=detached_query,
        qsa_compressed_key=detached_key,
    )
    assert output.shape == (8, 1, 6)
    assert len(captured) == 2
    for bound in captured:
        assert bound is not first and bound is not second
        assert bound.selected_ids is first.selected_ids
        assert bound.index_query is detached_query
        assert bound.compressed_key is detached_key
    with pytest.raises(RuntimeError, match="omitted differentiable indexer tensors"):
        core(query, key, key, None, qsa_selection=first)


@pytest.mark.parametrize("loss_coeff", [0.0, 0.7])
@pytest.mark.parametrize("raise_in_core", [False, True])
def test_qsa_forward_clears_selection_on_success_and_exception(
    monkeypatch, loss_coeff, raise_in_core
):
    attention = object.__new__(QwenSparseSelfAttention)
    torch.nn.Module.__init__(attention)
    attention.config = SimpleNamespace(
        qsa_indexer_loss_coeff=loss_coeff, attention_dropout=0.0, tensor_model_parallel_size=1
    )
    attention.pg_collection = SimpleNamespace(tp=None)
    selection = _selection(8)
    stored = []
    core = SimpleNamespace(sparse_backend="id_sparse", _selection=None)

    def set_selection(value):
        core._selection = value
        stored.append(value)

    core.set_selection = set_selection
    attention.core_attention = core
    formats = []

    def select(*args, output_format):
        formats.append(output_format)
        return selection

    attention.indexer = select
    output = torch.randn(8, 1, 4)

    def fake_attention_forward(self, *args, **kwargs):
        assert self.core_attention._selection is selection
        if raise_in_core:
            raise RuntimeError("synthetic core failure")
        return output, None

    monkeypatch.setattr(SelfAttention, "forward", fake_attention_forward)
    if raise_in_core:
        with pytest.raises(RuntimeError, match="synthetic core failure"):
            attention(torch.zeros_like(output), attention_mask=None)
    else:
        assert attention(torch.zeros_like(output), attention_mask=None)[0] is output
    assert formats == ["ids"]
    assert stored == [selection, None]
    assert attention.core_attention._selection is None
