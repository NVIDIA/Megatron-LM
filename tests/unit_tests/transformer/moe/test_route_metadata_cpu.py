# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.moe import moe_layer


def test_hash_ids_and_padding_use_owning_tp_group(monkeypatch):
    group = SimpleNamespace(size=lambda: 2)
    calls = []

    def scatter(value, group=None):
        calls.append(group)
        return value[2:]

    monkeypatch.setattr(moe_layer.tensor_parallel, "scatter_to_sequence_parallel_region", scatter)
    received = []

    class Router(torch.nn.Module):
        is_hash_layer = True

        def forward(self, hidden, padding, input_ids=None, packed_seq_params=None):
            received.append((padding, input_ids))
            return hidden, input_ids

    layer = object.__new__(moe_layer.MoELayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        sequence_parallel=True,
        tensor_model_parallel_size=2,
        cuda_graph_impl="none",
        moe_megakernel_backend=None,
    )
    layer.tp_group = group
    layer.router = Router()
    layer.token_dispatcher = SimpleNamespace(wrap_layer_input=lambda hidden_states: hidden_states)
    mask = torch.tensor([[False, True, False, True]])
    ids = torch.tensor([[11, 12, 13, 14]])
    hidden = torch.zeros(2, 1, 4)
    moe_layer.MoELayer.route(layer, hidden, padding_mask=mask, input_ids=ids)
    assert calls == [group, group]
    torch.testing.assert_close(received[0][0], mask[:, 2:].T)
    torch.testing.assert_close(received[0][1], ids[:, 2:])


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_flex_preprocess_aligns_padding_to_owning_sp_rank(monkeypatch, tp_rank):
    group = SimpleNamespace(size=lambda: 2)
    packed = PackedSeqParams(qkv_format="thd")
    calls = []

    def scatter(value, group=None):
        calls.append(group)
        return value.chunk(2, dim=0)[tp_rank]

    monkeypatch.setattr(moe_layer.tensor_parallel, "scatter_to_sequence_parallel_region", scatter)
    received = []

    class Dispatcher(moe_layer.MoEFlexTokenDispatcher):
        def __init__(self):
            pass

        def dispatch_preprocess(
            self, hidden, routing, probs, padding_mask=None, packed_seq_params=None
        ):
            assert packed_seq_params is packed
            received.append(padding_mask)
            assert padding_mask.shape == (hidden.shape[1], hidden.shape[0])
            return hidden, probs

    layer = object.__new__(moe_layer.MoELayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        sequence_parallel=True,
        cuda_graph_impl="none",
        moe_latent_size=None,
        moe_megakernel_backend=None,
    )
    layer.tp_group = group
    layer.token_dispatcher = Dispatcher()
    mask = torch.tensor([[False, True, True, False], [True, False, False, True]])
    hidden = torch.zeros(2, 2, 4)
    probs = torch.ones(4, 2)
    routing = torch.ones(4, 2, dtype=torch.bool)
    layer.preprocess(hidden, probs, routing, mask, packed_seq_params=packed)
    expected = mask[:, tp_rank * 2 : (tp_rank + 1) * 2]
    torch.testing.assert_close(received[0], expected)
    # Already local metadata must not be scattered a second time.
    layer.preprocess(hidden, probs, routing, expected, packed_seq_params=packed)
    torch.testing.assert_close(received[1], expected)
    assert calls == [group]


@pytest.mark.parametrize("fused_shared_expert", [True, False])
@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("mask_kind", ["full", "local", "none"])
def test_megakernel_backend_receives_sp_local_padding_mask(
    monkeypatch, mask_kind, tp_rank, recompute, fused_shared_expert
):
    """The MoK router and backend see the same SP-local batch-first padding mask.

    NOTE(nightly-sync 24_11_2026): dev #8497 runs MoK with TP/SP and main #8654 passes the
    padding mask to the backend, which requires one mask entry per local token.
    """
    group = SimpleNamespace(size=lambda: 2)

    def scatter(value, group=None):
        return value.chunk(2, dim=0)[tp_rank]

    monkeypatch.setattr(moe_layer.tensor_parallel, "scatter_to_sequence_parallel_region", scatter)
    checkpointed = []

    def checkpoint(function, config, tp_group, *args):
        checkpointed.append(True)
        return function(*args)

    monkeypatch.setattr(moe_layer, "precision_aware_checkpoint", checkpoint)
    routed, backend = [], []

    class Router(torch.nn.Module):
        def forward(self, hidden, padding, input_ids=None, packed_seq_params=None):
            routed.append(padding)
            return hidden, hidden

    class Experts(torch.nn.Module):
        def forward(self, hidden, probs, routing_map, padding_mask):
            tokens = hidden.shape[0] * hidden.shape[1]
            if padding_mask is not None and padding_mask.numel() != tokens:
                raise ValueError(f"got {tuple(padding_mask.shape)} for {tokens} tokens")
            backend.append(padding_mask)
            return hidden

    layer = object.__new__(moe_layer.MoELayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        sequence_parallel=True,
        tensor_model_parallel_size=2,
        cuda_graph_impl="none",
        moe_megakernel_backend="cutemok",
    )
    layer.training = recompute
    layer.tp_group = group
    layer.attn_tp_group = group
    layer.router = Router()
    layer.megakernel_experts = Experts()
    layer.megakernel_fuses_shared_expert = fused_shared_expert
    layer.moe_layer_recompute = recompute
    layer.select_token_dispatcher = lambda: None
    layer.shared_experts_compute = lambda hidden_states: None
    layer.preprocess = lambda hidden_states, probs, routing_map: (hidden_states, probs)
    layer.postprocess = lambda output, shared_output: output

    full = torch.tensor([[False, False, True, True], [False, True, False, True]])
    local = full[:, 2 * tp_rank : 2 * tp_rank + 2]
    mask = {"full": full, "local": local, "none": None}[mask_kind]
    hidden = torch.randn(2, 2, 4)
    output, bias = moe_layer.MoELayer.forward(layer, hidden, padding_mask=mask)

    torch.testing.assert_close(output, hidden)
    assert bias is None
    assert checkpointed == ([True] if recompute else [])
    if mask is None:
        assert routed == [None] and backend == [None]
    else:
        torch.testing.assert_close(backend[0], local)
        torch.testing.assert_close(routed[0], local.T)
