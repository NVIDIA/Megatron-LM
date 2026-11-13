# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer


@pytest.fixture(params=["transformer", "hybrid", "hybrid_grouped"])
def replay_layer(request):
    """Use the real replay boundary, replacing only the recorded GPU callable."""
    layer_type = TransformerLayer if request.param == "transformer" else HyperConnectionHybridLayer
    layer = layer_type.__new__(layer_type)
    torch.nn.Module.__init__(layer)
    layer.config = TransformerConfig(
        num_layers=1, hidden_size=4, num_attention_heads=1, cuda_graph_impl="transformer_engine"
    )
    layer.current_microbatch = 0
    layer.cuda_graph_manual_hooks = []
    if layer_type is TransformerLayer:
        layer.is_moe_layer = False
        layer._cuda_graph_padding_mask_shape = None
    else:
        layer.inner_layer = torch.nn.Identity()
        if request.param == "hybrid_grouped":
            layer._te_cuda_graph_group_tail = SimpleNamespace(
                _resume_partial_moe_cuda_graph=lambda outputs: (outputs[0] + 1, None)
            )
    return layer


@pytest.mark.parametrize("validation", [False, True, None])
def test_te_replay_keeps_runtime_validation_out_of_graph_inputs(replay_layer, validation):
    hidden = torch.ones(2, 1, 4)
    mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool)
    calls = []

    def graph(hidden_states, *, attention_mask, is_first_microbatch):
        assert hidden_states is hidden
        assert attention_mask is mask
        assert is_first_microbatch is True
        calls.append(True)
        return (hidden_states * 2,)

    replay_layer.cuda_graphs = [graph]
    output, context = replay_layer._te_cuda_graph_replay(
        hidden, attention_mask=mask, strict_runtime_validation=validation
    )
    expected = 3 if getattr(replay_layer, "_te_cuda_graph_group_tail", None) else 2
    torch.testing.assert_close(output, hidden * expected)
    assert context is None
    assert calls == [True]


def test_te_replay_still_rejects_unknown_non_tensor_inputs(replay_layer):
    replay_layer.cuda_graphs = [lambda *args, **kwargs: pytest.fail("invalid graph input")]
    with pytest.raises(AssertionError, match="CUDA graph accepts only Tensor inputs"):
        replay_layer._te_cuda_graph_replay(torch.ones(2, 1, 4), unsupported_control=True)


@pytest.mark.parametrize("validation", [False, True, None])
def test_partial_te_graph_preserves_eager_attention_validation(monkeypatch, validation):
    layer = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.config = TransformerConfig(
        num_layers=1,
        hidden_size=4,
        num_attention_heads=1,
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[CudaGraphModule.mlp],
    )
    layer.current_microbatch = 0
    layer.cuda_graph_manual_hooks = []
    layer.is_moe_layer = False
    layer._cuda_graph_padding_mask_shape = None
    seen = []

    def attention(hidden_states, *, strict_runtime_validation):
        seen.append(strict_runtime_validation)
        return hidden_states + 1, None

    monkeypatch.setattr(layer, "_forward_attention", attention)
    layer.cuda_graphs = [lambda hidden, *, is_first_microbatch: (hidden * 2,)]
    output, context = layer._te_cuda_graph_replay(
        torch.ones(2, 1, 4), strict_runtime_validation=validation
    )
    torch.testing.assert_close(output, torch.full((2, 1, 4), 4.0))
    assert context is None
    assert seen == [validation]
