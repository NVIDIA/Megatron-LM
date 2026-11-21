# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.fine_grained_callables import build_transformer_layer_callables
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_layer import TransformerLayer


class _DoubleNorm(torch.nn.Module):
    """Stands in for the pre-MLP norm so its output differs from the residual."""

    def forward(self, hidden_states):
        return hidden_states * 2


def test_moe_slots_unflatten_packed_microbatches_like_the_eager_path(monkeypatch):
    """A packed [mbs*s, 1, h] batch reaches MoE as [s, mbs, h] and leaves flattened again."""
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    mbs, seq_len, hidden = 2, 3, 4
    received = {}

    def shared_experts_compute(hidden_states):
        received["shared"] = hidden_states
        return hidden_states * 5

    def route(hidden_states, padding_mask=None, packed_seq_params=None):
        received["route"] = (hidden_states, padding_mask)
        return hidden_states, None

    def preprocess(hidden_states, probs, routing_map, padding_mask=None):
        received["preprocess"] = (hidden_states, padding_mask)
        return hidden_states, probs

    def postprocess(output, shared_expert_output):
        return output if shared_expert_output is None else output + shared_expert_output

    mlp = Mock(spec=MoELayer)
    mlp.use_shared_expert = True
    mlp.shared_expert_overlap = False
    mlp.token_dispatcher = SimpleNamespace()
    mlp.shared_experts_compute = shared_experts_compute
    mlp.route = route
    mlp.preprocess = preprocess
    mlp.dispatch = lambda tokens, probs: (tokens, probs)
    mlp.routed_experts_compute = lambda tokens, probs: (tokens * 3, None)
    mlp.combine = lambda output: output
    mlp.postprocess = postprocess
    layer = SimpleNamespace(
        config=SimpleNamespace(moe_token_dispatcher_type="alltoall", bias_dropout_fusion=False),
        mlp=mlp,
        is_moe_layer=True,
        offload_mlp_norm=False,
        recompute_pre_mlp_layernorm=False,
        _pre_mlp_layernorm_returns_residual=False,
        pre_mlp_layernorm=_DoubleNorm(),
        training=True,
        hidden_dropout=0.0,
        bias_dropout_add_exec_handler=nullcontext,
        # The real bda with no dropout, which otherwise selects the fused shared-expert and
        # residual add in moe_combine.
        mlp_bda=get_bias_dropout_add,
        init_backward_dw_wrapper=lambda: None,
        backward_dw_wrapper=None,
        _forward_attention=lambda hidden_states, **kwargs: (hidden_states, None),
        # Only read for the dense-layer wrapper's signature; MoE layers never call it.
        _forward_mlp=Mock(),
    )
    layer._maybe_unflatten_for_moe = partial(TransformerLayer._maybe_unflatten_for_moe, layer)
    layer._maybe_reflatten_from_moe = partial(TransformerLayer._maybe_reflatten_from_moe, layer)
    node = SimpleNamespace(
        layer_state=SimpleNamespace(),
        chunk_state=SimpleNamespace(
            model=SimpleNamespace(decoder=SimpleNamespace(final_layernorm=None)),
            attention_mask=None,
            rotary_pos_emb=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            sequence_len_offset=None,
            packed_seq_params=PackedSeqParams(tokens_per_sample=seq_len),
            # Batch-first, as GPTModel passes it for a packed batch: [1, mbs * s].
            padding_mask=torch.tensor([[False, False, False, False, True, True]]),
        ),
        detach=lambda tensor: tensor.detach(),
        is_mtp=False,
        is_last_layer=False,
    )
    # Sample 0's tokens, then sample 1's, as inter-document masking flattens them.
    hidden_states = torch.arange(mbs * seq_len * hidden, dtype=torch.float32).view(
        mbs * seq_len, 1, hidden
    )

    (pre_dispatch, dispatch, moe_experts, combine, _), _ = build_transformer_layer_callables(layer)
    tokens, probs = pre_dispatch(node, hidden_states)
    output = combine(node, moe_experts(node, dispatch(node, tokens, probs)))

    expected_moe_input = (hidden_states * 2).view(mbs, seq_len, hidden).transpose(0, 1)
    expected_mask = torch.tensor([[False, False, False], [False, True, True]])
    torch.testing.assert_close(received["shared"], expected_moe_input)
    for name in ("route", "preprocess"):
        moe_input, padding_mask = received[name]
        torch.testing.assert_close(moe_input, expected_moe_input)
        torch.testing.assert_close(padding_mask, expected_mask)
    # Routed experts add 3x and the shared expert 5x the normalized tokens; bda adds the
    # flattened residual back.
    torch.testing.assert_close(output, hidden_states * 17)
    assert node.layer_state.moe_unflatten_mbs is None
