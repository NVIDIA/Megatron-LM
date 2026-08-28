# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.models.hybrid.shortcut_block import (
    ShortcutExecutionMode,
    ShortcutMoEBlock,
    group_layers_into_shortcut_blocks,
)
from megatron.core.transformer.module import SplitOutputProjection
from megatron.core.transformer.transformer_config import TransformerConfig


def _shortcut_config(*, parallel: bool = False):
    return SimpleNamespace(
        moe_shortcut_parallel=parallel,
        fp32_residual_connection=False,
        hidden_size=8,
        layernorm_epsilon=1e-5,
        sequence_parallel=True,
    )


class _FakeCompute(torch.nn.Module, SplitOutputProjection):
    def __init__(self, config, *, supports_split: bool = True):
        super().__init__()
        self.config = config
        self.layer_number = 1
        self.is_first_layer = True
        self.is_last_layer = False
        self.supports_split = supports_split

    def supports_split_output_projection(self) -> bool:
        return self.supports_split

    def forward_pre_output_proj(self, hidden_states, **kwargs):
        return (hidden_states,)

    def forward_output_proj(self, hidden_states, **kwargs):
        return hidden_states


class _FakeMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tp_group = None


class _FakeMoE(torch.nn.Module):
    def __init__(self, config, *, layer_number: int = 2):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.is_first_layer = layer_number == 1
        self.is_last_layer = False
        self.pre_mlp_layernorm = torch.nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.submodules_config = SimpleNamespace(
            pre_mlp_layernorm=lambda **kwargs: torch.nn.RMSNorm(
                kwargs["hidden_size"], eps=kwargs["eps"]
            )
        )
        self.mlp = _FakeMLP()


@pytest.mark.parametrize(
    ("compute_symbol", "parallel"),
    [
        pytest.param(LayerSymbols.MAMBA, False, id="mamba-serial"),
        pytest.param(LayerSymbols.GDN, True, id="gdn-overlap"),
        pytest.param(LayerSymbols.ATTENTION, False, id="attention-serial"),
    ],
)
def test_group_layers_into_shortcut_blocks(compute_symbol, parallel):
    """Supported pairs are registered together while an initial MoE keeps its normal path."""
    config = _shortcut_config(parallel=parallel)
    leading_moe = _FakeMoE(config, layer_number=1)
    compute = _FakeCompute(config)
    paired_moe = _FakeMoE(config, layer_number=3)
    trailing_layer = torch.nn.Identity()

    grouped = group_layers_into_shortcut_blocks(
        torch.nn.ModuleList([leading_moe, compute, paired_moe, trailing_layer]),
        [LayerSymbols.MOE, compute_symbol, LayerSymbols.MOE, LayerSymbols.MLP],
        config,
    )

    assert len(grouped) == 3
    assert grouped[0] is leading_moe
    assert grouped[2] is trailing_layer
    shortcut = grouped[1]
    assert isinstance(shortcut, ShortcutMoEBlock)
    assert shortcut.compute_layer is compute
    assert shortcut.moe_layer is paired_moe
    assert shortcut.execution_mode == ShortcutExecutionMode.resolve(overlap_a2a=parallel)
    assert shortcut.shortcut_pre_mlp_layernorm is not paired_moe.pre_mlp_layernorm
    assert isinstance(shortcut.shortcut_post_norm, torch.nn.RMSNorm)
    assert all(
        parameter.sequence_parallel for parameter in shortcut.shortcut_post_norm.parameters()
    )

    state_keys = set(grouped.state_dict())
    assert "1.shortcut_pre_mlp_layernorm.weight" in state_keys
    assert "1.shortcut_post_norm.weight" in state_keys


@pytest.mark.parametrize(
    ("compute_symbol", "compute", "error"),
    [
        pytest.param(
            LayerSymbols.MLP,
            lambda config: _FakeCompute(config),
            "must be preceded",
            id="unsupported-predecessor",
        ),
        pytest.param(
            LayerSymbols.MAMBA,
            lambda config: _FakeCompute(config, supports_split=False),
            "does not support split output projection",
            id="atomic-forward-only",
        ),
    ],
)
def test_group_layers_rejects_invalid_shortcut_pair(compute_symbol, compute, error):
    config = _shortcut_config()
    with pytest.raises(ValueError, match=error):
        group_layers_into_shortcut_blocks(
            torch.nn.ModuleList([compute(config), _FakeMoE(config)]),
            [compute_symbol, LayerSymbols.MOE],
            config,
        )


def test_eager_overlap_matches_serial_output_and_gradients(monkeypatch):
    """The overlap schedule changes ordering, not Shortcut MoE numerics or autograd."""
    config = TransformerConfig(num_layers=2, hidden_size=4, num_attention_heads=1)

    class FakeCompute(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.layer_number = 1
            self.input_scale = torch.nn.Parameter(torch.tensor(2.0))
            self.output_scale = torch.nn.Parameter(torch.tensor(3.0))

        def forward_pre_output_proj(self, hidden_states, **kwargs):
            return hidden_states * self.input_scale, hidden_states

        def forward_output_proj(self, projected, residual, **kwargs):
            return projected * self.output_scale

    class FakeMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tp_group = None
            self.dispatch_scale = torch.nn.Parameter(torch.tensor(5.0))
            self.expert_scale = torch.nn.Parameter(torch.tensor(7.0))
            self.combine_scale = torch.nn.Parameter(torch.tensor(11.0))

        def dispatch(self, route_input, route_probs):
            return route_input * self.dispatch_scale, route_probs

        def routed_experts_compute(self, dispatched_input, dispatched_probs):
            return dispatched_input + dispatched_probs * self.expert_scale, None

        def combine(self, routed_output):
            return routed_output * self.combine_scale

        def postprocess(self, combined_output, shared_expert_output):
            return combined_output + shared_expert_output

    class FakeMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_number = 2
            self.submodules_config = SimpleNamespace(
                pre_mlp_layernorm=lambda **kwargs: torch.nn.Identity()
            )
            self.route_scale = torch.nn.Parameter(torch.tensor(13.0))
            self.prob_scale = torch.nn.Parameter(torch.tensor(17.0))
            self.shared_scale = torch.nn.Parameter(torch.tensor(19.0))
            self.mlp = FakeMLP()

        def shortcut_route_preprocess(self, shortcut_hidden, padding_mask=None):
            return shortcut_hidden * self.route_scale, shortcut_hidden * self.prob_scale

        def shortcut_shared_experts(self, hidden_states):
            return hidden_states * self.shared_scale

        def _apply_mlp_bda_step(self, output_with_bias, residual):
            return output_with_bias[0] + residual

    @contextmanager
    def quant_context_factory(config, layer_number):
        yield

    def run(overlap):
        block = ShortcutMoEBlock(FakeCompute(), FakeMoE(), overlap_a2a=overlap)
        monkeypatch.setattr(
            block, "_shortcut_route_preprocess", block.moe_layer.shortcut_route_preprocess
        )
        monkeypatch.setattr(
            block, "_shortcut_shared_experts", block.moe_layer.shortcut_shared_experts
        )
        if overlap:
            monkeypatch.setattr(
                block,
                "_launch_dispatch_async",
                lambda route_input, route_probs, ready_event: block.moe_layer.mlp.dispatch(
                    route_input, route_probs
                ),
            )
            monkeypatch.setattr(
                block,
                "_wait_dispatch_and_launch_combine",
                lambda dispatch_output: block.moe_layer.mlp.combine(
                    block.moe_layer.mlp.routed_experts_compute(*dispatch_output)[0]
                ),
            )
            monkeypatch.setattr(block, "_wait_combine", lambda combined_output: combined_output)

        hidden_states = torch.arange(1.0, 5.0, requires_grad=True)
        output = block(
            hidden_states=hidden_states,
            attention_mask=None,
            inference_context=None,
            rotary_pos_emb=None,
            sequence_len_offset=None,
            packed_seq_params=None,
            padding_mask=None,
            quant_context_factory=quant_context_factory,
            quant_config=None,
        )
        output.sum().backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in block.named_parameters()
            if parameter.grad is not None
        }
        gradients["hidden_states"] = hidden_states.grad.detach().clone()
        return output, gradients

    monkeypatch.setattr(
        torch.cuda, "Event", lambda *args, **kwargs: SimpleNamespace(record=lambda stream: None)
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    serial_output, serial_gradients = run(overlap=False)
    overlap_output, overlap_gradients = run(overlap=True)

    torch.testing.assert_close(overlap_output, serial_output)
    assert overlap_gradients.keys() == serial_gradients.keys()
    for name in overlap_gradients:
        torch.testing.assert_close(overlap_gradients[name], serial_gradients[name])
