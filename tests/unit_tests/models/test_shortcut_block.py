# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import (
    _allreduce_non_tensor_model_parallel_grads,
)
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.models.hybrid.shortcut_block import (
    ShortcutExecutionMode,
    ShortcutMoEBlock,
    group_layers_into_shortcut_blocks,
)
from megatron.core.transformer.module import SplitOutputProjection
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _fake_shortcut_config(*, parallel: bool = False):
    return SimpleNamespace(
        moe_shortcut_parallel=parallel,
        fp32_residual_connection=False,
        hidden_size=8,
        layernorm_epsilon=1e-5,
        sequence_parallel=False,
    )


def test_group_layers_into_registered_shortcut_blocks():
    config = _fake_shortcut_config()

    class FakeCompute(torch.nn.Module, SplitOutputProjection):
        def __init__(self):
            super().__init__()
            self.config = config
            self.layer_number = 1
            self.is_first_layer = True
            self.is_last_layer = False

        def forward_pre_output_proj(self, hidden_states, **kwargs):
            return (hidden_states,)

        def forward_output_proj(self, hidden_states, **kwargs):
            return hidden_states

    class FakeMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tp_group = None

    class FakeMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_number = 2
            self.is_first_layer = False
            self.is_last_layer = False
            self.submodules_config = SimpleNamespace(
                pre_mlp_layernorm=lambda **kwargs: torch.nn.Identity()
            )
            self.mlp = FakeMLP()

    compute = FakeCompute()
    moe = FakeMoE()
    standalone = torch.nn.Identity()
    grouped = group_layers_into_shortcut_blocks(
        torch.nn.ModuleList([compute, moe, standalone]),
        [LayerSymbols.MAMBA, LayerSymbols.MOE, LayerSymbols.ATTENTION],
        config,
    )

    assert len(grouped) == 2
    assert isinstance(grouped[0], ShortcutMoEBlock)
    assert grouped[0].execution_mode == ShortcutExecutionMode.EAGER_SERIAL
    assert grouped[0].compute_layer is compute
    assert grouped[0].moe_layer is moe
    assert grouped[1] is standalone
    registered_modules = dict(grouped.named_modules())
    assert registered_modules["0.compute_layer"] is compute
    assert registered_modules["0.moe_layer"] is moe
    assert registered_modules["1"] is standalone


def test_shortcut_block_registers_pair_and_post_norm():
    config = _fake_shortcut_config()
    config.sequence_parallel = True

    class FakeCompute(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.layer_number = 1
            self.is_first_layer = True
            self.is_last_layer = False

    class FakeMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tp_group = None

    class FakeMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_number = 2
            self.is_first_layer = False
            self.is_last_layer = True
            self.submodules_config = SimpleNamespace(
                pre_mlp_layernorm=lambda **kwargs: torch.nn.Identity()
            )
            self.mlp = FakeMLP()

    compute = FakeCompute()
    moe = FakeMoE()
    block = ShortcutMoEBlock(compute, moe, overlap_a2a=False)

    assert block.compute_layer is compute
    assert block.moe_layer is moe
    assert block.execution_mode == ShortcutExecutionMode.EAGER_SERIAL
    assert block.route_ready_event is None
    assert isinstance(block.shortcut_post_norm, torch.nn.RMSNorm)
    assert block.shortcut_post_norm.eps == config.layernorm_epsilon
    assert all(
        getattr(parameter, 'sequence_parallel', False)
        for parameter in block.shortcut_post_norm.parameters()
    )
    assert block.is_first_layer
    assert block.is_last_layer
    assert list(dict(block.named_children())) == [
        "compute_layer",
        "moe_layer",
        "shortcut_pre_mlp_layernorm",
        "shortcut_post_norm",
    ]


def test_shortcut_execution_mode_resolution():
    assert ShortcutExecutionMode.resolve(overlap_a2a=False) == ShortcutExecutionMode.EAGER_SERIAL
    assert ShortcutExecutionMode.resolve(overlap_a2a=True) == ShortcutExecutionMode.EAGER_OVERLAP


def test_shortcut_blocks_share_one_parallel_stream_per_device(monkeypatch):
    created_streams = []

    def make_stream(*, priority):
        stream = SimpleNamespace(priority=priority)
        created_streams.append(stream)
        return stream

    monkeypatch.setattr(ShortcutMoEBlock, "_parallel_streams", {})
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "Stream", make_stream)
    first_block = object.__new__(ShortcutMoEBlock)
    second_block = object.__new__(ShortcutMoEBlock)

    first_stream = first_block._get_parallel_stream()
    second_stream = second_block._get_parallel_stream()

    assert first_stream is second_stream
    assert first_stream.priority == -1
    assert created_streams == [first_stream]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Multiple CUDA devices required")
def test_shortcut_blocks_use_distinct_parallel_streams_across_devices(monkeypatch):
    monkeypatch.setattr(ShortcutMoEBlock, "_parallel_streams", {})

    with torch.cuda.device(0):
        first_stream = ShortcutMoEBlock._get_parallel_stream()
    with torch.cuda.device(1):
        second_stream = ShortcutMoEBlock._get_parallel_stream()

    assert first_stream is not second_stream


def test_eager_overlap_matches_serial_output_gradients_and_updates(monkeypatch):
    config = TransformerConfig(num_layers=2, hidden_size=4, num_attention_heads=1)

    class FakeCompute(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.layer_number = 1
            self.is_first_layer = True
            self.is_last_layer = False
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
            self.is_first_layer = False
            self.is_last_layer = True
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
        compute = FakeCompute()
        moe = FakeMoE()
        block = ShortcutMoEBlock(compute, moe, overlap_a2a=overlap)
        monkeypatch.setattr(block, "_shortcut_route_preprocess", moe.shortcut_route_preprocess)
        monkeypatch.setattr(block, "_shortcut_shared_experts", moe.shortcut_shared_experts)
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
        optimizer = torch.optim.SGD(block.parameters(), lr=0.01)
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
        optimizer.step()
        updated_parameters = {
            name: parameter.detach().clone() for name, parameter in block.named_parameters()
        }
        return output, gradients, updated_parameters

    monkeypatch.setattr(
        torch.cuda, "Event", lambda *args, **kwargs: SimpleNamespace(record=lambda stream: None)
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    serial_output, serial_gradients, serial_parameters = run(overlap=False)
    overlap_output, overlap_gradients, overlap_parameters = run(overlap=True)

    torch.testing.assert_close(overlap_output, serial_output, rtol=1e-5, atol=1e-6)
    assert overlap_gradients.keys() == serial_gradients.keys()
    for name in overlap_gradients:
        torch.testing.assert_close(
            overlap_gradients[name], serial_gradients[name], rtol=1e-5, atol=1e-6
        )
    assert overlap_parameters.keys() == serial_parameters.keys()
    for name in overlap_parameters:
        torch.testing.assert_close(
            overlap_parameters[name], serial_parameters[name], rtol=1e-5, atol=1e-6
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tp2_sequence_shards_reduce_post_norm_gradient_to_unsharded_reference():
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    try:
        config = TransformerConfig(
            num_layers=2,
            hidden_size=8,
            num_attention_heads=2,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
        )
        device = torch.device("cuda", torch.cuda.current_device())
        full_input = torch.arange(1, 65, dtype=torch.float32, device=device).view(8, 1, 8)
        tp_rank = parallel_state.get_tensor_model_parallel_rank()

        post_norm = torch.nn.RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon, device=device
        )
        post_norm.ddp_config = DistributedDataParallelConfig()
        post_norm.weight.sequence_parallel = True
        local_input = full_input.chunk(2, dim=0)[tp_rank].detach().requires_grad_(True)
        post_norm(local_input).sum().backward()

        reference_norm = torch.nn.RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon, device=device
        )
        reference_norm.load_state_dict(post_norm.state_dict())
        reference_norm(full_input).sum().backward()

        tp_group = parallel_state.get_tensor_model_parallel_group()
        _allreduce_non_tensor_model_parallel_grads([post_norm], config, tp_group)

        reduced_grads = [torch.empty_like(post_norm.weight.grad) for _ in range(2)]
        torch.distributed.all_gather(reduced_grads, post_norm.weight.grad, group=tp_group)
        assert torch.equal(reduced_grads[0], reduced_grads[1])
        torch.testing.assert_close(
            post_norm.weight.grad, reference_norm.weight.grad, rtol=1e-5, atol=1e-6
        )
    finally:
        Utils.destroy_model_parallel()


def test_shortcut_rejects_shared_expert_overlap():
    with pytest.raises(ValueError, match="mutually exclusive"):
        TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=2,
            num_moe_experts=2,
            moe_shortcut_connection=True,
            moe_shared_expert_overlap=True,
        )
