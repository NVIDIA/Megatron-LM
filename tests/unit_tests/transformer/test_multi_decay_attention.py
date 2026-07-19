# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from argparse import ArgumentParser
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_inference_stack_spec,
    hybrid_stack_spec,
)
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_decay_attention import (
    MultiDecayAttention,
    MultiDecaySelfAttention,
    MultiDecaySelfAttentionSubmodules,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class _FakeGroup:

    def size(self):
        return 1


class _FakeProcessGroups:

    tp = _FakeGroup()
    cp = _FakeGroup()


class _FakeParallelLinear(nn.Module):

    def __init__(self, input_size, output_size, bias=False, **_kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.zeros(output_size)) if bias else None
        self.return_layernorm_output = False
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, hidden_states):
        output = torch.nn.functional.linear(hidden_states, self.weight, self.bias)
        if self.return_layernorm_output:
            return (output, hidden_states), None
        return output, None


class _FakeNorm(nn.Module):

    def __init__(self, **_kwargs):
        super().__init__()

    def forward(self, hidden_states):
        return torch.nn.functional.normalize(hidden_states, dim=-1)


def _fake_parallel_mda(
    q, k, v, log_f, mixer_logits=None, aggregate_mode='query_mix', output_dtype=None, **_kwargs
):
    del k, v
    if aggregate_mode == 'concat':
        output = q.repeat_interleave(log_f.shape[-1], dim=2)
        control = log_f.reshape(*log_f.shape[:2], -1).unsqueeze(-1)
    else:
        output = q
        control = log_f.mean(dim=-1).unsqueeze(-1)
        if mixer_logits is not None:
            control = control + mixer_logits.mean(dim=-1).unsqueeze(-1)
    output = output + control.to(output.dtype)
    return output if output_dtype is None else output.to(output_dtype)


def _fake_reference_mda(q, k, v, log_f, **kwargs):
    return _fake_parallel_mda(q, k, v, log_f, **kwargs)


def _fake_fused_mda(q, k, v, log_f, mixer_logits, output_dtype=None, **_kwargs):
    return _fake_parallel_mda(q, k, v, log_f, mixer_logits=mixer_logits, output_dtype=output_dtype)


class _FakeFA4Availability:

    forward = False


def _fake_operators():
    return {
        'fa4': _fake_fused_mda,
        'fa4_availability': lambda _device: _FakeFA4Availability(),
        'fused': _fake_fused_mda,
        'parallel': _fake_parallel_mda,
        'reference': _fake_reference_mda,
    }


def _build_layer(operators=None, **config_overrides):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=4,
        add_bias_linear=False,
        attention_dropout=0.0,
        **config_overrides,
    )
    submodules = MultiDecaySelfAttentionSubmodules(
        linear_qkv=_FakeParallelLinear,
        core_attention=MultiDecayAttention,
        linear_proj=_FakeParallelLinear,
        q_layernorm=_FakeNorm,
        k_layernorm=_FakeNorm,
        f_proj=_FakeParallelLinear,
        mix_proj=_FakeParallelLinear,
        g_proj=_FakeParallelLinear,
    )
    with patch(
        'megatron.core.transformer.multi_decay_attention._load_mda_operators',
        return_value=_fake_operators() if operators is None else operators,
    ):
        return MultiDecaySelfAttention(
            config=config, submodules=submodules, layer_number=1, pg_collection=_FakeProcessGroups()
        )


@pytest.mark.parametrize('stack_spec', [hybrid_stack_spec, hybrid_inference_stack_spec])
def test_hybrid_spec_uses_one_general_self_attention_path(stack_spec):
    baseline = stack_spec.submodules.attention_layer.submodules.self_attention
    multi_decay = stack_spec.submodules.multi_decay_layer.submodules.self_attention

    assert multi_decay.module is MultiDecaySelfAttention
    assert multi_decay.submodules.core_attention is MultiDecayAttention
    if stack_spec is hybrid_stack_spec:
        assert multi_decay.submodules.linear_qkv is baseline.submodules.linear_qkv
        assert multi_decay.submodules.linear_proj is baseline.submodules.linear_proj


def test_multi_decay_cli_options_are_registered():
    from megatron.training.arguments import _add_experimental_args

    parser = _add_experimental_args(ArgumentParser())
    args = parser.parse_args(
        [
            '--multi-decay-num-channels',
            '4',
            '--multi-decay-decay-generation',
            'full',
            '--multi-decay-decay-type',
            'mamba2',
            '--multi-decay-aggregate-mode',
            'concat',
            '--multi-decay-training-kernel',
            'bridge',
            '--multi-decay-qkv-bias',
            '--multi-decay-qk-norm',
            '--multi-decay-window-size',
            '256',
            '--no-multi-decay-decay-bias',
            '--multi-decay-use-output-gate',
            '--multi-decay-use-nope',
        ]
    )

    assert args.multi_decay_num_channels == 4
    assert args.multi_decay_decay_generation == 'full'
    assert args.multi_decay_decay_type == 'mamba2'
    assert args.multi_decay_aggregate_mode == 'concat'
    assert args.multi_decay_training_kernel == 'bridge'
    assert args.multi_decay_qkv_bias is True
    assert args.multi_decay_qk_norm is True
    assert args.multi_decay_window_size == 256
    assert args.multi_decay_decay_bias is False
    assert args.multi_decay_use_output_gate is True
    assert args.multi_decay_use_nope is True


def test_r1_nope_has_regular_attention_parameter_shell():
    layer = _build_layer(multi_decay_num_channels=1, multi_decay_use_nope=True)

    assert layer.f_proj is None
    assert layer.mix_proj is None
    assert layer.g_proj is None
    assert layer.decay_log_scales is None
    assert set(layer.state_dict()) == {'linear_proj.weight', 'linear_qkv.weight'}


def test_auxiliary_parameters_do_not_advance_baseline_initialization_stream():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=4,
        add_bias_linear=False,
        attention_dropout=0.0,
        multi_decay_num_channels=4,
    )
    baseline_submodules = SelfAttentionSubmodules(
        linear_qkv=_FakeParallelLinear,
        core_attention=MultiDecayAttention,
        linear_proj=_FakeParallelLinear,
    )
    multi_decay_submodules = MultiDecaySelfAttentionSubmodules(
        linear_qkv=_FakeParallelLinear,
        core_attention=MultiDecayAttention,
        linear_proj=_FakeParallelLinear,
        f_proj=_FakeParallelLinear,
        mix_proj=_FakeParallelLinear,
        g_proj=_FakeParallelLinear,
    )

    with patch(
        'megatron.core.transformer.multi_decay_attention._load_mda_operators',
        side_effect=_fake_operators,
    ):
        torch.manual_seed(1234)
        baseline = SelfAttention(
            config=config,
            submodules=baseline_submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            pg_collection=_FakeProcessGroups(),
        )
        baseline_tail = torch.randn(8)

        torch.manual_seed(1234)
        multi_decay = MultiDecaySelfAttention(
            config=config,
            submodules=multi_decay_submodules,
            layer_number=1,
            pg_collection=_FakeProcessGroups(),
        )
        multi_decay_tail = torch.randn(8)

    torch.testing.assert_close(multi_decay.linear_qkv.weight, baseline.linear_qkv.weight)
    torch.testing.assert_close(multi_decay.linear_proj.weight, baseline.linear_proj.weight)
    torch.testing.assert_close(multi_decay_tail, baseline_tail)


@pytest.mark.parametrize('num_decay_channels', [1, 4, 8])
@pytest.mark.parametrize('decay_generation', ['scaled_basis', 'full'])
@pytest.mark.parametrize('use_nope', [False, True])
def test_general_r_config_preserves_shape_and_updates_controls(
    num_decay_channels, decay_generation, use_nope
):
    layer = _build_layer(
        multi_decay_num_channels=num_decay_channels,
        multi_decay_decay_generation=decay_generation,
        multi_decay_use_nope=use_nope,
    )
    hidden_states = torch.randn(7, 3, 16, requires_grad=True)

    output, bias = layer(hidden_states, attention_mask=None)
    output.sum().backward()

    assert output.shape == hidden_states.shape
    assert bias is None
    assert hidden_states.grad is not None
    if layer.f_proj is not None:
        assert layer.f_proj.weight.grad is not None
    if layer.mix_proj is not None:
        assert layer.mix_proj.weight.grad is not None
    if layer.decay_log_scales is not None:
        assert layer.decay_log_scales.grad is not None


@pytest.mark.parametrize('aggregate_mode', ['query_mix', 'mean', 'concat'])
def test_all_aggregate_modes_preserve_hidden_shape(aggregate_mode):
    layer = _build_layer(multi_decay_num_channels=4, multi_decay_aggregate_mode=aggregate_mode)

    output, _ = layer(torch.randn(5, 2, 16), attention_mask=None)

    assert output.shape == (5, 2, 16)
    expected_proj_input = 64 if aggregate_mode == 'concat' else 16
    assert layer.linear_proj.weight.shape[1] == expected_proj_input


def test_output_gate_and_qk_norm_are_general_options():
    layer = _build_layer(
        multi_decay_num_channels=4, multi_decay_qk_norm=True, multi_decay_use_output_gate=True
    )

    output, _ = layer(torch.randn(5, 2, 16), attention_mask=None)
    output.sum().backward()

    assert layer.q_layernorm is not None
    assert layer.k_layernorm is not None
    assert layer.g_proj.weight.grad is not None


def test_mamba2_decay_parameterization_updates_a_log():
    layer = _build_layer(multi_decay_num_channels=4, multi_decay_decay_type='mamba2')

    output, _ = layer(torch.randn(5, 2, 16), attention_mask=None)
    output.sum().backward()

    assert layer.decay_log_scales.grad is not None
    assert layer.decay_log_scales.dtype == torch.float32
    assert torch.all(layer._decay_scales() > 0)


def test_attention_mask_is_rejected():
    layer = _build_layer(multi_decay_num_channels=4)

    with pytest.raises(NotImplementedError, match='implicit causal mask'):
        layer(torch.randn(7, 3, 16), attention_mask=torch.ones(1))


def test_auto_cpu_uses_reference_backend():
    operators = _fake_operators()
    operators['reference'] = Mock(side_effect=_fake_reference_mda)
    operators['parallel'] = Mock(side_effect=AssertionError('parallel backend selected on CPU'))
    layer = _build_layer(operators=operators, multi_decay_num_channels=4)

    layer(torch.randn(7, 3, 16), attention_mask=None)

    operators['reference'].assert_called_once()


def test_explicit_bridge_requires_cuda():
    layer = _build_layer(multi_decay_num_channels=4, multi_decay_training_kernel='bridge')

    with pytest.raises(NotImplementedError, match='requires CUDA'):
        layer(torch.randn(7, 3, 16), attention_mask=None)


def test_explicit_fa4_validates_training_contract():
    layer = _build_layer(
        multi_decay_num_channels=8,
        multi_decay_decay_generation='scaled_basis',
        multi_decay_training_kernel='fa4',
    )

    with pytest.raises(NotImplementedError, match='requires CUDA'):
        layer(torch.randn(7, 3, 16), attention_mask=None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='R1 NoPE parity requires CUDA')
def test_multi_decay_attention_r1_nope_matches_te_dot_product_attention():
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', '0')))
    torch.cuda.set_device(device)
    torch.manual_seed(1234)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=4,
        attention_dropout=0.0,
        bf16=True,
        multi_decay_num_channels=1,
        multi_decay_use_nope=True,
        multi_decay_aggregate_mode='query_mix',
    )
    kwargs = dict(
        config=config,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type='self',
        pg_collection=_FakeProcessGroups(),
    )
    baseline = TEDotProductAttention(**kwargs)
    r1_nope = MultiDecayAttention(**kwargs)

    query = torch.randn(17, 2, 4, 4, device=device, dtype=torch.bfloat16)
    key = torch.randn(17, 2, 2, 4, device=device, dtype=torch.bfloat16)
    value = torch.randn(17, 2, 2, 4, device=device, dtype=torch.bfloat16)
    baseline_inputs = [
        tensor.detach().clone().requires_grad_(True) for tensor in (query, key, value)
    ]
    r1_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (query, key, value)]

    expected = baseline(*baseline_inputs, None, attn_mask_type=AttnMaskType.causal)
    actual = r1_nope(*r1_inputs, None, attn_mask_type=AttnMaskType.causal)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)

    upstream = torch.randn_like(expected)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    for actual_grad, expected_grad in zip(
        (tensor.grad for tensor in r1_inputs), (tensor.grad for tensor in baseline_inputs)
    ):
        torch.testing.assert_close(actual_grad.float(), expected_grad.float(), atol=3e-2, rtol=3e-2)
