# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.common.language_module.language_module import LanguageModule
from tests.unit_tests.test_utilities import Utils
from tools.checkpoint import loader_mixtral_hf


@pytest.mark.parametrize(
    'version', ['4.36.0', '4.49.0', '4.57.0rc1', '5.0.0.dev0', '5.8.1', '6.0.0']
)
def test_supported_transformers_versions(monkeypatch, version):
    monkeypatch.setattr(loader_mixtral_hf.transformers, '__version__', version)

    loader_mixtral_hf.verify_transformers_version()


@pytest.mark.parametrize('version', ['3.99.0', '4.35.2', '4.36.0rc1'])
def test_unsupported_transformers_versions(monkeypatch, version):
    monkeypatch.setattr(loader_mixtral_hf.transformers, '__version__', version)

    with pytest.raises(AssertionError, match=r'requires transformers>=4\.36\.0, found '):
        loader_mixtral_hf.verify_transformers_version()


def _make_mlp_layers(layout):
    num_experts, hidden_size, intermediate_size = 3, 4, 6
    args = SimpleNamespace(num_experts=num_experts)
    gate = torch.arange(num_experts * hidden_size, dtype=torch.float32).reshape(
        num_experts, hidden_size
    )
    w1 = torch.arange(num_experts * intermediate_size * hidden_size, dtype=torch.float32).reshape(
        num_experts, intermediate_size, hidden_size
    )
    w3 = w1 + 1000
    w2 = (
        torch.arange(num_experts * hidden_size * intermediate_size, dtype=torch.float32).reshape(
            num_experts, hidden_size, intermediate_size
        )
        + 2000
    )

    if layout == 'legacy':
        hf_experts = torch.nn.ModuleList()
        for expert_idx in range(num_experts):
            expert = torch.nn.Module()
            expert.w1 = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
            expert.w2 = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
            expert.w3 = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
            with torch.no_grad():
                expert.w1.weight.copy_(w1[expert_idx])
                expert.w2.weight.copy_(w2[expert_idx])
                expert.w3.weight.copy_(w3[expert_idx])
            hf_experts.append(expert)
        moe_attribute = 'block_sparse_moe'
    else:
        hf_experts = torch.nn.Module()
        hf_experts.gate_up_proj = torch.nn.Parameter(torch.cat([w1, w3], dim=1))
        hf_experts.down_proj = torch.nn.Parameter(w2)
        hf_experts.is_transposed = False
        moe_attribute = 'mlp'

    hf_moe = SimpleNamespace(gate=SimpleNamespace(weight=gate), experts=hf_experts)
    hf_layer = SimpleNamespace(**{moe_attribute: hf_moe})
    local_experts = [
        SimpleNamespace(
            linear_fc1=torch.nn.Linear(hidden_size, 2 * intermediate_size, bias=False),
            linear_fc2=torch.nn.Linear(intermediate_size, hidden_size, bias=False),
        )
        for _ in range(num_experts)
    ]
    layer = SimpleNamespace(
        mlp=SimpleNamespace(
            router=torch.nn.Linear(hidden_size, num_experts, bias=False),
            experts=SimpleNamespace(local_experts=local_experts),
        )
    )
    return args, layer, hf_layer, gate, w1, w2, w3


@pytest.mark.parametrize('layout', ['legacy', 'fused'])
def test_set_mlp_state_preserves_router_and_expert_weights(layout):
    args, layer, hf_layer, gate, w1, w2, w3 = _make_mlp_layers(layout)

    loader_mixtral_hf.set_mlp_state(args, layer, hf_layer)

    torch.testing.assert_close(layer.mlp.router.weight, gate, rtol=0, atol=0)
    for expert_idx, expert in enumerate(layer.mlp.experts.local_experts):
        gate_weight, up_weight = expert.linear_fc1.weight.chunk(2, dim=0)
        torch.testing.assert_close(gate_weight, w1[expert_idx], rtol=0, atol=0)
        torch.testing.assert_close(up_weight, w3[expert_idx], rtol=0, atol=0)
        torch.testing.assert_close(expert.linear_fc2.weight, w2[expert_idx], rtol=0, atol=0)


def test_transposed_fused_experts_are_rejected():
    args, layer, hf_layer, *_ = _make_mlp_layers('fused')
    hf_layer.mlp.experts.is_transposed = True

    with pytest.raises(AssertionError, match='Transposed HF expert weight layout is not supported'):
        loader_mixtral_hf.set_mlp_state(args, layer, hf_layer)


# The parallel_state setters that override the global ranks and sizes of the grid.
_RANK_AND_SIZE_OVERRIDES = (
    'set_tensor_model_parallel_rank',
    'set_pipeline_model_parallel_rank',
    'set_virtual_pipeline_model_parallel_rank',
    'set_expert_model_parallel_rank',
    'set_tensor_model_parallel_world_size',
    'set_pipeline_model_parallel_world_size',
    'set_virtual_pipeline_model_parallel_world_size',
    'set_expert_model_parallel_world_size',
)


class _ModelBuilt(Exception):
    """Stops the loader once it has built its model."""


@pytest.fixture
def no_global_process_groups():
    """torch.distributed is initialized, as in the loader process, and no grid is registered."""
    Utils.initialize_distributed()
    parallel_state.destroy_model_parallel()
    yield
    parallel_state.destroy_model_parallel()


def test_loader_builds_its_model_on_its_collection(monkeypatch, no_global_process_groups):
    """The loader builds its model on a collection it registers with parallel_state, without
    overriding the global ranks and sizes."""
    margs = SimpleNamespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        num_layers=2,
        hidden_size=16,
        seq_length=32,
        num_attention_heads=4,
        max_position_embeddings=32,
        position_embedding_type='rope',
        tokenizer_type='Llama2Tokenizer',
        disable_bias_linear=True,
        add_bias_linear=False,
        params_dtype=torch.bfloat16,
        swiglu=True,
        bert_binary_head=False,
        untie_embeddings_and_output_weights=True,
        vocab_size=32,
        num_experts=2,
    )
    built = {}

    def load_checkpoint_to_model(args, pg_collection=None):
        built.update(
            args=args,
            pg_collection=pg_collection,
            registered=parallel_state.get_registered_process_groups(),
        )
        raise _ModelBuilt

    def forbidden(name):
        def override(*args, **kwargs):
            raise AssertionError(f"the loader must not call parallel_state.{name}")

        return override

    for name in _RANK_AND_SIZE_OVERRIDES:
        monkeypatch.setattr(parallel_state, name, forbidden(name))
    # The loader rewrites sys.argv and sys.path and sets a LanguageModule class attribute.
    monkeypatch.setattr(sys, 'argv', list(sys.argv))
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setattr(LanguageModule, 'embedding_warning_printed', False, raising=False)
    monkeypatch.setattr(loader_mixtral_hf, 'verify_transformers_version', lambda: None)
    monkeypatch.setattr(loader_mixtral_hf, 'load_args_from_checkpoint', lambda args: None)
    monkeypatch.setattr(loader_mixtral_hf, 'load_checkpoint_to_model', load_checkpoint_to_model)
    monkeypatch.setattr('megatron.training.arguments.parse_args', lambda: margs)
    monkeypatch.setattr('megatron.training.arguments.validate_args', lambda args: args)
    monkeypatch.setattr(
        'megatron.training.argument_utils.inference_cfg_container_from_args', mock.Mock()
    )
    monkeypatch.setattr('megatron.training.global_vars.set_global_variables', mock.Mock())
    args = SimpleNamespace(
        megatron_path=None, load_dir='unused', tokenizer_model='unused', model_type='GPT'
    )

    with pytest.raises(_ModelBuilt):
        loader_mixtral_hf._load_checkpoint(mock.Mock(), args)

    assert built['args'] is margs
    assert built['pg_collection'] is not None
    assert built['registered'] is built['pg_collection']
    for name in ('tp', 'pp', 'ep'):
        assert getattr(built['pg_collection'], name).size() == 1, name
        assert getattr(built['pg_collection'], name).rank() == 0, name


def test_model_is_built_on_the_given_collection(monkeypatch):
    import gpt_builders
    import model_provider

    provider = mock.Mock(side_effect=_ModelBuilt)
    monkeypatch.setattr('transformers.MixtralForCausalLM.from_pretrained', mock.Mock())
    monkeypatch.setattr(model_provider, 'model_provider', provider)
    pg_collection = object()

    with pytest.raises(_ModelBuilt):
        loader_mixtral_hf.load_checkpoint_to_model(SimpleNamespace(load='unused'), pg_collection)

    provider.assert_called_once_with(
        gpt_builders.gpt_builder, pre_process=True, post_process=True, pg_collection=pg_collection
    )
