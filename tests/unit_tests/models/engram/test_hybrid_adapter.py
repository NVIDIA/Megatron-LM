# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused tests for layout, module ownership, and explicit microbatch context."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from megatron.core import recompute
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.engram import hybrid_adapter
from megatron.core.transformer.engram.addressing import NgramHashMapping
from megatron.core.transformer.engram.hybrid_adapter import (
    EngramAttentionLayer,
    EngramHybridProvider,
    resolve_engram_targets,
)
from megatron.core.transformer.engram.tokenizer import CompressedTokenizer
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


def _provider(**changes):
    values = dict(
        num_layers=4,
        engram_layer_ids=[1],
        engram_target_layer_indices=[2],
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        cuda_graph_impl='none',
        linear_cp_layout='zigzag',
        attention_cp_layout='zigzag',
    )
    values.update(changes)
    return EngramHybridProvider(
        config=SimpleNamespace(**values),
        tokenizer_lookup=torch.arange(8),
        pad_id=0,
        hybrid_layer_pattern='*-*E',
        pg_collection=SimpleNamespace(cp=SimpleNamespace(size=lambda: 1)),
    )


def test_target_mapping_preserves_memory_identity_and_hashes():
    pattern = '*-' + '*E' * 11
    assert resolve_engram_targets([1], pattern) == {2: 1}
    assert resolve_engram_targets([1], pattern, [4]) == {4: 1}
    tokenizer = CompressedTokenizer(torch.arange(8))

    def mapping(ids):
        return NgramHashMapping(
            hash_table_min_sizes=[17, 19],
            max_ngram_size=3,
            num_hash_heads_per_ngram=2,
            layer_ids=ids,
            pad_id=0,
            seed=123,
            compressed_tokenizer=tokenizer,
        )

    original = mapping([1])
    relocated = mapping(list(resolve_engram_targets([1], pattern, [4]).values()))
    tokens = torch.tensor([[1, 3, 2, 4]])
    assert torch.equal(
        original.forward_compressed(tokens, 1), relocated.forward_compressed(tokens, 1)
    )
    assert original.hash_moduli_by_layer == relocated.hash_moduli_by_layer


@pytest.mark.parametrize(
    ('memory_ids', 'pattern', 'targets'),
    [
        ([1], '*-*E', [3]),
        ([1, 2], '*-*E', [2, 2]),
        ([1, 1], '*-*E', None),
        ([2], '*-*E', None),
        ([1], '*-*E/Z', [2]),
        ([0], 'MGE', None),
        ([1], '*-*E', [-1]),
    ],
)
def test_invalid_targets_fail_before_model_construction(memory_ids, pattern, targets):
    with pytest.raises(ValueError):
        resolve_engram_targets(memory_ids, pattern, targets)


@pytest.mark.parametrize(
    'setting',
    [
        'enable_mhc_connections',
        'fine_grained_activation_offloading',
        'overlap_moe_expert_parallel_comm',
        'use_megatron_fsdp',
        'fp8',
    ],
)
def test_provider_rejects_unsupported_combinations(setting):
    with pytest.raises(ValueError, match='Engram does not support'):
        _provider(**{setting: True})


def test_specs_replace_only_selected_attention_without_mutating_defaults():
    provider = _provider()
    original = ModuleSpec(
        module=TransformerLayer,
        params={'hidden_dropout': 0.0},
        submodules=TransformerLayerSubmodules(),
    )
    layer_config = AttentionLayerConfig.__new__(AttentionLayerConfig)
    overrides = provider.layer_spec_overrides(
        SimpleNamespace(attention_layer=original), [None, None, layer_config, None], 0
    )
    assert set(overrides) == {2}
    assert overrides[2].module is EngramAttentionLayer
    assert overrides[2].params['hidden_dropout'] == 0.0
    assert overrides[2].params['memory_id'] == 1
    assert overrides[2].params['engram_model_config'] is provider.config
    assert overrides[2].submodules is original.submodules
    assert original.module is TransformerLayer
    assert original.params == {'hidden_dropout': 0.0}


def test_layer_owns_its_parameters_and_provider_has_no_stale_context(monkeypatch):
    def layer_init(self, config, layer_number, **kwargs):
        nn.Module.__init__(self)
        self.config = config
        self.layer_number = layer_number
        self.pg_collection = object()

    def parent_forward(self, hidden_states, **kwargs):
        assert 'token_context' not in kwargs
        return hidden_states * 3, None

    class TinyEngram(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.register_buffer('lookup', kwargs['tokenizer_lookup'])
            assert kwargs['local_layer_ids'] == [1]

        def compress_input_ids(self, tokens):
            return self.lookup[tokens]

        def forward(self, hidden, memory_id, compressed):
            assert memory_id == 1
            return hidden + self.weight * compressed.T.unsqueeze(-1)

    monkeypatch.setattr(TransformerLayer, '__init__', layer_init)
    monkeypatch.setattr(TransformerLayer, 'forward', parent_forward)
    monkeypatch.setattr(hybrid_adapter, 'Engram', TinyEngram)
    layer_config = AttentionLayerConfig.__new__(AttentionLayerConfig)
    provider = _provider()
    layer = EngramAttentionLayer(
        config=layer_config,
        layer_number=3,
        memory_id=1,
        engram_model_config=provider.config,
        tokenizer_lookup=provider.tokenizer_lookup,
        pad_id=provider.pad_id,
    )
    model = nn.Module()
    model.decoder = nn.ModuleList([layer])
    model.provider = provider
    assert set(model.state_dict()) == {'decoder.0.engram.weight', 'decoder.0.engram.lookup'}
    assert len(list(model.parameters())) == 1
    assert set(layer.state_dict()) == {'engram.weight', 'engram.lookup'}
    assert not any(isinstance(value, nn.Module) for value in vars(provider).values())

    prepare_kwargs = dict(inference_context=None, packed_seq_params=None, cp_batch=None)
    first = provider.prepare(torch.tensor([[1, 2]]), **prepare_kwargs)
    second = provider.prepare(torch.tensor([[4, 5]]), **prepare_kwargs)
    hidden = torch.zeros(2, 1, 1)
    first_output, _ = layer(hidden, token_context=first)
    second_output, _ = layer(hidden, token_context=second)
    torch.testing.assert_close(first_output, 3 * first.T.unsqueeze(-1).float())
    torch.testing.assert_close(second_output, 3 * second.T.unsqueeze(-1).float())
    with pytest.raises(ValueError, match='explicit token context'):
        layer(hidden)
    with pytest.raises(ValueError, match='requires input_ids'):
        provider.prepare(None, **prepare_kwargs)


@pytest.mark.parametrize('method', ['uniform', 'block'])
def test_recompute_retains_each_microbatch_and_only_dispatches_to_consumers(monkeypatch, method):
    class PlainLayer(TransformerLayer):
        def __init__(self):
            nn.Module.__init__(self)
            # GraphableMegatronModule.__call__ consults config even without CUDA graphs.
            self.config = SimpleNamespace(cuda_graph_impl='none')

        def forward(self, hidden_states, **kwargs):
            assert 'token_context' not in kwargs
            return hidden_states + 1, None

    class ConsumerLayer(PlainLayer):
        accepts_token_context = True

        def forward(self, hidden_states, *, token_context, **kwargs):
            return hidden_states.square() * token_context, None

    monkeypatch.setattr(
        recompute.tensor_parallel,
        'checkpoint',
        lambda function, distribute, *args: torch_checkpoint(function, *args, use_reentrant=False),
    )
    stack = SimpleNamespace(
        layers=[PlainLayer(), ConsumerLayer()],
        num_layers_per_pipeline_rank=2,
        config=SimpleNamespace(
            recompute_method=method,
            recompute_num_layers=2,
            fp8=False,
            fp4=False,
            distribute_saved_activations=False,
        ),
    )
    hidden = torch.tensor([2.0], requires_grad=True)
    outputs = []
    for context in (torch.tensor([3.0]), torch.tensor([7.0])):
        outputs.append(
            recompute.checkpointed_forward(
                stack,
                hidden,
                None,
                None,
                None,
                None,
                None,
                None,
                use_inner_quantization_context=False,
                token_context=context,
            )
        )
    (outputs[0] + outputs[1]).sum().backward()
    torch.testing.assert_close(outputs[0], torch.tensor([27.0]))
    torch.testing.assert_close(outputs[1], torch.tensor([63.0]))
    torch.testing.assert_close(hidden.grad, torch.tensor([60.0]))


def test_pipeline_and_mtp_patterns_preserve_main_stack_targets():
    assert resolve_engram_targets([1], 'M*G-|*E/*E') == {4: 1}
    assert resolve_engram_targets([7], 'M*G-|*E/*E', [4]) == {4: 7}
    provider = _provider(pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2)
    layer_config = AttentionLayerConfig.__new__(AttentionLayerConfig)
    spec = ModuleSpec(module=TransformerLayer, submodules=TransformerLayerSubmodules())
    submodules = SimpleNamespace(attention_layer=spec)
    assert not provider.layer_spec_overrides(submodules, [None, None], 0)
    assert set(provider.layer_spec_overrides(submodules, [layer_config, None], 2)) == {2}


def test_provider_uses_attention_view_instead_of_contiguous_boundary_tokens():
    from megatron.core.context_parallel import ContextParallelBatch

    provider = _provider(linear_cp_layout='contiguous')
    contiguous_tokens = torch.tensor([[0, 1, 2, 3]])
    attention_tokens = torch.tensor([[0, 1, 6, 7]])
    cp_batch = ContextParallelBatch(
        boundary_layout='contiguous',
        batches_by_layout={
            'contiguous': {'tokens': contiguous_tokens},
            'zigzag': {'tokens': attention_tokens},
        },
        packed_seq_params_by_layout={'contiguous': None, 'zigzag': None},
    )
    actual = provider.prepare(
        contiguous_tokens, inference_context=None, packed_seq_params=None, cp_batch=cp_batch
    )
    assert torch.equal(actual, attention_tokens)
