# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compare complete Hybrid losses and gradients across context-parallel layouts."""

import copy
import json

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.context_parallel import get_batches_on_this_cp_rank
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.engram.hybrid_adapter import EngramHybridProvider
from megatron.core.transformer.engram.layer import ParallelSequenceLayout
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(Utils.world_size != 2, reason='requires exactly two ranks')


def _model(cp_size, backend, memory_id=1):
    pattern = '*-*-'
    config = TransformerConfig(
        num_layers=4,
        hidden_size=128,
        num_attention_heads=2,
        kv_channels=64,
        ffn_hidden_size=256,
        context_parallel_size=cp_size,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        add_bias_linear=False,
        normalization='RMSNorm',
        hidden_dropout=0.0,
        attention_dropout=0.0,
        linear_cp_layout='contiguous',
        attention_cp_layout='zigzag',
        engram_layer_ids=[memory_id] if backend else None,
        engram_hash_table_min_sizes=[101, 103],
        engram_embedding_dim_per_ngram=16,
        engram_num_hash_heads_per_ngram=2,
        engram_table_backend=backend or 'local',
        engram_seed=7,
    )
    provider = None
    if backend:
        provider = ModuleSpec(
            module=EngramHybridProvider,
            params=dict(tokenizer_lookup=torch.arange(256), pad_id=0, hybrid_layer_pattern=pattern),
        )
    return (
        HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            hybrid_layer_pattern=pattern,
            vocab_size=256,
            max_sequence_length=128,
            position_embedding_type='rope',
            token_context_provider_spec=provider,
        )
        .cuda()
        .bfloat16()
    )


def _run(model):
    positions = torch.arange(128, device='cuda').view(1, -1)
    tokens = (positions * 37 + positions.square() * 11 + 5) % 255 + 1
    batch = get_batches_on_this_cp_rank(
        {
            'tokens': tokens,
            'position_ids': positions,
            'labels': (tokens + 31) % 256,
            'loss_mask': torch.ones_like(tokens, dtype=torch.float32),
            'attention_mask': None,
        },
        boundary_layout='contiguous',
        is_hybrid_cp=False,
        cp_group=model.pg_collection.cp,
        additional_layouts={'zigzag'},
        tp_group=model.pg_collection.tp,
    )
    local = batch.get_batch()
    losses = model(
        local['tokens'],
        local['position_ids'],
        None,
        labels=local['labels'],
        loss_mask=local['loss_mask'],
        cp_batch=batch,
    )
    # Local mean applies the CP compensation used by the native ordinary-loss
    # path. Replicated gradients then average over DP x CP, while row_a2a has
    # already reduced and normalized its owner-local sparse gradients.
    losses.mean().backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        if parameter.grad.is_sparse:
            gradients[name] = parameter.grad.to_dense().float().cpu()
        else:
            gradient = parameter.grad.float()
            torch.distributed.all_reduce(gradient, group=model.pg_collection.dp_cp)
            gradient.div_(model.pg_collection.dp_cp.size())
            gradients[name] = gradient.cpu()
    gathered = [torch.empty_like(losses) for _ in range(model.pg_collection.cp.size())]
    torch.distributed.all_gather(gathered, losses.detach(), group=model.pg_collection.cp)
    return torch.cat(gathered, dim=1).float().cpu(), gradients


@pytest.mark.parametrize('backend', [None, 'local', 'row_a2a'])
def test_first_attention_memory_cp_model_loss_and_gradients(monkeypatch, backend):
    """Compare the whole CP chain with identical inputs to the sensitive Engram gate.

    Placement before the first attention avoids amplifying an earlier BF16 attention
    kernel's rounding through the original sqrt(abs(score)) derivative near zero.
    The separate fixed-input test covers the second attention's memory module.
    """
    Utils.initialize_model_parallel()
    monkeypatch.setenv('NVTE_ALLOW_NONDETERMINISTIC_ALGO', '1')
    try:
        torch.manual_seed(2026)
        model_parallel_cuda_manual_seed(2026)
        reference = _model(1, backend, memory_id=0)
        state = copy.deepcopy(reference.state_dict())
        expected_loss, expected_gradients = _run(reference)
        del reference
    finally:
        Utils.destroy_model_parallel()

    Utils.initialize_model_parallel(context_parallel_size=2)
    monkeypatch.setenv('NVTE_ALLOW_NONDETERMINISTIC_ALGO', '1')
    try:
        actual = _model(2, backend, memory_id=0)
        actual.load_state_dict(state, strict=True)
        actual_loss, actual_gradients = _run(actual)
        errors = []
        for name, expected in expected_gradients.items():
            found = actual_gradients[name]
            difference = torch.linalg.vector_norm(found - expected).item()
            norm = torch.linalg.vector_norm(expected).item()
            errors.append(
                {'name': name, 'relative_l2': difference / max(norm, 1e-12), 'norm': norm}
            )
        print(json.dumps({'backend': backend, 'gradients': errors}, sort_keys=True), flush=True)
        # CP selects a different TE attention reduction kernel. A 2% relative
        # L2 limit bounds BF16 accumulation differences (about 2.6 BF16 eps),
        # while still detecting token permutations and missing CP scale factors.
        failures = [item for item in errors if item['relative_l2'] > 0.02]
        assert not failures, failures
        # Main CE is the token-weighted mean. Individual BF16 attention results
        # may differ by one logit ULP even when the feature is disabled.
        torch.testing.assert_close(actual_loss.mean(), expected_loss.mean(), rtol=0, atol=2e-4)
    finally:
        Utils.destroy_model_parallel()


def _fixed_engram(engram, hidden, upstream):
    cp_group = engram.pg_collection.cp
    cp_size = cp_group.size()
    positions = torch.arange(128, device='cuda').view(1, -1)
    tokens = (positions * 37 + positions.square() * 11 + 5) % 255 + 1
    local_hidden = (
        ParallelSequenceLayout._select_cp(hidden, 0, cp_group=cp_group)
        .detach()
        .requires_grad_(True)
    )
    local_tokens = ParallelSequenceLayout._select_cp(tokens, 1, cp_group=cp_group)
    output = engram(local_hidden, 1, engram.compress_input_ids(local_tokens))
    local_upstream = ParallelSequenceLayout._select_cp(upstream, 0, cp_group=cp_group)
    output.backward(local_upstream * cp_size)
    gradients = {}
    for name, parameter in engram.named_parameters():
        if parameter.grad.is_sparse:
            gradients[name] = parameter.grad.to_dense().float().cpu()
        else:
            gradient = parameter.grad.float()
            torch.distributed.all_reduce(gradient, group=engram.pg_collection.dp_cp)
            gradient.div_(engram.pg_collection.dp_cp.size())
            gradients[name] = gradient.cpu()
    chunks = [torch.empty_like(local_hidden.grad) for _ in range(cp_size)]
    torch.distributed.all_gather(chunks, local_hidden.grad, group=cp_group)
    gradients['input_hidden'] = (
        ParallelSequenceLayout._restore_cp(chunks, 0).float().cpu() / cp_size
    )
    outputs = [torch.empty_like(output) for _ in range(cp_size)]
    torch.distributed.all_gather(outputs, output.detach(), group=cp_group)
    return ParallelSequenceLayout._restore_cp(outputs, 0).cpu(), gradients


@pytest.mark.parametrize('backend', ['local', 'row_a2a'])
def test_second_attention_memory_cp_fixed_inputs_and_gradients(monkeypatch, backend):
    """Hold complete hidden states and upstream gradients fixed across CP layouts."""
    Utils.initialize_model_parallel()
    monkeypatch.setenv('NVTE_ALLOW_NONDETERMINISTIC_ALGO', '1')
    try:
        torch.manual_seed(2026)
        model_parallel_cuda_manual_seed(2026)
        reference = _model(1, backend)
        state = copy.deepcopy(reference.state_dict())
        generator = torch.Generator(device='cuda').manual_seed(2027)
        hidden = torch.randn(128, 1, 128, dtype=torch.bfloat16, device='cuda', generator=generator)
        upstream = torch.randn(hidden.shape, dtype=hidden.dtype, device='cuda', generator=generator)
        expected, gradients = _fixed_engram(reference.decoder.layers[2].engram, hidden, upstream)
        del reference
    finally:
        Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(context_parallel_size=2)
    monkeypatch.setenv('NVTE_ALLOW_NONDETERMINISTIC_ALGO', '1')
    try:
        model = _model(2, backend)
        model.load_state_dict(state, strict=True)
        actual, found_gradients = _fixed_engram(model.decoder.layers[2].engram, hidden, upstream)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        failures = {}
        for name, gradient in gradients.items():
            relative_l2 = (
                torch.linalg.vector_norm(found_gradients[name] - gradient)
                / torch.linalg.vector_norm(gradient).clamp_min(1e-12)
            ).item()
            if relative_l2 > 0.02:
                failures[name] = relative_l2
        assert not failures, failures
        torch.testing.assert_close(
            found_gradients['input_hidden'], gradients['input_hidden'], rtol=0, atol=0
        )
        if backend == 'row_a2a':
            name = 'layers.1.multi_head_embedding.embedding.weight'
            torch.testing.assert_close(found_gradients[name], gradients[name], rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()
