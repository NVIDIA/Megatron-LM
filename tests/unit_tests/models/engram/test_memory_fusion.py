# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy
import math

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.engram import Engram, MultiHeadEmbedding, ShortConv
from tests.unit_tests.models.engram.test_addressing import make_config


def _lookup():
    return torch.arange(64)


def test_multi_head_embedding_uses_layer_local_offsets():
    memory = MultiHeadEmbedding([5, 7, 11, 13], D=3)
    assert torch.equal(memory.offsets, torch.tensor([0, 5, 12, 23]))
    ids = torch.tensor([[[1, 2, 3, 4]]])
    expected = F.embedding(ids + memory.offsets, memory.embedding.weight)
    torch.testing.assert_close(memory(ids), expected)
    assert memory.embedding.num_embeddings == 36
    assert memory.embedding.weight.engram_table_metadata.lr_multiplier == 5.0
    assert memory.embedding.weight.engram_table_metadata.weight_decay == 0.0


def test_short_conv_matches_padding_crop_norm_and_activation():
    module = ShortConv(hidden_size=4, kernel_size=3, dilation=2)
    with torch.no_grad():
        module.conv.weight.copy_(torch.arange(12).view(4, 1, 3) / 10)
    values = torch.randn(2, 5, 4)
    normalized = F.rms_norm(values, (4,), module.norm.weight, module.norm.eps)
    convolution = F.conv1d(
        normalized.transpose(1, 2), module.conv.weight, padding=4, dilation=2, groups=4
    )
    expected = F.silu(convolution[..., :5].transpose(1, 2))
    torch.testing.assert_close(module(values), expected)
    assert module.conv.bias is None


def test_engram_forward_matches_direct_equations_and_roundtrips():
    config = make_config()
    module = Engram(engram_config=config, tokenizer_lookup=_lookup())
    layer = module.layers["0"]
    hidden = torch.randn(4, 2, config.hidden_size)
    raw_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    compressed_input_ids = module.compress_input_ids(raw_ids)

    hashes = module.hash_mapping.forward_compressed(compressed_input_ids, 0)
    embeddings = layer.multi_head_embedding(hashes).flatten(start_dim=-2)
    key = layer.key_proj(embeddings)
    query = hidden.transpose(0, 1)
    score = (layer.norm1(key) * layer.norm2(query)).sum(-1) / math.sqrt(config.hidden_size)
    score = score.abs().clamp_min(1e-6).sqrt() * score.sign()
    value = score.sigmoid().unsqueeze(-1) * layer.value_proj(embeddings)
    delta = value + layer.short_conv(value)
    expected = hidden + delta.transpose(0, 1)
    torch.testing.assert_close(module(hidden, 0, compressed_input_ids), expected)

    restored = Engram(engram_config=config, tokenizer_lookup=_lookup())
    result = restored.load_state_dict(copy.deepcopy(module.state_dict()), strict=True)
    assert not result.missing_keys and not result.unexpected_keys


def test_single_stream_fusion_matches_independent_forward_and_gradient_reference():
    config = make_config()
    layer = Engram(engram_config=config, tokenizer_lookup=_lookup()).layers["0"]
    with torch.no_grad():
        layer.short_conv.conv.weight.uniform_(-0.2, 0.2)
    hidden = torch.randn(7, 3, config.hidden_size, requires_grad=True)
    embeddings = torch.randn(3, 7, config.engram_hidden_size, requires_grad=True)
    expected_hidden = hidden.detach().clone().requires_grad_()
    expected_embeddings = embeddings.detach().clone().requires_grad_()
    expected_params = {
        name: param.detach().clone().requires_grad_()
        for name, param in layer.named_parameters()
        if not name.startswith("multi_head_embedding.")
    }
    width = (config.hidden_size,)
    key = F.linear(
        expected_embeddings, expected_params["key_proj.weight"], expected_params["key_proj.bias"]
    )
    key = F.rms_norm(key, width, expected_params["norm1.weight"], layer.norm1.eps)
    query = F.rms_norm(
        expected_hidden.transpose(0, 1), width, expected_params["norm2.weight"], layer.norm2.eps
    )
    score = (key * query).sum(-1) / math.sqrt(config.hidden_size)
    gate = (score.sign() * score.abs().clamp_min(1.0e-6).sqrt()).sigmoid()
    value = gate.unsqueeze(-1) * F.linear(
        expected_embeddings,
        expected_params["value_proj.weight"],
        expected_params["value_proj.bias"],
    )
    normalized = F.rms_norm(
        value, width, expected_params["short_conv.norm.weight"], layer.short_conv.norm.eps
    )
    convolution = F.conv1d(
        normalized.transpose(1, 2),
        expected_params["short_conv.conv.weight"],
        padding=(config.kernel_size - 1) * config.max_ngram_size,
        dilation=config.max_ngram_size,
        groups=config.hidden_size,
    )[..., : hidden.shape[0]].transpose(1, 2)
    expected = expected_hidden + (value + F.silu(convolution)).transpose(0, 1)
    actual = layer(hidden, embeddings)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(hidden.grad, expected_hidden.grad)
    torch.testing.assert_close(embeddings.grad, expected_embeddings.grad)
    actual_params = dict(layer.named_parameters())
    for name, param in expected_params.items():
        torch.testing.assert_close(actual_params[name].grad, param.grad)


def test_checkpoint_validates_architecture_and_rejects_unknown_schema():
    config = make_config()
    source = Engram(engram_config=config, tokenizer_lookup=_lookup())
    state = copy.deepcopy(source.state_dict())

    changed_lookup = _lookup()
    changed_lookup[-1] = 0
    restored = Engram(engram_config=config, tokenizer_lookup=changed_lookup)
    result = restored.load_state_dict(copy.deepcopy(state), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    torch.testing.assert_close(restored.compressed_tokenizer.lookup_table, _lookup())

    mismatched_state = copy.deepcopy(state)
    mismatched_state["_extra_state"]["seed"] += 1
    with pytest.raises(ValueError, match="checkpoint architecture mismatch"):
        restored.load_state_dict(mismatched_state, strict=True)

    state["_extra_state"]["schema_version"] = -1
    with pytest.raises(ValueError, match="Unsupported Engram checkpoint schema version"):
        restored.load_state_dict(state, strict=True)


def test_engram_keeps_continuous_sequence_history():
    config = make_config()
    module = Engram(engram_config=config, tokenizer_lookup=_lookup())
    raw_ids = torch.tensor([[1, 2, 3, 4, 5]])
    hidden = torch.randn(5, 1, config.hidden_size)
    continuous = module(hidden, 0, module.compress_input_ids(raw_ids))
    separate = torch.cat(
        (
            module(hidden[:2], 0, module.compress_input_ids(raw_ids[:, :2])),
            module(hidden[2:], 0, module.compress_input_ids(raw_ids[:, 2:])),
        ),
        dim=0,
    )
    assert not torch.allclose(continuous[2:], separate[2:])


def test_biases_initialization_and_selected_layer_gradients():
    config = make_config()
    module = Engram(engram_config=config, tokenizer_lookup=_lookup())
    layer = module.layers["0"]
    assert layer.value_proj.bias is not None
    assert layer.key_proj.bias is not None
    assert torch.count_nonzero(layer.short_conv.conv.weight).item() == 0

    hidden = torch.randn(4, 2, config.hidden_size, requires_grad=True)
    output = module(
        hidden, 0, module.compress_input_ids(torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]))
    )
    output.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in layer.parameters()
    )
    assert all(parameter.grad is None for parameter in module.layers["2"].parameters())
