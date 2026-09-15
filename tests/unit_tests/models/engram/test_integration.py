# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.engram import Engram
from megatron.core.transformer.engram.hybrid_adapter import (
    EngramAttentionLayer,
    EngramHybridProvider,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.models.engram import engram_context_provider_spec
from tests.unit_tests.test_utilities import Utils


def make_transformer_config(**kwargs):
    values = dict(
        num_layers=6,
        hidden_size=32,
        num_attention_heads=4,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        add_bias_linear=False,
        engram_hash_table_min_sizes=[17, 19],
        engram_max_ngram_size=3,
        engram_embedding_dim_per_ngram=8,
        engram_num_hash_heads_per_ngram=2,
        engram_layer_ids=[0, 2],
        engram_seed=7,
    )
    values.update(kwargs)
    return TransformerConfig(**values)


def make_hybrid(config, pattern="*-*-*-"):
    provider = None
    if config.engram_layer_ids:
        provider = ModuleSpec(
            module=EngramHybridProvider,
            params=dict(tokenizer_lookup=torch.arange(64), pad_id=2, hybrid_layer_pattern=pattern),
        )
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        hybrid_layer_pattern=pattern,
        vocab_size=64,
        max_sequence_length=8,
        position_embedding_type="none",
        token_context_provider_spec=provider,
    ).cuda()


def model_inputs():
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device="cuda")
    positions = torch.arange(8, device="cuda").unsqueeze(0)
    mask = torch.triu(torch.ones(8, 8, dtype=torch.bool, device="cuda"), diagonal=1).view(
        1, 1, 8, 8
    )
    return tokens, positions, mask


def test_feature_off_builder_does_not_access_tokenizer(monkeypatch):
    def unexpected_tokenizer_access():
        raise AssertionError("feature-off must not access the tokenizer")

    monkeypatch.setattr("megatron.training.get_tokenizer", unexpected_tokenizer_access)
    assert (
        engram_context_provider_spec(make_transformer_config(engram_layer_ids=None), "*-*-*-")
        is None
    )


def test_engram_requires_lookup_and_main_tokenizer_pad_id():
    config = make_transformer_config()
    with pytest.raises(ValueError, match="precomputed tokenizer lookup"):
        Engram(config=config)
    with pytest.raises(ValueError, match="raw pad token ID"):
        Engram(config=config, tokenizer_lookup=torch.arange(64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_feature_off_hybrid_has_no_engram_state_and_forwards():
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        model = make_hybrid(make_transformer_config(engram_layer_ids=None))
        assert not any(isinstance(layer, EngramAttentionLayer) for layer in model.decoder.layers)
        assert not any("engram" in name for name in model.state_dict())
        with torch.no_grad():
            output = model(*model_inputs())
        assert torch.isfinite(output).all()
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_selected_layers_forward_and_strict_state_roundtrip():
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = make_transformer_config()
        model = make_hybrid(config)
        assert not hasattr(model, 'engram')
        assert list(model.decoder.layers[0].engram.layers) == ['0']
        assert list(model.decoder.layers[4].engram.layers) == ['2']
        first_hashes = model.decoder.layers[0].engram.hash_mapping.hash_moduli_by_layer
        last_hashes = model.decoder.layers[4].engram.hash_mapping.hash_moduli_by_layer
        assert first_hashes == last_hashes
        assert first_hashes[0] != first_hashes[2]
        selected = [
            i
            for i, layer in enumerate(model.decoder.layers)
            if isinstance(layer, EngramAttentionLayer)
        ]
        assert selected == [0, 4]
        parameter_ids = [id(p) for _, p in model.named_parameters(remove_duplicate=False)]
        assert len(parameter_ids) == len(set(parameter_ids))
        with torch.no_grad():
            output = model(*model_inputs())
        assert torch.isfinite(output).all()
        restored = make_hybrid(config)
        result = restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        with torch.no_grad():
            torch.testing.assert_close(restored(*model_inputs()), output, rtol=0, atol=0)
        with pytest.raises(NotImplementedError, match="inference"):
            model(*model_inputs(), inference_context=object())
        with pytest.raises(ValueError, match="packed sequences"):
            model(*model_inputs(), packed_seq_params=object())

        original_decoder_forward = model.decoder.forward

        def fail_after_compression(*args, **kwargs):
            assert kwargs["token_context"] is not None
            raise RuntimeError("decoder failed")

        model.decoder.forward = fail_after_compression
        with pytest.raises(RuntimeError, match="decoder failed"):
            model(*model_inputs())
        model.decoder.forward = original_decoder_forward
        with torch.no_grad():
            torch.testing.assert_close(model(*model_inputs()), output, rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()


def test_cli_exposes_only_supported_engram_options(monkeypatch):
    from megatron.training.arguments import parse_args

    argv = [
        "pretrain_hybrid.py",
        "--hybrid-layer-pattern",
        "*-" * 16,
        "--hidden-size",
        "8",
        "--num-attention-heads",
        "2",
        "--seq-length",
        "4",
        "--max-position-embeddings",
        "4",
        "--micro-batch-size",
        "1",
        "--num-experts",
        "8",
        "--engram-layer-ids",
        "1",
        "9",
        "--engram-table-backend",
        "row_a2a",
        "--engram-hash-table-min-sizes",
        "17",
        "19",
    ]
    monkeypatch.setattr("sys.argv", argv)
    args = parse_args(ignore_unknown_args=True)
    assert not hasattr(args, "engram_enabled")
    assert not hasattr(args, "engram_schema_version")
    assert not hasattr(args, "engram_recompute_policy")
    assert not hasattr(args, "hc_mult")
    assert args.engram_layer_ids == [1, 9]
    assert args.engram_table_backend == "row_a2a"
    assert args.engram_hash_table_min_sizes == [17, 19]


def test_engram_cli_validation_rejects_incompatible_modes(monkeypatch):
    from megatron.training.arguments import parse_args, validate_engram_args

    def parse(extra):
        argv = [
            "pretrain_hybrid.py",
            "--hybrid-layer-pattern",
            "*-*-",
            "--hidden-size",
            "8",
            "--num-attention-heads",
            "2",
            "--max-position-embeddings",
            "4",
            "--engram-layer-ids",
            "0",
            "--engram-hash-table-min-sizes",
            "17",
            "19",
            "--tokenizer-model",
            "/local/tokenizer",
            *extra,
        ]
        monkeypatch.setattr("sys.argv", argv)
        return parse_args(ignore_unknown_args=True)

    with pytest.raises(ValueError, match="init-model-with-meta-device"):
        validate_engram_args(parse(["--init-model-with-meta-device"]))
    validate_engram_args(parse([]))
    with pytest.raises(ValueError, match="overlap-moe-expert-parallel-comm"):
        validate_engram_args(parse(["--overlap-moe-expert-parallel-comm"]))
    validate_engram_args(
        parse(["--engram-table-backend", "row_a2a", "--ckpt-format", "torch_dist", "--bf16"])
    )
    with pytest.raises(ValueError, match="unity loss scaling"):
        validate_engram_args(parse(["--engram-table-backend", "row_a2a", "--loss-scale", "128"]))
    with pytest.raises(ValueError, match="ckpt-format torch_dist"):
        validate_engram_args(parse(["--engram-table-backend", "row_a2a", "--ckpt-format", "torch"]))
    with pytest.raises(ValueError, match="RowSparseAdam"):
        validate_engram_args(parse(["--engram-table-backend", "row_a2a", "--optimizer", "sgd"]))
    with pytest.raises(ValueError, match="enable-mhc-connections"):
        validate_engram_args(parse(["--enable-mhc-connections"]))


@pytest.mark.parametrize('method', [None, 'uniform', 'block'])
def test_mtp_forward_backward_and_recompute_matches_eager(method):
    Utils.initialize_model_parallel(1, 1)
    try:
        torch.manual_seed(2026)
        model_parallel_cuda_manual_seed(2026)
        config = make_transformer_config(
            mtp_num_layers=1, mtp_loss_scaling_factor=0.3, recompute_granularity=None
        )
        reference = make_hybrid(config, '*-*-*-/*-')
        candidate_config = make_transformer_config(
            mtp_num_layers=1,
            mtp_loss_scaling_factor=0.3,
            recompute_granularity='full' if method else None,
            recompute_method=method,
            recompute_num_layers=1 if method else None,
        )
        candidate = make_hybrid(candidate_config, '*-*-*-/*-')
        candidate.load_state_dict(reference.state_dict(), strict=True)
        assert not any(isinstance(module, Engram) for module in candidate.mtp.modules())
        inputs = model_inputs()
        labels = inputs[0].roll(-1, dims=1)
        mask = torch.ones_like(labels, dtype=torch.float)
        expected = reference(*inputs, labels=labels, loss_mask=mask)
        expected.mean().backward()
        actual = candidate(*inputs, labels=labels, loss_mask=mask)
        actual.mean().backward()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        expected_parameters = dict(reference.named_parameters())
        for name, parameter in candidate.named_parameters():
            other = expected_parameters[name]
            if other.grad is None:
                assert parameter.grad is None, name
            else:
                torch.testing.assert_close(parameter.grad, other.grad, rtol=2e-4, atol=1e-6)
        assert candidate.decoder.layers[0].engram.layers['0'].value_proj.weight.grad is not None
    finally:
        Utils.destroy_model_parallel()
