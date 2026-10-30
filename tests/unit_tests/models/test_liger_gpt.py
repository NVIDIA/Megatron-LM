# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native Liger wiring through local GPT specs and model execution."""

import argparse
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_decoder_layer_specs,
    get_gpt_layer_local_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mla_qk_norm_config import QKNormConfigResolver
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _config(**overrides):
    options = dict(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        ffn_hidden_size=64,
        transformer_impl="local",
        normalization="RMSNorm",
        use_liger=True,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        masked_softmax_fusion=False,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        gradient_accumulation_fusion=False,
        apply_rope_fusion=False,
    )
    return TransformerConfig(**(options | overrides))


def _liger_norm():
    pytest.importorskip("liger_kernel.megatron")
    from megatron.core.ops.norm.liger_norm import LigerRMSNorm

    return LigerRMSNorm


@pytest.fixture
def model_parallel(request):
    tp = getattr(request, "param", 1)
    if Utils.world_size % tp:
        pytest.skip(f"requires a world size divisible by TP={tp}")
    Utils.initialize_model_parallel(tp, 1)
    model_parallel_cuda_manual_seed(123)
    yield tp
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("num_experts", [None, 2])
def test_layer_and_block_norm_slots(model_parallel, num_experts):
    norm = _liger_norm()
    config = _config(num_moe_experts=num_experts, qk_layernorm=True)
    block = get_gpt_decoder_block_spec(config, use_transformer_engine=False)
    assert block.layer_norm is norm
    for layer in block.layer_specs:
        assert layer.submodules.input_layernorm is norm
        assert layer.submodules.pre_mlp_layernorm is norm
        attention = layer.submodules.self_attention.submodules
        assert attention.q_layernorm is norm
        assert attention.k_layernorm is norm


def test_mtp_norm_slots(model_parallel):
    norm = _liger_norm()
    config = _config(mtp_num_layers=1)
    block = get_gpt_decoder_block_spec(config, use_transformer_engine=False)
    mtp = get_gpt_mtp_block_spec(config, block, use_transformer_engine=False)
    for layer in mtp.layer_specs:
        for slot in ("enorm", "hnorm", "layer_norm"):
            assert getattr(layer.submodules, slot) is norm


def test_low_level_layer_spec_option():
    norm = _liger_norm()
    layer = get_gpt_layer_local_spec(normalization="RMSNorm", use_liger=True)
    assert layer.submodules.input_layernorm is norm
    assert layer.submodules.pre_mlp_layernorm is norm


def test_local_mla_norm_resolution():
    norm = _liger_norm()
    config = MLATransformerConfig(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        normalization="RMSNorm",
        transformer_impl="local",
        use_liger=True,
        multi_latent_attention=True,
        qk_layernorm=True,
        q_lora_rank=16,
        kv_lora_rank=16,
    )
    layer = get_gpt_decoder_layer_specs(config, use_transformer_engine=False)[0]
    resolved = QKNormConfigResolver(config, layer.submodules.self_attention.submodules).resolve()
    assert resolved["q_layernorm"] is norm
    assert resolved["kv_layernorm"] is norm


def test_training_builder_includes_final_norm(model_parallel):
    from megatron.training.models.gpt import GPTModelConfig, default_layer_spec

    config = GPTModelConfig(transformer=_config())
    block = default_layer_spec(config, vp_stage=None)
    assert block.layer_norm is _liger_norm()


def test_training_builder_rejects_unimplemented_specs():
    from megatron.training.models.gpt import default_layer_spec

    config = SimpleNamespace(transformer=_config(), restore_modelopt_state=True)
    with pytest.raises(ValueError, match="standard local GPT specs"):
        default_layer_spec(config, vp_stage=None)


def test_cli_exposes_liger_norm_option():
    from megatron.training.arguments import add_megatron_arguments

    parser = add_megatron_arguments(argparse.ArgumentParser())
    args = parser.parse_args(["--use-liger"])
    assert args.use_liger


def _model(config):
    block = get_gpt_decoder_block_spec(config, use_transformer_engine=False)
    return GPTModel(
        config=config,
        transformer_layer_spec=block,
        vocab_size=128,
        max_sequence_length=16,
        position_embedding_type="rope",
        share_embeddings_and_output_weights=False,
        mtp_block_spec=(
            get_gpt_mtp_block_spec(config, block, use_transformer_engine=False)
            if config.mtp_num_layers
            else None
        ),
    ).cuda()


def _inputs():
    torch.manual_seed(456)
    tokens = torch.randint(0, 128, (2, 16), device="cuda")
    positions = torch.arange(16, device="cuda").expand_as(tokens)
    mask = torch.triu(torch.ones(1, 1, 16, 16, device="cuda", dtype=torch.bool), diagonal=1)
    labels = torch.randint(0, 128, tokens.shape, device="cuda")
    return tokens, positions, mask, labels


@pytest.mark.parametrize("model_parallel", [1, 2], indirect=True)
@pytest.mark.parametrize("mtp_layers", [None, 1])
def test_gpt_forward_backward_parity(model_parallel, mtp_layers):
    if mtp_layers:
        pytest.importorskip(
            "transformer_engine",
            reason="MTP construction currently checks TE inference-layer types",
        )
    norm = _liger_norm()
    config = _config(tensor_model_parallel_size=model_parallel, mtp_num_layers=mtp_layers)
    actual = _model(config)
    reference = _model(replace(config, use_liger=False))
    reference.load_state_dict(actual.state_dict(), strict=True)
    assert isinstance(actual.decoder.final_layernorm, norm)
    tokens, positions, mask, labels = _inputs()
    expected_loss = reference(tokens, positions, mask, labels=labels)
    actual_loss = actual(tokens, positions, mask, labels=labels)
    torch.testing.assert_close(actual_loss, expected_loss, atol=2e-5, rtol=2e-5)
    gradient = torch.linspace(0.1, 1.0, labels.numel(), device="cuda").reshape_as(labels)
    (actual_loss * gradient).sum().backward()
    (expected_loss * gradient).sum().backward()
    for (name, parameter), (ref_name, ref_parameter) in zip(
        actual.named_parameters(), reference.named_parameters()
    ):
        assert name == ref_name
        assert parameter.grad is not None, name
        torch.testing.assert_close(
            parameter.grad, ref_parameter.grad, atol=2e-4, rtol=2e-4, msg=name
        )


@pytest.mark.parametrize("model_parallel", [2], indirect=True)
def test_sequence_parallel_gpt_backward(model_parallel):
    norm = _liger_norm()
    model = _model(_config(tensor_model_parallel_size=model_parallel, sequence_parallel=True))
    tokens, positions, mask, labels = _inputs()
    loss = model(tokens, positions, mask, labels=labels)
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    norms = [module for module in model.modules() if isinstance(module, norm)]
    assert len(norms) == 2 * model.config.num_layers + 1
    for module in norms:
        assert module.weight.sequence_parallel
        assert module.weight.grad is not None
        assert torch.isfinite(module.weight.grad).all()
