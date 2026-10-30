# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Liger loss selection through GPT, including MTP training and predictions."""

import argparse
from dataclasses import replace

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.multi_token_prediction import _compute_mtp_acceptance_counts
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def test_cli_exposes_liger_cross_entropy():
    from megatron.training.arguments import add_megatron_arguments

    parser = add_megatron_arguments(argparse.ArgumentParser())
    args = parser.parse_args(
        ["--cross-entropy-loss-fusion", "--cross-entropy-fusion-impl", "liger"]
    )
    assert args.cross_entropy_loss_fusion
    assert args.cross_entropy_fusion_impl == "liger"


@pytest.fixture(params=[1, 2], ids=["tp1", "tp2"])
def model_parallel(request):
    if Utils.world_size % request.param:
        pytest.skip(f"requires world size divisible by TP={request.param}")
    Utils.initialize_model_parallel(request.param, 1)
    model_parallel_cuda_manual_seed(123)
    yield request.param
    Utils.destroy_model_parallel()


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


def test_language_loss_preserves_mtp_acceptance_counts(model_parallel):
    pytest.importorskip("liger_kernel.megatron")
    from megatron.core.models.common.language_module.language_module import LanguageModule

    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        transformer_impl="local",
        tensor_model_parallel_size=model_parallel,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="liger",
    )
    model = LanguageModule(config)
    torch.manual_seed(817)
    full = torch.randn(5, 2, 128, device="cuda")
    expected = full.argmax(-1)
    labels = expected.t().contiguous()
    labels[0, 0] = (labels[0, 0] + 1) % 128
    mask = torch.ones_like(labels)
    mask[-1, -1] = 0
    logits = full.chunk(model_parallel, dim=-1)[model.tp_group.rank()].clone()
    _, predictions = model.compute_language_model_loss(labels, logits, return_argmax=True)
    correct, total = _compute_mtp_acceptance_counts(
        logits, labels, mask, tp_group=model.tp_group, preds=predictions
    )
    torch.testing.assert_close(predictions, expected)
    assert correct.item() == 8
    assert total.item() == 9


@pytest.mark.parametrize("mtp_layers", [None, 1])
def test_gpt_loss_and_parameter_gradients(model_parallel, mtp_layers):
    pytest.importorskip("liger_kernel.megatron")
    if mtp_layers:
        pytest.importorskip(
            "transformer_engine",
            reason="MTP construction currently checks TE inference-layer types",
        )
    config = TransformerConfig(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        ffn_hidden_size=64,
        transformer_impl="local",
        normalization="RMSNorm",
        use_cpu_initialization=True,
        tensor_model_parallel_size=model_parallel,
        mtp_num_layers=mtp_layers,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="liger",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        masked_softmax_fusion=False,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        gradient_accumulation_fusion=False,
        apply_rope_fusion=False,
    )
    actual = _model(config)
    reference = _model(
        replace(config, cross_entropy_loss_fusion=False, cross_entropy_fusion_impl="native")
    )
    reference.load_state_dict(actual.state_dict(), strict=True)
    torch.manual_seed(456)
    tokens = torch.randint(0, 128, (2, 16), device="cuda")
    positions = torch.arange(16, device="cuda").expand_as(tokens)
    mask = torch.triu(torch.ones(1, 1, 16, 16, device="cuda", dtype=torch.bool), diagonal=1)
    labels = torch.randint(0, 128, tokens.shape, device="cuda")
    expected_loss = reference(tokens, positions, mask, labels=labels)
    actual_loss = actual(tokens, positions, mask, labels=labels)
    torch.testing.assert_close(actual_loss, expected_loss, atol=2e-5, rtol=2e-5)
    upstream = torch.linspace(0.1, 1.0, labels.numel(), device="cuda").reshape_as(labels)
    (actual_loss * upstream).sum().backward()
    (expected_loss * upstream).sum().backward()
    for (name, parameter), (ref_name, ref_parameter) in zip(
        actual.named_parameters(), reference.named_parameters()
    ):
        assert name == ref_name
        assert parameter.grad is not None, name
        torch.testing.assert_close(
            parameter.grad, ref_parameter.grad, atol=2e-4, rtol=2e-4, msg=name
        )
