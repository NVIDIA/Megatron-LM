# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Liger norm wiring and forward/backward parity for optional Liger backends in local GPT."""

from dataclasses import replace

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_decoder_layer_specs,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.ops.norm.liger_norm import LigerRMSNorm
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mla_qk_norm_config import QKNormConfigResolver
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from tests.unit_tests.test_utilities import Utils

_LIGER_RMSNORM = dict(
    num_layers=2,
    hidden_size=32,
    num_attention_heads=4,
    transformer_impl="local",
    normalization="RMSNorm",
    use_liger=True,
)


@pytest.mark.parametrize("num_experts", [None, 2])
def test_every_norm_slot_uses_liger(num_experts):
    pytest.importorskip("liger_kernel.megatron")
    config = TransformerConfig(
        **_LIGER_RMSNORM, num_moe_experts=num_experts, qk_layernorm=True, mtp_num_layers=1
    )
    block = get_gpt_decoder_block_spec(config, use_transformer_engine=False, pp_rank=0)
    mtp = get_gpt_mtp_block_spec(config, block, use_transformer_engine=False, pp_rank=0)
    norms = [block.layer_norm]
    for layer in block.layer_specs:
        attention = layer.submodules.self_attention.submodules
        norms += [layer.submodules.input_layernorm, layer.submodules.pre_mlp_layernorm]
        norms += [attention.q_layernorm, attention.k_layernorm]
    for layer in mtp.layer_specs:
        norms += [layer.submodules.enorm, layer.submodules.hnorm, layer.submodules.layer_norm]
    assert norms == [LigerRMSNorm] * (1 + 4 * config.num_layers + 3)


def test_mla_qk_norms_use_liger():
    pytest.importorskip("liger_kernel.megatron")
    config = MLATransformerConfig(
        **_LIGER_RMSNORM, qk_layernorm=True, q_lora_rank=16, kv_lora_rank=16
    )
    layer = get_gpt_decoder_layer_specs(config, use_transformer_engine=False)[0]
    resolved = QKNormConfigResolver(config, layer.submodules.self_attention.submodules).resolve()
    assert resolved["q_layernorm"] is resolved["kv_layernorm"] is LigerRMSNorm


@pytest.fixture(params=[1, 2], ids=["tp1", "tp2"])
def model_parallel(request):
    if Utils.world_size % request.param:
        pytest.skip(f"requires world size divisible by TP={request.param}")
    Utils.initialize_model_parallel(request.param, 1)
    model_parallel_cuda_manual_seed(123)
    yield request.param
    Utils.destroy_model_parallel()


@pytest.fixture
def full_fp32_matmuls():
    """TF32 would turn tiny norm differences into visible output-layer gradient gaps."""
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(precision)


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


@pytest.mark.parametrize(
    "liger_options",
    [
        pytest.param({"use_liger": True}, id="rmsnorm"),
        pytest.param({"use_liger": True, "sequence_parallel": True}, id="rmsnorm_sp"),
    ],
)
@pytest.mark.parametrize("mtp_layers", [None, 1])
def test_gpt_forward_backward_parity(model_parallel, mtp_layers, liger_options, full_fp32_matmuls):
    pytest.importorskip("liger_kernel.megatron")
    if mtp_layers:
        pytest.importorskip(
            "transformer_engine",
            reason="MTP construction currently checks TE inference-layer types",
        )
    if liger_options.get("sequence_parallel") and (model_parallel == 1 or mtp_layers):
        pytest.skip("sequence parallelism needs TP>1; MTP is covered without it")
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
        hidden_dropout=0.0,
        attention_dropout=0.0,
        masked_softmax_fusion=False,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        gradient_accumulation_fusion=False,
        apply_rope_fusion=False,
    )
    actual = _model(replace(config, **liger_options))
    reference = _model(config)
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
    for parameter in actual.parameters():
        if getattr(parameter, "sequence_parallel", False):  # as finalize_model_grads does
            torch.distributed.all_reduce(parameter.grad, group=actual.tp_group)
    for (name, parameter), (ref_name, ref_parameter) in zip(
        actual.named_parameters(), reference.named_parameters()
    ):
        assert name == ref_name
        assert parameter.grad is not None, name
        torch.testing.assert_close(
            parameter.grad, ref_parameter.grad, atol=2e-4, rtol=2e-4, msg=name
        )
