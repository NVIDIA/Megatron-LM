# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MTP compute can be disabled without changing model checkpoint topology."""

from unittest.mock import patch

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def test_disable_mtp_loss_rejects_frozen_base():
    with pytest.raises(ValueError, match="cannot both be enabled"):
        TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            mtp_num_layers=1,
            disable_mtp_loss=True,
            freeze_base_model_for_mtp=True,
        )


@pytest.fixture
def model_parallel():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


def _build_model(kind, *, disabled):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        disable_mtp_loss=disabled,
    )
    if kind == "hybrid":
        return HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            hybrid_layer_pattern="**/*",
            vocab_size=128,
            max_sequence_length=8,
        )
    spec = get_gpt_layer_with_transformer_engine_spec()
    return GPTModel(
        config=config,
        transformer_layer_spec=spec,
        mtp_block_spec=get_gpt_mtp_block_spec(config, spec, use_transformer_engine=True),
        vocab_size=128,
        max_sequence_length=8,
    )


@pytest.mark.parametrize("kind", ["gpt", "hybrid"])
def test_disabled_mtp_keeps_checkpoint_parameters(model_parallel, kind):
    enabled = _build_model(kind, disabled=False)
    disabled = _build_model(kind, disabled=True)
    disabled.load_state_dict(enabled.state_dict(), strict=True)
    assert enabled.state_dict().keys() == disabled.state_dict().keys()
    enabled_mtp = list(enabled.mtp.parameters())
    disabled_mtp = list(disabled.mtp.parameters())
    assert enabled_mtp and disabled_mtp
    assert all(parameter.requires_grad for parameter in enabled_mtp)
    assert all(not parameter.requires_grad for parameter in disabled_mtp)
    assert any(parameter.requires_grad for parameter in disabled.decoder.parameters())


@pytest.mark.parametrize("kind", ["gpt", "hybrid"])
def test_disabled_mtp_skips_forward_and_loss(model_parallel, kind):
    model = _build_model(kind, disabled=True).cuda().train()
    input_ids = torch.arange(8, device="cuda").unsqueeze(0)
    module = (
        "megatron.core.models.gpt.gpt_model"
        if kind == "gpt"
        else "megatron.core.models.hybrid.hybrid_model"
    )
    with (
        patch.object(
            model.mtp, "forward", side_effect=AssertionError("MTP forward ran")
        ) as forward,
        patch(f"{module}.process_mtp_loss", side_effect=AssertionError("MTP loss ran")) as loss,
    ):
        output = model(
            input_ids=input_ids,
            position_ids=input_ids.clone(),
            attention_mask=None,
            labels=(input_ids + 1) % 128,
            loss_mask=torch.ones_like(input_ids, dtype=torch.float32),
        )
        assert output.shape == input_ids.shape
        assert torch.isfinite(output).all()
        output.sum().backward()
        forward.assert_not_called()
        loss.assert_not_called()
    assert all(parameter.grad is None for parameter in model.mtp.parameters())
    assert any(parameter.grad is not None for parameter in model.decoder.parameters())
