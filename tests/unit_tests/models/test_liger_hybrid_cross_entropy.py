# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""HybridModel forward/backward parity with Liger cross entropy, with and without MTP."""

from dataclasses import replace

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.ops.loss.liger_cross_entropy import liger_cross_entropy
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(params=[1, 2], ids=["tp1", "tp2"])
def tensor_parallel(request):
    pytest.importorskip("liger_kernel.megatron")
    pytest.importorskip("transformer_engine")
    if Utils.world_size % request.param:
        pytest.skip(f"requires world size divisible by TP={request.param}")
    Utils.initialize_model_parallel(request.param, 1)
    model_parallel_cuda_manual_seed(123)
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")  # keep TF32 from amplifying CE rounding
    yield request.param
    torch.set_float32_matmul_precision(precision)
    Utils.destroy_model_parallel()


def _model(config, pattern):
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=16,
        hybrid_layer_pattern=pattern,
    ).cuda()


@pytest.mark.parametrize("pattern", ["*-", "*-/*-"], ids=["no_mtp", "mtp"])
def test_hybrid_forward_backward_parity(tensor_parallel, pattern):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        normalization="RMSNorm",
        tensor_model_parallel_size=tensor_parallel,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
    )
    liger = replace(config, cross_entropy_loss_fusion=True, cross_entropy_fusion_impl="liger")
    actual, reference = _model(liger, pattern), _model(config, pattern)
    assert actual.vocab_parallel_cross_entropy is liger_cross_entropy
    reference.load_state_dict(actual.state_dict(), strict=True)
    torch.manual_seed(456)
    tokens = torch.randint(0, 128, (2, 16), device="cuda")
    positions = torch.arange(16, device="cuda").expand_as(tokens)
    mask = torch.ones(2, 1, 16, 16, device="cuda", dtype=torch.bool)
    labels = torch.randint(0, 128, tokens.shape, device="cuda")
    upstream = torch.linspace(0.1, 1.0, labels.numel(), device="cuda").reshape_as(labels)
    losses = []
    for model in (actual, reference):
        loss = model(input_ids=tokens, position_ids=positions, attention_mask=mask, labels=labels)
        (loss * upstream).sum().backward()  # MTP adds its loss through the backward pass
        losses.append(loss)
    torch.testing.assert_close(losses[0], losses[1], atol=2e-5, rtol=2e-5)
    expected = dict(reference.named_parameters())
    assert expected.keys() == dict(actual.named_parameters()).keys()
    for name, parameter in actual.named_parameters():
        if parameter.grad is None or expected[name].grad is None:
            assert parameter.grad is expected[name].grad is None, name
            continue
        torch.testing.assert_close(
            parameter.grad, expected[name].grad, atol=2e-4, rtol=2e-4, msg=name
        )
