# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""HybridModel with the Liger norm backend: the spec's norm slots, and forward/backward parity."""

import pytest
import torch

from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_stack_spec,
    wide_residual_hybrid_stack_spec,
    with_norm_backend,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.ops.norm.liger_norm import LigerRMSNorm
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("stack_spec", [hybrid_stack_spec, wide_residual_hybrid_stack_spec])
def test_with_norm_backend_names_liger_in_every_norm_slot(stack_spec):
    assert with_norm_backend(stack_spec, "te") is stack_spec
    liger = with_norm_backend(stack_spec, "liger")
    # Every TENorm slot at any depth (ModuleSpecs, submodule dataclasses, partials, MTP layer
    # lists) now names Liger; the shared default spec is left as it was.
    assert "TENorm" in repr(stack_spec) and "TENorm" not in repr(liger)
    assert stack_spec.submodules.final_norm is TENorm
    submodules = liger.submodules
    assert submodules.final_norm is LigerRMSNorm
    assert submodules.moe_layer.submodules.pre_mlp_layernorm is LigerRMSNorm
    mtp_layer = submodules.mtp_block_spec.submodules.layer_specs[0].submodules
    assert mtp_layer.enorm is mtp_layer.hnorm is mtp_layer.layer_norm is LigerRMSNorm
    assert submodules.attention_layer == stack_spec.submodules.attention_layer  # no norm slots


def _rms_norms(model):
    return {name: type(m) for name, m in model.named_modules() if "RMSNorm" in type(m).__name__}


@pytest.fixture
def hybrid_models():
    pytest.importorskip("liger_kernel.megatron")
    pytest.importorskip("transformer_engine")
    Utils.initialize_model_parallel(1, 1)
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")  # keep TF32 from amplifying norm rounding
    models = {}
    for impl in ("te", "liger"):
        # The Liger model takes HybridModel's default spec, which follows config.rmsnorm_impl.
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            ffn_hidden_size=128,
            normalization="RMSNorm",
            rmsnorm_impl=impl,
            num_moe_experts=2,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            use_cpu_initialization=True,
        )
        models[impl] = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec if impl == "te" else None,
            vocab_size=128,
            max_sequence_length=16,
            hybrid_layer_pattern="*E/*-",  # attention, MoE, and one MTP depth
        ).cuda()
    yield models
    torch.set_float32_matmul_precision(precision)
    Utils.destroy_model_parallel()


def test_hybrid_norm_slots(hybrid_models):
    native, liger = _rms_norms(hybrid_models["te"]), _rms_norms(hybrid_models["liger"])
    # Final, MoE pre-MLP and MTP norms; norms fused into TE linears are not RMSNorm modules.
    assert {"decoder.final_norm", "decoder.layers.1.pre_mlp_layernorm"} <= native.keys()
    assert any(name.startswith("mtp.") for name in native)
    assert native.keys() == liger.keys()
    assert {cls.__module__.split(".")[0] for cls in native.values()} == {"transformer_engine"}
    assert {cls.__name__ for cls in liger.values()} == {"LigerMegatronRMSNorm"}


def test_hybrid_forward_backward_parity(hybrid_models):
    native, liger = hybrid_models["te"], hybrid_models["liger"]
    skipped = liger.load_state_dict(native.state_dict(), strict=False)
    assert all(
        key.endswith("_extra_state") for key in skipped.missing_keys + skipped.unexpected_keys
    )
    torch.manual_seed(456)
    tokens = torch.randint(0, 128, (2, 16), device="cuda")
    positions = torch.arange(16, device="cuda").expand_as(tokens)
    mask = torch.ones(2, 1, 16, 16, device="cuda", dtype=torch.bool)
    labels = torch.randint(0, 128, tokens.shape, device="cuda")
    losses = {}
    for impl, model in hybrid_models.items():
        losses[impl] = model(
            input_ids=tokens, position_ids=positions, attention_mask=mask, labels=labels
        )
        losses[impl].sum().backward()
    torch.testing.assert_close(losses["liger"], losses["te"], atol=2e-5, rtol=2e-5)
    native_params = dict(native.named_parameters())
    for name, parameter in liger.named_parameters():
        expected = native_params[name].grad
        if parameter.grad is None or expected is None:  # e.g. an expert that got no tokens
            assert parameter.grad is expected is None, name
            continue
        torch.testing.assert_close(parameter.grad, expected, atol=2e-4, rtol=2e-4, msg=name)
