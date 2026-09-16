# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from functools import wraps

import pytest
import torch
import torch.nn.functional as F

from megatron.core.extensions import transformer_engine as te_ext
from megatron.core.fp8_utils import is_mxfp8tensor
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.quantization.quant_config import RecipeConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import multi_token_prediction as mtp_module
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock, TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(autouse=True)
def model_parallel():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("mtp_num_layers", [1, 2])
@pytest.mark.skipif(
    not te_ext.HAVE_TE
    or not is_te_min_version("2.1.0")
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 10,
    reason="MXFP8 parameter construction requires a compatible TE and Blackwell GPU",
)
def test_native_shared_bf16_at_construction_with_mxfp8_routed(monkeypatch, mtp_num_layers):
    """The recipe must affect parameter init, before GPT's final finish_init pass."""
    constructors = []

    def capture_constructor(cls):
        original = cls.__init__

        @wraps(original)
        def wrapped(module, *args, **kwargs):
            original(module, *args, **kwargs)
            name = kwargs.get("name")
            constructors.append((module, name))
            # Check immediately: finish_init after complete GPT construction cannot
            # repair an MXFP8 parameter initialized before the BF16 override matched.
            if name is not None and ".shared_experts." in name:
                assert isinstance(module.weight, torch.nn.Parameter)
                assert not is_mxfp8tensor(module.weight)
                assert module.weight.dtype == torch.bfloat16
            if name is not None and ".experts." in name:
                assert all(is_mxfp8tensor(p) for p in module.parameters(recurse=False))

        monkeypatch.setattr(cls, "__init__", wrapped)

    for cls in (te_ext.TELinear, te_ext.TELayerNormColumnParallelLinear, te_ext.TEGroupedLinear):
        capture_constructor(cls)

    config = TransformerConfig(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        use_cpu_initialization=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        fp8="e4m3",
        fp8_recipe="mxfp8",
        fp8_param=True,
        num_moe_experts=4,
        moe_ffn_hidden_size=256,
        moe_shared_expert_intermediate_size=256,
        moe_shared_expert_gate=True,
        moe_router_topk=2,
        moe_grouped_gemm=True,
        moe_single_grouped_weight=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        mtp_num_layers=mtp_num_layers,
        quant_recipe=RecipeConfig.from_config_dict(
            {
                "configs": {
                    "shared_bf16": {
                        "transformer_engine_config_type": "TEQuantizationParams",
                        "training_recipe": {
                            "override_quantized_autocast": True,
                            "fp8_param": False,
                        },
                    }
                },
                "matchers": {
                    "shared": {
                        "type": "glob",
                        "pattern": "*.shared_experts.*",
                        "config": "shared_bf16",
                        "enabled": True,
                    }
                },
            }
        ),
    )
    layer_spec = get_gpt_layer_with_transformer_engine_spec(num_experts=4, moe_grouped_gemm=True)
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        mtp_block_spec=get_gpt_mtp_block_spec(config, layer_spec, use_transformer_engine=True),
        vocab_size=512,
        max_sequence_length=32,
    )
    canonical_names = {module: name for name, module in model.named_modules()}
    assert constructors
    for module, constructor_name in constructors:
        assert constructor_name == canonical_names[module]

    prefixes = [f"decoder.layers.{i}.mlp" for i in range(2)] + [
        f"mtp.layers.{i}.mtp_model_layer.mlp" for i in range(mtp_num_layers)
    ]
    expected_shared = {
        f"{prefix}.shared_experts.linear_fc{i}" for prefix in prefixes for i in (1, 2)
    }
    expected_routed = {f"{prefix}.experts.linear_fc{i}" for prefix in prefixes for i in (1, 2)}
    assert {name for _, name in constructors if ".shared_experts." in name} == expected_shared
    assert {name for _, name in constructors if ".experts." in name} == expected_routed
    for prefix in prefixes:
        shared = model.get_submodule(f"{prefix}.shared_experts")
        assert shared.gate_weight.dtype == torch.bfloat16
        for fc in (shared.linear_fc1, shared.linear_fc2):
            assert not fc.will_execute_quantized(is_context_quantized=True)
            assert not is_mxfp8tensor(fc.weight)


def test_unnamed_transformer_block_accepts_legacy_layer_without_name():
    class LegacyLayer(torch.nn.Module):
        def __init__(self, config, layer_number, pg_collection, vp_stage):
            super().__init__()
            self.layer_number = layer_number

    config = TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4)
    spec = TransformerBlockSubmodules(layer_specs=[ModuleSpec(module=LegacyLayer)] * 2)
    block = TransformerBlock(config, spec, post_layer_norm=False)
    assert block.name is None
    assert [layer.layer_number for layer in block.layers] == [1, 2]


def test_unnamed_mtp_preserves_default_constructor_names(monkeypatch):
    names = []
    original = mtp_module.build_module

    def capture_build(spec, *args, **kwargs):
        if getattr(spec, "module", None) is mtp_module.MultiTokenPredictionLayer:
            names.append(kwargs.get("name"))
        return original(spec, *args, **kwargs)

    monkeypatch.setattr(mtp_module, "build_module", capture_build)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        mtp_num_layers=2,
        use_cpu_initialization=True,
    )
    spec = get_gpt_mtp_block_spec(config, get_gpt_layer_local_spec(), use_transformer_engine=False)
    block = mtp_module.MultiTokenPredictionBlock(config, spec)
    assert block.name is None
    assert names == [None, None]
    assert [layer.layer_number for layer in block.layers] == [1, 2]


def test_gpt_without_quant_recipe_accepts_legacy_layer_without_name():
    class LegacyLayer(torch.nn.Module):
        def __init__(self, config, layer_number, pg_collection, vp_stage):
            super().__init__()
            self.layer_number = layer_number

    config = TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4)
    spec = TransformerBlockSubmodules(
        layer_specs=[ModuleSpec(module=LegacyLayer)] * 2, layer_norm=IdentityOp
    )
    model = GPTModel(config, spec, vocab_size=128, max_sequence_length=32)
    assert model.decoder.name is None
    assert [layer.layer_number for layer in model.decoder.layers] == [1, 2]
