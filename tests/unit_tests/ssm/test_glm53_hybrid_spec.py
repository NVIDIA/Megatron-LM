# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU construction contracts for the config-aware GLM-5.3-Flash module spec."""

from types import SimpleNamespace

import pytest

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TELayerNormColumnParallelLinear,
    TELinear,
    TENorm,
)
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_glm53_stack_spec,
    hybrid_stack_spec,
)
from megatron.core.ssm.gated_delta_net import KimiDeltaAttention
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)
from megatron.core.transformer.identity_op import IdentityOp


def _config(**overrides):
    fields = dict(
        multi_latent_attention=True,
        experimental_attention_variant="dsa",
        qk_pos_emb_head_dim=0,
        kda_two_stage_gates=True,
        enable_hyper_connections=True,
        mhc_norm_eps_inside_sqrt=True,
        mhc_keep_mappings_in_fp32=True,
        mhc_learned_output_contract=False,
        rmsnorm_cast_before_weight=True,
        normalization="RMSNorm",
        dsa_indexer_kpool=4,
        dsa_indexer_loss_coeff=0.0,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("mhc_keep_mappings_in_fp32", [False, True])
@pytest.mark.parametrize("rmsnorm_cast_before_weight", [False, True])
def test_glm53_spec_has_explicit_latent_and_dense_norms(
    mhc_keep_mappings_in_fp32, rmsnorm_cast_before_weight
):
    spec = hybrid_glm53_stack_spec(
        _config(
            mhc_keep_mappings_in_fp32=mhc_keep_mappings_in_fp32,
            rmsnorm_cast_before_weight=rmsnorm_cast_before_weight,
        )
    )
    attention = spec.submodules.dsa_layer.submodules.self_attention
    assert attention.module is AbsorbedMLASelfAttention
    assert attention.submodules.q_layernorm is TENorm
    assert attention.submodules.kv_layernorm is TENorm
    dense_ffn = spec.submodules.mlp_layer.submodules
    assert dense_ffn.pre_mlp_layernorm is TENorm
    assert dense_ffn.mlp.keywords["submodules"].linear_fc1 is TEColumnParallelLinear
    assert dense_ffn.sharded_state_dict_keys_map == {
        "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_"
    }
    kda = spec.submodules.kda_layer.submodules.self_attention
    assert kda.module is KimiDeltaAttention
    assert kda.submodules.f_a_proj is TELinear
    assert kda.submodules.g_a_proj is TELinear
    assert kda.submodules.f_b_proj is TEColumnParallelLinear
    assert kda.submodules.g_b_proj is TEColumnParallelLinear


def test_glm53_spec_does_not_mutate_generic_or_other_instances():
    first = hybrid_glm53_stack_spec(_config())
    second = hybrid_glm53_stack_spec(_config())
    first.submodules.dsa_layer.submodules.self_attention.submodules.q_layernorm = IdentityOp
    assert second.submodules.dsa_layer.submodules.self_attention.submodules.q_layernorm is TENorm
    generic = hybrid_stack_spec.submodules
    assert generic.dsa_layer.submodules.self_attention.submodules.q_layernorm is IdentityOp
    assert generic.dsa_layer.submodules.self_attention.submodules.kv_layernorm is IdentityOp
    assert generic.mlp_layer.submodules.pre_mlp_layernorm is IdentityOp
    assert (
        generic.mlp_layer.submodules.mlp.keywords["submodules"].linear_fc1
        is TELayerNormColumnParallelLinear
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"multi_latent_attention": False},
        {"experimental_attention_variant": "glm5_next"},
        {"qk_pos_emb_head_dim": 64},
        {"kda_two_stage_gates": False},
        {"enable_hyper_connections": False},
        {"mhc_norm_eps_inside_sqrt": False},
        {"mhc_learned_output_contract": True},
        {"normalization": "LayerNorm"},
        {"dsa_indexer_kpool": 1},
        {"dsa_indexer_loss_coeff": 0.01},
    ],
)
def test_glm53_spec_rejects_non_glm_modes(overrides):
    with pytest.raises(ValueError, match="hybrid_glm53_stack_spec requires"):
        hybrid_glm53_stack_spec(_config(**overrides))


def test_glm53_attention_and_ffn_pattern_uses_physical_module_counts():
    # 34 KDA + 11 DSA attention blocks, each followed by one FFN. This is 45
    # released HF blocks but 90 hybrid modules; the first three FFNs are dense.
    attention = "KKKD" * 11 + "K"
    pattern = "".join(kind + ("-" if block < 3 else "E") for block, kind in enumerate(attention))
    layers = validate_segment_layers(pattern)
    assert len(layers) == 90
    assert layers.count(Symbols.KDA) == 34
    assert layers.count(Symbols.DS_ATTENTION) == 11
    assert layers.count(Symbols.MLP) == 3
    assert layers.count(Symbols.MOE) == 42
