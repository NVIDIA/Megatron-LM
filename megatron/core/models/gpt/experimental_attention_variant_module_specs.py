# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import BackendSpecProvider
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerSubmodules,
    DSAttention,
    DSAttentionSubmodules,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


def get_dsa_module_spec_for_backend(
    backend: BackendSpecProvider,
    qk_layernorm: Optional[bool] = False,
    qk_l2_norm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    num_experts: Optional[int] = None,
    mlp: Optional[ModuleSpec] = None,
) -> ModuleSpec:
    """Helper function to get module spec for Sparse Attention."""
    assert multi_latent_attention, "Currently only MLA supports sparse attention."
    assert qk_l2_norm is False, "qk_l2_norm is not supported with MLA."

    linear_q_up_proj = (
        backend.column_parallel_layer_norm_linear()
        if qk_layernorm
        else backend.column_parallel_linear()
    )
    linear_kv_up_proj = (
        backend.column_parallel_layer_norm_linear()
        if qk_layernorm
        else backend.column_parallel_linear()
    )

    # Because TransformerEngine does not support sparse attention yet, we use local
    # implementation whether the backend is TransformerEngine or not.
    core_attention = ModuleSpec(
        module=DSAttention,
        submodules=DSAttentionSubmodules(
            indexer=ModuleSpec(
                module=DSAIndexer,
                submodules=DSAIndexerSubmodules(
                    linear_wq_b=backend.linear(),
                    linear_wk=backend.linear(),
                    k_norm=backend.layer_norm(rms_norm=False, for_qk=True),
                    linear_weights_proj=backend.linear(),
                ),
            )
        ),
    )

    attention = ModuleSpec(
        module=MLASelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=MLASelfAttentionSubmodules(
            linear_q_proj=backend.column_parallel_linear(),
            linear_q_down_proj=backend.linear(),
            linear_q_up_proj=linear_q_up_proj,
            linear_kv_down_proj=backend.linear(),
            linear_kv_up_proj=linear_kv_up_proj,
            core_attention=core_attention,
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=IdentityOp,
            kv_layernorm=IdentityOp,
        ),
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=backend.layer_norm(),
            self_attention=attention,
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=backend.layer_norm() if num_experts else IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def get_experimental_attention_variant_module_spec_for_backend(
    backend: BackendSpecProvider,
    experimental_attention_variant: Optional[str] = None,
    qk_layernorm: Optional[bool] = False,
    qk_l2_norm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    num_experts: Optional[int] = None,
    mlp: Optional[ModuleSpec] = None,
) -> ModuleSpec:
    """Helper function to get module spec for Attention"""
    if experimental_attention_variant == "dsa":
        return get_dsa_module_spec_for_backend(
            backend=backend,
            qk_layernorm=qk_layernorm,
            qk_l2_norm=qk_l2_norm,
            multi_latent_attention=multi_latent_attention,
            num_experts=num_experts,
            mlp=mlp,
        )
    else:
        raise ValueError(
            f"Invalid experimental attention variant: {experimental_attention_variant}"
        )
