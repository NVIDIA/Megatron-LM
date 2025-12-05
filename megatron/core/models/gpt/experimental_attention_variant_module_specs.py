# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

from megatron.core.models.backends import BackendSpecProvider
from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules
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


def is_linear_attention_variant(experimental_attention_variant: str) -> bool:
    """Check if the experimental attention variant is a linear attention variant."""
    linear_attention_variants = ["gated_delta_net"]
    return experimental_attention_variant in linear_attention_variants


def get_gated_delta_net_module_spec_for_backend(
    backend: BackendSpecProvider, normalization: Optional[str] = None
) -> ModuleSpec:
    """Helper function to get module spec for Linear Attention"""
    rms_norm = normalization == "RMSNorm"
    attention = ModuleSpec(
        module=GatedDeltaNet,
        submodules=GatedDeltaNetSubmodules(
            in_proj=backend.column_parallel_layer_norm_linear(),
            out_norm=backend.layer_norm(rms_norm=rms_norm, for_qk=False),
            out_proj=backend.row_parallel_linear(),
        ),
        metainfo={"fuse_input_layernorm": True},
    )
    return attention


def get_dsa_module_spec_for_backend(
    backend: BackendSpecProvider,
    qk_layernorm: Optional[bool] = False,
    qk_l2_norm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    mla_down_proj_use_column_parallel: Optional[bool] = False,
    normalization: Optional[str] = None,
    fallback_to_eager_attn: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for Sparse Attention."""
    assert multi_latent_attention, "Currently only MLA supports sparse attention."
    assert qk_l2_norm is False, "qk_l2_norm is not supported with MLA."
    assert fallback_to_eager_attn is False, "Fallback to eager attention is not supported with DSA."

    linear_q_down_proj = (
        backend.column_parallel_linear() if mla_down_proj_use_column_parallel else backend.linear()
    )
    linear_kv_down_proj = (
        backend.column_parallel_linear() if mla_down_proj_use_column_parallel else backend.linear()
    )
    linear_q_up_proj = backend.column_parallel_linear()
    linear_kv_up_proj = backend.column_parallel_linear()

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

    # Adjust for RMS norm.
    rms_norm = normalization == "RMSNorm"
    qk_norm = backend.layer_norm(rms_norm=rms_norm, for_qk=True) if qk_layernorm else IdentityOp

    attention = ModuleSpec(
        module=MLASelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=MLASelfAttentionSubmodules(
            linear_q_proj=backend.column_parallel_linear(),
            linear_q_down_proj=linear_q_down_proj,
            linear_q_up_proj=linear_q_up_proj,
            linear_kv_down_proj=linear_kv_down_proj,
            linear_kv_up_proj=linear_kv_up_proj,
            core_attention=core_attention,
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=qk_norm,
            kv_layernorm=qk_norm,
        ),
        metainfo={"fuse_input_layernorm": False},
    )

    return attention


def get_experimental_attention_variant_module_spec_for_backend(
    backend: BackendSpecProvider,
    sharded_state_dict_keys_map: dict,
    experimental_attention_variant: Optional[str] = None,
    qk_layernorm: Optional[bool] = False,
    qk_l2_norm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    mla_down_proj_use_column_parallel: Optional[bool] = False,
    normalization: Optional[str] = None,
    fallback_to_eager_attn: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for Attention"""
    if experimental_attention_variant == "gated_delta_net":
        return get_gated_delta_net_module_spec_for_backend(
            backend=backend, normalization=normalization
        )
    elif experimental_attention_variant == "dsa":
        return get_dsa_module_spec_for_backend(
            backend=backend,
            qk_layernorm=qk_layernorm,
            qk_l2_norm=qk_l2_norm,
            multi_latent_attention=multi_latent_attention,
            mla_down_proj_use_column_parallel=mla_down_proj_use_column_parallel,
            normalization=normalization,
            fallback_to_eager_attn=fallback_to_eager_attn,
        )
    else:
        raise ValueError(
            f"Invalid experimental attention variant: {experimental_attention_variant}"
        )
