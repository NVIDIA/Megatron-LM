# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import warnings
from functools import partial

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import get_backend
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


def get_bert_layer_with_transformer_engine_submodules() -> TransformerLayerSubmodules:
    """Use these submodules to use lower-level Transformer Engine modules (required for fp8
    training).

    Returns:
        TransformerLayerSubmodules: Submodules with TE modules.
    """
    backend = get_backend("transformer_engine")

    return TransformerLayerSubmodules(
        self_attention=ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.padding},
            submodules=SelfAttentionSubmodules(
                linear_qkv=backend.column_parallel_layer_norm_linear(),
                core_attention=backend.core_attention(),
                linear_proj=backend.row_parallel_linear(),
                # Leave q_layernorm/k_layernorm unset (None) rather than IdentityOp so that
                # TransformerConfig.qk_layernorm can select the default TENorm through the
                # shared SelfAttention fallback (`submodules.q_layernorm or TENorm`).
                q_layernorm=None,
                k_layernorm=None,
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
        mlp=partial(
            MLP.as_mlp_submodule,
            submodules=MLPSubmodules(
                linear_fc1=backend.column_parallel_layer_norm_linear(),
                linear_fc2=backend.row_parallel_linear(),
            ),
        ),
        mlp_bda=get_bias_dropout_add,
    )


def get_bert_layer_with_transformer_engine_spec():
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).

    Returns:
        ModuleSpec: Module specification with TE modules
    """
    return ModuleSpec(
        module=TransformerLayer, submodules=get_bert_layer_with_transformer_engine_submodules()
    )


def __getattr__(name):
    if name == "bert_layer_with_transformer_engine_spec":
        warnings.warn("""Attribute bert_layer_specs.bert_layer_with_transformer_engine_spec is on a
            deprecation track and will be removed in future releases. Please migrate to
            bert_layer_specs.get_bert_layer_with_transformer_engine_spec().""")

        return get_bert_layer_with_transformer_engine_spec()


_local_backend = get_backend("local")

# Use this spec for an implementation using only modules in megatron core
bert_layer_local_spec = ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        input_layernorm=_local_backend.layer_norm(),
        self_attention=ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.padding},
            submodules=SelfAttentionSubmodules(
                linear_qkv=_local_backend.column_parallel_linear(),
                core_attention=_local_backend.core_attention(),
                linear_proj=_local_backend.row_parallel_linear(),
                q_layernorm=IdentityOp,
                k_layernorm=IdentityOp,
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=_local_backend.layer_norm(),
        mlp=partial(
            MLP.as_mlp_submodule,
            submodules=MLPSubmodules(
                linear_fc1=_local_backend.column_parallel_linear(),
                linear_fc2=_local_backend.row_parallel_linear(),
            ),
        ),
        mlp_bda=get_bias_dropout_add,
        sharded_state_dict_keys_map={
            "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
            "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
        },
    ),
)
