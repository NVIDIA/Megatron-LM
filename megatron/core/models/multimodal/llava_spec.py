# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from functools import cache
from typing import Optional

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import get_backend
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


# One provider per backend; every choice below comes from the provider, not from a flag.
@cache
def _te():
    """The Transformer Engine provider, built on first use.

    Deferred so that importing this module does not require Transformer Engine -- only
    building a Transformer Engine spec does. That is what the ``HAVE_TE`` guard used to buy.
    """
    return get_backend("transformer_engine")


_local = get_backend("local")


def decoder_model_with_transformer_engine_default_spec(
    num_experts: Optional[int] = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    """LLava decoder TE spec (uses Transformer Engine components)."""
    mlp = get_mlp_module_spec(
        use_te=True, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=_te().column_parallel_layer_norm_linear(),
                    core_attention=_te().core_attention(),
                    linear_proj=_te().row_parallel_linear(),
                    q_layernorm=_te().layer_norm() if qk_layernorm else IdentityOp,
                    k_layernorm=_te().layer_norm() if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def decoder_model_with_local_default_spec(
    num_experts: Optional[int] = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    """LLava decoder local spec."""
    mlp = get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=_local.layer_norm(),
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=_local.column_parallel_linear(),
                    core_attention=_local.core_attention(),
                    linear_proj=_local.row_parallel_linear(),
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=_local.layer_norm(),
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )
