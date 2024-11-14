#forked from alibaba/Pai-Megatron-Patch for deepseekv2 implementation
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

#from megatron.core.transformer.custom_layers.transformer_engine import (
#    TEDotProductAttention,
#    TEDotProductAttentionMLA,
#    TELayerNormColumnParallelLinear,
#    TENorm,
#    TERowParallelLinear,
#    TEColumnParallelLinear,
#)

from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec


from .transformer.mlp import MLP, MLPSubmodules
from .transformer.attention import SelfAttention, SelfAttentionSubmodules
from .moe.moe_layer import MoELayer
from .transformer_layer import TransformerLayer, TransformerLayerSubmodules
from .rms_norm import DeepseekV2RMSNorm

# Use this spec to use lower level Transformer Engine modules (required for fp8 training)
# def get_gpt_layer_with_transformer_engine_spec(
#     num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
# ) -> ModuleSpec:

#     mlp = _get_mlp_module_spec(
#         use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
#     )

#     mlp_dense = _get_mlp_module_spec(
#         use_te=True, num_experts=None, moe_grouped_gemm=moe_grouped_gemm
#     )

#     return ModuleSpec(
#         module=TransformerLayer,
#         submodules=TransformerLayerSubmodules(
#             self_attention=ModuleSpec(
#                 module=SelfAttention,
#                 params={"attn_mask_type": AttnMaskType.causal},
#                 submodules=SelfAttentionSubmodules(
#                     linear_q_proj=TEColumnParallelLinear,
#                     linear_q_a_proj=TEColumnParallelLinear,
#                     linear_q_b_proj=ColumnParallelLinear,
#                     linear_kv_a_proj_with_mqa=TEColumnParallelLinear,
#                     linear_kv_b_proj=ColumnParallelLinear,
#                     linear_proj=TERowParallelLinear,
#                     q_a_layernorm=TENorm if qk_layernorm else IdentityOp,
#                     kv_a_layernorm=TENorm if qk_layernorm else IdentityOp,
#                     core_attention=TEDotProductAttentionMLA,
#                 ),
#             ),
#             self_attn_bda=get_bias_dropout_add,
#             pre_mlp_layernorm=TENorm if num_experts else IdentityOp,
#             input_layernorm=TENorm if num_experts else IdentityOp,
#             mlp=mlp,
#             mlp_dense=mlp_dense,
#             mlp_bda=get_bias_dropout_add,
#         ),
#     )

# Use this spec to use lower level Transformer Engine modules (required for fp8 training)
def get_gpt_layer_with_local_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:

    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )

    mlp_dense = _get_mlp_module_spec(
        use_te=False, num_experts=None, moe_grouped_gemm=moe_grouped_gemm
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_q_proj=ColumnParallelLinear,
                    linear_q_a_proj=ColumnParallelLinear,
                    linear_q_b_proj=ColumnParallelLinear,
                    linear_kv_a_proj_with_mqa=ColumnParallelLinear,
                    linear_kv_b_proj=ColumnParallelLinear,
                    linear_proj=RowParallelLinear,
                    q_a_layernorm=DeepseekV2RMSNorm if qk_layernorm else IdentityOp,
                    kv_a_layernorm=DeepseekV2RMSNorm if qk_layernorm else IdentityOp,
                    # core_attention=TEDotProductAttention,
                    core_attention=DotProductAttention,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=DeepseekV2RMSNorm if num_experts else IdentityOp,
            input_layernorm=DeepseekV2RMSNorm if num_experts else IdentityOp,
            mlp=mlp,
            mlp_dense=mlp_dense,
            mlp_bda=get_bias_dropout_add,
        ),
    )

# Helper function to get module spec for MLP/MoE
def _get_mlp_module_spec(
    use_te: bool = True, num_experts: int = None, moe_grouped_gemm: bool = False
) -> ModuleSpec:
    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TEColumnParallelLinear if use_te else ColumnParallelLinear,
                linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
            ),
        )
    else:
        # Mixture of experts with modules in megatron core.
        return ModuleSpec(
            module=MoELayer,
            submodules=MLPSubmodules(
                linear_fc1=ColumnParallelLinear,
                linear_fc2=RowParallelLinear,)
            if not moe_grouped_gemm
            else None,
        )