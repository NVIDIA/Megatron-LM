from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.mlp import MLP, MLPSubmodules

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TEColumnParallelLinear,
    TELinear,
    TENorm,
    TERowParallelLinear,
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)

from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)

def get_deepseek_layer_with_spec(
    num_experts: int=None, moe_grouped_gemm: bool=False, qk_layernorm: bool=False, multi_latent_attention=False
) -> ModuleSpec:
    """
    Args:
        num_experts (int, optional): The number of experts. If set to a non-zero value,
            the multi-head attention mechanism (MoE) is used.
            Otherwise, the standard self-attention mechanism will be used.
        moe_grouped_gemm (bool, optional): Whether to use grouped matrix multiplication
            for multi-head attention mechanism. The default is False.
        It is only valid when num_experts is greater than 0.
        qk_layernorm (bool, optional): Whether to apply flat normalization between the query and the keywords.
            The default is False.
        It is only valid when num_experts is greater than 0.

    Returns:
        ModuleSpec: ModuleSpec instance containing the TransformerLayerDeepSeek model and corresponding submodules.
    """

    mlp_moe = _get_mlp_module_spec(
        num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )

    mlp_dense = _get_mlp_module_spec(
        num_experts=None, moe_grouped_gemm=moe_grouped_gemm
    )

    return ModuleSpec(
        module=DeepSeekTransformerLayer,
        submodules=DeepSeekTransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=MLASelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=MLASelfAttentionSubmodules(
                    linear_q_proj=TEColumnParallelLinear,
                    linear_q_down_proj=TELinear,
                    linear_q_up_proj=(TELayerNormColumnParallelLinear
                            if qk_layernorm
                            else TEColumnParallelLinear),
                    linear_kv_up_proj_with_mqa=(
                            TELayerNormColumnParallelLinear
                            if qk_layernorm
                            else TEColumnParallelLinear
                    ),
                    linear_kv_down_proj=TELinear,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=IdentityOp,
                    kv_layernorm=IdentityOp,
                    core_attention=TEDotProductAttention,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm if num_experts else IdentityOp,
            input_layernorm=TENorm,
            moe_mlp=mlp,
            mlp_dense=mlp_dense,
            mlp_bda=get_bias_dropout_add,
        ),
    )

# Helper function to get module spec for MLP/MoE
def _get_mlp_module_spec(
    num_experts: int=None, moe_grouped_gemm: bool=False
) -> ModuleSpec:

    mlp = MLPSubmodules(
        linear_fc1=TEColumnParallelLinear if use_te else ColumnParallelLinear,
        linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
    )
    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(
            module=MLP,
            submodules=mlp,
        )
    else:
        # Mixture of experts with modules in megatron core.
        if moe_grouped_gemm:
            if use_te and TEColumnParallelGroupedLinear is not None: 
                expert_module = TEGroupedMLP
                expert_submodule = MLPSubmodules(
                    linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear
                )
            else:
                expert_module = GroupedMLP
                expert_submodule = None
        else:
            expert_module = SequentialMLP
            if use_te and not is_te_min_version("1.7.0.dev0"):
                expert_submodule = MLPSubmodules(
                    linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear
                )
            else:
                expert_submodule = mlp

        experts = ModuleSpec(module=expert_module, submodules=expert_submodule)

        # shared experts spec
        shared_experts = ModuleSpec(module=SharedExpertMLP, params={"gate": False}, submodules=mlp)

        # MoE module spec
        moe_module_spec = ModuleSpec(
            module=MoELayer, submodules=MoESubmodules(experts=experts, shared_experts=shared_experts)
        )

        return moe_module_spec


