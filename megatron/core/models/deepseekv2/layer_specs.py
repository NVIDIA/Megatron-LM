# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from typing import Optional
import warnings
from megatron.core.utils import get_te_version, is_te_min_version
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules, # same as SelfAttentionSubmodules
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules, TransformerLayer
from megatron.core.transformer.moe.experts import SequentialMLP, GroupedMLP, TEGroupedMLP

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEColumnParallelLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TELinear,
        TENorm,
        TERowParallelGroupedLinear,
        TERowParallelLinear,
        TEDotProductAttentionMLA # same as TEDotProductAttention
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from megatron.legacy.model.rms_norm import RMSNorm


def get_gpt_layer_with_transformer_engine_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,
) -> ModuleSpec:
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fp8 (str, optional): Flag to decide the linear layer spec for MoE. Defaults to None.

    Returns:
        ModuleSpec: Module specification with TE modules
    """
    #mlp_moe = _get_mlp_module_spec(
    #    use_te=True, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm, fp8=fp8
    #)
    mlp_moe = _get_moe_module_spec(
        use_te=True,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )

    mlp_dense = _get_mlp_module_spec(
        use_te=True, 
        num_experts=None, 
        moe_grouped_gemm=moe_grouped_gemm, 
        fp8=fp8
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=TENorm if num_experts else IdentityOp,
            self_attention=ModuleSpec(
                module=MLASelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_q_proj=TEColumnParallelLinear,
                    linear_q_down_proj=TELinear,
                    linear_q_up_proj=(
                        TELayerNormColumnParallelLinear
                        if qk_layernorm
                        else TEColumnParallelLinear
                    ),
                    linear_kv_down_proj=TELinear,
                    linear_kv_up_proj=(
                        TELayerNormColumnParallelLinear
                        if qk_layernorm
                        else TEColumnParallelLinear
                    ),
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=IdentityOp,
                    kv_layernorm=IdentityOp,
                    ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm if num_experts else IdentityOp,
            mlp=mlp_moe,
            mlp_dense=mlp_dense,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def get_gpt_layer_local_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,
) -> ModuleSpec:
    """Use this spec for an implementation using only modules in Megatron-Core.


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.

    Returns:
        ModuleSpec: Module specification with Megatron-Core modules
    """
    mlp_moe = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm, fp8=fp8
    )
    mlp_dense = _get_mlp_module_spec(
        use_te=False, num_experts=None, moe_grouped_gemm=moe_grouped_gemm, fp8=fp8
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=MLASelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_q_proj=ColumnParallelLinear,
                    linear_q_down_proj=ColumnParallelLinear,
                    linear_q_up_proj=ColumnParallelLinear,
                    linear_kv_down_proj=ColumnParallelLinear,
                    linear_kv_up_proj=ColumnParallelLinear,
                    linear_proj=RowParallelLinear,
                    q_layernorm=RMSNorm if qk_layernorm else IdentityOp,
                    kv_layernorm=RMSNorm if qk_layernorm else IdentityOp,
                    core_attention=DotProductAttention,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=RMSNorm if num_experts else IdentityOp,
            input_layernorm=RMSNorm if num_experts else IdentityOp,
            mlp=mlp_moe,
            mlp_dense=mlp_dense,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
            },
        ),
    )


def _get_mlp_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    fp8: Optional[str] = None,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""
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
        if use_te and moe_grouped_gemm:
            linear_fc1 = TEColumnParallelGroupedLinear
            linear_fc2 = TERowParallelGroupedLinear
        elif use_te and fp8:
            linear_fc1 = TEColumnParallelLinear
            linear_fc2 = TERowParallelLinear
        else:
            linear_fc1 = ColumnParallelLinear
            linear_fc2 = RowParallelLinear

        use_te_grouped_gemm = use_te and TEColumnParallelGroupedLinear is not None

        return ModuleSpec(
            module=MoELayer,
            submodules=MoESubmodules(
                experts=(
                    MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2)
                    if not moe_grouped_gemm or use_te_grouped_gemm
                    else None
                ),
                # experts=ModuleSpec(
                    # module=TEGroupedMLP if use_te_grouped_gemm else GroupedMLP,
                # ),
                shared_experts=ModuleSpec(
                    module=SharedExpertMLP,
                    params={"gate": False},
                    submodules=MLPSubmodules(
                        linear_fc1=TEColumnParallelLinear if use_te else ColumnParallelLinear,
                        linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
                    ),
                ),
            ),
        )


def _get_moe_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MoE"""
    assert num_experts is not None

    mlp = MLPSubmodules(
        linear_fc1=TEColumnParallelLinear if use_te else ColumnParallelLinear,
        linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
    )

    # experts spec
    if moe_grouped_gemm:
        ## use GroupedMLP
        if use_te and TEColumnParallelGroupedLinear is not None and not moe_use_legacy_grouped_gemm:
            ## use TEGroupedLinear
            expert_module = TEGroupedMLP
            expert_submodule = MLPSubmodules(
                linear_fc1=TEColumnParallelGroupedLinear, 
                linear_fc2=TERowParallelGroupedLinear
            )
        else:
            ## use legacy GroupedMLP
            expert_module = GroupedMLP
            expert_submodule = None
            warnings.warn(
                'The legacy GroupedMLP will be deprecated in Megatron-Core v0.12.0. '
                'Please update the TransformerEngine to version>=1.7.0 and use TEGroupedMLP.'
            )
    else:
        ## use SequentialMLP
        expert_module = SequentialMLP
        if use_te and not is_te_min_version("1.7.0.dev0"):
            warnings.warn(
                "Only transformer-engine>=1.7.0 supports MoE experts, "
                f"but your version is {get_te_version()}. Use local linear implementation instead."
            )
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