# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add, get_bias_dropout_norm_add
from megatron.core.fusions.fused_dot_product_attention import FusedDotProductAttention
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.rmsnorm import RMSNorm
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import is_real_cuda_device_available

try:
    from megatron.core.transformer.custom_layers.intel_transformer_engine import (
        IntelTEColumnParallelLinear,
        IntelTEDotProductAttention,
        IntelTENorm,
        IntelTERowParallelLinear,
    )
except:
    pass

try:
    from megatron.core.transformer.custom_layers.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelGroupedLinear,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm

    warnings.warn(f'Apex is not installed. Falling back to Torch LayerNorm')
    LNImpl = WrappedTorchLayerNorm


# Use this spec to use lower level Transformer Engine modules (required for fp8 training)
def get_gpt_layer_with_transformer_engine_spec(
    num_experts: int = None,
    moe_grouped_gemm: bool = False,
    qk_layernorm: bool = False,
    enable_fsdpa: bool = False,
    fp8_coverage: dict = {},
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=True,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        fp8_coverage=fp8_coverage,
    )

    use_intel_te = not is_real_cuda_device_available()
    if use_intel_te:
        try:
            from intel_transformer_engine.utils import is_gaudi3
        except:
            from habana_transformer_engine.utils import is_gaudi3
        if is_gaudi3() and enable_fsdpa and fp8_coverage.get('attention', True):
            core_attention_class = IntelTEDotProductAttention
        elif enable_fsdpa:
            core_attention_class = FusedDotProductAttention
        else:
            core_attention_class = DotProductAttention
        linear_proj = IntelTERowParallelLinear
        linear_qkv = IntelTEColumnParallelLinear
        normalization_class = IntelTENorm
    else:
        core_attention_class = TEDotProductAttention
        linear_proj = TERowParallelLinear
        linear_qkv = TELayerNormColumnParallelLinear
        normalization_class = TENorm
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=normalization_class if use_intel_te else IdentityOp,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=linear_qkv,
                    core_attention=core_attention_class,
                    linear_proj=linear_proj,
                    # TENorm significantly harms convergence when used
                    # for QKLayerNorm; we instead use the Apex implementation.
                    q_layernorm=LNImpl if qk_layernorm else IdentityOp,
                    k_layernorm=LNImpl if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=normalization_class if use_intel_te or num_experts else IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


# Use this spec for an implementation using only modules in megatron core
def get_gpt_layer_local_spec(
    num_experts: int = None,
    moe_grouped_gemm: bool = False,
    qk_layernorm: bool = False,
    normalization_type: str = 'LayerNorm',
    enable_fsdpa: bool = False,
    use_pre_norm=True,
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    if normalization_type not in ('LayerNorm', 'RMSNorm'):
        raise Exception(
            f'Only LayerNorm and RMSNorm are currently supported, configured {normalization_type}'
        )
    normalization_class = None
    if normalization_type == "LayerNorm":
        normalization_class = LNImpl
    elif normalization_type == "RMSNorm":
        normalization_class = RMSNorm
    core_attention_class = None
    if is_real_cuda_device_available() or not enable_fsdpa:
        core_attention_class = DotProductAttention
    else:
        core_attention_class = FusedDotProductAttention
    get_bda = get_bias_dropout_add if use_pre_norm else get_bias_dropout_norm_add
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=normalization_class,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=core_attention_class,
                    linear_proj=RowParallelLinear,
                    q_layernorm=LNImpl if qk_layernorm else IdentityOp,
                    k_layernorm=LNImpl if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bda,
            pre_mlp_layernorm=normalization_class,
            mlp=mlp,
            mlp_bda=get_bda,
            sharded_state_dict_keys_map={
                'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
            },
        ),
    )


# Helper function to get module spec for MLP/MoE
def _get_mlp_module_spec(
    use_te: bool = True,
    num_experts: int = None,
    moe_grouped_gemm: bool = False,
    fp8_coverage: dict = {},
) -> ModuleSpec:
    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        if use_te:
            if is_real_cuda_device_available():
                linear_fc1 = TELayerNormColumnParallelLinear
                linear_fc2 = TERowParallelLinear
            else:
                linear_fc1 = IntelTEColumnParallelLinear
                linear_fc2 = (
                    IntelTERowParallelLinear
                    if fp8_coverage.get('mlp_row_parallel', True)
                    else RowParallelLinear
                )
        else:
            linear_fc1 = ColumnParallelLinear
            linear_fc2 = RowParallelLinear
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=linear_fc1,
                linear_fc2=linear_fc2,
            ),
        )
    else:
        # Mixture of experts with modules in megatron core.
        if use_te and moe_grouped_gemm:
            linear_fc1 = TEColumnParallelGroupedLinear
            linear_fc2 = TERowParallelGroupedLinear
        else:
            linear_fc1 = ColumnParallelLinear
            linear_fc2 = RowParallelLinear

        use_te_grouped_gemm = use_te and HAVE_TE and TEColumnParallelGroupedLinear is not None

        return ModuleSpec(
            module=MoELayer,
            submodules=(
                MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2)
                if not moe_grouped_gemm or use_te_grouped_gemm
                else None
            ),
        )
