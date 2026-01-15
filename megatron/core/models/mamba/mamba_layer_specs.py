# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.ssm.mamba_block import MambaStack, MambaStackSubmodules
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from megatron.core.ssm.mlp_layer import MLPLayer
from megatron.core.tensor_parallel import (
    InferenceLayerNormColumnParallelLinear,
    InferenceRowParallelLinear,
)
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    MultiTokenPredictionBlockSubmodules,
    MultiTokenPredictionLayer,
    MultiTokenPredictionLayerSubmodules,
)

moe = get_moe_module_spec(
    use_te=True,
    num_experts=8,  # Can be any positive integer (must not be None).
    moe_grouped_gemm=True,
    moe_use_legacy_grouped_gemm=False,
)

mamba_stack_spec = ModuleSpec(
    module=MambaStack,
    submodules=MambaStackSubmodules(
        mamba_layer=ModuleSpec(
            module=MambaLayer,
            submodules=MambaLayerSubmodules(
                mixer=ModuleSpec(
                    module=MambaMixer,
                    submodules=MambaMixerSubmodules(
                        in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear
                    ),
                ),
                mamba_bda=get_bias_dropout_add,
            ),
        ),
        # Started with spec from gpt_layer_specs.py (with MLP removed)
        # Using the TE spec because we had problems getting the non-TE spec
        # working
        attention_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=TELayerNormColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        linear_proj=TERowParallelLinear,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        # Started with spec from gpt_layer_specs.py
        # Using the TE spec because we had problems getting the non-TE spec
        # working
        mlp_layer=ModuleSpec(
            module=MLPLayer,
            submodules=TransformerLayerSubmodules(
                mlp=ModuleSpec(
                    module=MLP,
                    submodules=MLPSubmodules(
                        linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
                    ),
                ),
                mlp_bda=get_bias_dropout_add,
            ),
        ),
        moe_layer=ModuleSpec(
            # TODO (rwaleffe): change this to be an "MoELayer" to work with CudaGraphs?
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                pre_mlp_layernorm=TENorm, mlp=moe, mlp_bda=get_bias_dropout_add
            ),
        ),
    ),
)


mamba_inference_stack_spec = ModuleSpec(
    module=MambaStack,
    submodules=MambaStackSubmodules(
        mamba_layer=ModuleSpec(
            module=MambaLayer,
            submodules=MambaLayerSubmodules(
                mixer=ModuleSpec(
                    module=MambaMixer,
                    submodules=MambaMixerSubmodules(
                        in_proj=InferenceLayerNormColumnParallelLinear,
                        out_proj=InferenceRowParallelLinear,
                    ),
                ),
                mamba_bda=get_bias_dropout_add,
            ),
        ),
        # Started with spec from gpt_layer_specs.py (with MLP removed)
        # Using the TE spec because we had problems getting the non-TE spec
        # working
        attention_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=InferenceLayerNormColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        linear_proj=InferenceRowParallelLinear,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        # Started with spec from gpt_layer_specs.py
        # Using the TE spec because we had problems getting the non-TE spec
        # working
        mlp_layer=ModuleSpec(
            module=MLPLayer,
            submodules=TransformerLayerSubmodules(
                mlp=ModuleSpec(
                    module=MLP,
                    submodules=MLPSubmodules(
                        linear_fc1=InferenceLayerNormColumnParallelLinear,
                        linear_fc2=InferenceRowParallelLinear,
                    ),
                ),
                mlp_bda=get_bias_dropout_add,
            ),
        ),
        moe_layer=ModuleSpec(
            # TODO (rwaleffe): change this to be an "MoELayer" to work with CudaGraphs?
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                pre_mlp_layernorm=TENorm, mlp=moe, mlp_bda=get_bias_dropout_add
            ),
        ),
    ),
)


def get_mamba_mtp_block_spec(use_te: bool = True) -> ModuleSpec:
    """MTP block spec for Mamba using unified pattern syntax.

    This spec provides norms and projection only - inner layers are built
    by MultiTokenPredictionLayer using the shared layer_builder with
    mtp_layer_pattern and mamba_submodules passed from MambaModel.

    The number of MTP depths is determined by the parsed unified pattern
    (e.g., "M*M*/MM/MM" -> main="M*M*", mtp="MM", 2 depths).

    Args:
        use_te: Whether to use TransformerEngine modules (default: True)

    Returns:
        ModuleSpec for MultiTokenPredictionBlock

    Example:
        >>> mtp_spec = get_mamba_mtp_block_spec()
        >>> mtp_block = MultiTokenPredictionBlock(
        ...     config=config,
        ...     spec=mtp_spec,
        ...     mtp_layer_pattern="MM",
        ...     mtp_num_depths=2,
        ...     mamba_submodules=mamba_stack_spec.submodules,
        ... )
    """
    norm = TENorm if use_te else TENorm  # Fallback to TENorm for now
    linear = TELayerNormColumnParallelLinear if use_te else TELayerNormColumnParallelLinear

    return ModuleSpec(
        module=MultiTokenPredictionBlock,
        submodules=MultiTokenPredictionBlockSubmodules(
            layer_specs=[
                ModuleSpec(
                    module=MultiTokenPredictionLayer,
                    submodules=MultiTokenPredictionLayerSubmodules(
                        enorm=norm,
                        hnorm=norm,
                        eh_proj=linear,
                        mtp_model_layer=None,  # Built via pattern + mamba_submodules
                        layer_norm=norm,
                    ),
                )
            ]
        ),
    )
