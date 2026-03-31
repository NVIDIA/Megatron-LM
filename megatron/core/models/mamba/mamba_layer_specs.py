# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.moe_module_specs import (
    get_inference_optimized_moe_spec,
    get_moe_module_spec,
)
from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules
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
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    MultiTokenPredictionBlockSubmodules,
    MultiTokenPredictionLayer,
    MultiTokenPredictionLayerSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import L2Norm
from megatron.core.transformer.transformer_layer import (
    MoETransformerLayer,
    TransformerLayer,
    TransformerLayerSubmodules,
)

# This should be private and should not be used outside of this file.
moe = get_moe_module_spec(
    use_te=True,
    num_experts=8,  # Can be any positive integer (must not be None).
    moe_grouped_gemm=True,
)

# Inference-optimized MoE spec
moe_inference = get_inference_optimized_moe_spec()


# MTP block spec for Mamba - provides norms and projection only.
# Inner layers are built by MultiTokenPredictionLayer using nested MambaStack
_mamba_mtp_block_spec = ModuleSpec(
    module=MultiTokenPredictionBlock,
    submodules=MultiTokenPredictionBlockSubmodules(
        layer_specs=[
            ModuleSpec(
                module=MultiTokenPredictionLayer,
                submodules=MultiTokenPredictionLayerSubmodules(
                    enorm=TENorm,
                    hnorm=TENorm,
                    eh_proj=TEColumnParallelLinear,
                    mtp_model_layer=None,  # Built via pattern + mamba_submodules
                    layer_norm=TENorm,
                ),
            )
        ]
    ),
)


def _get_qk_layernorm(config):
    """Resolve q/k layernorm class from config."""
    if config and config.qk_l2_norm:
        return L2Norm
    if config and config.qk_layernorm:
        return TENorm
    return IdentityOp


def get_mamba_stack_spec(config=None):
    """Get the Mamba stack spec with Transformer Engine modules.

    Args:
        config: TransformerConfig. When provided, config-dependent features
            (e.g. qk_layernorm) are applied. When None, defaults are used.
    """
    qk_norm = _get_qk_layernorm(config)

    return ModuleSpec(
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
            gdn_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    self_attention=ModuleSpec(
                        module=GatedDeltaNet,
                        submodules=GatedDeltaNetSubmodules(
                            in_proj=TELayerNormColumnParallelLinear,
                            out_norm=TENorm,
                            out_proj=TERowParallelLinear,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
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
                            q_layernorm=qk_norm,
                            k_layernorm=qk_norm,
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
                            linear_fc1=TELayerNormColumnParallelLinear,
                            linear_fc2=TERowParallelLinear
                        ),
                    ),
                    mlp_bda=get_bias_dropout_add,
                ),
            ),
            moe_layer=ModuleSpec(
                module=MoETransformerLayer,
                submodules=TransformerLayerSubmodules(
                    pre_mlp_layernorm=TENorm, mlp=moe, mlp_bda=get_bias_dropout_add
                ),
            ),
            mtp_block_spec=_mamba_mtp_block_spec,
        ),
    )


# Backward-compatible constant (no QK norm).
mamba_stack_spec = get_mamba_stack_spec()


def get_mamba_inference_stack_spec(config=None):
    """Get the inference-optimized Mamba stack spec.

    Args:
        config: TransformerConfig. When provided, config-dependent features
            (e.g. qk_layernorm) are applied. When None, defaults are used.
    """
    qk_norm = _get_qk_layernorm(config)

    return ModuleSpec(
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
                            q_layernorm=qk_norm,
                            k_layernorm=qk_norm,
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
                # Use inference-optimized MoE layer for end-to-end CUDA graph support
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    pre_mlp_layernorm=TENorm, mlp=moe_inference, mlp_bda=get_bias_dropout_add
                ),
            ),
            mtp_block_spec=_mamba_mtp_block_spec,
        ),
    )


# Backward-compatible constant (no QK norm).
mamba_inference_stack_spec = get_mamba_inference_stack_spec()
