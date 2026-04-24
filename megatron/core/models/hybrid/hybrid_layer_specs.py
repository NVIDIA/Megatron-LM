# Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TELinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.moe_module_specs import (
    get_inference_optimized_moe_spec,
    get_moe_module_spec,
)
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_fusion import MambaMixerForTransformerLayer
from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from megatron.core.ssm.mlp_layer import MLPLayer
from megatron.core.tensor_parallel import (
    InferenceColumnParallelLinear,
    InferenceLayerNormColumnParallelLinear,
    InferenceRowParallelLinear,
)
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerSubmodules,
    DSAttention,
    DSAttentionSubmodules,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    MultiTokenPredictionBlockSubmodules,
    MultiTokenPredictionLayer,
    MultiTokenPredictionLayerSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
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


# Primitive (non-block) specs used when building a fused TransformerLayer at
# runtime. Each primitive is the exact same `ModuleSpec` that already sits
# inside the corresponding stand-alone block spec below – defining them once
# up here lets both places share a single source of truth and lets
# `hybrid_block.py` assemble a `TransformerLayer` on demand without knowing
# the internal wiring of any individual primitive.

# Sequence mixers (fill the `self_attention` slot of a `TransformerLayer`).
_mamba_mixer_spec = ModuleSpec(
    module=MambaMixer,
    submodules=MambaMixerSubmodules(
        in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear
    ),
)

# Fusion-slot variant: same submodules as `_mamba_mixer_spec`, but the
# top-level module is the `MambaMixerForTransformerLayer` subclass that
# adapts `__init__` and `forward` signatures to what TransformerLayer expects.
_mamba_mixer_fusion_spec = ModuleSpec(
    module=MambaMixerForTransformerLayer,
    submodules=MambaMixerSubmodules(
        in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear
    ),
)

_gdn_mixer_spec = ModuleSpec(
    module=GatedDeltaNet,
    submodules=GatedDeltaNetSubmodules(
        in_proj=TELayerNormColumnParallelLinear, out_norm=TENorm, out_proj=TERowParallelLinear
    ),
)

_attention_mixer_spec = ModuleSpec(
    module=SelfAttention,
    params={"attn_mask_type": AttnMaskType.causal},
    submodules=SelfAttentionSubmodules(
        linear_qkv=TELayerNormColumnParallelLinear,
        core_attention=TEDotProductAttention,
        linear_proj=TERowParallelLinear,
    ),
)

_dsa_mixer_spec = ModuleSpec(
    module=MLASelfAttention,
    params={"attn_mask_type": AttnMaskType.causal},
    submodules=MLASelfAttentionSubmodules(
        linear_q_proj=TEColumnParallelLinear,
        linear_q_down_proj=TELinear,
        linear_q_up_proj=TEColumnParallelLinear,
        linear_kv_down_proj=TELinear,
        linear_kv_up_proj=TEColumnParallelLinear,
        core_attention=ModuleSpec(
            module=DSAttention,
            submodules=DSAttentionSubmodules(
                indexer=ModuleSpec(
                    module=DSAIndexer,
                    submodules=DSAIndexerSubmodules(
                        linear_wq_b=TELinear,
                        linear_wk=TELinear,
                        k_norm=TENorm,
                        linear_weights_proj=TELinear,
                    ),
                )
            ),
        ),
        linear_proj=TERowParallelLinear,
        q_layernorm=IdentityOp,
        kv_layernorm=IdentityOp,
    ),
)

# Channel mixers (fill the `mlp` slot of a `TransformerLayer`).
_mlp_mixer_spec = ModuleSpec(
    module=MLP,
    submodules=MLPSubmodules(
        linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
    ),
)

_moe_mixer_spec = moe

# Inference-variant primitives – identical shape, but route through the
# inference-optimised linear classes where the training specs use TE ones.
_mamba_mixer_inference_spec = ModuleSpec(
    module=MambaMixer,
    submodules=MambaMixerSubmodules(
        in_proj=InferenceLayerNormColumnParallelLinear, out_proj=InferenceRowParallelLinear
    ),
)

# Fusion-slot inference variant: same as `_mamba_mixer_inference_spec` but
# with the TransformerLayer-adapted subclass as the top-level module.
_mamba_mixer_inference_fusion_spec = ModuleSpec(
    module=MambaMixerForTransformerLayer,
    submodules=MambaMixerSubmodules(
        in_proj=InferenceLayerNormColumnParallelLinear, out_proj=InferenceRowParallelLinear
    ),
)

_attention_mixer_inference_spec = ModuleSpec(
    module=SelfAttention,
    params={"attn_mask_type": AttnMaskType.causal},
    submodules=SelfAttentionSubmodules(
        linear_qkv=InferenceLayerNormColumnParallelLinear,
        core_attention=TEDotProductAttention,
        linear_proj=InferenceRowParallelLinear,
    ),
)

_dsa_mixer_inference_spec = ModuleSpec(
    module=MLASelfAttention,
    params={"attn_mask_type": AttnMaskType.causal},
    submodules=MLASelfAttentionSubmodules(
        linear_q_proj=TEColumnParallelLinear,
        linear_q_down_proj=TELinear,
        linear_q_up_proj=TEColumnParallelLinear,
        linear_kv_down_proj=TELinear,
        linear_kv_up_proj=TEColumnParallelLinear,
        core_attention=ModuleSpec(
            module=DSAttention,
            submodules=DSAttentionSubmodules(
                indexer=ModuleSpec(
                    module=DSAIndexer,
                    submodules=DSAIndexerSubmodules(
                        linear_wq_b=TELinear,
                        linear_wk=TELinear,
                        k_norm=TENorm,
                        linear_weights_proj=TELinear,
                    ),
                )
            ),
        ),
        linear_proj=InferenceRowParallelLinear,
        q_layernorm=IdentityOp,
        kv_layernorm=IdentityOp,
    ),
)

_mlp_mixer_inference_spec = ModuleSpec(
    module=MLP,
    submodules=MLPSubmodules(
        linear_fc1=InferenceLayerNormColumnParallelLinear, linear_fc2=InferenceRowParallelLinear
    ),
)

_moe_mixer_inference_spec = moe_inference


# MTP block spec - provides norms and projection only.
# Inner layers are built by MultiTokenPredictionLayer using nested HybridStack
_hybrid_mtp_block_spec = ModuleSpec(
    module=MultiTokenPredictionBlock,
    submodules=MultiTokenPredictionBlockSubmodules(
        layer_specs=[
            ModuleSpec(
                module=MultiTokenPredictionLayer,
                submodules=MultiTokenPredictionLayerSubmodules(
                    enorm=TENorm,
                    hnorm=TENorm,
                    eh_proj=TEColumnParallelLinear,
                    mtp_model_layer=None,  # Built via pattern + hybrid_submodules
                    layer_norm=TENorm,
                ),
            )
        ]
    ),
)


hybrid_stack_spec = ModuleSpec(
    module=HybridStack,
    submodules=HybridStackSubmodules(
        mamba_layer=ModuleSpec(
            module=MambaLayer,
            submodules=MambaLayerSubmodules(
                mixer=_mamba_mixer_spec, mamba_bda=get_bias_dropout_add
            ),
        ),
        gdn_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=_gdn_mixer_spec, self_attn_bda=get_bias_dropout_add
            ),
        ),
        # Started with spec from gpt_layer_specs.py (with MLP removed)
        # Using the TE spec because we had problems getting the non-TE spec
        # working
        attention_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=_attention_mixer_spec, self_attn_bda=get_bias_dropout_add
            ),
        ),
        dsa_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=TENorm,
                self_attention=_dsa_mixer_spec,
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        # Started with spec from gpt_layer_specs.py
        # Using the TE spec because we had problems getting the non-TE spec
        # working
        mlp_layer=ModuleSpec(
            module=MLPLayer,
            submodules=TransformerLayerSubmodules(
                mlp=_mlp_mixer_spec, mlp_bda=get_bias_dropout_add
            ),
        ),
        moe_layer=ModuleSpec(
            module=MoETransformerLayer,
            submodules=TransformerLayerSubmodules(
                pre_mlp_layernorm=TENorm, mlp=_moe_mixer_spec, mlp_bda=get_bias_dropout_add
            ),
        ),
        mtp_block_spec=_hybrid_mtp_block_spec,
        # Primitives reused when the hybrid-layer pattern fuses two adjacent
        # layers via the `[XY]` syntax; see `hybrid_layer_fusion.build_fused_layer`.
        # Sequence mixers. MambaMixer uses the TransformerLayer-adapter
        # subclass so its __init__/forward signatures match what
        # TransformerLayer emits in its `self_attention` slot.
        mamba_mixer=_mamba_mixer_fusion_spec,
        gdn_mixer=_gdn_mixer_spec,
        attention_mixer=_attention_mixer_spec,
        dsa_mixer=_dsa_mixer_spec,
        # Channel mixers
        mlp_mixer=_mlp_mixer_spec,
        moe_mixer=_moe_mixer_spec,
    ),
)


hybrid_inference_stack_spec = ModuleSpec(
    module=HybridStack,
    submodules=HybridStackSubmodules(
        mamba_layer=ModuleSpec(
            module=MambaLayer,
            submodules=MambaLayerSubmodules(
                mixer=_mamba_mixer_inference_spec, mamba_bda=get_bias_dropout_add
            ),
        ),
        # Started with spec from gpt_layer_specs.py (with MLP removed)
        # Using the TE spec because we had problems getting the non-TE spec
        # working
        attention_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=_attention_mixer_inference_spec, self_attn_bda=get_bias_dropout_add
            ),
        ),
        dsa_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=TENorm,
                self_attention=_dsa_mixer_inference_spec,
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        # Started with spec from gpt_layer_specs.py
        # Using the TE spec because we had problems getting the non-TE spec
        # working
        mlp_layer=ModuleSpec(
            module=MLPLayer,
            submodules=TransformerLayerSubmodules(
                mlp=_mlp_mixer_inference_spec, mlp_bda=get_bias_dropout_add
            ),
        ),
        moe_layer=ModuleSpec(
            # Use inference-optimized MoE layer for end-to-end CUDA graph support
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                pre_mlp_layernorm=TENorm,
                mlp=_moe_mixer_inference_spec,
                mlp_bda=get_bias_dropout_add,
            ),
        ),
        mtp_block_spec=ModuleSpec(
            module=MultiTokenPredictionBlock,
            submodules=MultiTokenPredictionBlockSubmodules(
                layer_specs=[
                    ModuleSpec(
                        module=MultiTokenPredictionLayer,
                        submodules=MultiTokenPredictionLayerSubmodules(
                            enorm=TENorm,
                            hnorm=TENorm,
                            eh_proj=InferenceColumnParallelLinear,
                            mtp_model_layer=None,  # Built via pattern + hybrid_submodules
                            layer_norm=TENorm,
                        ),
                    )
                ]
            ),
        ),
        # Inference variants of the fusion primitives; GDN is intentionally
        # omitted (the inference stack does not support GDN layers). MambaMixer
        # uses the TransformerLayer-adapter subclass, same as the training spec.
        mamba_mixer=_mamba_mixer_inference_fusion_spec,
        attention_mixer=_attention_mixer_inference_spec,
        dsa_mixer=_dsa_mixer_inference_spec,
        mlp_mixer=_mlp_mixer_inference_spec,
        moe_mixer=_moe_mixer_inference_spec,
    ),
)


# Backward-compatible aliases
mamba_stack_spec = hybrid_stack_spec
mamba_inference_stack_spec = hybrid_inference_stack_spec
