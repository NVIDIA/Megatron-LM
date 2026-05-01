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
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttention,
    CompressedSparseAttentionSubmodules,
    Compressor,
    CompressorSubmodules,
    CSAIndexer,
    CSAIndexerSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerSubmodules,
    DSAttention,
    DSAttentionSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.dsv4_hybrid_attention import (
    DSv4HybridSelfAttention,
    DSv4HybridSelfAttentionSubmodules,
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


def _make_dsv4_hybrid_attention_spec():
    """Build a self_attention spec for a DSv4 hybrid (CSA/HCA) layer.

    The same spec is used for both 'C' and 'H' symbols; the per-layer compress
    ratio is selected at module-build time from ``config.csa_compress_ratios``.
    """
    compressor_spec = ModuleSpec(
        module=Compressor,
        submodules=CompressorSubmodules(linear_wkv=TELinear, linear_wgate=TELinear, norm=TENorm),
    )

    indexer_spec = ModuleSpec(
        module=CSAIndexer,
        submodules=CSAIndexerSubmodules(
            linear_wq_b=TELinear, linear_weights_proj=TELinear, compressor=compressor_spec
        ),
    )

    core_attention = ModuleSpec(
        module=CompressedSparseAttention,
        submodules=CompressedSparseAttentionSubmodules(
            compressor=compressor_spec, indexer=indexer_spec
        ),
    )

    return ModuleSpec(
        module=DSv4HybridSelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=DSv4HybridSelfAttentionSubmodules(
            linear_q_down_proj=TELinear,
            linear_q_up_proj=TEColumnParallelLinear,
            linear_kv_proj=TEColumnParallelLinear,
            core_attention=core_attention,
            linear_proj=TERowParallelLinear,
            q_layernorm=TENorm,
            kv_layernorm=TENorm,
        ),
    )


_dsv4_hybrid_self_attention_spec = _make_dsv4_hybrid_attention_spec()

# Inference-optimized MoE spec
moe_inference = get_inference_optimized_moe_spec()


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
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        dsa_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=TENorm,
                self_attention=ModuleSpec(
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
                ),
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        csa_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=TENorm,
                self_attention=_dsv4_hybrid_self_attention_spec,
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        hca_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=TENorm,
                self_attention=_dsv4_hybrid_self_attention_spec,
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
            module=MoETransformerLayer,
            submodules=TransformerLayerSubmodules(
                pre_mlp_layernorm=TENorm, mlp=moe, mlp_bda=get_bias_dropout_add
            ),
        ),
        mtp_block_spec=_hybrid_mtp_block_spec,
    ),
)


hybrid_inference_stack_spec = ModuleSpec(
    module=HybridStack,
    submodules=HybridStackSubmodules(
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
        dsa_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=TENorm,
                self_attention=ModuleSpec(
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
    ),
)


# Backward-compatible aliases
mamba_stack_spec = hybrid_stack_spec
mamba_inference_stack_spec = hybrid_inference_stack_spec
