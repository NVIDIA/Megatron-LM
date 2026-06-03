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
                    # Both projection forms are populated so the layer can switch on
                    # `config.enable_hyper_connections` at runtime: mHC=True uses
                    # `e_proj`+`h_proj` per-stream, mHC=False uses fused `eh_proj`.
                    eh_proj=TEColumnParallelLinear,
                    e_proj=TEColumnParallelLinear,
                    h_proj=TEColumnParallelLinear,
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
                            # Populate both projection forms so the layer can switch on
                            # `config.enable_hyper_connections` at runtime.
                            eh_proj=InferenceColumnParallelLinear,
                            e_proj=InferenceColumnParallelLinear,
                            h_proj=InferenceColumnParallelLinear,
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


def hybrid_dsv4_stack_spec(config):
    """Config-aware hybrid stack spec whose ``D`` (DS_ATTENTION) layer runs the DSv4
    ``CompressedSparseAttention`` (CSA/HCA + ``CSAIndexer``), identical to the GPT
    ``dsv4_hybrid`` path, instead of the legacy ``DSAttention``.

    The default ``hybrid_stack_spec`` wires the ``D`` layer to ``MLASelfAttention +
    DSAttention`` (DSv3-style sparse attention with no CSA/HCA compression). To run real
    DSv4 on HybridModel — and to be numerically equivalent to a GPT ``dsv4_hybrid``
    attention layer — we reuse GPT's own ``get_dsv4_hybrid_module_spec_for_backend``
    (which is config-aware, e.g. picks the qk-layernorm form from ``config``) so the two
    model paths build the *same* attention module. Selected via
    ``--spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_dsv4_stack_spec``;
    ``hybrid_builder`` invokes this function with ``config`` when the spec is callable.
    """
    import dataclasses

    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        _get_backend_spec_provider,
        get_dsv4_hybrid_module_spec_for_backend,
    )

    backend = _get_backend_spec_provider(config)
    dsv4_attention = get_dsv4_hybrid_module_spec_for_backend(config, backend)

    def _wrap_dsv4_layer(compress_ratio=None):
        # Wrap the DSv4 attention in a hybrid TransformerLayer. When compress_ratio is given
        # (the 'C'/'H'/'W' layer symbols), bake it into the attention params so the layer uses a
        # fixed CSA(4)/HCA(128)/window-only(0) ratio regardless of csa_compress_ratios; otherwise
        # the layer reads its ratio from config.csa_compress_ratios (array-driven 'D' / GPT-parity
        # path). compress_ratio=0 builds neither the compressor nor the top-k indexer, so the
        # 'W' layer reuses the entire CSA/HCA code path as pure sliding-window attention.
        attn = dsv4_attention
        if compress_ratio is not None:
            attn = dataclasses.replace(
                dsv4_attention, params={**dsv4_attention.params, "compress_ratio": compress_ratio}
            )
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=TENorm, self_attention=attn, self_attn_bda=get_bias_dropout_add
            ),
        )

    submodules = dataclasses.replace(
        hybrid_stack_spec.submodules,
        dsa_layer=_wrap_dsv4_layer(),  # 'D': array-driven (or window) DSv4 attention
        csa_layer=_wrap_dsv4_layer(compress_ratio=4),  # 'C': CSA
        hca_layer=_wrap_dsv4_layer(compress_ratio=128),  # 'H': HCA
        window_layer=_wrap_dsv4_layer(compress_ratio=0),  # 'W': sliding-window-only
    )
    return ModuleSpec(module=HybridStack, submodules=submodules)
