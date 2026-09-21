# Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
import copy
from functools import partial

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TELinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.moe_module_specs import (
    get_inference_optimized_moe_spec,
    get_moe_module_spec,
)
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNet2, GatedDeltaNetSubmodules
from megatron.core.ssm.gated_delta_product import (
    GatedDeltaProductMixer,
    GatedDeltaProductMixerSubmodules,
)
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
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
    AbsorbedMLASelfAttentionSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttention,
    CompressedSparseAttentionSubmodules,
    Compressor,
    CompressorSubmodules,
    CSAIndexer,
    CSAIndexerSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    DSv4HybridSelfAttention,
    DSv4HybridSelfAttentionSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerSubmodules,
    DSAttention,
    DSAttentionSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.qsa import (
    QSAIndexer,
    QSAIndexerSubmodules,
    QSASelfAttention,
    QSASelfAttentionSubmodules,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.multi_latent_attention import (
    FusedMLASelfAttention,
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

_csa_compressor = partial(
    Compressor,
    submodules=CompressorSubmodules(linear_wkv=TELinear, linear_wgate=TELinear, norm=TENorm),
)
_csa_indexer = partial(
    CSAIndexer,
    submodules=CSAIndexerSubmodules(
        linear_wq_b=TELinear, linear_weights_proj=TELinear, compressor=_csa_compressor
    ),
)
_csa_qk_norm = TESpecProvider().layer_norm(for_qk=True)

# QSA (Qwen Sparse Attention). The indexer consumes the same post-input-layernorm hidden
# states as the main q/k/v projections (HF semantics), so the input layernorm stays
# unfused (``fuse_input_layernorm=False`` + a separate ``input_layernorm=TENorm``) and
# ``linear_qkv`` is a plain column-parallel linear rather than the fused-LN form.
_qsa_qk_norm = TESpecProvider().layer_norm(for_qk=True)
_qsa_indexer = ModuleSpec(
    module=QSAIndexer,
    submodules=QSAIndexerSubmodules(
        index_qk_proj=TELinear, q_layernorm=_qsa_qk_norm, k_layernorm=_qsa_qk_norm
    ),
)


def _qsa_layer_spec(qk_layernorm: bool) -> ModuleSpec:
    qk_norm = _qsa_qk_norm if qk_layernorm else IdentityOp
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=TENorm,
            self_attention=ModuleSpec(
                module=QSASelfAttention,
                params={"attn_mask_type": AttnMaskType.arbitrary},
                submodules=QSASelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=qk_norm,
                    k_layernorm=qk_norm,
                    indexer=_qsa_indexer,
                ),
                metainfo={"fuse_input_layernorm": False},
            ),
            self_attn_bda=get_bias_dropout_add,
        ),
    )


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
                    # Hybrid MTP selects the combined projection normally and
                    # per-stream projections when mHC is enabled.
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


def _get_gated_delta_product_mamba_layer_spec(in_proj, out_proj):
    return ModuleSpec(
        module=MambaLayer,
        submodules=MambaLayerSubmodules(
            mixer=ModuleSpec(
                module=GatedDeltaProductMixer,
                submodules=GatedDeltaProductMixerSubmodules(in_proj=in_proj, out_proj=out_proj),
            ),
            mamba_bda=get_bias_dropout_add,
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
        gdn2_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=GatedDeltaNet2,
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
                    module=AbsorbedMLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=AbsorbedMLASelfAttentionSubmodules(
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
                self_attention=ModuleSpec(
                    module=DSv4HybridSelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=DSv4HybridSelfAttentionSubmodules(
                        linear_q_down_proj=TELinear,
                        linear_q_up_proj=TEColumnParallelLinear,
                        linear_kv_proj=TEColumnParallelLinear,
                        core_attention=partial(
                            CompressedSparseAttention,
                            submodules=CompressedSparseAttentionSubmodules(
                                compressor=_csa_compressor, indexer=_csa_indexer
                            ),
                        ),
                        linear_proj=TERowParallelLinear,
                        q_layernorm=IdentityOp,
                        kv_layernorm=IdentityOp,
                    ),
                    metainfo={"fuse_input_layernorm": False},
                ),
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        csa_qk_layernorm_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=TENorm,
                self_attention=ModuleSpec(
                    module=DSv4HybridSelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=DSv4HybridSelfAttentionSubmodules(
                        linear_q_down_proj=TELinear,
                        linear_q_up_proj=TEColumnParallelLinear,
                        linear_kv_proj=TEColumnParallelLinear,
                        core_attention=partial(
                            CompressedSparseAttention,
                            submodules=CompressedSparseAttentionSubmodules(
                                compressor=_csa_compressor, indexer=_csa_indexer
                            ),
                        ),
                        linear_proj=TERowParallelLinear,
                        q_layernorm=_csa_qk_norm,
                        kv_layernorm=_csa_qk_norm,
                    ),
                    metainfo={"fuse_input_layernorm": False},
                ),
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        mla_layer=ModuleSpec(
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
                        core_attention=TEDotProductAttention,
                        linear_proj=TERowParallelLinear,
                        q_layernorm=IdentityOp,
                        kv_layernorm=IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        mla_fused_down_proj_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=IdentityOp,
                self_attention=ModuleSpec(
                    module=FusedMLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=TEColumnParallelLinear,
                        linear_qkv_down_proj=TELayerNormColumnParallelLinear,
                        linear_q_up_proj=TEColumnParallelLinear,
                        linear_kv_up_proj=TEColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        linear_proj=TERowParallelLinear,
                        q_layernorm=IdentityOp,
                        kv_layernorm=IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                sharded_state_dict_keys_map={
                    "self_attention.linear_q_down_proj.layer_norm_": "input_layernorm.",
                    "self_attention.linear_kv_down_proj.layer_norm_": "input_layernorm.",
                    "self_attention.linear_qkv_down_proj.layer_norm_": "input_layernorm.",
                },
            ),
        ),
        # Started with spec from gpt_layer_specs.py
        # Using the TE spec because we had problems getting the non-TE spec
        # working
        qsa_layer=_qsa_layer_spec(qk_layernorm=False),
        qsa_qk_layernorm_layer=_qsa_layer_spec(qk_layernorm=True),
        mlp_layer=ModuleSpec(
            module=MLPLayer,
            submodules=TransformerLayerSubmodules(
                mlp=partial(
                    MLP.as_mlp_submodule,
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


gated_delta_product_stack_spec = ModuleSpec(
    module=HybridStack,
    submodules=HybridStackSubmodules(
        mamba_layer=_get_gated_delta_product_mamba_layer_spec(
            TELayerNormColumnParallelLinear, TERowParallelLinear
        ),
        gdn_layer=hybrid_stack_spec.submodules.gdn_layer,
        gdn2_layer=hybrid_stack_spec.submodules.gdn2_layer,
        attention_layer=hybrid_stack_spec.submodules.attention_layer,
        dsa_layer=hybrid_stack_spec.submodules.dsa_layer,
        mlp_layer=hybrid_stack_spec.submodules.mlp_layer,
        moe_layer=hybrid_stack_spec.submodules.moe_layer,
        mtp_block_spec=hybrid_stack_spec.submodules.mtp_block_spec,
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
        gdn_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=GatedDeltaNet,
                    submodules=GatedDeltaNetSubmodules(
                        in_proj=InferenceLayerNormColumnParallelLinear,
                        out_norm=TENorm,
                        out_proj=InferenceRowParallelLinear,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        gdn2_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=GatedDeltaNet2,
                    submodules=GatedDeltaNetSubmodules(
                        in_proj=InferenceLayerNormColumnParallelLinear,
                        out_norm=TENorm,
                        out_proj=InferenceRowParallelLinear,
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
        mla_layer=ModuleSpec(
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
                        core_attention=TEDotProductAttention,
                        linear_proj=InferenceRowParallelLinear,
                        q_layernorm=IdentityOp,
                        kv_layernorm=IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        mla_fused_down_proj_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=IdentityOp,
                self_attention=ModuleSpec(
                    module=FusedMLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=TEColumnParallelLinear,
                        linear_qkv_down_proj=TELayerNormColumnParallelLinear,
                        linear_q_up_proj=TEColumnParallelLinear,
                        linear_kv_up_proj=TEColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        linear_proj=InferenceRowParallelLinear,
                        q_layernorm=IdentityOp,
                        kv_layernorm=IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                sharded_state_dict_keys_map={
                    "self_attention.linear_q_down_proj.layer_norm_": "input_layernorm.",
                    "self_attention.linear_kv_down_proj.layer_norm_": "input_layernorm.",
                    "self_attention.linear_qkv_down_proj.layer_norm_": "input_layernorm.",
                },
            ),
        ),
        # Started with spec from gpt_layer_specs.py
        # Using the TE spec because we had problems getting the non-TE spec
        # working
        mlp_layer=ModuleSpec(
            module=MLPLayer,
            submodules=TransformerLayerSubmodules(
                mlp=partial(
                    MLP.as_mlp_submodule,
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
                            # Keep both projection forms available for Hybrid MTP.
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


gated_delta_product_inference_stack_spec = ModuleSpec(
    module=HybridStack,
    submodules=HybridStackSubmodules(
        mamba_layer=_get_gated_delta_product_mamba_layer_spec(
            InferenceLayerNormColumnParallelLinear, InferenceRowParallelLinear
        ),
        gdn_layer=hybrid_inference_stack_spec.submodules.gdn_layer,
        gdn2_layer=hybrid_inference_stack_spec.submodules.gdn2_layer,
        attention_layer=hybrid_inference_stack_spec.submodules.attention_layer,
        dsa_layer=hybrid_inference_stack_spec.submodules.dsa_layer,
        mlp_layer=hybrid_inference_stack_spec.submodules.mlp_layer,
        moe_layer=hybrid_inference_stack_spec.submodules.moe_layer,
        mtp_block_spec=hybrid_inference_stack_spec.submodules.mtp_block_spec,
    ),
)


# Backward-compatible aliases
mamba_stack_spec = hybrid_stack_spec
mamba_inference_stack_spec = hybrid_inference_stack_spec
gdp_stack_spec = gated_delta_product_stack_spec
gdp_inference_stack_spec = gated_delta_product_inference_stack_spec


# Preserve the existing --spec import path; C/H/W use the standard static stack spec.
hybrid_dsv4_stack_spec = hybrid_stack_spec


def _strip_input_norms(spec: ModuleSpec) -> ModuleSpec:
    """Return a copy of ``spec`` with every pre-sublayer normalization removed.

    Gated-residual layers carry no pre-sublayer norms (the gated-residual group norm owns
    that role), so fused-LN projections (GDN ``in_proj``, attention ``linear_qkv``, dense-MLP
    ``linear_fc1``) become plain column-parallel linears and explicit ``input_layernorm`` /
    ``pre_mlp_layernorm`` entries become ``IdentityOp``.

    This runs once at import time, not per model construction: the result below is a single
    concrete ModuleSpec with one precise behavior, as `hybrid/CLAUDE.md` requires.
    """
    spec = copy.deepcopy(spec)
    sub = spec.submodules

    for name in ("gdn_layer", "gdn2_layer"):
        layer = getattr(sub, name, None)
        if layer is not None and hasattr(layer, "submodules"):
            layer.submodules.self_attention.submodules.in_proj = TEColumnParallelLinear
    for name in ("attention_layer", "qsa_layer", "qsa_qk_layernorm_layer"):
        layer = getattr(sub, name, None)
        if layer is not None and layer is not IdentityOp and hasattr(layer, "submodules"):
            layer.submodules.self_attention.submodules.linear_qkv = TEColumnParallelLinear
            layer.submodules.input_layernorm = IdentityOp
    for name in ("mla_layer", "dsa_layer", "csa_layer", "csa_qk_layernorm_layer"):
        layer = getattr(sub, name, None)
        if layer is not None and layer is not IdentityOp and hasattr(layer, "submodules"):
            layer.submodules.input_layernorm = IdentityOp
    sub.moe_layer.submodules.pre_mlp_layernorm = IdentityOp
    sub.mlp_layer.submodules.mlp = partial(
        MLP.as_mlp_submodule,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear
        ),
    )
    # Lets a builder that was handed this spec explicitly (via --spec) tell it apart from an
    # arbitrary user spec, which must still be warned about as possibly double-normalizing.
    spec.metainfo = {**(spec.metainfo or {}), "gated_residual_norm_free": True}
    return spec


# Norm-free stack for ``mhc_connection_variant='gated_residual'``.
gated_residual_hybrid_stack_spec = _strip_input_norms(hybrid_stack_spec)


def is_gated_residual_norm_free(spec) -> bool:
    """True for stack specs built by :func:`_strip_input_norms`.

    The hybrid builder warns about explicitly provided specs under the gated-residual
    variant (fused input layernorms would normalize the residual streams twice); specs that
    carry the ``gated_residual_norm_free`` metainfo are exempt.
    """
    return bool((getattr(spec, "metainfo", None) or {}).get("gated_residual_norm_free", False))
