# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import List, Optional

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import BackendSpecProvider
from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules
from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerSubmodules,
    DSAttention,
    DSAttentionSubmodules,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)

try:
    import transformer_engine as te  # type: ignore[import-untyped]  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import nvidia_kitchen  # type: ignore[import-not-found]  # pylint: disable=unused-import

    from megatron.core.extensions.kitchen import KitchenSpecProvider

    HAVE_KITCHEN = True
except ImportError:
    HAVE_KITCHEN = False


##########
# Experimental Attention Variant Module Specs
##########


def get_gated_delta_net_module_spec(
    config: TransformerConfig, backend: BackendSpecProvider = None
) -> ModuleSpec:
    """Build module spec for GatedDeltaNet attention."""

    if backend is None:
        backend = _get_backend_spec_provider(config=config)

    rms_norm = config.normalization == "RMSNorm"
    attention = ModuleSpec(
        module=GatedDeltaNet,
        submodules=GatedDeltaNetSubmodules(
            in_proj=backend.column_parallel_layer_norm_linear(),
            out_norm=backend.layer_norm(rms_norm=rms_norm, for_qk=False),
            out_proj=backend.row_parallel_linear(),
        ),
        metainfo={"fuse_input_layernorm": True},
    )
    return attention


def get_dsa_module_spec_for_backend(
    config: TransformerConfig, backend: BackendSpecProvider = None
) -> ModuleSpec:
    """Helper function to get module spec for Sparse Attention."""
    assert config.multi_latent_attention, "Currently only MLA supports sparse attention."
    assert config.qk_l2_norm is False, "qk_l2_norm is not supported with MLA."

    linear_q_up_proj = (
        backend.column_parallel_layer_norm_linear()
        if config.qk_layernorm
        else backend.column_parallel_linear()
    )
    linear_kv_up_proj = (
        backend.column_parallel_layer_norm_linear()
        if config.qk_layernorm
        else backend.column_parallel_linear()
    )

    # Because TransformerEngine does not support sparse attention yet, we use local
    # implementation whether the backend is TransformerEngine or not.
    core_attention = ModuleSpec(
        module=DSAttention,
        submodules=DSAttentionSubmodules(
            indexer=ModuleSpec(
                module=DSAIndexer,
                submodules=DSAIndexerSubmodules(
                    linear_wq_b=backend.linear(),
                    linear_wk=backend.linear(),
                    k_norm=backend.layer_norm(rms_norm=False, for_qk=True),
                    linear_weights_proj=backend.linear(),
                ),
            )
        ),
    )

    attention = ModuleSpec(
        module=MLASelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=MLASelfAttentionSubmodules(
            linear_q_proj=backend.column_parallel_linear(),
            linear_q_down_proj=backend.linear(),
            linear_q_up_proj=linear_q_up_proj,
            linear_kv_down_proj=backend.linear(),
            linear_kv_up_proj=linear_kv_up_proj,
            core_attention=core_attention,
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=IdentityOp,
            kv_layernorm=IdentityOp,
        ),
    )

    return attention


def get_experimental_attention_variant_module_spec(
    config: TransformerConfig, backend: BackendSpecProvider = None
) -> ModuleSpec:
    """Helper function to get module spec for experimental attention variant"""

    if backend is None:
        backend = _get_backend_spec_provider(config=config)

    if config.experimental_attention_variant == "gated_delta_net":
        return get_gated_delta_net_module_spec(config=config, backend=backend)
    else:
        raise ValueError(
            f"Invalid experimental attention variant: {config.experimental_attention_variant}"
        )


##########
# Experimental GPT Decoder Block Spec
##########


def get_transformer_block_with_experimental_attention_variant_spec(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> TransformerBlockSubmodules:
    """Build transformer block spec with experimental attention variants (e.g., linear attention).

    This function constructs a heterogeneous transformer block that supports mixing different
    attention mechanisms (experimental vs standard) and MLP types (MoE vs dense) across layers.
    **Note that, this API is a experimental API in the short term, and might be deprecated in the
    future. In the long run, we will move to a new design that better support hybrid models.**

    Key Design:
        1. Attention and MLP patterns: The attention pattern and MLP pattern are orthogonal
           and determined independently. This allows flexible combinations (e.g., linear attention
           with MoE, or standard attention with dense MLP).
           - Attention pattern: derived from `config.linear_attention_freq` or
             `config.experimental_attention_variant`.
           - MLP pattern: derived from `config.moe_layer_freq`.

        2. Per-Layer Spec Construction: Iterates through layers, constructing transformer
           layer specs based on attention and MLP patterns.

        3. Pipeline Slicing: Extracts layer specs for the current pipeline stage.

    Args:
        config: Transformer configuration containing model hyperparameters and feature flags.
        vp_stage: Virtual pipeline stage index for interleaved pipeline parallelism.
        pp_rank: Pipeline model parallel rank.

    Returns:
        TransformerBlockSubmodules containing per-layer specs and final layer norm.

    Note:
        Currently only supports transformer_engine backend. Kitchen backend can be used as a
        wrapper with TE fallback for unsupported operations.
    """

    backend = _get_backend_spec_provider(config=config)

    # Get attention patterns and specs
    experimental_attention_pattern = [0] * config.num_layers
    if is_linear_attention_variant(config.experimental_attention_variant):
        experimental_attention_pattern = get_linear_attention_pattern(config=config)
    elif config.experimental_attention_variant is not None:
        experimental_attention_pattern = [1] * config.num_layers

    if 1 in experimental_attention_pattern:
        experimental_attention_spec = get_experimental_attention_variant_module_spec(
            config=config, backend=backend
        )
    else:
        experimental_attention_spec = None

    if 0 in experimental_attention_pattern:
        standard_attention_spec = _get_self_attention_module_spec(config=config, backend=backend)
    else:
        standard_attention_spec = None

    # Get MLP patterns and specs
    if config.num_moe_experts is not None:
        moe_layer_pattern = get_moe_layer_pattern(config=config)
    else:
        moe_layer_pattern = [0] * config.num_layers

    if 1 in moe_layer_pattern:
        moe_layer_spec = _get_moe_module_spec(config=config, backend=backend)
    else:
        moe_layer_spec = None

    if 0 in moe_layer_pattern:
        dense_mlp_layer_spec = _get_dense_mlp_module_spec(config=config, backend=backend)
    else:
        dense_mlp_layer_spec = None

    # Get GPT decoder block layer specs
    rms_norm = config.normalization == "RMSNorm"
    layer_specs = []
    for layer_number in range(config.num_layers):
        attention = (
            experimental_attention_spec
            if experimental_attention_pattern[layer_number] == 1
            else standard_attention_spec
        )
        mlp = moe_layer_spec if moe_layer_pattern[layer_number] == 1 else dense_mlp_layer_spec
        input_layernorm = (
            IdentityOp
            if attention.metainfo["fuse_input_layernorm"]
            else backend.layer_norm(rms_norm=rms_norm, for_qk=False)
        )
        pre_mlp_layernorm = (
            IdentityOp
            if mlp.metainfo["fuse_pre_mlp_layernorm"]
            else backend.layer_norm(rms_norm=rms_norm, for_qk=False)
        )

        layer_specs.append(
            ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=input_layernorm,
                    self_attention=attention,
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=pre_mlp_layernorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                ),
            )
        )

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    if config.pipeline_model_parallel_layout is not None:
        local_layer_ids = config.pipeline_model_parallel_layout.get_layer_id_list(
            layer_type=LayerType.decoder, vp_stage=vp_stage, pp_rank=pp_rank
        )
    else:
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank)
        num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)
        local_layer_ids = range(offset, offset + num_layers_to_build)

    layer_specs = [layer_specs[layer_id] for layer_id in local_layer_ids]

    # Get GPT decoder block spec
    gpt_decoder_block_spec = TransformerBlockSubmodules(
        layer_specs=layer_specs, layer_norm=backend.layer_norm(rms_norm=rms_norm, for_qk=False)
    )

    return gpt_decoder_block_spec


##########
# Utilities
##########


def is_linear_attention_variant(experimental_attention_variant: Optional[str]) -> bool:
    """Check if the experimental attention variant is a linear attention variant."""
    linear_attention_variants = ["gated_delta_net"]
    return experimental_attention_variant in linear_attention_variants


def get_moe_layer_pattern(config: TransformerConfig) -> List[int]:
    """Parse config.moe_layer_freq to get per-layer MoE pattern (1=MoE, 0=dense).

    - int N: one MoE layer every N layers (e.g., N=2 -> [1,0,1,0,...])
    - list: use directly as the pattern."""

    if isinstance(config.moe_layer_freq, int):
        # [1,0,0,...,0,1,0,0,...,0,...]
        moe_layer_pattern = [
            1 if (i % config.moe_layer_freq == 0) else 0 for i in range(config.num_layers)
        ]
    elif isinstance(config.moe_layer_freq, list):
        moe_layer_pattern = config.moe_layer_freq
        assert len(moe_layer_pattern) == config.num_layers, (
            f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
            f"expected {config.num_layers}, "
            f"current moe layer pattern: {config.moe_layer_freq}"
        )
    else:
        raise ValueError(
            f"Invalid moe_layer_freq: {type(config.moe_layer_freq)}, {config.moe_layer_freq}"
        )
    return moe_layer_pattern


def get_linear_attention_pattern(config: TransformerConfig) -> List[int]:
    """Parse config.linear_attention_freq to get per-layer attention pattern (1=LA, 0=SDPA).

    - int N: one SDPA layer every N layers (e.g., N=4 -> [1,1,1,0,1,1,1,0,...])
    - list: use directly as the pattern."""

    if isinstance(config.linear_attention_freq, int):
        linear_attention_pattern = [
            # [1,1,...,1,0,1,1,...,1,0,...]
            0 if ((i + 1) % config.linear_attention_freq == 0) else 1
            for i in range(config.num_layers)
        ]
    elif isinstance(config.linear_attention_freq, list):
        linear_attention_pattern = config.linear_attention_freq
        assert len(linear_attention_pattern) == config.num_layers, (
            f"Invalid length of linear_attention_pattern: {len(linear_attention_pattern)}, "
            f"expected {config.num_layers}, "
            f"current linear attention pattern: {config.linear_attention_freq}"
        )
    elif config.linear_attention_freq is None:
        if not is_linear_attention_variant(config.experimental_attention_variant):
            linear_attention_pattern = [0] * config.num_layers
        else:
            # This should be caught by config validation, but raise here as a safety check
            raise ValueError(
                f"Linear attention type {config.experimental_attention_variant} is specified "
                "but linear_attention_freq is None. "
                "Please set linear_attention_freq to specify the LA/SDPA layer pattern."
            )
    else:
        raise ValueError(
            f"Invalid linear_attention_freq: {type(config.linear_attention_freq)},"
            f" {config.linear_attention_freq}"
        )
    return linear_attention_pattern


def _get_backend_spec_provider(config: TransformerConfig) -> BackendSpecProvider:
    """Get backend spec provider for experimental attention variant."""

    assert config.transformer_impl == "transformer_engine", (
        "Experimental GPT decoder block spec only supports "
        "transformer engine implementation for now."
    )
    backend: BackendSpecProvider = (
        KitchenSpecProvider(
            fallback=TESpecProvider(),
            use_kitchen_attention=config.use_kitchen_attention,
            kitchen_attention_backend=config.kitchen_attention_backend,
        )
        if config.use_kitchen
        else TESpecProvider()
    )
    return backend


##########
# Spec functions for non-experimental self attention and MLP layer.
##########


def _get_self_attention_module_spec(
    config: TransformerConfig, backend: BackendSpecProvider = None
) -> ModuleSpec:
    """Get non-experimental self-attention module spec.
    For hybrid models that mix experimental and non-experimental attention architectures.

    Warning: This function may be deprecated in the future."""

    if backend is None:
        backend = _get_backend_spec_provider(config=config)

    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

    layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        qk_l2_norm=config.qk_l2_norm,
        use_kitchen=config.use_kitchen,
        use_te_activation_func=config.use_te_activation_func,
        use_kitchen_attention=config.use_kitchen_attention,
        kitchen_attention_backend=config.kitchen_attention_backend,
    )
    attn_spec = layer_spec.submodules.self_attention
    if config.multi_latent_attention:
        attn_spec.metainfo["fuse_input_layernorm"] = False
    else:
        attn_spec.metainfo["fuse_input_layernorm"] = backend.fuse_layernorm_and_linear()

    return attn_spec


def _get_dense_mlp_module_spec(
    config: TransformerConfig, backend: BackendSpecProvider = None
) -> ModuleSpec:
    """Get dense MLP module spec.
    For hybrid models that mix dense MLP and experimental attention architectures.

    Warning: This function may be deprecated in the future."""

    if backend is None:
        backend = _get_backend_spec_provider(config=config)

    from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec_for_backend

    mlp_spec = get_mlp_module_spec_for_backend(backend=backend, num_experts=None)
    mlp_spec.metainfo["fuse_pre_mlp_layernorm"] = backend.fuse_layernorm_and_linear()

    return mlp_spec


def _get_moe_module_spec(
    config: TransformerConfig, backend: BackendSpecProvider = None
) -> ModuleSpec:
    """Get MoE module spec.
    For hybrid models that mix MoE and experimental attention architectures.

    Warning: This function may be deprecated in the future."""

    if backend is None:
        backend = _get_backend_spec_provider(config=config)

    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend

    moe_spec = get_moe_module_spec_for_backend(
        backend=backend,
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        use_te_activation_func=config.use_te_activation_func,
    )
    moe_spec.metainfo["fuse_pre_mlp_layernorm"] = False
    return moe_spec
