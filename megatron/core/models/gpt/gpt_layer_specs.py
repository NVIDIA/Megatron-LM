# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import warnings
from typing import Optional

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
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
from megatron.core.utils import is_te_min_version

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TELinear,
        TENorm,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedApexNorm

    HAVE_APEX = True
    LNImpl = FusedApexNorm
except ImportError:
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn('Apex is not installed. Falling back to Torch Norm')
    LNImpl = WrappedTorchNorm


def get_gpt_layer_with_transformer_engine_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-arguments
    attn_layernorm: bool = True,
    mlp_layernorm: bool = True,
    qknorm_impl: str = 'te',
    post_layer_norm: bool = False,
    pre_layer_norm: bool = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
) -> ModuleSpec:
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fp8 (str, optional): Deprecated. For temporary Nemo compatibility.
        moe_use_legacy_grouped_gemm (bool, optional): Force use the legacy GroupedMLP.
                                                      Defaults to False.

    Returns:
        ModuleSpec: Module specification with TE modules
    """
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated'
            ' and will be removed soon. Please update your code accordingly.'
        )

    mlp = get_mlp_module_spec(
        use_te=True,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        mlp_layernorm=mlp_layernorm, 
        post_layer_norm=post_layer_norm,
        pre_layer_norm=pre_layer_norm,
    )

    if multi_latent_attention:
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=TENorm,
                self_attention=ModuleSpec(
                    module=MLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
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
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
            ),
        )
    else:

        # TENorm significantly harms convergence when used
        # for QKLayerNorm if TE Version < 1.9;
        # we instead use the Apex implementation.
        if qknorm_impl == "te":
            qk_norm = TENorm if is_te_min_version("1.9.0") else FusedApexNorm
        elif qknorm_impl == "apex":
            qk_norm = FusedApexNorm
        elif qknorm_impl == "torch":
            qk_norm = WrappedTorchNorm
        else:
            raise KeyError(f"Unknown qknorm_impl {qknorm_impl}")

        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=TELayerNormColumnParallelLinear if attn_layernorm and pre_layer_norm else TEColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        linear_proj=TERowParallelLinear,
                        q_layernorm=qk_norm if qk_layernorm else IdentityOp,
                        k_layernorm=qk_norm if qk_layernorm else IdentityOp,
                    ),
                ),
                input_layernorm=IdentityOp,
                pre_mlp_layernorm=IdentityOp, 
                self_attn_bda=get_bias_dropout_add,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
                post_attention_layernorm=TENorm if attn_layernorm and post_layer_norm else IdentityOp,
                post_mlp_layernorm=TENorm if mlp_layernorm and post_layer_norm else IdentityOp,
            ),
        )


def get_gpt_layer_local_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-arguments
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
) -> ModuleSpec:
    """Use this spec for an implementation using only modules in Megatron-Core.


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fp8 (str, optional): Deprecated. For temporary Nemo compatibility.
        moe_use_legacy_grouped_gemm (bool, optional): Force use the legacy GroupedMLP.
                                                      Defaults to False.

    Returns:
        ModuleSpec: Module specification with Megatron-Core modules
    """
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "get_gpt_layer_local_spec" has been deprecated'
            ' and will be removed soon. Please update your code accordingly.'
        )

    mlp = get_mlp_module_spec(
        use_te=False,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )

    if multi_latent_attention:
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=LNImpl,
                self_attention=ModuleSpec(
                    module=MLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=ColumnParallelLinear,
                        linear_q_down_proj=ColumnParallelLinear,
                        linear_q_up_proj=ColumnParallelLinear,
                        linear_kv_down_proj=ColumnParallelLinear,
                        linear_kv_up_proj=ColumnParallelLinear,
                        core_attention=DotProductAttention,
                        linear_proj=RowParallelLinear,
                        q_layernorm=LNImpl if qk_layernorm else IdentityOp,
                        kv_layernorm=LNImpl if qk_layernorm else IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=LNImpl,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
            ),
        )
    else:
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=LNImpl,
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=ColumnParallelLinear,
                        core_attention=DotProductAttention,
                        linear_proj=RowParallelLinear,
                        q_layernorm=LNImpl if qk_layernorm else IdentityOp,
                        k_layernorm=LNImpl if qk_layernorm else IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=LNImpl,
                mlp=mlp,
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
    fp8: Optional[str] = None,  # pylint: disable=unused-arguments
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
):
    warnings.warn(
        """This private function is on a deprecation track. Please switch to `get_mlp_module_spec`
        since it will be removed in a future release."""
    )

    return get_mlp_module_spec(
        use_te=use_te,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        fp8=fp8,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )


def get_mlp_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    mlp_layernorm: bool = True,
    post_layer_norm: bool = False,
    pre_layer_norm: bool = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-arguments
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "_get_mlp_module_spec" has been deprecated'
            ' and will be removed soon. Please update your code accordingly.'
        )

    if use_te:
        if mlp_layernorm and pre_layer_norm:
            fc1 = TELayerNormColumnParallelLinear
        else:
            fc1 = TEColumnParallelLinear
    else:  # No OP arguments can be given without te.
        fc1 = ColumnParallelLinear
        
    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=fc1,
                linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
            ),
        )
    else:
        # Mixture of experts with modules in megatron core.
        return get_moe_module_spec(
            use_te=use_te,
            num_experts=num_experts,
            moe_grouped_gemm=moe_grouped_gemm,
            moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        )


def get_gpt_decoder_block_spec(
    config: TransformerConfig, use_transformer_engine: bool
) -> TransformerBlockSubmodules:
    """GPT block spec."""
    if use_transformer_engine:
        layer_norm_impl = TENorm
    else:
        layer_norm_impl = LNImpl

    # Layer specs.
    dense_layer_spec = (
        get_gpt_layer_with_transformer_engine_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        )
        if use_transformer_engine
        else get_gpt_layer_local_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        )
    )
    moe_layer_spec = (
        get_gpt_layer_with_transformer_engine_spec(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        )
        if use_transformer_engine
        else get_gpt_layer_local_spec(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        )
    )

    # Parse config.moe_layer_freq to determine the pattern of expert/dense layers.
    # 0 stands for dense layers, 1 stands for expert layers.
    # For integer N: Creates a pattern with one expert layer every N layers.
    # For string pattern: Evaluates the str directly (e.g. "[1,0,1]" for alternating expert/dense).
    if isinstance(config.moe_layer_freq, int):
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

    # Create the layer specs for the model.
    layer_specs = []
    for layer_number in range(config.num_layers):
        if moe_layer_pattern[layer_number] == 1:
            layer_specs.append(moe_layer_spec)
        elif moe_layer_pattern[layer_number] == 0:
            layer_specs.append(dense_layer_spec)
        else:
            raise ValueError(f"Invalid layer pattern: {moe_layer_pattern}")

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    # Note: MCore layer_number starts at 1
    offset = get_transformer_layer_offset(config)
    num_layers_to_build = get_num_layers_to_build(config)
    layer_specs = layer_specs[offset : offset + num_layers_to_build]

    # Block spec.
    block_spec = TransformerBlockSubmodules(layer_specs=layer_specs, layer_norm=layer_norm_impl)

    return block_spec
