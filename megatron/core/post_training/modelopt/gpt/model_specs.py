# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec
from megatron.core.post_training.modelopt.layers import (
    BlockwiseFP8WeightTransformerLayer,
    FP8WeightTransformerLayer,
    Linear,
    Norm,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import L2Norm
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


def get_gpt_modelopt_spec(
    config: TransformerConfig,
    local_core_attention: bool = False,
    remap_te_layernorm: bool = False,
    real_quant_cfg: str = "None",
    qk_l2_norm: bool = False,
    use_arbitrary_attention_mask: bool = False,
):
    """Mix the native spec with TENorm.

    This is essentially the native local spec except for the layernorm implementation
    is using TENorm from Transformer-Engine. The issue is that FusedLayerNorm from apex
    has stopped supporting RMSNorm needed by llama.

    Args:
        config: model's transformer config
        local_core_attention: whether to use local DotProductAttention or TEDotProductAttention
        remap_te_layernorm: whether to perform sharded state_dict prefix mapping on layernorm
        real_quant_cfg: TensorRT Model Optimizer real quantization config
        qk_l2_norm: whether to use Llama4 L2 norm for Q and K
        use_arbitrary_attention_mask: whether to use arbitrary attention mask instead of causal
    """
    num_layers_to_build = get_num_layers_to_build(config)

    # Llama4 Scout-16E support for NeMo. NeMo's GPTConfig is using attribute .qk_l2_norm directly.
    qk_l2_norm = getattr(config, "qk_l2_norm", qk_l2_norm)
    if use_arbitrary_attention_mask:
        attn_mask_type = AttnMaskType.arbitrary
    else:
        attn_mask_type = AttnMaskType.causal

    moe_sharded_state_dict_keys_map = {}
    dense_sharded_state_dict_keys_map = {}
    if remap_te_layernorm:
        input_layernorm_map = {'input_layernorm.': 'self_attention.linear_qkv.layer_norm_'}
        mla_qk_layernorm_map = {
            "self_attention.q_layernorm.": 'self_attention.linear_q_up_proj.layer_norm_',
            "self_attention.kv_layernorm.": 'self_attention.linear_kv_up_proj.layer_norm_',
        }
        dense_sharded_state_dict_keys_map = {'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_'}
        if not config.multi_latent_attention:
            moe_sharded_state_dict_keys_map.update(input_layernorm_map)
            dense_sharded_state_dict_keys_map.update(input_layernorm_map)
        else:
            if config.qk_layernorm:
                moe_sharded_state_dict_keys_map.update(mla_qk_layernorm_map)
                dense_sharded_state_dict_keys_map.update(mla_qk_layernorm_map)

    if real_quant_cfg == "None":
        transformer_layer = TransformerLayer
    elif real_quant_cfg == "fp8_real_quant":
        transformer_layer = FP8WeightTransformerLayer
    elif real_quant_cfg == "fp8_blockwise_real_quant":
        transformer_layer = BlockwiseFP8WeightTransformerLayer
    else:
        raise ValueError("RealQuantTransformerLayer does not support {}".format(real_quant_cfg))

    core_attention = DotProductAttention if local_core_attention else TEDotProductAttention

    if config.multi_latent_attention:
        attn_module = MLASelfAttention
        attn_submodules = MLASelfAttentionSubmodules(
            linear_q_proj=ColumnParallelLinear,
            linear_q_down_proj=Linear,
            q_layernorm=Norm,
            linear_q_up_proj=ColumnParallelLinear,
            linear_kv_down_proj=Linear,
            kv_layernorm=Norm,
            linear_kv_up_proj=ColumnParallelLinear,
            core_attention=core_attention,
            linear_proj=RowParallelLinear,
        )
    else:
        norm = L2Norm if qk_l2_norm else Norm if config.qk_layernorm else IdentityOp
        attn_module = SelfAttention
        attn_submodules = SelfAttentionSubmodules(
            linear_qkv=ColumnParallelLinear,
            core_attention=core_attention,
            linear_proj=RowParallelLinear,
            q_layernorm=norm,
            k_layernorm=norm,
        )

    dense_mlp_spec = get_mlp_module_spec(use_te=False)

    dense_layer_spec = ModuleSpec(
        module=transformer_layer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=Norm,
            self_attention=ModuleSpec(
                module=attn_module,
                params={"attn_mask_type": attn_mask_type},
                submodules=attn_submodules,
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=Norm,
            mlp=dense_mlp_spec,
            mlp_bda=get_bias_dropout_add,
            # Map TE-layernorm-fusion keys back
            sharded_state_dict_keys_map=dense_sharded_state_dict_keys_map,
        ),
    )

    if config.num_moe_experts is None:
        return TransformerBlockSubmodules(
            layer_specs=[dense_layer_spec] * num_layers_to_build, layer_norm=Norm
        )

    moe_mlp_spec = get_mlp_module_spec(
        use_te=False,
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=False,
        # use_te=True, num_experts=config.num_moe_experts, moe_grouped_gemm=True,
    )

    moe_layer_spec = ModuleSpec(
        module=transformer_layer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=Norm,
            self_attention=ModuleSpec(
                module=attn_module,
                params={"attn_mask_type": attn_mask_type},
                submodules=attn_submodules,
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=Norm,
            mlp=moe_mlp_spec,
            mlp_bda=get_bias_dropout_add,
            # Map TE-layernorm-fusion keys back
            sharded_state_dict_keys_map=moe_sharded_state_dict_keys_map,
        ),
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
    layer_specs = layer_specs[offset : offset + num_layers_to_build]

    # Block spec.
    block_spec = TransformerBlockSubmodules(layer_specs=layer_specs, layer_norm=Norm)

    return block_spec
