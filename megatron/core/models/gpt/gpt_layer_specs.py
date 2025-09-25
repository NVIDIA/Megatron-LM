# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import warnings
from typing import Optional, Union

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import BackendSpecProvider, LocalSpecProvider
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlockSubmodules,
    get_mtp_layer_offset,
    get_mtp_layer_spec_for_backend,
    get_mtp_num_layers_to_build,
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
from megatron.core.ssm.gated_delta_net_mixer import (
    GatedDeltaNetMixer,
    GatedDeltaNetMixerSubmodules,
)
from megatron.core.ssm.mamba_mixer import (
    MambaMixer,
    MambaMixerSubmodules,
)

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import TEFusedMLP, TENorm
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import nvidia_kitchen  # pylint: disable=unused-import

    from megatron.core.extensions.kitchen import KitchenSpecProvider

    HAVE_KITCHEN = True
except ImportError:
    HAVE_KITCHEN = False

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn("Apex is not installed. Falling back to Torch Norm")
    LNImpl = WrappedTorchNorm
    HAVE_APEX = False


def get_gpt_layer_with_transformer_engine_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    linear_attention_type: Optional[str] = None,
    fp8: Optional[str] = None,  # pylint: disable=unused-argument
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    qk_l2_norm: Optional[bool] = False,
    use_te_op_fuser: Optional[bool] = False,
    use_kitchen: bool = False,
    use_te_activation_func: bool = False,
) -> ModuleSpec:
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fp8 (str, optional): Deprecated. For temporary Nemo compatibility.
        moe_use_legacy_grouped_gemm (bool, optional): Force use the legacy GroupedMLP.
                                                      Defaults to False.
        qk_l2_norm (bool, optional): To use l2 norm for queries/keys. Defaults to False.
        use_te_op_fuser (bool, optional): Use Transformer Engine's operation-based API, which may
                                          enable certain operation fusions. Defaults to False.

    Returns:
        ModuleSpec: Module specification with TE modules

    """
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated'
            " and will be removed soon. Please update your code accordingly."
        )

    if use_kitchen:
        assert HAVE_KITCHEN
        backend: BackendSpecProvider = KitchenSpecProvider(fallback=TESpecProvider())
        if use_te_op_fuser:
            raise AssertionError("use_te_op_fuser not compatible with using kitchen in mlp.")
        if use_te_activation_func:
            raise AssertionError("use_te_activation_func not compatible with using kitchen.")
    else:
        backend = TESpecProvider()

    mlp = get_mlp_module_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        use_te_op_fuser=use_te_op_fuser,
        use_te_activation_func=use_te_activation_func,
    )

    if linear_attention_type is not None:
        if linear_attention_type == "mamba":
            attention_spec = ModuleSpec(
                module=MambaMixer,
                submodules=MambaMixerSubmodules(
                    in_proj=backend.column_parallel_layer_norm_linear(),
                    out_proj=backend.row_parallel_linear(),
                ),
            ),
        elif linear_attention_type == "gated_delta_net":
            attention_spec = ModuleSpec(
                module=GatedDeltaNetMixer,
                submodules=GatedDeltaNetMixerSubmodules(
                    in_proj=backend.column_parallel_layer_norm_linear(),
                    out_norm=backend.layer_norm(),
                    out_proj=backend.row_parallel_linear(),
                    # do not need q_layernorm and k_layernorm
                    # because they are fused with Gated Delta Rule
                )
            )
        else:
            raise ValueError(f"Invalid linear attention type: {linear_attention_type}")
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=attention_spec,
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=backend.layer_norm() if num_experts else IdentityOp,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
            ),
        )
    elif multi_latent_attention:
        assert qk_l2_norm is False, "qk_l2_norm is not supported with MLA."
        linear_q_up_proj = (
            backend.column_parallel_layer_norm_linear()
            if qk_layernorm
            else backend.column_parallel_linear()
        )
        linear_kv_up_proj = (
            backend.column_parallel_layer_norm_linear()
            if qk_layernorm
            else backend.column_parallel_linear()
        )
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=backend.layer_norm(),
                self_attention=ModuleSpec(
                    module=MLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=backend.column_parallel_linear(),
                        linear_q_down_proj=backend.linear(),
                        linear_q_up_proj=linear_q_up_proj,
                        linear_kv_down_proj=backend.linear(),
                        linear_kv_up_proj=linear_kv_up_proj,
                        core_attention=backend.core_attention(),
                        linear_proj=backend.row_parallel_linear(),
                        q_layernorm=IdentityOp,
                        kv_layernorm=IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=backend.layer_norm() if num_experts else IdentityOp,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
            ),
        )
    else:
        qk_norm = backend.layer_norm(for_qk=True)
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=backend.column_parallel_layer_norm_linear(),
                        core_attention=backend.core_attention(),
                        linear_proj=backend.row_parallel_linear(),
                        q_layernorm=(
                            L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                        ),
                        k_layernorm=(
                            L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                        ),
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=backend.layer_norm() if num_experts else IdentityOp,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
                sharded_state_dict_keys_map={
                    "mlp.0.weight": "mlp.linear_fc1.layer_norm_weight",
                    "mlp.0.bias": "mlp.linear_fc1.layer_norm_bias",
                    "mlp.1.basic_ops.0.weight": "mlp.linear_fc1.weight",
                    "mlp.1.basic_ops.1.bias": "mlp.linear_fc1.bias",
                    "mlp.3.basic_ops.0.weight": "mlp.linear_fc2.weight",
                    "mlp.3.basic_ops.1.bias": "mlp.linear_fc2.bias",
                },
            ),
        )


def get_gpt_layer_local_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    linear_attention_type: Optional[str] = None,
    fp8: Optional[str] = None,  # pylint: disable=unused-argument
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
    use_kitchen: bool = False,
) -> ModuleSpec:
    """Use this spec for an implementation using only modules in Megatron-Core.


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fp8 (str, optional): Deprecated. For temporary Nemo compatibility.
        moe_use_legacy_grouped_gemm (bool, optional): Force use the legacy GroupedMLP.
                                                      Defaults to False.
        qk_l2_norm (bool, optional): To use l2 norm for queries/keys. Defaults to False.

    Returns:
        ModuleSpec: Module specification with Megatron-Core modules
    """

    if use_kitchen:
        assert HAVE_KITCHEN
        backend = KitchenSpecProvider(fallback=LocalSpecProvider())
    else:
        backend = LocalSpecProvider()
    # Adjust for RMS norm.
    if normalization == "RMSNorm":
        layer_norm = backend.layer_norm(rms_norm=True, for_qk=False)
        qk_norm = backend.layer_norm(rms_norm=True, for_qk=True)
    else:
        layer_norm = backend.layer_norm(rms_norm=False, for_qk=False)
        qk_norm = backend.layer_norm(rms_norm=False, for_qk=True)

    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "get_gpt_layer_local_spec" has been deprecated'
            " and will be removed soon. Please update your code accordingly."
        )
    
    if linear_attention_type is not None:
        raise NotImplementedError("Linear attention is not supported with local spec yet.")

    mlp = get_mlp_module_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )

    if multi_latent_attention:
        assert qk_l2_norm is False, "qk_l2_norm is not supported with MLA."
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=layer_norm,
                self_attention=ModuleSpec(
                    module=MLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=backend.column_parallel_linear(),
                        linear_q_down_proj=backend.column_parallel_linear(),
                        linear_q_up_proj=backend.column_parallel_linear(),
                        linear_kv_down_proj=backend.column_parallel_linear(),
                        linear_kv_up_proj=backend.column_parallel_linear(),
                        core_attention=backend.core_attention(),
                        linear_proj=backend.row_parallel_linear(),
                        q_layernorm=qk_norm if qk_layernorm else IdentityOp,
                        kv_layernorm=qk_norm if qk_layernorm else IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=layer_norm,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
            ),
        )
    else:
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=layer_norm,
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=backend.column_parallel_linear(),
                        core_attention=backend.core_attention(),
                        linear_proj=backend.row_parallel_linear(),
                        q_layernorm=(
                            L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                        ),
                        k_layernorm=(
                            L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                        ),
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=layer_norm,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
                sharded_state_dict_keys_map={
                    "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
                    "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
                },
            ),
        )


def _get_mlp_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-argument
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
    fp8: Optional[str] = None,  # pylint: disable=unused-argument
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    use_te_op_fuser: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "_get_mlp_module_spec" has been deprecated'
            " and will be removed soon. Please update your code accordingly."
        )
    if use_te_op_fuser:
        if not is_te_min_version("1.13.0"):
            raise ValueError(
                "Transformer Engine operation-based API requires Transformer Engine 1.13+"
            )
        if num_experts is not None:
            raise ValueError(
                "Transformer Engine operation-based API does not support mixture-of-experts"
            )

    return get_mlp_module_spec_for_backend(
        backend=TESpecProvider() if use_te else LocalSpecProvider(),
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        use_te_op_fuser=use_te_op_fuser,
    )


def get_mlp_module_spec_for_backend(
    backend: BackendSpecProvider,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    use_te_op_fuser: Optional[bool] = False,
    use_te_activation_func: bool = False,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""

    linear_fc2 = backend.row_parallel_linear()
    activation_func = backend.activation_func() if use_te_activation_func else None

    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        module = TEFusedMLP if use_te_op_fuser else MLP
        if backend.fuse_layernorm_and_linear():
            linear_fc1 = backend.column_parallel_layer_norm_linear()
            assert linear_fc1 is not None
        else:
            linear_fc1 = backend.column_parallel_linear()
        return ModuleSpec(
            module=module,
            submodules=MLPSubmodules(
                linear_fc1=linear_fc1, linear_fc2=linear_fc2, activation_func=activation_func
            ),
        )
    else:
        # Mixture of experts with modules in megatron core.
        return get_moe_module_spec_for_backend(
            backend=backend,
            num_experts=num_experts,
            moe_grouped_gemm=moe_grouped_gemm,
            moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
            use_te_activation_func=use_te_activation_func,
        )


def get_gpt_decoder_block_spec(
    config: TransformerConfig,
    use_transformer_engine: bool,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """GPT block spec."""
    get_layer_spec_kwargs = {
        "qk_layernorm": config.qk_layernorm,
        "moe_use_legacy_grouped_gemm": config.moe_use_legacy_grouped_gemm,
        "qk_l2_norm": qk_l2_norm,
        "use_kitchen": config.use_kitchen,
    }
    if use_transformer_engine:
        layer_norm_impl = TENorm
        get_layer_spec_kwargs["use_te_activation_func"] = config.use_te_activation_func
        get_layer_spec_fn = get_gpt_layer_with_transformer_engine_spec
    else:
        layer_norm_impl = LNImpl
        get_layer_spec_kwargs["normalization"] = normalization
        get_layer_spec_fn = get_gpt_layer_local_spec
    
    layer_spec_dict = {}
    for moe in [0, 1]:
        for la in [0, 1]:
            if moe and config.moe_layer_freq is None:
                continue
            if la and config.linear_attention_type is None:
                continue
            
            if moe:
                num_experts = config.num_moe_experts
                moe_grouped_gemm = config.moe_grouped_gemm
            else:
                num_experts = None
                moe_grouped_gemm = None
            if la:
                linear_attention_type = config.linear_attention_type
                multi_latent_attention = None
            else:
                linear_attention_type = None
                multi_latent_attention = config.multi_latent_attention

            layer_spec_dict[f"moe{moe}_la{la}"] = get_layer_spec_fn(
                num_experts=num_experts,
                moe_grouped_gemm=moe_grouped_gemm,
                multi_latent_attention=multi_latent_attention,
                linear_attention_type=linear_attention_type,
                **get_layer_spec_kwargs,
            )

    # Parse config.moe_layer_freq to determine the pattern of expert/dense layers.
    # 0 stands for dense layers, 1 stands for expert layers.
    # For integer N: Creates a pattern with one expert layer every N layers.
    # For string pattern: Evaluates the str directly (e.g. "[1,0,1]" for alternating expert/dense).
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
    
    # Parse config.linear_attention_freq to determine the pattern of expert/dense layers.
    # 0 stands for SDPA layers, 1 stands for LA layers.
    # For integer N: Creates a pattern with (N-1) LA layers and 1 SDPA layer every N layers.
    # For string pattern: Evaluates the str directly (e.g. "[1,0,1]" for alternating LA/SDPA).
    if isinstance(config.linear_attention_freq, int):
        linear_attention_pattern = [
            # [1,1,...,1,0,1,1,...,1,0,...]
            0 if ((i + 1) % config.linear_attention_freq == 0)
            else 1 for i in range(config.num_layers)
        ]
    elif isinstance(config.linear_attention_freq, list):
        linear_attention_pattern = config.linear_attention_freq
        assert len(linear_attention_pattern) == config.num_layers, (
            f"Invalid length of linear_attention_pattern: {len(linear_attention_pattern)}, "
            f"expected {config.num_layers}, "
            f"current linear attention pattern: {config.linear_attention_freq}"
        )
    else:
        raise ValueError(
            f"Invalid linear_attention_freq: {type(config.linear_attention_freq)},"
            f" {config.linear_attention_freq}"
        )

    # Create the layer specs for the model.
    layer_specs = []
    for layer_number in range(config.num_layers):
        moe = moe_layer_pattern[layer_number]
        la = linear_attention_pattern[layer_number]
        layer_spec_key = f"moe{moe}_la{la}"
        if layer_spec_key not in layer_spec_dict:
            raise ValueError(f"Invalid layer spec key: {layer_spec_key}")
        layer_specs.append(layer_spec_dict[layer_spec_key])

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    # Note: MCore layer_number starts at 1
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)

    if config.pipeline_model_parallel_layout is not None:
        local_layer_specs = [
            layer_specs[layer_id]
            for layer_id in config.pipeline_model_parallel_layout.get_layer_id_list(
                layer_type=LayerType.decoder, vp_stage=vp_stage, pp_rank=pp_rank
            )
        ]
    else:
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank)
        local_layer_specs = layer_specs[offset : offset + num_layers_to_build]

    # Block spec.
    block_spec = TransformerBlockSubmodules(
        layer_specs=local_layer_specs, layer_norm=layer_norm_impl
    )

    return block_spec


def get_gpt_mtp_block_spec(
    config: TransformerConfig,
    spec: Union[TransformerBlockSubmodules, ModuleSpec],
    use_transformer_engine: bool,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> MultiTokenPredictionBlockSubmodules:
    """GPT Multi-Token Prediction (MTP) block spec."""
    if use_transformer_engine:
        backend: BackendSpecProvider = (
            KitchenSpecProvider(fallback=TESpecProvider())
            if config.use_kitchen
            else TESpecProvider()
        )
    else:
        backend = (
            KitchenSpecProvider(fallback=LocalSpecProvider())
            if config.use_kitchen
            else LocalSpecProvider()
        )
    return get_gpt_mtp_block_spec_for_backend(
        config=config, spec=spec, backend=backend, vp_stage=vp_stage, pp_rank=pp_rank
    )


def get_gpt_mtp_block_spec_for_backend(
    config: TransformerConfig,
    spec: Union[TransformerBlockSubmodules, ModuleSpec],
    backend: BackendSpecProvider,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> MultiTokenPredictionBlockSubmodules:
    """GPT Multi-Token Prediction (MTP) block spec."""
    num_layers_to_build = get_mtp_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)
    if num_layers_to_build == 0:
        return None

    if isinstance(spec, TransformerBlockSubmodules):
        # get the spec for the last layer of decoder block
        transformer_layer_spec = spec.layer_specs[-1]
    elif isinstance(spec, ModuleSpec) and spec.module == TransformerLayer:
        transformer_layer_spec = spec
    else:
        raise ValueError(f"Invalid spec: {spec}")

    mtp_layer_spec = get_mtp_layer_spec_for_backend(
        transformer_layer_spec=transformer_layer_spec, backend=backend
    )
    mtp_num_layers = config.mtp_num_layers if config.mtp_num_layers else 0
    mtp_layer_specs = [mtp_layer_spec] * mtp_num_layers

    offset = get_mtp_layer_offset(config)
    # split the mtp layer specs to only include the layers that are built in this pipeline stage.
    mtp_layer_specs = mtp_layer_specs[offset : offset + num_layers_to_build]
    if len(mtp_layer_specs) > 0:
        assert (
            len(mtp_layer_specs) == config.mtp_num_layers
        ), +f"currently all of the mtp layers must stage in the same pipeline stage."
        mtp_block_spec = MultiTokenPredictionBlockSubmodules(layer_specs=mtp_layer_specs)
    else:
        mtp_block_spec = None

    return mtp_block_spec
