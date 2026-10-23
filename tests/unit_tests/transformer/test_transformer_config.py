# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.activations import squared_relu
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version


def test_gdn2_defaults_to_headwise_cp():
    """GDN2 cannot inherit GDN1/KDA's dev chunkwise default."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        experimental_attention_variant="gdn2",
        linear_attention_freq=1,
        context_parallel_size=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
    )
    assert config.linear_cp_mode == "headwise"


def test_gdn2_rejects_chunkwise_cp():
    """Reject a layout that GDN2's all-to-all implementation does not implement."""
    with pytest.raises(ValueError, match="GDN2 context parallelism requires"):
        TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            experimental_attention_variant="gdn2",
            linear_attention_freq=1,
            context_parallel_size=2,
            linear_cp_mode="chunkwise",
            linear_cp_layout="contiguous",
        )


def test_gdn2_rejects_heads_not_divisible_by_cp():
    """The witness from PR8112 must fail before GDN2 builds zero-width head slices."""
    with pytest.raises(AssertionError, match="linear_head_parallel_size"):
        TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            experimental_attention_variant="gdn2",
            linear_attention_freq=1,
            context_parallel_size=2,
            linear_num_key_heads=1,
            linear_num_value_heads=2,
        )


def _make_glu_interleave_config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        gated_linear_unit=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


@pytest.mark.parametrize("kind", ["routed", "shared"])
def test_glu_interleave_rejects_fp32_model_parameters(kind):
    interleave_kwargs = (
        {"moe_mlp_glu_interleave_size": 32}
        if kind == "routed"
        else {
            "use_grouped_gemm_for_shared_expert": True,
            "moe_shared_expert_intermediate_size": 256,
            "moe_shared_expert_glu_interleave_size": 32,
        }
    )
    with pytest.raises(
        ValueError, match="GLU interleave is not supported with FP32 model parameters"
    ):
        _make_glu_interleave_config(params_dtype=torch.float32, **interleave_kwargs)


@pytest.mark.parametrize("params_dtype", [torch.bfloat16, torch.float16])
def test_glu_interleave_accepts_low_precision_model_parameters(params_dtype):
    config = _make_glu_interleave_config(
        params_dtype=params_dtype,
        moe_mlp_glu_interleave_size=32,
        use_grouped_gemm_for_shared_expert=True,
        moe_shared_expert_intermediate_size=256,
        moe_shared_expert_glu_interleave_size=32,
    )

    assert config.params_dtype == params_dtype


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({}, id="interleave-disabled"),
        pytest.param(
            {"gated_linear_unit": False, "moe_mlp_glu_interleave_size": 32}, id="non-glu-routed"
        ),
        pytest.param(
            {
                "gated_linear_unit": False,
                "use_grouped_gemm_for_shared_expert": True,
                "moe_shared_expert_intermediate_size": 256,
                "moe_shared_expert_glu_interleave_size": 32,
            },
            id="non-glu-shared",
        ),
        pytest.param(
            {
                "moe_shared_expert_intermediate_size": 256,
                "moe_shared_expert_glu_interleave_size": 32,
            },
            id="shared-without-grouped-gemm",
        ),
    ],
)
def test_inactive_glu_interleave_accepts_fp32_model_parameters(overrides):
    config = _make_glu_interleave_config(params_dtype=torch.float32, **overrides)

    assert config.params_dtype == torch.float32


def _make_overlap_config(mtp_num_layers: int | None) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        expert_model_parallel_size=2,
        moe_token_dispatcher_type="alltoall",
        overlap_moe_expert_parallel_comm=True,
        bf16=True,
        mtp_num_layers=mtp_num_layers,
    )


@pytest.mark.parametrize("mtp_num_layers", [None, 0, 1])
def test_ep_a2a_overlap_accepts_supported_mtp_layer_counts(mtp_num_layers: int | None):
    config = _make_overlap_config(mtp_num_layers)

    assert config.mtp_num_layers == mtp_num_layers


@pytest.mark.parametrize("mtp_num_layers", [-1, 2])
def test_ep_a2a_overlap_rejects_unsupported_mtp_layer_counts(mtp_num_layers: int):
    with pytest.raises(AssertionError, match="MTP supports at most one layer"):
        _make_overlap_config(mtp_num_layers)


def _make_dsa_kpool_config(**overrides) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        experimental_attention_variant="dsa",
        add_bias_linear=False,
        **overrides,
    )


@pytest.mark.parametrize("kpool", [0, -1])
def test_dsa_kpool_requires_positive_pool_size(kpool):
    with pytest.raises(ValueError, match="dsa_indexer_kpool must be positive"):
        _make_dsa_kpool_config(dsa_indexer_kpool=kpool)


def test_dsa_kpool_requires_divisible_topk():
    with pytest.raises(ValueError, match="dsa_indexer_topk must be divisible"):
        _make_dsa_kpool_config(dsa_indexer_kpool=4, dsa_indexer_topk=6)


def test_dsa_kpool_requires_topk():
    with pytest.raises(ValueError, match="dsa_indexer_topk must be set"):
        _make_dsa_kpool_config(dsa_indexer_kpool=4)


def test_dsa_kpool_cudnn_accepts_unquantized_fp32_weights_projection():
    config = _make_dsa_kpool_config(
        dsa_indexer_kpool=4,
        dsa_indexer_topk=8,
        dsa_kernel_backend="cudnn",
        dsa_indexer_weights_proj_output_dtype="fp32",
        dsa_indexer_weights_proj_use_quantization=False,
    )
    assert config.dsa_indexer_kpool == 4


def test_per_token_cudnn_still_rejects_fp32_weights_projection():
    with pytest.raises(ValueError, match="for per-token DSA"):
        _make_dsa_kpool_config(
            dsa_indexer_kpool=1,
            dsa_indexer_topk=8,
            dsa_kernel_backend="cudnn",
            dsa_indexer_weights_proj_output_dtype="fp32",
            dsa_indexer_weights_proj_use_quantization=False,
        )


def test_hybrid_gdn_does_not_require_gpt_layer_frequency():
    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        is_hybrid_model=True,
        experimental_attention_variant="gated_delta_net",
    )
    assert config.linear_attention_freq is None


@pytest.mark.parametrize(
    "legacy_fusion, expected", [(None, "cudnn"), (True, "cudnn"), (False, "none")]
)
def test_programmatic_dsv4_kernel_default_respects_legacy_switch(legacy_fusion, expected):
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    config = MLATransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        experimental_attention_variant="dsv4_hybrid",
        apply_dsa_kernel_fusion=legacy_fusion,
        csa_compress_ratios=[4],
    )
    assert config.dsa_kernel_backend == expected


def test_static_dsv4_contiguous_cp_accepts_explicit_packed_metadata_without_scheduler():
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    config = MLATransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        experimental_attention_variant="dsv4_hybrid",
        context_parallel_size=2,
        attention_cp_layout="contiguous",
        csa_compress_ratios=[4],
    )
    assert config.sequence_packing_scheduler is None
    assert config.cp_partition_mode == "contiguous"


def test_deprecated_dsv4_selector_normalizes_layout_before_validation():
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    common = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        context_parallel_size=2,
        cp_partition_mode="contiguous",
        apply_dsa_kernel_fusion=False,
        csa_compress_ratios=[4],
    )
    canonical = MLATransformerConfig(experimental_attention_variant="dsv4_hybrid", **common)
    with pytest.warns(UserWarning, match="linear_attention_type is deprecated"):
        legacy = MLATransformerConfig(linear_attention_type="dsv4_hybrid", **common)
    assert legacy.linear_attention_type is None
    for name in (
        "experimental_attention_variant",
        "attention_cp_layout",
        "linear_cp_layout",
        "dsa_kernel_backend",
    ):
        assert getattr(legacy, name) == getattr(canonical, name)


def test_ncclep_static_shape_flag_cannot_silently_ignore_a_capacity():
    with pytest.raises(ValueError, match="moe_ncclep_static_shape=False conflicts"):
        TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            num_moe_experts=4,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="ncclep",
            moe_expert_rank_capacity_factor=1.0,
            moe_ncclep_static_shape=False,
        )


def _make_scheduled_release_config(moe_ncclep_zero_copy: bool) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        expert_model_parallel_size=2,
        moe_token_dispatcher_type="alltoall",
        overlap_moe_expert_parallel_comm=True,
        ep_overlap_use_scheduled_tensor_release=True,
        moe_ncclep_zero_copy=moe_ncclep_zero_copy,
        bf16=True,
    )


def test_scheduled_tensor_release_accepts_staged_ncclep_buffers():
    config = _make_scheduled_release_config(moe_ncclep_zero_copy=False)

    assert config.ep_overlap_use_scheduled_tensor_release


def test_scheduled_tensor_release_rejects_ncclep_zero_copy():
    with pytest.raises(ValueError, match="moe_ncclep_zero_copy is not supported"):
        _make_scheduled_release_config(moe_ncclep_zero_copy=True)


def test_batch_invariant_backend_rejects_unknown_value_at_construction():
    # Programmatic construction bypasses argparse's Literal choices, so
    # __post_init__ must catch typos before model init.
    with pytest.raises(AssertionError, match="Unknown batch_invariant_backend"):
        TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            batch_invariant_mode=True,
            batch_invariant_backend="te-native",
        )


def test_mhc_fused_backend_defaults_to_auto():
    config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4)

    assert config.mhc_fused_backend == "auto"


@pytest.mark.parametrize("backend", ["native", "triton", "cutile"])
def test_mhc_fused_backend_accepts_explicit_policy(backend: str):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        enable_mhc_connections=True,
        use_fused_mhc=True,
        mhc_fused_backend=backend,
    )

    assert config.mhc_fused_backend == backend


def test_mhc_fused_backend_rejects_unknown_value_at_construction():
    with pytest.raises(ValueError, match="Unknown mhc_fused_backend"):
        TransformerConfig(
            num_layers=1, hidden_size=128, num_attention_heads=4, mhc_fused_backend="cuda"
        )


def test_explicit_mhc_fused_backend_requires_fused_mhc():
    with pytest.raises(ValueError, match="requires use_fused_mhc"):
        TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            enable_mhc_connections=True,
            mhc_fused_backend="native",
        )


def test_gdp_num_householder_defaults_to_three():
    config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4)

    assert config.gdp_num_householder == 3


def test_gdp_num_householder_accepts_positive_values():
    config = TransformerConfig(
        num_layers=1, hidden_size=128, num_attention_heads=4, gdp_num_householder=5
    )

    assert config.gdp_num_householder == 5


def test_from_config_creates_independent_target_config_without_reinitializing():
    class LayerConfig(TransformerConfig):

        def __post_init__(self):
            raise AssertionError("from_config must not reinitialize the target config")

    config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4)
    config.dynamic_value = {"items": []}
    config.dynamic_alias = config.dynamic_value
    config.self_reference = config
    config.state_reference = config.__dict__

    layer_config = LayerConfig.from_config(config)

    assert type(layer_config) is LayerConfig
    assert vars(layer_config).keys() == vars(config).keys()
    assert layer_config.dynamic_value == config.dynamic_value
    assert layer_config.dynamic_value is not config.dynamic_value
    assert layer_config.dynamic_alias is layer_config.dynamic_value
    assert layer_config.self_reference is layer_config
    assert layer_config.state_reference is layer_config.__dict__

    layer_config.dynamic_value["items"].append("changed")
    assert config.dynamic_value == {"items": []}


def _make_softcap_config(attn_logit_softcapping) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        attn_logit_softcapping=attn_logit_softcapping,
    )


@pytest.mark.parametrize("softcap", [None, 50.0])
def test_attn_logit_softcapping_accepts_none_and_positive_values(softcap):
    """None disables softcapping; any positive finite cap is a real cap."""
    assert _make_softcap_config(softcap).attn_logit_softcapping == softcap


@pytest.mark.parametrize("softcap", [0.0, -50.0, float("inf")])
def test_attn_logit_softcapping_rejects_invalid_values(softcap):
    """Values the attention backends disagree about must not reach a kernel.

    0.0 disables softcapping in TransformerEngine but collapses every logit to zero
    in the local path, a negative cap is applied as its absolute value locally while
    FlashAttention ignores it, and a non-finite cap yields NaN logits.
    """
    with pytest.raises(ValueError, match="attn_logit_softcapping must be"):
        _make_softcap_config(softcap)


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        pytest.param(
            {"moe_shortcut_connection": True},
            AssertionError,
            "requires MoE to be enabled",
            id="requires-moe",
        ),
        pytest.param(
            {"num_moe_experts": 2, "moe_shortcut_parallel": True},
            AssertionError,
            "requires moe_shortcut_connection",
            id="parallel-requires-shortcut",
        ),
        pytest.param(
            {
                "num_moe_experts": 2,
                "moe_shortcut_connection": True,
                "recompute_granularity": "full",
            },
            ValueError,
            "not supported with full activation recomputation",
            id="full-recompute",
        ),
        pytest.param(
            {
                "num_moe_experts": 2,
                "moe_shortcut_connection": True,
                "moe_shared_expert_overlap": True,
            },
            ValueError,
            "mutually exclusive",
            id="shared-expert-overlap",
        ),
        pytest.param(
            {"num_moe_experts": 2, "moe_shortcut_connection": True, "cuda_graph_impl": "local"},
            AssertionError,
            "supports CUDA graphs only",
            id="cuda-graphs-layer-scope",
        ),
    ],
)
def test_shortcut_rejects_incompatible_configurations(overrides, error, message):
    with pytest.raises(error, match=message):
        TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            moe_router_topk=1,
            moe_router_pre_softmax=True,
            **overrides,
        )


def test_shortcut_accepts_local_block_cuda_graphs():
    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        moe_shortcut_connection=True,
        cuda_graph_impl="local",
        inference_cuda_graph_scope="block",
    )
    assert config.cuda_graph_impl == "local"
    assert config.inference_cuda_graph_scope.name == "block"


def _make_shortcut_memory_config(**overrides):
    kwargs = dict(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        moe_shortcut_connection=True,
        moe_shortcut_post_norm=True,
        recompute_modules=[],
        offload_modules=[],
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


@pytest.mark.parametrize(
    ("recompute_pre_norm", "offload_post_norm"),
    [
        pytest.param(True, False, id="pre-recompute"),
        pytest.param(False, True, id="post-offload"),
        pytest.param(True, True, id="combined"),
    ],
)
def test_shortcut_memory_controls_accept_independent_and_combined_selection(
    recompute_pre_norm, offload_post_norm
):
    config = _make_shortcut_memory_config(
        recompute_granularity="selective" if recompute_pre_norm else None,
        recompute_modules=["shortcut_pre_mlp_layernorm"] if recompute_pre_norm else [],
        fine_grained_activation_offloading=offload_post_norm,
        offload_modules=["shortcut_post_norm"] if offload_post_norm else [],
    )

    assert ("shortcut_pre_mlp_layernorm" in config.recompute_modules) is recompute_pre_norm
    assert ("shortcut_post_norm" in config.offload_modules) is offload_post_norm


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param(
            {
                "recompute_granularity": "selective",
                "recompute_modules": ["shortcut_pre_mlp_layernorm"],
            },
            id="pre-recompute",
        ),
        pytest.param(
            {"fine_grained_activation_offloading": True, "offload_modules": ["shortcut_post_norm"]},
            id="post-offload",
        ),
    ],
)
def test_shortcut_memory_controls_require_shortcut_connection(overrides):
    with pytest.raises(ValueError, match="require moe_shortcut_connection"):
        _make_shortcut_memory_config(moe_shortcut_connection=False, **overrides)


@pytest.mark.parametrize("recompute_granularity", [None, "full"])
def test_shortcut_pre_norm_recompute_requires_selective_granularity(recompute_granularity):
    with pytest.raises(ValueError, match="requires recompute_granularity='selective'"):
        _make_shortcut_memory_config(
            recompute_granularity=recompute_granularity,
            recompute_modules=["shortcut_pre_mlp_layernorm"],
        )


def test_shortcut_pre_norm_recompute_rejects_fp8_delayed_scaling():
    with pytest.raises(ValueError, match="Delayed scaling does not support"):
        _make_shortcut_memory_config(
            recompute_granularity="selective",
            recompute_modules=["shortcut_pre_mlp_layernorm"],
            fp8="e4m3",
            fp8_recipe="delayed",
        )


def test_shortcut_pre_norm_recompute_applies_fp8_te_version_requirement(monkeypatch):
    from megatron.core.transformer import transformer_config as transformer_config_module

    monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda _version: False)
    monkeypatch.setattr(transformer_config_module, "get_te_version", lambda: "2.5.0")
    with pytest.raises(ValueError, match="transformer-engine>=2.6.0dev0"):
        _make_shortcut_memory_config(
            recompute_granularity="selective",
            recompute_modules=["shortcut_pre_mlp_layernorm"],
            fp8="e4m3",
            fp8_recipe="tensorwise",
        )


@pytest.mark.parametrize("num_householder", [0, -1])
def test_gdp_num_householder_rejects_non_positive_values(num_householder: int):
    with pytest.raises(ValueError, match="gdp_num_householder must be positive"):
        TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            gdp_num_householder=num_householder,
        )


def test_squared_relu_tanh_clamp_allows_te_op_fuser():
    """Whether the fused path can clamp depends on the installed TE; TEGroupedMLP decides."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_grouped_gemm=True,
        activation_func=squared_relu,
        use_fused_weighted_squared_relu=True,
        activation_func_tanh_clamp_scale=16.0,
        use_transformer_engine_op_fuser=True,
    )

    assert config.activation_func_tanh_clamp_scale == 16.0
    assert config.use_transformer_engine_op_fuser is True


def _make_mxfp8_wire_config(**overrides) -> TransformerConfig:
    kwargs = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        expert_model_parallel_size=2,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="ncclep",
        moe_grouped_gemm=True,
        use_transformer_engine_op_fuser=True,
        moe_dispatch_fwd_dtype='mxfp8',
        moe_combine_bwd_dtype='mxfp8',
        bf16=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_mxfp8_wire_dtypes_accept_valid_ncclep_config():
    config = _make_mxfp8_wire_config()

    assert config.moe_dispatch_fwd_dtype == 'mxfp8'
    assert config.moe_combine_bwd_dtype == 'mxfp8'


def test_mxfp8_wire_dtypes_accept_a2a_overlap():
    # The 1F1B a2a overlap schedule only moves/stages the dispatch output as an opaque block,
    # which the plain-tensor MXFP8 carrier survives; the combination is deliberately allowed.
    config = _make_mxfp8_wire_config(overlap_moe_expert_parallel_comm=True)

    assert config.overlap_moe_expert_parallel_comm


@pytest.mark.parametrize(
    "overrides",
    [
        dict(moe_flex_dispatcher_backend="hybridep"),
        dict(moe_token_dispatcher_type="alltoall", moe_flex_dispatcher_backend=None),
    ],
)
def test_mxfp8_wire_dtypes_reject_non_ncclep_dispatcher(overrides):
    with pytest.raises(ValueError, match="require the 'ncclep' flex"):
        _make_mxfp8_wire_config(**overrides)


@pytest.mark.parametrize(
    "overrides", [dict(use_transformer_engine_op_fuser=False), dict(moe_grouped_gemm=False)]
)
def test_mxfp8_wire_dtypes_require_op_fuser_grouped_gemm(overrides):
    with pytest.raises(ValueError, match="require BOTH"):
        _make_mxfp8_wire_config(**overrides)


requires_te_2_9 = pytest.mark.skipif(
    not is_te_min_version("2.9.0"), reason="sequence packing requires Transformer Engine >= 2.9.0"
)


def _make_packing_config(**kwargs) -> TransformerConfig:
    defaults = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        sequence_packing_scheduler="dp_balanced",
        max_seqlen_per_dp_cp_rank=4096,
    )
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


@requires_te_2_9
def test_sequence_packing_dense_config_passes():
    # Dense models have no MoE dispatcher; the (unused) allgather default
    # must not fail sequence-packing validation.
    config = _make_packing_config()
    assert config.variable_seq_lengths is True


@requires_te_2_9
def test_sequence_packing_moe_rejects_allgather_dispatcher():
    # The general allgather-vs-variable_seq_lengths check fires first, since
    # sequence packing derives variable_seq_lengths=True.
    with pytest.raises(ValueError, match="alltoall"):
        _make_packing_config(num_moe_experts=2, moe_token_dispatcher_type="allgather")


@requires_te_2_9
def test_sequence_packing_moe_alltoall_dispatcher_passes():
    config = _make_packing_config(num_moe_experts=2, moe_token_dispatcher_type="alltoall")
    assert config.variable_seq_lengths is True


@requires_te_2_9
def test_sequence_packing_moe_hybridep_requires_uneven_input_padding():
    with pytest.raises(ValueError, match="HybridEP requires"):
        _make_packing_config(
            num_moe_experts=2,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="hybridep",
        )


@requires_te_2_9
def test_sequence_packing_moe_hybridep_with_uneven_input_padding_passes():
    config = _make_packing_config(
        num_moe_experts=2,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_hybridep_pad_uneven_dispatch_inputs=True,
    )

    assert config.variable_seq_lengths is True
    assert config.moe_hybridep_pad_variable_tokens is True
    assert config.moe_hybridep_pad_uneven_dispatch_inputs is True


@pytest.mark.parametrize(
    "dev_flag,main_flag", [(False, False), (True, False), (False, True), (True, True)]
)
def test_hybridep_padding_flag_spellings_share_one_setting(dev_flag, main_flag):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        moe_hybridep_pad_variable_tokens=dev_flag,
        moe_hybridep_pad_uneven_dispatch_inputs=main_flag,
    )
    assert config.moe_hybridep_pad_variable_tokens == (dev_flag or main_flag)
    assert config.moe_hybridep_pad_uneven_dispatch_inputs == (dev_flag or main_flag)


def test_sequence_packing_rejects_unknown_scheduler():
    # Raised by ModelParallelConfig.__post_init__ before any TE check runs.
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        _make_packing_config(sequence_packing_scheduler="bogus")


def test_sequence_packing_requires_max_seqlen_per_dp_cp_rank():
    with pytest.raises(ValueError, match="max_seqlen_per_dp_cp_rank"):
        _make_packing_config(max_seqlen_per_dp_cp_rank=None)


class TestTransformerConfig:
    def test_num_query_groups_divides_num_attention_heads(self):
        config = TransformerConfig(
            num_layers=2, hidden_size=128, num_attention_heads=32, num_query_groups=8
        )
        assert config.num_query_groups == 8

    def test_num_query_groups_defaults_to_num_attention_heads(self):
        config = TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=32)
        assert config.num_query_groups == 32

    def test_num_query_groups_not_dividing_num_attention_heads_raises(self):
        with pytest.raises(ValueError, match="must be a divisor of num_attention_heads"):
            TransformerConfig(
                num_layers=2, hidden_size=128, num_attention_heads=32, num_query_groups=5
            )

    def test_num_query_groups_larger_than_num_attention_heads_raises(self):
        with pytest.raises(ValueError, match="must be a divisor of num_attention_heads"):
            TransformerConfig(
                num_layers=2, hidden_size=128, num_attention_heads=4, num_query_groups=8
            )

    def test_minimal_config_without_attention_heads_is_allowed(self):
        # num_attention_heads defaults to 0 in minimal configs used by many non-attention tests;
        # num_query_groups then defaults to 0 and the check is skipped.
        config = TransformerConfig(num_layers=1, kv_channels=1)
        assert config.num_query_groups == 0


def _hybrid_gdn_config(**overrides):
    return TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        is_hybrid_model=True,
        experimental_attention_variant="gated_delta_net",
        **overrides,
    )


def test_repeated_post_init_keeps_derived_linear_cp_fields():
    """A second __post_init__ must not read its own filled-in defaults as explicit choices."""
    config = _hybrid_gdn_config()
    assert (config.linear_cp_mode, config.linear_cp_layout) == ("chunkwise", "zigzag")
    for _ in range(2):
        config.__post_init__()
        assert (config.linear_cp_mode, config.linear_cp_layout) == ("chunkwise", "zigzag")
        assert config._linear_cp_mode_was_default is True
        assert config._linear_cp_layout_explicit is False


def test_repeated_post_init_keeps_gdn2_headwise_default():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        experimental_attention_variant="gdn2",
        linear_attention_freq=1,
        context_parallel_size=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
    )
    config.__post_init__()
    assert (config.linear_cp_mode, config.linear_cp_layout) == ("headwise", "zigzag")
    assert config._linear_cp_layout_explicit is False


def test_repeated_post_init_preserves_explicit_linear_cp_fields():
    config = _hybrid_gdn_config(linear_cp_mode="chunkwise", linear_cp_layout="contiguous")
    for _ in range(2):
        config.__post_init__()
    assert (config.linear_cp_mode, config.linear_cp_layout) == ("chunkwise", "contiguous")
    assert config._linear_cp_mode_was_default is False
    assert config._linear_cp_layout_explicit is True


def test_repeated_post_init_honors_linear_cp_layout_set_between_passes():
    config = _hybrid_gdn_config()
    config.linear_cp_layout = "contiguous"
    config.__post_init__()
    assert config.linear_cp_layout == "contiguous"
    assert config._linear_cp_layout_explicit is True


def test_repeated_post_init_honors_linear_cp_mode_set_between_passes():
    config = _hybrid_gdn_config()
    config.linear_cp_mode = "headwise"
    config.__post_init__()
    assert (config.linear_cp_mode, config.linear_cp_layout) == ("headwise", "zigzag")
    assert config._linear_cp_mode_was_default is False
    assert config._linear_cp_layout_explicit is False


def test_explicit_chunkwise_with_non_contiguous_layout_is_still_rejected():
    with pytest.raises(ValueError, match="requires linear_cp_layout='contiguous'"):
        _hybrid_gdn_config(linear_cp_mode="chunkwise", linear_cp_layout="zigzag")
