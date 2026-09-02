# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version


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
            "CUDA graphs are not supported",
            id="cuda-graphs",
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
def test_sequence_packing_moe_requires_alltoall_dispatcher():
    # The general allgather-vs-variable_seq_lengths check fires first, since
    # sequence packing derives variable_seq_lengths=True.
    with pytest.raises(ValueError, match="alltoall"):
        _make_packing_config(num_moe_experts=2, moe_token_dispatcher_type="allgather")


@requires_te_2_9
def test_sequence_packing_moe_alltoall_dispatcher_passes():
    config = _make_packing_config(num_moe_experts=2, moe_token_dispatcher_type="alltoall")
    assert config.variable_seq_lengths is True


def test_sequence_packing_rejects_unknown_scheduler():
    # Raised by ModelParallelConfig.__post_init__ before any TE check runs.
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        _make_packing_config(sequence_packing_scheduler="bogus")


def test_sequence_packing_requires_max_seqlen_per_dp_cp_rank():
    with pytest.raises(ValueError, match="max_seqlen_per_dp_cp_rank"):
        _make_packing_config(max_seqlen_per_dp_cp_rank=None)
