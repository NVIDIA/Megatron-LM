# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for first-class HybridModel architecture descriptions."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.config import MambaInferenceStateConfig
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_architecture import (
    HYBRID_LAYER_TYPE,
    HybridLayerSpec,
    PipelineSplit,
    flatten_hybrid_layer_pattern,
    resolve_hybrid_architecture,
)
from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_inference_stack_spec,
    hybrid_stack_spec,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec


class _Layer:
    """Stand-in module class; resolver tests only inspect ModuleSpec metadata."""


class _Stack:
    """Stand-in HybridStack class for the resolver's legacy symbol mapping."""


def _tagged_spec(layer_type: str) -> ModuleSpec:
    return ModuleSpec(module=_Layer, metainfo={HYBRID_LAYER_TYPE: layer_type})


STACK_SPEC = ModuleSpec(
    module=_Stack,
    submodules=SimpleNamespace(
        mamba_layer=_tagged_spec("mamba"),
        gdn_layer=_tagged_spec("gdn"),
        attention_layer=_tagged_spec("attention"),
        dsa_layer=_tagged_spec("dsa"),
        mla_layer=_tagged_spec("mla"),
        mlp_layer=_tagged_spec("mlp"),
        moe_layer=_tagged_spec("moe"),
    ),
)

SPEC_BY_TYPE = {
    "mamba": STACK_SPEC.submodules.mamba_layer,
    "gdn": STACK_SPEC.submodules.gdn_layer,
    "attention": STACK_SPEC.submodules.attention_layer,
    "dsa": STACK_SPEC.submodules.dsa_layer,
    "mla": STACK_SPEC.submodules.mla_layer,
    "mlp": STACK_SPEC.submodules.mlp_layer,
    "moe": STACK_SPEC.submodules.moe_layer,
}

STOCK_LAYER_TYPES = {
    "mamba_layer": "mamba",
    "gdn_layer": "gdn",
    "attention_layer": "attention",
    "dsa_layer": "dsa",
    "mla_layer": "mla",
    "mlp_layer": "mlp",
    "moe_layer": "moe",
}


def _config(num_layers: int, *, pp_size: int = 1, mtp_num_layers: int | None = None):
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=64,
        num_attention_heads=8,
        num_query_groups=4,
        kv_channels=8,
        ffn_hidden_size=128,
        num_moe_experts=8,
        moe_ffn_hidden_size=96,
        moe_router_topk=2,
        mamba_state_dim=16,
        mamba_head_dim=8,
        mamba_num_heads=8,
        mamba_num_groups=2,
        add_bias_linear=False,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=torch.float32,
        mtp_num_layers=mtp_num_layers,
    )


def _layer(layer_type: str, config: TransformerConfig) -> HybridLayerSpec:
    return HybridLayerSpec(module_spec=SPEC_BY_TYPE[layer_type], config=config)


def _types(layers) -> list[str]:
    return [layer.layer_type for layer in layers]


def _model_shell(architecture, config):
    """Create enough of a HybridModel to exercise its early inference guards."""

    model = HybridModel.__new__(HybridModel)
    torch.nn.Module.__init__(model)
    model.config = config
    if architecture is not None:
        model.resolved_hybrid_architecture = architecture
    return model


@pytest.mark.parametrize(
    ("stack_spec", "expected_types"),
    [
        pytest.param(hybrid_stack_spec, STOCK_LAYER_TYPES, id="training"),
        pytest.param(
            hybrid_inference_stack_spec,
            {name: value for name, value in STOCK_LAYER_TYPES.items() if name != "gdn_layer"},
            id="inference",
        ),
    ],
)
def test_every_stock_hybrid_layer_module_spec_has_stable_semantic_tag(stack_spec, expected_types):
    tagged_types = {
        field_name: module_spec.metainfo.get(HYBRID_LAYER_TYPE)
        for field_name in STOCK_LAYER_TYPES
        if isinstance((module_spec := getattr(stack_spec.submodules, field_name)), ModuleSpec)
    }

    assert tagged_types == expected_types


def test_flatten_preserves_nested_alias_multiplication_and_empty_segments():
    config = _config(6)
    mamba = _layer("mamba", config)
    attention = _layer("attention", config)
    moe = _layer("moe", config)

    segments = flatten_hybrid_layer_pattern(
        [[mamba, attention] * 2, PipelineSplit(), [], PipelineSplit(), (moe, [mamba])]
    )

    assert [_types(segment) for segment in segments] == [
        ["mamba", "attention", "mamba", "attention"],
        [],
        ["moe", "mamba"],
    ]
    assert segments[0][0] is mamba
    assert segments[0][2] is mamba


def test_pp2_vpp2_selection_is_vpp_major_and_offsets_include_empty_chunks():
    config = _config(6, pp_size=2)
    mamba = _layer("mamba", config)
    attention = _layer("attention", config)
    moe = _layer("moe", config)
    mlp = _layer("mlp", config)

    architecture = resolve_hybrid_architecture(
        config=config,
        hybrid_stack_spec=STACK_SPEC,
        layer_specs=[
            [mamba],
            PipelineSplit(),
            [attention, moe],
            PipelineSplit(),
            [],
            PipelineSplit(),
            [mlp, mamba, attention],
        ],
    )

    assert config.virtual_pipeline_model_parallel_size == 2
    expected = {
        (0, 0): (["mamba"], 0),
        (1, 0): (["attention", "moe"], 1),
        (0, 1): ([], 3),
        (1, 1): (["mlp", "mamba", "attention"], 3),
    }
    for (pp_rank, vp_stage), (expected_types, expected_offset) in expected.items():
        layers, offset = architecture.select_segment(pp_rank=pp_rank, pp_size=2, vp_stage=vp_stage)
        assert _types(layers) == expected_types
        assert offset == expected_offset


def test_resolver_materializes_an_isolated_config_for_every_occurrence():
    config = _config(3)
    alias_config = deepcopy(config)
    alias_config.mamba_state_dim = 24
    alias = _layer("mamba", alias_config)

    architecture = resolve_hybrid_architecture(
        config=config, hybrid_stack_spec=STACK_SPEC, layer_specs=[[alias] * 3]
    )

    resolved_configs = [layer.config for layer in architecture.main_layers]
    assert len({id(layer_config) for layer_config in resolved_configs}) == 3
    assert all(layer_config is not alias_config for layer_config in resolved_configs)
    assert [layer_config.mamba_state_dim for layer_config in resolved_configs] == [24, 24, 24]

    resolved_configs[0].mamba_state_dim = 32
    assert resolved_configs[1].mamba_state_dim == 24
    assert alias_config.mamba_state_dim == 24


def test_resolver_materializes_pipeline_and_mtp_topology_from_base_config():
    config = _config(2, pp_size=2, mtp_num_layers=2)
    config.mtp_use_repeated_layer = True
    occurrence_config = deepcopy(config)
    occurrence_config.pipeline_model_parallel_layout = object()
    occurrence_config.mtp_num_layers = 7
    occurrence_config.mtp_use_repeated_layer = False
    occurrence_config.mtp_standalone = True

    architecture = resolve_hybrid_architecture(
        config=config,
        hybrid_stack_spec=STACK_SPEC,
        layer_specs=[
            _layer("mamba", occurrence_config),
            PipelineSplit(),
            _layer("mamba", occurrence_config),
        ],
        mtp_layer_specs=[_layer("attention", occurrence_config)],
    )

    for layer in architecture.main_layers + architecture.mtp_layers:
        assert layer.config.pipeline_model_parallel_layout is None
        assert layer.config.mtp_num_layers == 2
        assert layer.config.mtp_use_repeated_layer is True
        assert layer.config.mtp_standalone is False


def test_resolver_preserves_permitted_per_layer_shape_differences():
    config = _config(8)
    configs = [deepcopy(config) for _ in range(8)]

    configs[0].mamba_state_dim = 24
    configs[0].mamba_head_dim = 4
    configs[0].mamba_num_heads = 16
    configs[0].mamba_num_groups = 4
    configs[1].mamba_state_dim = 32

    configs[2].num_attention_heads = 16
    configs[2].num_query_groups = 8
    configs[2].kv_channels = 4
    configs[3].num_attention_heads = 4
    configs[3].num_query_groups = 2
    configs[3].kv_channels = 16

    configs[4].ffn_hidden_size = 160
    configs[5].ffn_hidden_size = 192

    configs[6].moe_ffn_hidden_size = 112
    configs[6].moe_router_topk = 3
    configs[7].moe_ffn_hidden_size = 144
    configs[7].moe_router_topk = 4

    architecture = resolve_hybrid_architecture(
        config=config,
        hybrid_stack_spec=STACK_SPEC,
        layer_specs=[
            _layer("mamba", configs[0]),
            _layer("mamba", configs[1]),
            _layer("attention", configs[2]),
            _layer("attention", configs[3]),
            _layer("mlp", configs[4]),
            _layer("mlp", configs[5]),
            _layer("moe", configs[6]),
            _layer("moe", configs[7]),
        ],
    )

    layers = architecture.main_layers
    assert (
        layers[0].config.mamba_state_dim,
        layers[0].config.mamba_head_dim,
        layers[0].config.mamba_num_heads,
        layers[0].config.mamba_num_groups,
    ) == (24, 4, 16, 4)
    assert layers[1].config.mamba_state_dim == 32
    assert (
        layers[2].config.num_attention_heads,
        layers[2].config.num_query_groups,
        layers[2].config.kv_channels,
    ) == (16, 8, 4)
    assert (
        layers[3].config.num_attention_heads,
        layers[3].config.num_query_groups,
        layers[3].config.kv_channels,
    ) == (4, 2, 16)
    assert [layers[index].config.ffn_hidden_size for index in (4, 5)] == [160, 192]
    assert [
        (layers[index].config.moe_ffn_hidden_size, layers[index].config.moe_router_topk)
        for index in (6, 7)
    ] == [(112, 3), (144, 4)]


@pytest.mark.parametrize(
    ("layer_type", "field_name", "heterogeneous_value"),
    [
        pytest.param("mamba", "mamba_state_dim", 24, id="mamba-cache"),
        pytest.param("attention", "kv_channels", 16, id="attention-kv-cache"),
        pytest.param("moe", "moe_router_topk", 3, id="moe-router-buffer"),
    ],
)
def test_dynamic_inference_rejects_heterogeneous_runtime_shapes(
    layer_type, field_name, heterogeneous_value
):
    config = _config(2)
    first_config = deepcopy(config)
    second_config = deepcopy(config)
    setattr(second_config, field_name, heterogeneous_value)
    architecture = resolve_hybrid_architecture(
        config=config,
        hybrid_stack_spec=STACK_SPEC,
        layer_specs=[_layer(layer_type, first_config), _layer(layer_type, second_config)],
    )
    model = _model_shell(architecture, config)
    inference_context = SimpleNamespace(is_dynamic_batching=lambda: True)

    with (
        InferenceMode.active(),
        pytest.raises(
            NotImplementedError, match="incompatible with dynamic inference's model-global"
        ),
    ):
        model.forward(
            input_ids=None,
            position_ids=None,
            attention_mask=None,
            inference_context=inference_context,
            runtime_gather_output=False,
        )


@pytest.mark.parametrize(
    ("layer_type", "field_name", "override_value"),
    [
        pytest.param("attention", "kv_channels", 16, id="attention-kv-cache"),
        pytest.param("moe", "moe_router_topk", 3, id="moe-router-buffer"),
    ],
)
def test_dynamic_inference_rejects_uniform_overrides_incompatible_with_global_buffers(
    layer_type, field_name, override_value
):
    config = _config(2)
    layer_config = deepcopy(config)
    setattr(layer_config, field_name, override_value)
    architecture = resolve_hybrid_architecture(
        config=config,
        hybrid_stack_spec=STACK_SPEC,
        layer_specs=[_layer(layer_type, layer_config), _layer(layer_type, layer_config)],
    )
    model = _model_shell(architecture, config)
    inference_context = SimpleNamespace(is_dynamic_batching=lambda: True)

    with (
        InferenceMode.active(),
        pytest.raises(
            NotImplementedError, match="incompatible with dynamic inference's model-global"
        ),
    ):
        model.forward(
            input_ids=None,
            position_ids=None,
            attention_mask=None,
            inference_context=inference_context,
            runtime_gather_output=False,
        )


def test_dynamic_inference_allows_uniform_runtime_shapes_past_heterogeneity_guard():
    config = _config(2)
    architecture = resolve_hybrid_architecture(
        config=config,
        hybrid_stack_spec=STACK_SPEC,
        layer_specs=[_layer("mamba", config), _layer("mamba", config)],
    )
    model = _model_shell(architecture, config)
    inference_context = SimpleNamespace(is_dynamic_batching=lambda: True)

    # The next inference invariant proves the heterogeneity guard allowed this
    # uniform architecture through without needing to construct model layers.
    with (
        InferenceMode.active(),
        pytest.raises(AssertionError, match="Inference must always gather TP logits"),
    ):
        model.forward(
            input_ids=None,
            position_ids=None,
            attention_mask=None,
            inference_context=inference_context,
            runtime_gather_output=False,
        )


def test_dynamic_inference_rejects_before_reading_layer_state_shapes():
    config = _config(2)
    second_config = deepcopy(config)
    second_config.mamba_state_dim = 24
    architecture = resolve_hybrid_architecture(
        config=config,
        hybrid_stack_spec=STACK_SPEC,
        layer_specs=[_layer("mamba", config), _layer("mamba", second_config)],
    )

    class ModelWithUnallocatedState:
        resolved_hybrid_architecture = architecture

        def __init__(self):
            self.config = config

        @property
        def decoder(self):
            raise AssertionError("Mamba state shapes must not be read before rejection")

    with pytest.raises(
        NotImplementedError, match="incompatible with dynamic inference's model-global"
    ):
        MambaInferenceStateConfig.from_model(ModelWithUnallocatedState())


def test_explicit_static_engine_path_skips_dynamic_shape_validation():
    config = _config(2)
    second_config = deepcopy(config)
    second_config.mamba_state_dim = 24
    architecture = resolve_hybrid_architecture(
        config=config,
        hybrid_stack_spec=STACK_SPEC,
        layer_specs=[_layer("mamba", config), _layer("mamba", second_config)],
    )
    decoder = SimpleNamespace(
        layer_type_list=["mamba", "mamba"],
        layers=[SimpleNamespace(mixer=SimpleNamespace(chunk_size=64)), SimpleNamespace()],
        mamba_state_shapes_per_request=lambda: ((8, 4), (8, 16)),
    )
    model = SimpleNamespace(
        resolved_hybrid_architecture=architecture,
        decoder=decoder,
        config=SimpleNamespace(params_dtype=torch.bfloat16, batch_invariant_mode=False),
    )

    inference_config = MambaInferenceStateConfig.from_model(model, validate_dynamic_inference=False)

    assert inference_config is not None
    assert inference_config.layer_type_list == ["M", "M"]


def test_inference_state_normalizes_semantic_layer_types_to_legacy_maps():
    decoder = SimpleNamespace(
        layer_type_list=["mamba", "attention"],
        layers=[SimpleNamespace(mixer=SimpleNamespace(chunk_size=64)), SimpleNamespace()],
        mamba_state_shapes_per_request=lambda: ((8, 4), (8, 16)),
    )
    model = SimpleNamespace(
        decoder=decoder,
        config=SimpleNamespace(params_dtype=torch.bfloat16, batch_invariant_mode=False),
    )

    inference_config = MambaInferenceStateConfig.from_model(model)

    assert inference_config is not None
    assert inference_config.layer_type_list == ["M", "*"]
    assert inference_config.mamba_chunk_size == 64


def test_legacy_inference_keeps_symbol_list_identity_and_skips_direct_shape_rejection():
    layer_type_list = ["M", "M"]
    decoder = SimpleNamespace(
        layer_type_list=layer_type_list,
        layers=[SimpleNamespace(mixer=SimpleNamespace(chunk_size=64)), SimpleNamespace()],
        mamba_state_shapes_per_request=lambda: ((8, 4), (8, 16)),
    )
    model = SimpleNamespace(
        decoder=decoder,
        config=SimpleNamespace(params_dtype=torch.bfloat16, batch_invariant_mode=False),
    )

    assert not hasattr(model, "resolved_hybrid_architecture")

    inference_config = MambaInferenceStateConfig.from_model(model)

    assert inference_config is not None
    assert inference_config.layer_type_list is layer_type_list


def test_legacy_forward_does_not_run_direct_heterogeneity_guard():
    config = _config(2)
    model = _model_shell(None, config)
    inference_context = SimpleNamespace(is_dynamic_batching=lambda: True)

    assert not hasattr(model, "resolved_hybrid_architecture")

    with (
        InferenceMode.active(),
        pytest.raises(AssertionError, match="Inference must always gather TP logits"),
    ):
        model.forward(
            input_ids=None,
            position_ids=None,
            attention_mask=None,
            inference_context=inference_context,
            runtime_gather_output=False,
        )


@pytest.mark.parametrize(
    ("layer_specs", "mtp_layer_specs", "legacy_pattern", "message"),
    [
        pytest.param(["direct"], None, "M", "mutually exclusive", id="direct-and-legacy"),
        pytest.param(None, ["mtp"], "M", "requires direct layer_specs", id="mtp-with-legacy"),
        pytest.param(None, None, None, "Exactly one", id="missing-architecture"),
    ],
)
def test_resolver_rejects_invalid_architecture_source_combinations(
    layer_specs, mtp_layer_specs, legacy_pattern, message
):
    config = _config(1)
    direct_layer = _layer("mamba", config)
    if layer_specs is not None:
        layer_specs = [direct_layer]
    if mtp_layer_specs is not None:
        mtp_layer_specs = [direct_layer]

    with pytest.raises(ValueError, match=message):
        resolve_hybrid_architecture(
            config=config,
            hybrid_stack_spec=STACK_SPEC,
            layer_specs=layer_specs,
            mtp_layer_specs=mtp_layer_specs,
            hybrid_layer_pattern=legacy_pattern,
        )


def test_direct_pp_requires_splits_and_segment_count_divisible_by_pp_size():
    config = _config(2, pp_size=2)
    mamba = _layer("mamba", config)

    with pytest.raises(ValueError, match="must contain explicit PipelineSplit"):
        resolve_hybrid_architecture(
            config=config, hybrid_stack_spec=STACK_SPEC, layer_specs=[mamba, mamba]
        )

    config = _config(2, pp_size=2)
    mamba = _layer("mamba", config)
    with pytest.raises(ValueError, match="not divisible"):
        resolve_hybrid_architecture(
            config=config,
            hybrid_stack_spec=STACK_SPEC,
            layer_specs=[mamba, PipelineSplit(), mamba, PipelineSplit()],
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("pipeline_model_parallel_layout", object()),
        ("num_layers_in_first_pipeline_stage", 1),
        ("num_layers_in_last_pipeline_stage", 1),
        ("account_for_embedding_in_pipeline_split", True),
        ("account_for_loss_in_pipeline_split", True),
    ],
)
def test_direct_splits_reject_other_pipeline_ownership_controls(field_name, value):
    config = _config(2, pp_size=2)
    setattr(config, field_name, value)
    mamba = _layer("mamba", config)

    with pytest.raises(ValueError, match=field_name):
        resolve_hybrid_architecture(
            config=config, hybrid_stack_spec=STACK_SPEC, layer_specs=[mamba, PipelineSplit(), mamba]
        )


def test_direct_splits_reject_configured_vpp_mismatch():
    config = _config(4, pp_size=2)
    config.virtual_pipeline_model_parallel_size = 3
    mamba = _layer("mamba", config)

    with pytest.raises(ValueError, match="imply virtual_pipeline_model_parallel_size=2"):
        resolve_hybrid_architecture(
            config=config,
            hybrid_stack_spec=STACK_SPEC,
            layer_specs=[
                mamba,
                PipelineSplit(),
                mamba,
                PipelineSplit(),
                mamba,
                PipelineSplit(),
                mamba,
            ],
        )


def test_mtp_rejects_pipeline_splits():
    config = _config(1, mtp_num_layers=1)
    mamba = _layer("mamba", config)

    with pytest.raises(ValueError, match="not allowed in an MTP pattern"):
        resolve_hybrid_architecture(
            config=config,
            hybrid_stack_spec=STACK_SPEC,
            layer_specs=[mamba],
            mtp_layer_specs=[mamba, PipelineSplit(), mamba],
        )


def test_direct_mtp_rejects_an_empty_final_decoder_chunk():
    config = _config(1, pp_size=2, mtp_num_layers=1)
    mamba = _layer("mamba", config)

    with pytest.raises(ValueError, match="standalone MTP placement"):
        resolve_hybrid_architecture(
            config=config,
            hybrid_stack_spec=STACK_SPEC,
            layer_specs=[mamba, PipelineSplit()],
            mtp_layer_specs=[mamba],
        )


def test_direct_architecture_rejects_standalone_mtp_placement():
    config = _config(1, mtp_num_layers=1)
    config.mtp_standalone = True
    mamba = _layer("mamba", config)

    with pytest.raises(ValueError, match="do not support standalone MTP"):
        resolve_hybrid_architecture(
            config=config,
            hybrid_stack_spec=STACK_SPEC,
            layer_specs=[mamba],
            mtp_layer_specs=[mamba],
        )


@pytest.mark.parametrize(
    ("num_layers", "layer_count"),
    [pytest.param(2, 1, id="too-few"), pytest.param(1, 2, id="too-many")],
)
def test_direct_layer_count_must_match_transformer_config(num_layers, layer_count):
    config = _config(num_layers)
    mamba = _layer("mamba", config)

    with pytest.raises(ValueError, match=f"contains {layer_count} decoder layers"):
        resolve_hybrid_architecture(
            config=config, hybrid_stack_spec=STACK_SPEC, layer_specs=[mamba] * layer_count
        )


def test_direct_mtp_presence_must_match_configured_depth():
    config = _config(1, mtp_num_layers=1)
    mamba = _layer("mamba", config)
    with pytest.raises(ValueError, match="requires mtp_layer_specs"):
        resolve_hybrid_architecture(
            config=config, hybrid_stack_spec=STACK_SPEC, layer_specs=[mamba]
        )


def test_legacy_summary_does_not_validate_or_mutate_mtp_depth():
    config = _config(1, mtp_num_layers=1)

    architecture = resolve_hybrid_architecture(
        config=config, hybrid_stack_spec=STACK_SPEC, hybrid_layer_pattern="M/M/M"
    )

    assert architecture.mtp_num_layers == 2
    assert config.mtp_num_layers == 1

    config = _config(1)
    mamba = _layer("mamba", config)
    with pytest.raises(ValueError, match="requires config.mtp_num_layers > 0"):
        resolve_hybrid_architecture(
            config=config,
            hybrid_stack_spec=STACK_SPEC,
            layer_specs=[mamba],
            mtp_layer_specs=[mamba],
        )


def test_legacy_summary_preserves_shared_config_identity():
    config = _config(4, mtp_num_layers=2)

    architecture = resolve_hybrid_architecture(
        config=config, hybrid_stack_spec=STACK_SPEC, hybrid_layer_pattern="M*E-/M*/M*"
    )

    assert all(layer.config is config for layer in architecture.main_layers)
    assert all(layer.config is config for layer in architecture.mtp_layers)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("pipeline_model_parallel_layout", object()),
        ("account_for_embedding_in_pipeline_split", True),
        ("account_for_loss_in_pipeline_split", True),
    ],
)
def test_legacy_summary_does_not_reject_new_direct_split_conflicts(field_name, value):
    config = _config(2, pp_size=2)
    setattr(config, field_name, value)

    architecture = resolve_hybrid_architecture(
        config=config, hybrid_stack_spec=STACK_SPEC, hybrid_layer_pattern="M|M"
    )

    assert [_types(segment) for segment in architecture.segments] == [["mamba"], ["mamba"]]
    assert getattr(config, field_name) is value


def test_legacy_summary_does_not_infer_or_mutate_vpp():
    config = _config(4, pp_size=2)

    architecture = resolve_hybrid_architecture(
        config=config, hybrid_stack_spec=STACK_SPEC, hybrid_layer_pattern="M|M|M|M"
    )

    assert len(architecture.segments) == 4
    assert config.virtual_pipeline_model_parallel_size is None


def test_legacy_summary_does_not_add_a_layer_count_validation():
    config = _config(3)

    architecture = resolve_hybrid_architecture(
        config=config, hybrid_stack_spec=STACK_SPEC, hybrid_layer_pattern="M"
    )

    assert _types(architecture.main_layers) == ["mamba"]
    assert config.num_layers == 3


def test_legacy_summary_accepts_layer_classes_supported_by_hybrid_stack():
    stack_spec = deepcopy(STACK_SPEC)
    stack_spec.submodules.mamba_layer = _Layer
    config = _config(1)

    architecture = resolve_hybrid_architecture(
        config=config, hybrid_stack_spec=stack_spec, hybrid_layer_pattern="M"
    )

    assert architecture.main_layers[0].module_spec.module is _Layer
    assert architecture.main_layers[0].config is config


def test_legacy_summary_only_reads_submodule_fields_used_by_the_pattern():
    stack_spec = ModuleSpec(
        module=_Stack, submodules=SimpleNamespace(mamba_layer=_tagged_spec("mamba"))
    )
    config = _config(1)

    architecture = resolve_hybrid_architecture(
        config=config, hybrid_stack_spec=stack_spec, hybrid_layer_pattern="M"
    )

    assert _types(architecture.main_layers) == ["mamba"]


def test_legacy_summary_preserves_empty_final_chunk_with_mtp():
    config = _config(1, pp_size=2, mtp_num_layers=1)

    architecture = resolve_hybrid_architecture(
        config=config, hybrid_stack_spec=STACK_SPEC, hybrid_layer_pattern="M|/M"
    )

    assert [len(segment) for segment in architecture.segments] == [1, 0]
    assert _types(architecture.mtp_layers) == ["mamba"]


def test_direct_architecture_rejects_nonuniform_total_expert_count():
    config = _config(1, mtp_num_layers=1)
    first_config = deepcopy(config)
    second_config = deepcopy(config)
    second_config.num_moe_experts = 16

    with pytest.raises(ValueError, match="uniform num_moe_experts"):
        resolve_hybrid_architecture(
            config=config,
            hybrid_stack_spec=STACK_SPEC,
            layer_specs=[_layer("moe", first_config)],
            mtp_layer_specs=[_layer("moe", second_config)],
        )


def test_direct_architecture_expert_count_must_match_base_config():
    config = _config(1)
    occurrence_config = deepcopy(config)
    occurrence_config.num_moe_experts = 16

    with pytest.raises(ValueError, match="must match the model-wide config"):
        resolve_hybrid_architecture(
            config=config,
            hybrid_stack_spec=STACK_SPEC,
            layer_specs=[_layer("moe", occurrence_config)],
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("fp32_residual_connection", True),
        ("attention_softmax_in_fp32", False),
        ("enable_autocast", True),
        ("fp8_margin", 1),
        ("layernorm_zero_centered_gamma", True),
        ("moe_router_dtype", "fp32"),
        ("mamba_training_ssm_states_dtype", torch.float32),
        ("dsa_indexer_k_norm_fp32", True),
        ("tensor_parallel_num_weight_shards", 2),
        ("expert_tensor_parallel_num_weight_shards", 2),
        ("hierarchical_context_parallel_sizes", [1]),
        ("hybrid_context_parallel", True),
    ],
)
def test_direct_architecture_rejects_per_layer_model_wide_changes(field_name, value):
    config = _config(1)
    occurrence_config = deepcopy(config)
    setattr(occurrence_config, field_name, value)

    with pytest.raises(ValueError, match=field_name):
        resolve_hybrid_architecture(
            config=config,
            hybrid_stack_spec=STACK_SPEC,
            layer_specs=[_layer("mamba", occurrence_config)],
        )


def test_equivalent_legacy_and_direct_architectures_have_same_types_and_order():
    direct_config = _config(4, pp_size=2)
    direct_architecture = resolve_hybrid_architecture(
        config=direct_config,
        hybrid_stack_spec=STACK_SPEC,
        layer_specs=[
            _layer("mamba", direct_config),
            _layer("attention", direct_config),
            PipelineSplit(),
            _layer("moe", direct_config),
            _layer("mlp", direct_config),
        ],
    )
    legacy_architecture = resolve_hybrid_architecture(
        config=_config(4, pp_size=2), hybrid_stack_spec=STACK_SPEC, hybrid_layer_pattern="M*|E-"
    )

    assert [_types(segment) for segment in direct_architecture.segments] == [
        _types(segment) for segment in legacy_architecture.segments
    ]
    assert [layer.module_spec.module for layer in direct_architecture.main_layers] == [
        layer.module_spec.module for layer in legacy_architecture.main_layers
    ]
    assert direct_architecture.source == "direct"
    assert legacy_architecture.source == "legacy"
