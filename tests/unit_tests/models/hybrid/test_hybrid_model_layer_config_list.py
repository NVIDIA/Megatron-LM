# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import megatron.core.models.hybrid.hybrid_layer_allocation as hybrid_allocation_module
import megatron.core.models.hybrid.hybrid_model as hybrid_model_module
import megatron.core.transformer.multi_token_prediction as mtp_module
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.models.hybrid import MTPSplit, PipelineSplit, scan_hybrid_layer_config_list
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.ssm.mlp_layer_config import MLPLayerConfig
from megatron.core.transformer import ModuleSpec, TransformerConfig
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    MultiTokenPredictionLayer,
)


class _FakeGroup:

    def __init__(self, rank: int = 0, size: int = 1):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


class _CapturedModule(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs


class _DummyHybridLayer(nn.Module):

    def __init__(self, config: TransformerConfig, layer_number: int, **kwargs):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.scale = nn.Parameter(torch.randn(config.hidden_size))
        self.bias = nn.Parameter(torch.randn(config.hidden_size))

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return hidden_states * self.scale + self.bias


def _config(
    *,
    num_layers: int,
    pp_size: int = 1,
    vp_size: int | None = None,
    mtp_num_layers: int | None = None,
    mtp_use_repeated_layer: bool = False,
) -> TransformerConfig:
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=vp_size,
        pipeline_dtype=torch.float32 if pp_size > 1 else None,
        mtp_num_layers=mtp_num_layers,
        mtp_use_repeated_layer=mtp_use_repeated_layer,
    )


def _layer(config_type, root_config: TransformerConfig):
    return config_type.from_config(root_config)


def _pg_collection(*, pp_rank: int = 0, pp_size: int = 1):
    return SimpleNamespace(
        tp=_FakeGroup(),
        cp=_FakeGroup(),
        pp=_FakeGroup(pp_rank, pp_size),
        tp_cp=_FakeGroup(),
        embd=None,
        dp_cp=_FakeGroup(),
    )


@pytest.fixture
def patch_cpu_model_construction(monkeypatch):
    build_calls = []

    def fake_build_module(module_spec, *args, **kwargs):
        built = _CapturedModule(module_spec=module_spec, args=args, **kwargs)
        build_calls.append(built)
        return built

    monkeypatch.setattr(LanguageModule, "_set_attention_backend", lambda self: None)
    monkeypatch.setattr(hybrid_model_module, "get_pg_rank", lambda group: group.rank())
    monkeypatch.setattr(hybrid_model_module, "get_pg_size", lambda group: group.size())
    monkeypatch.setattr(
        hybrid_model_module, "log_on_each_pipeline_stage", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(hybrid_model_module, "build_module", fake_build_module)
    return build_calls


def test_legacy_pattern_and_flat_list_build_equivalent_real_hybrid_stacks(monkeypatch):
    monkeypatch.setattr(LanguageModule, "_set_attention_backend", lambda self: None)
    monkeypatch.setattr(hybrid_model_module, "get_pg_rank", lambda group: group.rank())
    monkeypatch.setattr(hybrid_model_module, "get_pg_size", lambda group: group.size())
    monkeypatch.setattr(
        hybrid_model_module, "log_on_each_pipeline_stage", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        hybrid_allocation_module, "log_on_each_pipeline_stage", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        hybrid_allocation_module,
        "torch",
        SimpleNamespace(
            distributed=SimpleNamespace(
                get_rank=lambda group: group.rank(), get_world_size=lambda group: group.size()
            )
        ),
    )

    dummy_layer_spec = ModuleSpec(module=_DummyHybridLayer)
    stack_spec = ModuleSpec(
        module=HybridStack,
        submodules=HybridStackSubmodules(mamba_layer=dummy_layer_spec, mlp_layer=dummy_layer_spec),
    )
    legacy_config = _config(num_layers=4, pp_size=2)
    list_config = _config(num_layers=4, pp_size=2)
    architecture = [
        _layer(MambaLayerConfig, list_config),
        _layer(MLPLayerConfig, list_config),
        PipelineSplit,
        _layer(MambaLayerConfig, list_config),
        _layer(MLPLayerConfig, list_config),
    ]

    torch.manual_seed(1234)
    legacy_model = HybridModel(
        config=legacy_config,
        hybrid_stack_spec=stack_spec,
        vocab_size=32,
        max_sequence_length=8,
        hybrid_layer_pattern="M-|M-",
        pre_process=False,
        post_process=False,
        pg_collection=_pg_collection(pp_rank=1, pp_size=2),
    )
    torch.manual_seed(1234)
    list_model = HybridModel(
        config=list_config,
        hybrid_stack_spec=stack_spec,
        vocab_size=32,
        max_sequence_length=8,
        pre_process=False,
        post_process=False,
        pg_collection=_pg_collection(pp_rank=1, pp_size=2),
        layer_config_list=architecture,
    )

    expected_config_types = [MambaLayerConfig, MLPLayerConfig]
    assert type(legacy_model.decoder) is HybridStack
    assert type(list_model.decoder) is HybridStack
    assert [type(config) for config in legacy_model.decoder.layer_config_list] == (
        expected_config_types
    )
    assert [type(config) for config in list_model.decoder.layer_config_list] == (
        expected_config_types
    )
    assert legacy_model.decoder.layer_type_list == list_model.decoder.layer_type_list == list("M-")
    assert [type(layer) for layer in legacy_model.decoder.layers] == [_DummyHybridLayer] * 2
    assert [type(layer) for layer in list_model.decoder.layers] == [_DummyHybridLayer] * 2
    assert [layer.layer_number for layer in legacy_model.decoder.layers] == [3, 4]
    assert [layer.layer_number for layer in list_model.decoder.layers] == [3, 4]
    assert all(
        layer.config is layer_config
        for layer, layer_config in zip(
            list_model.decoder.layers, list_model.decoder.layer_config_list, strict=True
        )
    )

    legacy_state = {key: value.detach().clone() for key, value in legacy_model.state_dict().items()}
    list_state = {key: value.detach().clone() for key, value in list_model.state_dict().items()}
    assert legacy_state
    assert legacy_state.keys() == list_state.keys()
    assert {key: value.shape for key, value in legacy_state.items()} == {
        key: value.shape for key, value in list_state.items()
    }
    for key in legacy_state:
        torch.testing.assert_close(legacy_state[key], list_state[key])

    with torch.no_grad():
        next(list_model.parameters()).add_(1.0)
        next(legacy_model.parameters()).sub_(1.0)
    list_load_result = list_model.load_state_dict(legacy_state, strict=True)
    legacy_load_result = legacy_model.load_state_dict(list_state, strict=True)
    assert list_load_result.missing_keys == list_load_result.unexpected_keys == []
    assert legacy_load_result.missing_keys == legacy_load_result.unexpected_keys == []

    torch.manual_seed(5678)
    hidden_states = torch.randn(5, 2, legacy_config.hidden_size)
    legacy_model.decoder.set_input_tensor(hidden_states.clone())
    list_model.decoder.set_input_tensor(hidden_states.clone())
    legacy_output = legacy_model.decoder(hidden_states=None, attention_mask=None)
    list_output = list_model.decoder(hidden_states=None, attention_mask=None)
    torch.testing.assert_close(legacy_output, list_output)


@pytest.mark.parametrize(
    ("pp_rank", "vp_stage", "expected_type", "expected_offset"),
    [
        (0, 0, MambaLayerConfig, 0),
        (1, 0, AttentionLayerConfig, 1),
        (0, 1, MoELayerConfig, 2),
        (1, 1, MLPLayerConfig, 3),
    ],
)
def test_pipeline_split_selection_is_vp_major_and_clones_configs(
    monkeypatch, pp_rank, vp_stage, expected_type, expected_offset
):
    config = _config(num_layers=4, pp_size=2, vp_size=2)
    source_configs = [
        _layer(MambaLayerConfig, config),
        _layer(AttentionLayerConfig, config),
        _layer(MoELayerConfig, config),
        _layer(MLPLayerConfig, config),
    ]
    architecture = [
        source_configs[0],
        PipelineSplit,
        source_configs[1],
        PipelineSplit,
        source_configs[2],
        PipelineSplit,
        source_configs[3],
    ]
    metadata = scan_hybrid_layer_config_list(architecture, pp_size=2)
    monkeypatch.setattr(hybrid_model_module, "get_pg_rank", lambda group: pp_rank)
    monkeypatch.setattr(hybrid_model_module, "get_pg_size", lambda group: 2)
    monkeypatch.setattr(
        hybrid_model_module, "log_on_each_pipeline_stage", lambda *args, **kwargs: None
    )

    selected, offset = hybrid_model_module._select_pipeline_config_segment(
        architecture, metadata, config, _FakeGroup(pp_rank, 2), vp_stage
    )

    assert offset == expected_offset
    assert [type(layer_config) for layer_config in selected] == [expected_type]
    assert selected[0] is not source_configs[expected_offset]


@pytest.mark.parametrize(
    ("pp_rank", "vp_stage", "expected_indices", "expected_offset"),
    [(0, 0, [0, 1], 0), (1, 0, [2], 2), (0, 1, [3, 4, 5], 3), (1, 1, [6, 7], 6)],
)
def test_uneven_pipeline_split_segments_preserve_ownership_and_offsets(
    monkeypatch, pp_rank, vp_stage, expected_indices, expected_offset
):
    config = _config(num_layers=8, pp_size=2, vp_size=2)
    source_configs = [_layer(MambaLayerConfig, config) for _ in range(8)]
    for index, source_config in enumerate(source_configs):
        source_config.architecture_index = index
    segments = [source_configs[0:2], source_configs[2:3], source_configs[3:6], source_configs[6:8]]
    architecture = []
    for segment_index, segment in enumerate(segments):
        if segment_index:
            architecture.append(PipelineSplit)
        architecture.extend(segment)

    metadata = scan_hybrid_layer_config_list(architecture, pp_size=2)
    monkeypatch.setattr(hybrid_model_module, "get_pg_rank", lambda group: pp_rank)
    monkeypatch.setattr(hybrid_model_module, "get_pg_size", lambda group: 2)
    monkeypatch.setattr(
        hybrid_model_module, "log_on_each_pipeline_stage", lambda *args, **kwargs: None
    )

    selected, offset = hybrid_model_module._select_pipeline_config_segment(
        architecture, metadata, config, _FakeGroup(pp_rank, 2), vp_stage
    )

    assert offset == expected_offset
    assert [layer_config.architecture_index for layer_config in selected] == expected_indices
    assert all(
        selected_config is not source_configs[source_index]
        for selected_config, source_index in zip(selected, expected_indices, strict=True)
    )


def test_implicit_pipeline_selection_evenly_slices_pp_and_vpp(monkeypatch):
    config = _config(num_layers=4, pp_size=2, vp_size=2)
    architecture = [
        _layer(MambaLayerConfig, config),
        _layer(AttentionLayerConfig, config),
        _layer(MoELayerConfig, config),
        _layer(MLPLayerConfig, config),
    ]
    metadata = scan_hybrid_layer_config_list(architecture, pp_size=2)
    monkeypatch.setattr(hybrid_model_module, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(hybrid_model_module, "get_pg_size", lambda group: 2)
    monkeypatch.setattr(
        hybrid_model_module, "log_on_each_pipeline_stage", lambda *args, **kwargs: None
    )

    selected, offset = hybrid_model_module._select_pipeline_config_segment(
        architecture, metadata, config, _FakeGroup(0, 2), vp_stage=1
    )

    assert offset == 2
    assert [type(layer_config) for layer_config in selected] == [MoELayerConfig]
    assert selected[0] is not architecture[2]


def test_implicit_pipeline_selection_requires_exact_divisibility(monkeypatch):
    config = _config(num_layers=3, pp_size=2)
    architecture = [
        _layer(MambaLayerConfig, config),
        _layer(AttentionLayerConfig, config),
        _layer(MLPLayerConfig, config),
    ]
    metadata = scan_hybrid_layer_config_list(architecture, pp_size=2)
    monkeypatch.setattr(hybrid_model_module, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(hybrid_model_module, "get_pg_size", lambda group: 2)

    with pytest.raises(ValueError, match="must be evenly divisible across PP=2 and VPP=1"):
        hybrid_model_module._select_pipeline_config_segment(
            architecture, metadata, config, _FakeGroup(0, 2), vp_stage=None
        )


def test_pipeline_selection_rejects_config_and_group_size_mismatch(monkeypatch):
    config = _config(num_layers=2, pp_size=2)
    architecture = [_layer(MambaLayerConfig, config), _layer(AttentionLayerConfig, config)]
    metadata = scan_hybrid_layer_config_list(architecture)
    monkeypatch.setattr(hybrid_model_module, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(hybrid_model_module, "get_pg_size", lambda group: 1)

    with pytest.raises(
        ValueError, match="config.pipeline_model_parallel_size is 2.*process group has size 1"
    ):
        hybrid_model_module._select_pipeline_config_segment(
            architecture, metadata, config, _FakeGroup(0, 1), vp_stage=None
        )


def test_pipeline_selection_requires_vp_stage_when_vpp_is_enabled(monkeypatch):
    config = _config(num_layers=4, pp_size=2, vp_size=2)
    architecture = [_layer(MambaLayerConfig, config) for _ in range(4)]
    metadata = scan_hybrid_layer_config_list(architecture, pp_size=2)
    monkeypatch.setattr(hybrid_model_module, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(hybrid_model_module, "get_pg_size", lambda group: 2)

    with pytest.raises(ValueError, match="vp_stage must be provided"):
        hybrid_model_module._select_pipeline_config_segment(
            architecture, metadata, config, _FakeGroup(0, 2), vp_stage=None
        )


def test_pipeline_selection_rejects_decoder_count_mismatch(monkeypatch):
    config = _config(num_layers=2)
    architecture = [_layer(MambaLayerConfig, config)]
    metadata = scan_hybrid_layer_config_list(architecture)
    monkeypatch.setattr(hybrid_model_module, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(hybrid_model_module, "get_pg_size", lambda group: 1)

    with pytest.raises(ValueError, match="defines 1 decoder layers.*config.num_layers is 2"):
        hybrid_model_module._select_pipeline_config_segment(
            architecture, metadata, config, _FakeGroup(), vp_stage=None
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pipeline_model_parallel_layout", object()),
        ("num_layers_in_first_pipeline_stage", 1),
        ("num_layers_in_last_pipeline_stage", 1),
        ("account_for_embedding_in_pipeline_split", True),
        ("account_for_loss_in_pipeline_split", True),
    ],
)
def test_pipeline_selection_rejects_other_topology_controls(monkeypatch, field, value):
    config = _config(num_layers=1)
    setattr(config, field, value)
    architecture = [_layer(MambaLayerConfig, config)]
    metadata = scan_hybrid_layer_config_list(architecture)
    monkeypatch.setattr(hybrid_model_module, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(hybrid_model_module, "get_pg_size", lambda group: 1)

    with pytest.raises(ValueError, match=field):
        hybrid_model_module._select_pipeline_config_segment(
            architecture, metadata, config, _FakeGroup(), vp_stage=None
        )


def test_pipeline_selection_rejects_conflicting_inferred_vpp(monkeypatch):
    config = _config(num_layers=2)
    architecture = [
        _layer(MambaLayerConfig, config),
        PipelineSplit,
        _layer(AttentionLayerConfig, config),
    ]
    metadata = scan_hybrid_layer_config_list(architecture)
    monkeypatch.setattr(hybrid_model_module, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(hybrid_model_module, "get_pg_size", lambda group: 1)

    with pytest.raises(ValueError, match="infers virtual pipeline size 2"):
        hybrid_model_module._select_pipeline_config_segment(
            architecture, metadata, config, _FakeGroup(), vp_stage=0
        )


def test_pipeline_selection_rejects_out_of_range_vp_stage(monkeypatch):
    config = _config(num_layers=1)
    architecture = [_layer(MambaLayerConfig, config)]
    metadata = scan_hybrid_layer_config_list(architecture)
    monkeypatch.setattr(hybrid_model_module, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(hybrid_model_module, "get_pg_size", lambda group: 1)

    with pytest.raises(ValueError, match=r"vp_stage must be in \[0, 1\)"):
        hybrid_model_module._select_pipeline_config_segment(
            architecture, metadata, config, _FakeGroup(), vp_stage=1
        )


def test_list_model_never_calls_pattern_parser(patch_cpu_model_construction, monkeypatch):
    config = _config(num_layers=1)
    source_config = _layer(MambaLayerConfig, config)
    architecture = [source_config]
    for function_name in ("parse_hybrid_pattern", "select_pipeline_segment"):
        monkeypatch.setattr(
            f"megatron.core.models.hybrid.hybrid_layer_allocation.{function_name}",
            lambda *_args, **_kwargs: pytest.fail(
                "list construction called hybrid-layer-pattern allocation"
            ),
        )

    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=32,
        max_sequence_length=8,
        pre_process=False,
        post_process=False,
        pg_collection=_pg_collection(),
        layer_config_list=architecture,
    )
    architecture.append(MTPSplit)

    decoder_call = patch_cpu_model_construction[0].kwargs
    assert model.hybrid_layer_pattern is None
    assert model.layer_config_list == (source_config,)
    assert len(decoder_call["layer_config_list"]) == 1
    assert type(decoder_call["layer_config_list"][0]) is MambaLayerConfig
    assert decoder_call["layer_config_list"][0] is not source_config


def test_list_model_clones_repeated_decoder_aliases_and_preserves_overrides(
    patch_cpu_model_construction,
):
    config = _config(num_layers=2)
    source_config = _layer(AttentionLayerConfig, config)
    source_config.attention_dropout = 0.314
    source_config.custom_options = {"items": []}

    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=32,
        max_sequence_length=8,
        pre_process=False,
        post_process=False,
        pg_collection=_pg_collection(),
        layer_config_list=[source_config, source_config],
    )

    decoder_configs = patch_cpu_model_construction[0].kwargs["layer_config_list"]
    assert model.layer_config_list == (source_config, source_config)
    assert len(decoder_configs) == 2
    assert decoder_configs[0] is not source_config
    assert decoder_configs[1] is not source_config
    assert decoder_configs[0] is not decoder_configs[1]
    assert [layer_config.attention_dropout for layer_config in decoder_configs] == [0.314, 0.314]
    assert decoder_configs[0].custom_options == {"items": []}
    assert decoder_configs[1].custom_options == {"items": []}
    assert decoder_configs[0].custom_options is not decoder_configs[1].custom_options
    assert decoder_configs[0].custom_options is not source_config.custom_options

    decoder_configs[0].custom_options["items"].append("changed")
    assert decoder_configs[1].custom_options == {"items": []}
    assert source_config.custom_options == {"items": []}


@pytest.mark.parametrize(
    "conflicting_kwargs",
    [
        {"hybrid_layer_pattern": "M"},
        {"hybrid_override_pattern": "M"},
        {"hybrid_attention_ratio": 0.5},
        {"hybrid_mlp_ratio": 0.5},
        {"hybrid_attention_ratio": 0.0},
        {"hybrid_mlp_ratio": 0.0},
    ],
    ids=[
        "pattern",
        "deprecated-pattern",
        "positive-attention-ratio",
        "positive-mlp-ratio",
        "zero-attention-ratio",
        "zero-mlp-ratio",
    ],
)
def test_list_model_rejects_other_architecture_sources(
    patch_cpu_model_construction, conflicting_kwargs
):
    config = _config(num_layers=1)

    with pytest.raises(ValueError, match="layer_config_list cannot be combined"):
        HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=32,
            max_sequence_length=8,
            pre_process=False,
            post_process=False,
            pg_collection=_pg_collection(),
            layer_config_list=[_layer(MambaLayerConfig, config)],
            **conflicting_kwargs,
        )


@pytest.mark.parametrize("configured_depths", [None, 1, 3])
def test_list_model_rejects_mtp_depth_count_mismatch(
    patch_cpu_model_construction, configured_depths
):
    config = _config(num_layers=1, mtp_num_layers=configured_depths)
    decoder = _layer(MambaLayerConfig, config)
    head = _layer(AttentionLayerConfig, config)

    with pytest.raises(ValueError, match="defines 2 MTP depths.*config.mtp_num_layers"):
        HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=32,
            max_sequence_length=8,
            pre_process=False,
            post_process=False,
            pg_collection=_pg_collection(),
            layer_config_list=[decoder, MTPSplit, head, MTPSplit, head],
        )


@pytest.mark.parametrize(
    ("pp_rank", "vp_stage", "expected_mtp_process"),
    [(0, 0, False), (1, 0, False), (0, 1, False), (1, 1, True)],
)
def test_list_model_places_mtp_only_on_final_pp_and_vpp_stage(
    patch_cpu_model_construction, monkeypatch, pp_rank, vp_stage, expected_mtp_process
):
    config = _config(num_layers=4, pp_size=2, vp_size=2, mtp_num_layers=1)
    decoder_configs = [_layer(MambaLayerConfig, config) for _ in range(4)]
    for index, decoder_config in enumerate(decoder_configs):
        decoder_config.architecture_index = index
    mtp_config = _layer(AttentionLayerConfig, config)
    architecture = [
        decoder_configs[0],
        PipelineSplit,
        decoder_configs[1],
        PipelineSplit,
        decoder_configs[2],
        PipelineSplit,
        decoder_configs[3],
        MTPSplit,
        mtp_config,
    ]
    placement_calls = []
    mtp_build_calls = []

    def fake_mtp_on_this_rank(**kwargs):
        placement_calls.append(kwargs)
        pp_group = kwargs["pp_group"]
        return (
            pp_group.rank() == pp_group.size() - 1 and kwargs["vp_stage"] == kwargs["vp_size"] - 1
        )

    class DummyEmbedding(nn.Module):

        def __init__(self, **kwargs):
            super().__init__()

    class DummyOutput(nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__()

    class DummyMTP(nn.Module):

        def __init__(self, **kwargs):
            super().__init__()
            mtp_build_calls.append(kwargs)

    monkeypatch.setattr(hybrid_model_module, "mtp_on_this_rank", fake_mtp_on_this_rank)
    monkeypatch.setattr(hybrid_model_module, "LanguageModelEmbedding", DummyEmbedding)
    monkeypatch.setattr(hybrid_model_module, "MultiTokenPredictionBlock", DummyMTP)
    monkeypatch.setattr(hybrid_model_module.tensor_parallel, "ColumnParallelLinear", DummyOutput)
    monkeypatch.setattr(LanguageModule, "setup_embeddings_and_output_layer", lambda self: None)

    pg_collection = _pg_collection(pp_rank=pp_rank, pp_size=2)
    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=32,
        max_sequence_length=8,
        pre_process=False,
        post_process=False,
        pg_collection=pg_collection,
        vp_stage=vp_stage,
        layer_config_list=architecture,
    )

    segment_index = vp_stage * 2 + pp_rank
    decoder_call = patch_cpu_model_construction[0].kwargs
    assert decoder_call["pp_layer_offset"] == segment_index
    assert [
        layer_config.architecture_index for layer_config in decoder_call["layer_config_list"]
    ] == [segment_index]
    assert len(placement_calls) == 1
    assert placement_calls[0]["pp_group"] is pg_collection.pp
    assert placement_calls[0]["vp_stage"] == vp_stage
    assert placement_calls[0]["vp_size"] == 2
    assert model.mtp_process is expected_mtp_process
    assert hasattr(model, "mtp") is expected_mtp_process
    assert hasattr(model, "output_layer") is expected_mtp_process
    assert len(mtp_build_calls) == int(expected_mtp_process)


def test_list_model_passes_source_mtp_template_without_a_pattern(
    patch_cpu_model_construction, monkeypatch
):
    config = _config(num_layers=1, mtp_num_layers=2)
    decoder = _layer(MambaLayerConfig, config)
    attention = _layer(AttentionLayerConfig, config)
    moe = _layer(MoELayerConfig, config)
    captured_mtp = {}

    class DummyEmbedding(nn.Module):

        def __init__(self, **kwargs):
            super().__init__()

    class DummyOutput(nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__()

    class DummyMTP(nn.Module):

        def __init__(self, **kwargs):
            super().__init__()
            captured_mtp.update(kwargs)

    monkeypatch.setattr(hybrid_model_module, "mtp_on_this_rank", lambda **kwargs: True)
    monkeypatch.setattr(hybrid_model_module, "LanguageModelEmbedding", DummyEmbedding)
    monkeypatch.setattr(hybrid_model_module, "MultiTokenPredictionBlock", DummyMTP)
    monkeypatch.setattr(hybrid_model_module.tensor_parallel, "ColumnParallelLinear", DummyOutput)
    monkeypatch.setattr(LanguageModule, "setup_embeddings_and_output_layer", lambda self: None)

    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=32,
        max_sequence_length=8,
        pre_process=False,
        post_process=False,
        pg_collection=_pg_collection(),
        layer_config_list=[decoder, MTPSplit, attention, moe, MTPSplit, attention, moe],
    )

    assert captured_mtp["mtp_layer_pattern"] is None
    assert captured_mtp["mtp_num_depths"] == 2
    assert captured_mtp["is_hybrid_mtp"] is True
    assert [type(layer_config) for layer_config in captured_mtp["mtp_layer_config_list"]] == [
        AttentionLayerConfig,
        MoELayerConfig,
    ]
    assert all(
        forwarded is source
        for forwarded, source in zip(
            captured_mtp["mtp_layer_config_list"], (attention, moe), strict=True
        )
    )
    assert model.layer_config_list[2] is model.layer_config_list[5]
    assert model.layer_config_list[3] is model.layer_config_list[6]


@pytest.mark.parametrize(
    ("repeated_layer", "expected_layer_numbers"),
    [(False, [1, 2]), (True, [1])],
    ids=["independent", "repeated"],
)
def test_list_defined_mtp_builds_independent_or_repeated_physical_layers(
    monkeypatch, repeated_layer, expected_layer_numbers
):
    config = _config(num_layers=1, mtp_num_layers=2, mtp_use_repeated_layer=repeated_layer)
    attention = _layer(AttentionLayerConfig, config)
    moe = _layer(MoELayerConfig, config)
    source_template = [attention, moe]
    build_calls = []

    def fake_build_module(layer_spec, **kwargs):
        build_calls.append((layer_spec, kwargs))
        return _CapturedModule(layer_spec=layer_spec, **kwargs)

    monkeypatch.setattr(
        mtp_module,
        "_get_mtp_block_submodules",
        lambda config, spec: SimpleNamespace(layer_specs=[object(), object()]),
    )
    monkeypatch.setattr(mtp_module, "get_fp8_context", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(mtp_module, "build_module", fake_build_module)
    monkeypatch.setattr(mtp_module, "is_vp_last_stage", lambda **kwargs: True)

    block = MultiTokenPredictionBlock(
        config=config,
        spec=object(),
        pg_collection=SimpleNamespace(tp=_FakeGroup(), cp=_FakeGroup(), pp=_FakeGroup()),
        mtp_num_depths=2,
        mtp_layer_config_list=source_template,
        hybrid_submodules=object(),
        is_hybrid_mtp=True,
    )

    assert len(block.layers) == len(expected_layer_numbers)
    assert [kwargs["layer_number"] for _, kwargs in build_calls] == expected_layer_numbers
    assert all(kwargs["is_hybrid_mtp"] is True for _, kwargs in build_calls)
    assert all(kwargs["mtp_layer_pattern"] is None for _, kwargs in build_calls)
    assert all(kwargs["mtp_layer_config_list"] is source_template for _, kwargs in build_calls)


def test_each_physical_mtp_layer_clones_the_source_template(monkeypatch):
    config = _config(num_layers=1, mtp_num_layers=2)
    source_config = _layer(AttentionLayerConfig, config)
    source_config.test_mutable_value = {"items": []}
    captured_config_lists = []

    class CapturingHybridStack(nn.Module):

        def __init__(self, *, layer_config_list, **kwargs):
            super().__init__()
            captured_config_lists.append(layer_config_list)
            self.layers = nn.ModuleList([nn.Identity()])

    submodules = SimpleNamespace(
        mtp_model_layer=None,
        enorm=lambda **kwargs: nn.Identity(),
        hnorm=lambda **kwargs: nn.Identity(),
        eh_proj=object(),
        e_proj=None,
        h_proj=None,
        layer_norm=lambda **kwargs: nn.Identity(),
    )
    pg_collection = SimpleNamespace(tp=_FakeGroup(), cp=_FakeGroup(), pp=_FakeGroup())
    monkeypatch.setattr(mtp_module, "build_module", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(
        "megatron.core.models.hybrid.hybrid_block.HybridStack", CapturingHybridStack
    )

    for layer_number in (1, 2):
        MultiTokenPredictionLayer(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            pg_collection=pg_collection,
            mtp_layer_config_list=[source_config],
            hybrid_submodules=object(),
            is_hybrid_mtp=True,
        )

    first_config = captured_config_lists[0][0]
    second_config = captured_config_lists[1][0]
    assert first_config is not source_config
    assert second_config is not source_config
    assert first_config is not second_config
    assert first_config.test_mutable_value == {"items": []}
    assert second_config.test_mutable_value == {"items": []}
    assert first_config.test_mutable_value is not second_config.test_mutable_value

    first_config.test_mutable_value["items"].append("changed")
    assert second_config.test_mutable_value == {"items": []}
    assert source_config.test_mutable_value == {"items": []}
