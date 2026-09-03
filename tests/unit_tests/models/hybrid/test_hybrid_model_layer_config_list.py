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
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.ssm.mlp_layer_config import MLPLayerConfig
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    MultiTokenPredictionLayer,
)
from megatron.core.utils import init_method_normal


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


def _config(
    *,
    num_layers: int,
    pp_size: int = 1,
    vp_size: int | None = None,
    mtp_num_layers: int | None = None,
    mtp_use_repeated_layer: bool = False,
    first_stage_layers: int | None = None,
    last_stage_layers: int | None = None,
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
        num_layers_in_first_pipeline_stage=first_stage_layers,
        num_layers_in_last_pipeline_stage=last_stage_layers,
        output_layer_init_method=init_method_normal(0.02),
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
    monkeypatch.setattr(hybrid_model_module, "get_pg_size", lambda group: group.size())
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_rank", lambda group: group.rank())
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_size", lambda group: group.size())
    monkeypatch.setattr(
        hybrid_allocation_module, "log_on_each_pipeline_stage", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(hybrid_model_module, "build_module", fake_build_module)
    return build_calls


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

    monkeypatch.setattr(hybrid_allocation_module, "get_pg_rank", lambda group: pp_rank)
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_size", lambda group: 2)
    monkeypatch.setattr(
        hybrid_allocation_module, "log_on_each_pipeline_stage", lambda *args, **kwargs: None
    )

    selected, offset = hybrid_allocation_module.select_pipeline_config_segment(
        architecture, config, _FakeGroup(pp_rank, 2), vp_stage
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
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_size", lambda group: 2)
    monkeypatch.setattr(
        hybrid_allocation_module, "log_on_each_pipeline_stage", lambda *args, **kwargs: None
    )

    selected, offset = hybrid_allocation_module.select_pipeline_config_segment(
        architecture, config, _FakeGroup(0, 2), vp_stage=1
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
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_size", lambda group: 2)

    with pytest.raises(ValueError, match="should be divisible"):
        hybrid_allocation_module.select_pipeline_config_segment(
            architecture, config, _FakeGroup(0, 2), vp_stage=None
        )


def test_pipeline_selection_rejects_config_and_group_size_mismatch(monkeypatch):
    config = _config(num_layers=2, pp_size=2)
    architecture = [_layer(MambaLayerConfig, config), _layer(AttentionLayerConfig, config)]
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_size", lambda group: 1)

    with pytest.raises(ValueError, match="process group has size 1"):
        hybrid_allocation_module.select_pipeline_config_segment(
            architecture, config, _FakeGroup(), vp_stage=None
        )


def test_pipeline_selection_rejects_decoder_count_mismatch(monkeypatch):
    config = _config(num_layers=2)
    architecture = [_layer(MambaLayerConfig, config)]
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_size", lambda group: 1)

    with pytest.raises(ValueError, match="defines 1 decoder layers.*config.num_layers is 2"):
        hybrid_allocation_module.select_pipeline_config_segment(
            architecture, config, _FakeGroup(), vp_stage=None
        )


@pytest.mark.parametrize(
    ("vp_stage", "error_match"),
    [(None, "vp_stage must be provided"), (2, r"vp_stage must be in \[0, 2\)")],
)
def test_pipeline_selection_validates_vp_stage(monkeypatch, vp_stage, error_match):
    config = _config(num_layers=4, pp_size=2, vp_size=2)
    architecture = [_layer(MambaLayerConfig, config) for _ in range(4)]
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_size", lambda group: 2)

    with pytest.raises(ValueError, match=error_match):
        hybrid_allocation_module.select_pipeline_config_segment(
            architecture, config, _FakeGroup(0, 2), vp_stage
        )


@pytest.mark.parametrize(
    ("pp_size", "first_stage_layers", "last_stage_layers"), [(1, 2, 2), (2, 10, None)]
)
def test_pipeline_selection_validates_uneven_stage_counts(
    monkeypatch, pp_size, first_stage_layers, last_stage_layers
):
    config = _config(
        num_layers=6,
        pp_size=pp_size,
        first_stage_layers=first_stage_layers,
        last_stage_layers=last_stage_layers,
    )
    architecture = [_layer(MambaLayerConfig, config) for _ in range(6)]
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_size", lambda group: pp_size)

    with pytest.raises(ValueError, match="overrides are incompatible"):
        hybrid_allocation_module.select_pipeline_config_segment(
            architecture, config, _FakeGroup(0, pp_size), vp_stage=None
        )


@pytest.mark.parametrize("vp_size", [0, -1])
def test_pipeline_selection_requires_positive_vpp(monkeypatch, vp_size):
    config = _config(num_layers=1)
    config.virtual_pipeline_model_parallel_size = vp_size
    architecture = [_layer(MambaLayerConfig, config)]
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_size", lambda group: 1)

    with pytest.raises(ValueError, match="virtual_pipeline_model_parallel_size must be positive"):
        hybrid_allocation_module.select_pipeline_config_segment(
            architecture, config, _FakeGroup(), vp_stage=None
        )


def test_pipeline_selection_requires_all_decoder_layers_to_be_owned(monkeypatch):
    config = _config(num_layers=4, pp_size=2)
    architecture = [_layer(MambaLayerConfig, config) for _ in range(4)]
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_size", lambda group: 2)
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_block.get_num_layers_to_build",
        lambda *args, **kwargs: 1,
    )

    with pytest.raises(ValueError, match="owns 2 decoder layers.*defines 4"):
        hybrid_allocation_module.select_pipeline_config_segment(
            architecture, config, _FakeGroup(0, 2), vp_stage=None
        )


@pytest.mark.parametrize(
    ("pp_rank", "expected_indices", "expected_offset"),
    [(0, [0], 0), (1, [1, 2], 1), (2, [3, 4], 3), (3, [5, 6], 5)],
)
def test_converted_pattern_preserves_uneven_first_and_last_stages(
    monkeypatch, pp_rank, expected_indices, expected_offset
):
    config = _config(num_layers=7, pp_size=4, first_stage_layers=1, last_stage_layers=2)
    architecture = hybrid_allocation_module.layer_config_list_from_hybrid_layer_pattern(
        "MMMMMMM", config
    )
    for index, layer_config in enumerate(architecture):
        layer_config.architecture_index = index
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_rank", lambda group: pp_rank)
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_size", lambda group: 4)
    monkeypatch.setattr(
        hybrid_allocation_module, "log_on_each_pipeline_stage", lambda *args, **kwargs: None
    )

    selected, offset = hybrid_allocation_module.select_pipeline_config_segment(
        architecture, config, _FakeGroup(pp_rank, 4), vp_stage=None
    )

    assert offset == expected_offset
    assert [layer_config.architecture_index for layer_config in selected] == expected_indices


@pytest.mark.parametrize(
    ("pp_rank", "vp_stage", "expected_indices", "expected_offset"),
    [
        (0, 0, [0], 0),
        (1, 0, [1, 2, 3], 1),
        (2, 0, [4, 5, 6], 4),
        (3, 0, [7], 7),
        (0, 1, [8], 8),
        (1, 1, [9, 10, 11], 9),
        (2, 1, [12, 13, 14], 12),
        (3, 1, [15], 15),
    ],
)
def test_marker_free_selection_uses_canonical_uneven_vpp_allocation(
    monkeypatch, pp_rank, vp_stage, expected_indices, expected_offset
):
    config = _config(num_layers=16, pp_size=4, vp_size=2, first_stage_layers=2, last_stage_layers=2)
    architecture = [_layer(MambaLayerConfig, config) for _ in range(16)]
    for index, layer_config in enumerate(architecture):
        layer_config.architecture_index = index

    monkeypatch.setattr(hybrid_allocation_module, "get_pg_rank", lambda group: pp_rank)
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_size", lambda group: 4)
    monkeypatch.setattr(
        hybrid_allocation_module, "log_on_each_pipeline_stage", lambda *args, **kwargs: None
    )

    selected, offset = hybrid_allocation_module.select_pipeline_config_segment(
        architecture, config, _FakeGroup(pp_rank, 4), vp_stage
    )

    assert offset == expected_offset
    assert [layer_config.architecture_index for layer_config in selected] == expected_indices


@pytest.mark.parametrize(
    ("pp_rank", "vp_stage", "expected_offset"), [(0, 0, 0), (1, 0, 1), (0, 1, 2), (1, 1, 3)]
)
def test_marker_free_uneven_vpp_without_middle_stages(
    monkeypatch, pp_rank, vp_stage, expected_offset
):
    config = _config(num_layers=4, pp_size=2, vp_size=2, first_stage_layers=2, last_stage_layers=2)
    architecture = [_layer(MambaLayerConfig, config) for _ in range(4)]
    for index, layer_config in enumerate(architecture):
        layer_config.architecture_index = index

    monkeypatch.setattr(hybrid_allocation_module, "get_pg_rank", lambda group: pp_rank)
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_size", lambda group: 2)
    monkeypatch.setattr(
        hybrid_allocation_module, "log_on_each_pipeline_stage", lambda *args, **kwargs: None
    )

    selected, offset = hybrid_allocation_module.select_pipeline_config_segment(
        architecture, config, _FakeGroup(pp_rank, 2), vp_stage
    )

    assert offset == expected_offset
    assert [layer_config.architecture_index for layer_config in selected] == [expected_offset]


def test_pipeline_config_selector_rejects_mtp_markers(monkeypatch):
    config = _config(num_layers=1, mtp_num_layers=1)
    architecture = [
        _layer(MambaLayerConfig, config),
        MTPSplit,
        _layer(AttentionLayerConfig, config),
    ]
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(hybrid_allocation_module, "get_pg_size", lambda group: 1)

    with pytest.raises(ValueError, match="must not contain MTPSplit"):
        hybrid_allocation_module.select_pipeline_config_segment(
            architecture, config, _FakeGroup(), vp_stage=None
        )


def test_pattern_adapter_emits_split_markers_and_reuses_the_mtp_template():
    config = _config(num_layers=2, mtp_num_layers=2)

    architecture = hybrid_allocation_module.layer_config_list_from_hybrid_layer_pattern(
        "M|*/-E/-E", config
    )

    assert [
        type(entry) if isinstance(entry, TransformerConfig) else entry for entry in architecture
    ] == [
        MambaLayerConfig,
        PipelineSplit,
        AttentionLayerConfig,
        MTPSplit,
        MLPLayerConfig,
        MoELayerConfig,
        MTPSplit,
        MLPLayerConfig,
        MoELayerConfig,
    ]
    assert architecture[4] is architecture[7]
    assert architecture[5] is architecture[8]


def test_pattern_is_converted_once_to_the_authoritative_config_tuple(
    patch_cpu_model_construction, monkeypatch
):
    config = _config(num_layers=2)
    adapter_calls = []
    scanner_inputs = []
    pattern_adapter = hybrid_allocation_module.layer_config_list_from_hybrid_layer_pattern
    scanner = hybrid_model_module.scan_hybrid_layer_config_list

    def record_adapter(pattern, root_config):
        adapter_calls.append((pattern, root_config))
        return pattern_adapter(pattern, root_config)

    def record_scanner(layer_config_list, **kwargs):
        scanner_inputs.append(layer_config_list)
        return scanner(layer_config_list, **kwargs)

    monkeypatch.setattr(
        hybrid_allocation_module, "layer_config_list_from_hybrid_layer_pattern", record_adapter
    )
    monkeypatch.setattr(hybrid_model_module, "scan_hybrid_layer_config_list", record_scanner)

    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=32,
        max_sequence_length=8,
        hybrid_layer_pattern="M*",
        pre_process=False,
        post_process=False,
        pg_collection=_pg_collection(),
    )

    decoder_call = patch_cpu_model_construction[0].kwargs
    assert adapter_calls == [("M*", config)]
    assert isinstance(model.layer_config_list, tuple)
    assert [type(entry) for entry in model.layer_config_list] == [
        MambaLayerConfig,
        AttentionLayerConfig,
    ]
    assert len(scanner_inputs) == 1
    assert scanner_inputs[0] is model.layer_config_list
    assert not hasattr(model, "hybrid_layer_pattern")
    assert not hasattr(model, "mtp_pattern")
    assert [type(entry) for entry in decoder_call["layer_config_list"]] == [
        MambaLayerConfig,
        AttentionLayerConfig,
    ]
    assert all(
        physical is not source
        for physical, source in zip(
            decoder_call["layer_config_list"], model.layer_config_list, strict=True
        )
    )


def test_list_is_snapshotted_and_repeated_entries_get_independent_physical_clones(
    patch_cpu_model_construction, monkeypatch
):
    config = _config(num_layers=2)
    source_config = _layer(AttentionLayerConfig, config)
    source_config.custom_options = {"items": []}
    architecture = [source_config, source_config]
    monkeypatch.setattr(
        hybrid_allocation_module,
        "layer_config_list_from_hybrid_layer_pattern",
        lambda *_args, **_kwargs: pytest.fail(
            "explicit config lists must skip the pattern adapter"
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

    decoder_configs = patch_cpu_model_construction[0].kwargs["layer_config_list"]
    assert model.layer_config_list == (source_config, source_config)
    assert all(layer_config is not source_config for layer_config in decoder_configs)
    assert decoder_configs[0] is not decoder_configs[1]
    assert decoder_configs[0].custom_options is not decoder_configs[1].custom_options
    assert decoder_configs[0].custom_options is not source_config.custom_options

    decoder_configs[0].custom_options["items"].append("changed")
    assert decoder_configs[1].custom_options == {"items": []}
    assert source_config.custom_options == {"items": []}


def test_pattern_and_config_list_are_mutually_exclusive(patch_cpu_model_construction):
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
            hybrid_layer_pattern="M",
        )


def test_model_requires_pattern_or_config_list():
    config = _config(num_layers=1)

    with pytest.raises(
        ValueError, match="Either hybrid_layer_pattern or layer_config_list must be provided"
    ):
        HybridModel(
            config=config, hybrid_stack_spec=hybrid_stack_spec, vocab_size=32, max_sequence_length=8
        )


def test_deprecated_override_pattern_and_ratios_are_mutually_exclusive():
    config = _config(num_layers=1)

    with pytest.raises(
        ValueError, match="hybrid_layer_pattern cannot be used together with hybrid_attention_ratio"
    ):
        HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=32,
            max_sequence_length=8,
            hybrid_override_pattern="M",
            hybrid_attention_ratio=0.5,
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

    assert "mtp_num_depths" not in captured_mtp
    assert "mtp_layer_pattern" not in captured_mtp
    assert "is_hybrid_mtp" not in captured_mtp
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


def test_leading_mtp_split_builds_the_zero_decoder_case(patch_cpu_model_construction, monkeypatch):
    config = _config(num_layers=0, mtp_num_layers=1)
    mtp_config = _layer(AttentionLayerConfig, config)
    monkeypatch.setattr(hybrid_model_module, "mtp_on_this_rank", lambda **kwargs: True)
    monkeypatch.setattr(hybrid_model_module, "LanguageModelEmbedding", _CapturedModule)
    monkeypatch.setattr(hybrid_model_module, "MultiTokenPredictionBlock", _CapturedModule)
    monkeypatch.setattr(
        hybrid_model_module.tensor_parallel,
        "ColumnParallelLinear",
        lambda *args, **kwargs: _CapturedModule(args=args, **kwargs),
    )
    monkeypatch.setattr(LanguageModule, "setup_embeddings_and_output_layer", lambda self: None)

    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=32,
        max_sequence_length=8,
        pre_process=False,
        post_process=False,
        pg_collection=_pg_collection(),
        layer_config_list=[MTPSplit, mtp_config],
    )

    assert patch_cpu_model_construction[0].kwargs["layer_config_list"] == []
    assert model.mtp.kwargs["mtp_layer_config_list"] == (mtp_config,)
    assert "mtp_num_depths" not in model.mtp.kwargs


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
        mtp_layer_config_list=source_template,
        hybrid_submodules=object(),
    )

    assert len(block.layers) == len(expected_layer_numbers)
    assert [kwargs["layer_number"] for _, kwargs in build_calls] == expected_layer_numbers
    assert all("is_hybrid_mtp" not in kwargs for _, kwargs in build_calls)
    assert all("mtp_layer_pattern" not in kwargs for _, kwargs in build_calls)
    assert all(kwargs["mtp_layer_config_list"] is source_template for _, kwargs in build_calls)


def test_gpt_mtp_build_does_not_receive_hybrid_arguments(monkeypatch):
    config = _config(num_layers=1, mtp_num_layers=2)
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
    )

    assert block.is_hybrid_mtp is False
    assert len(block.layers) == 2
    assert all("hybrid_submodules" not in kwargs for _, kwargs in build_calls)
    assert all("mtp_layer_config_list" not in kwargs for _, kwargs in build_calls)


@pytest.mark.parametrize("constructor", ["block", "layer"])
@pytest.mark.parametrize("provided", ["config-list", "submodules"])
def test_hybrid_mtp_requires_config_list_and_submodules_together(constructor, provided):
    config = _config(num_layers=1, mtp_num_layers=1)
    kwargs = (
        {"mtp_layer_config_list": [_layer(AttentionLayerConfig, config)]}
        if provided == "config-list"
        else {"hybrid_submodules": object()}
    )

    with pytest.raises(
        ValueError, match="mtp_layer_config_list and hybrid_submodules must be provided together"
    ):
        if constructor == "block":
            MultiTokenPredictionBlock(config=config, spec=object(), **kwargs)
        else:
            MultiTokenPredictionLayer(config=config, submodules=object(), **kwargs)


@pytest.mark.parametrize("mtp_num_layers", [None, 0, -1, 1.5])
def test_hybrid_mtp_block_requires_positive_depth_count(mtp_num_layers):
    config = _config(num_layers=1)
    config.mtp_num_layers = mtp_num_layers

    with pytest.raises(ValueError, match="mtp_num_layers to be a positive integer"):
        MultiTokenPredictionBlock(
            config=config,
            spec=object(),
            mtp_layer_config_list=[_layer(AttentionLayerConfig, config)],
            hybrid_submodules=object(),
        )


def test_mtp_num_depths_compatibility_argument_must_match_config():
    config = _config(num_layers=1, mtp_num_layers=2)

    with pytest.raises(ValueError, match="mtp_num_depths=1 conflicts"):
        MultiTokenPredictionBlock(config=config, spec=object(), mtp_num_depths=1)


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
