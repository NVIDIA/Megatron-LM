# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Model module declarations add ownership and matching names to ordinary refit."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from megatron.core.resharding.planner import _extract_module_metadata, _find_source_metadata
from megatron.core.resharding.refit import (
    _get_parallel_config,
    _harmonize_buffer_dtypes,
    _unwrap_model_cores,
)
from megatron.core.resharding.utils import ReshardPlan, TransferOp, get_refit_tensor_dict
from megatron.core.transformer.module import Float16Module, MegatronModule


def groups(ranks=(0,)):
    return SimpleNamespace(tp=ranks, pp=ranks, dp=ranks, ep=ranks, expt_tp=ranks)


def linear(ranks=(0,)):
    module = torch.nn.Linear(2, 2, bias=False)
    module.pg_collection = groups(ranks)
    return module


class Composite(MegatronModule):
    def __init__(self) -> None:
        super().__init__(SimpleNamespace(num_moe_experts=None))
        self.pg_collection = groups((9,))
        self.first, self.second = linear((0, 1)), linear((2,))

    def refit_modules(self):
        return [
            ("left", self.first, self.first.pg_collection),
            ("right", self.second, self.second.pg_collection),
        ]


@pytest.fixture(autouse=True)
def group_ranks(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: list(group))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)


def metadata(module, offset=0):
    return _extract_module_metadata(module, 0, None, offset, {})


def test_ordinary_models_keep_native_names_and_root_ownership():
    ordinary = MegatronModule(SimpleNamespace())
    ordinary.pg_collection = groups()
    ordinary.child = linear()
    entry = metadata(ordinary)[0]
    assert entry.name == entry.resolved_name == "child.weight"
    with pytest.raises(ValueError, match="Module must have pg_collection"):
        metadata(torch.nn.Module())


@pytest.mark.parametrize("explicit_none", [False, True])
def test_ordinary_models_require_root_groups(explicit_none):
    model = MegatronModule(SimpleNamespace(num_moe_experts=None))
    if explicit_none:
        model.pg_collection = None
    with pytest.raises(RuntimeError, match="Source model missing pg_collection"):
        _unwrap_model_cores(model, None)
    with pytest.raises(RuntimeError, match="Target model missing pg_collection"):
        _unwrap_model_cores(None, model)


def test_provider_supplies_matching_names_groups_and_expert_count():
    model = Composite()
    model.first.config = SimpleNamespace(num_moe_experts=8)
    entries = {entry.name: entry for entry in metadata(model, offset=3)}
    assert entries["first.weight"].resolved_name == "left.weight"
    assert entries["first.weight"].tensor_parallel_group_ranks == [3, 4]
    assert entries["second.weight"].tensor_parallel_group_ranks == [5]
    assert entries["first.weight"].num_experts == 8
    assert entries["second.weight"].num_experts is None


@pytest.mark.parametrize("root_groups", ["missing", "none", "partial"])
def test_provider_owns_groups_independently_of_root(monkeypatch, root_groups):
    model = Composite()
    if root_groups == "missing":
        del model.pg_collection
    elif root_groups == "none":
        model.pg_collection = None
    else:
        model.pg_collection.dp = None
    monkeypatch.setattr(
        "megatron.core.resharding.refit.parallel_state.get_data_parallel_group",
        lambda **kwargs: pytest.fail("Module declarations must not read global DP groups"),
    )
    assert _unwrap_model_cores(model, None) == (model, None, None)
    assert _unwrap_model_cores(None, model) == (None, model, None)
    if root_groups == "partial":
        assert model.pg_collection.dp is None
    assert len(metadata(model)) == 2


def test_wrappers_and_pipeline_indices_only_change_matching_names():
    model = Composite()
    layer = linear()
    layer.layer_number = 7
    wrapper = Float16Module.__new__(Float16Module)
    torch.nn.Module.__init__(wrapper)
    wrapper.module = torch.nn.ModuleDict({"layers": torch.nn.ModuleList([layer])})
    wrapper.pg_collection = groups()
    model.first = wrapper
    keys = tuple(model.state_dict())
    entry = metadata(model)[0]
    assert entry.name == "first.module.layers.0.weight"
    assert entry.resolved_name == "left.layers.6.weight"
    assert dict(model.named_parameters())[entry.name] is layer.weight
    assert tuple(model.state_dict()) == keys


def test_persistent_state_coverage_and_nonpersistent_scratch():
    model = Composite()
    model.first.register_buffer("state", torch.ones(1))
    model.register_buffer("scratch", torch.zeros(1), persistent=False)
    assert {entry.name for entry in metadata(model)} == {
        "first.weight",
        "first.state",
        "second.weight",
    }
    model.register_buffer("uncovered", torch.zeros(1))
    with pytest.raises(ValueError, match="cover every parameter and persistent buffer once"):
        metadata(model)
    empty = torch.nn.Module()
    empty.refit_modules = lambda: []
    assert metadata(empty) == []


@pytest.mark.parametrize(
    "fault,match",
    [
        ("missing", "cover every parameter"),
        ("unregistered", "original model"),
        ("duplicate", "cover every parameter"),
        ("collision", "duplicate matching"),
        ("shared", "cover every parameter"),
    ],
)
def test_invalid_module_declarations_fail(fault, match):
    model = Composite()
    components = model.refit_modules()
    if fault == "missing":
        components.pop()
    elif fault == "unregistered":
        components[0] = ("left", linear(), groups())
    elif fault == "duplicate":
        components.append(components[0])
    elif fault == "collision":
        components[1] = ("left", model.second, model.second.pg_collection)
    else:
        model.second.weight = model.first.weight
    model.refit_modules = lambda: components
    with pytest.raises(ValueError, match=match):
        metadata(model)


@pytest.mark.parametrize("root", [False, True])
def test_empty_matching_prefix_preserves_native_names(root):
    model = MegatronModule(SimpleNamespace(num_moe_experts=None))
    model.pg_collection = groups()
    model.llava_model = torch.nn.Module()
    model.llava_model.language_model = linear()
    declared = model if root else model.llava_model
    model.refit_modules = lambda: [("", declared, model.pg_collection)]
    keys = tuple(model.state_dict())
    entry = metadata(model)[0]
    assert entry.name == "llava_model.language_model.weight"
    assert entry.resolved_name == (
        "llava_model.language_model.weight" if root else "language_model.weight"
    )
    assert tuple(model.state_dict()) == keys


def mimo_model():
    from megatron.core.models.mimo import MimoModel
    from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules

    model = MimoModel.__new__(MimoModel)
    torch.nn.Module.__init__(model)
    model.language_model = linear()
    tower = VisionModalitySubmodules(
        encoders={"clip": linear((1,))},
        input_projections=[linear((2,))],
        pg_collection=groups((3,)),
    )
    model.modality_submodules = torch.nn.ModuleDict({"images": tower})
    return model, tower


def test_mimo_provider_uses_actual_component_groups_and_projector_fallback():
    model, tower = mimo_model()
    entries = {entry.resolved_name: entry for entry in metadata(model)}
    assert set(entries) == {
        "language_model.weight",
        "vision_model.weight",
        "vision_projection.weight",
    }
    assert entries["vision_model.weight"].tensor_parallel_group_ranks == [1]
    assert entries["vision_projection.weight"].tensor_parallel_group_ranks == [2]
    tower.input_projections[0].pg_collection = None
    assert metadata(model)[2].tensor_parallel_group_ranks == [3]
    del tower.input_projections[0].pg_collection
    assert metadata(model)[2].tensor_parallel_group_ranks == [3]


@pytest.mark.parametrize(
    "fault,match",
    [
        ("layout", "one image encoder"),
        ("groups", "requires explicit"),
        ("tp", "TP group disagrees"),
        ("fp8", "Quantized component"),
        ("fp4", "Quantized component"),
        ("quant_recipe", "Quantized component"),
        ("use_kitchen", "Quantized component"),
    ],
)
def test_mimo_rejects_unsupported_metadata(fault, match):
    model, tower = mimo_model()
    if fault == "layout":
        tower.encoders["extra"] = linear()
    elif fault == "groups":
        model.language_model.pg_collection.dp = None
    elif fault == "tp":
        tower.input_projections[0].pg_collection = None
        tower.input_projections[0].tp_group = (9,)
    else:
        model.language_model.config = SimpleNamespace(**{fault: True})
    with pytest.raises(ValueError, match=match):
        metadata(model)


def test_provider_cache_identity_does_not_execute_hook_or_retain_models():
    first, second = Composite(), Composite()
    first.refit_modules = lambda: pytest.fail("Cache lookup must not enumerate modules")
    key = _get_parallel_config(first)
    assert key == _get_parallel_config(first)
    assert key != _get_parallel_config(second)
    cached_plan = {key: object()}
    reference = weakref.ref(first)
    del first
    gc.collect()
    assert reference() is None
    assert key in cached_plan


def test_tied_embedding_alias_stays_inside_its_component():
    embedding = object()
    roster = {"language.embedding.word_embeddings.weight": [embedding]}
    assert _find_source_metadata(roster, "language.output_layer.weight") == [embedding]
    assert _find_source_metadata(roster, "other.output_layer.weight") is None


@pytest.mark.parametrize("conflicting_shard", [False, True])
def test_buffer_dtypes_match_transfer_ids_across_different_storage_names(
    monkeypatch, conflicting_shard
):
    source, target = torch.nn.ModuleDict({"source_tower": torch.nn.Module()}), torch.nn.ModuleDict(
        {"inference": torch.nn.Module()}
    )
    source["source_tower"].register_buffer("state", torch.ones(2, dtype=torch.float32))
    target["inference"].register_buffer("state", torch.zeros(2, dtype=torch.bfloat16))
    old_tensors = get_refit_tensor_dict(target)
    plan = ReshardPlan(
        [TransferOp("source_tower.state", 1, True, (slice(None),), (slice(None),), task_id=7)],
        [
            TransferOp("inference.state", 0, False, (slice(None),), (slice(None),), task_id=task)
            for task in (7, 9)
        ],
    )
    calls = []

    def gather(gathered, local, group=None):
        calls.append(local)
        assert local == {7: torch.float32}
        gathered[:] = [local, {9: torch.float16 if conflicting_shard else torch.float32}]

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    if conflicting_shard:
        with pytest.raises(ValueError, match="Source shards disagree on buffer dtype"):
            _harmonize_buffer_dtypes(plan, source, target)
        return
    _harmonize_buffer_dtypes(plan, source, target)
    assert plan.buffer_dtypes == {"inference.state": torch.float32}
    assert target["inference"].state.dtype == torch.float32
    assert get_refit_tensor_dict(target) is not old_tensors
    assert get_refit_tensor_dict(target)["inference.state"] is target["inference"].state
    _harmonize_buffer_dtypes(plan, source, target)
    assert len(calls) == 1
