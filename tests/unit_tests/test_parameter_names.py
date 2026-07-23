# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.parameter_names import CanonicalParameterNameIndex, CanonicalParameterNameMap


class _Layer(torch.nn.Module):
    def __init__(self, layer_number):
        super().__init__()
        self.layer_number = layer_number
        self.weight = torch.nn.Parameter(torch.zeros(1))


class _PipelineModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Module()
        self.decoder.layers = torch.nn.ModuleList([_Layer(5), _Layer(8)])


class _GroupedExpertModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"num_moe_experts": 4})()
        self.mlp = torch.nn.Module()
        self.mlp.experts = torch.nn.Module()
        self.mlp.experts.linear_fc1 = torch.nn.Module()
        self.mlp.experts.linear_fc1.register_parameter(
            "weight0", torch.nn.Parameter(torch.zeros(1))
        )
        self.mlp.experts.linear_fc1.register_parameter("bias1", torch.nn.Parameter(torch.zeros(1)))


class _SequentialExpertModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"num_moe_experts": 4})()
        self.mlp = torch.nn.Module()
        self.mlp.experts = torch.nn.Module()
        self.mlp.experts.local_experts = torch.nn.ModuleList(
            [torch.nn.Linear(1, 1, bias=False), torch.nn.Linear(1, 1, bias=False)]
        )


class _SingleParamModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


class _TwoParamModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.z = torch.nn.Parameter(torch.zeros(1))
        self.a = torch.nn.Parameter(torch.zeros(1))


def test_canonical_parameter_name_index_is_sorted_and_deduplicated():
    index = CanonicalParameterNameIndex(["z", "a", "z"])

    assert index.names == ("a", "z")
    assert dict(index) == {"a": 0, "z": 1}


def test_canonical_parameter_names_have_deterministic_local_index():
    names = CanonicalParameterNameMap(_TwoParamModel())

    assert names.local_index.names == ("a", "z")
    assert names.local_index["a"] == 0
    assert names.local_index["z"] == 1


def test_canonical_parameter_names_use_global_layer_numbers():
    model = _PipelineModel()
    names = CanonicalParameterNameMap(model)

    assert names.name_for_param(model.decoder.layers[0].weight) == "decoder.layers.4.weight"
    assert names.name_for_param(model.decoder.layers[1].weight) == "decoder.layers.7.weight"


def test_canonical_parameter_names_use_global_grouped_expert_numbers():
    model = _GroupedExpertModel()
    names = CanonicalParameterNameMap(model, expert_parallel_rank=1, expert_parallel_size=2)

    assert set(names.values()) == {"mlp.experts.linear_fc1.weight2", "mlp.experts.linear_fc1.bias3"}


def test_canonical_parameter_names_use_global_sequential_expert_numbers():
    model = _SequentialExpertModel()
    names = CanonicalParameterNameMap(model, expert_parallel_rank=1, expert_parallel_size=2)

    assert set(names.values()) == {
        "mlp.experts.local_experts.2.weight",
        "mlp.experts.local_experts.3.weight",
    }


def test_canonical_parameter_names_reject_collisions_between_chunks():
    with pytest.raises(ValueError, match="multiple distinct local parameters"):
        CanonicalParameterNameMap([_SingleParamModel(), _SingleParamModel()])


@pytest.mark.parametrize(
    ("rank", "size", "match"), [(0, 0, "at least 1"), (-1, 2, "must be in"), (2, 2, "must be in")]
)
def test_canonical_parameter_names_validate_expert_topology(rank, size, match):
    with pytest.raises(ValueError, match=match):
        CanonicalParameterNameMap(
            _SingleParamModel(), expert_parallel_rank=rank, expert_parallel_size=size
        )


def test_canonical_parameter_names_reject_uneven_expert_partition():
    model = _GroupedExpertModel()
    model.config.num_moe_experts = 5

    with pytest.raises(ValueError, match="must be divisible"):
        CanonicalParameterNameMap(model, expert_parallel_rank=0, expert_parallel_size=2)


def test_all_gather_index_returns_local_index_without_distributed(monkeypatch):
    names = CanonicalParameterNameMap(_TwoParamModel())
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    assert names.all_gather_index() is names.local_index


def test_all_gather_index_collects_deduplicates_and_orders_names(monkeypatch):
    names = CanonicalParameterNameMap(_SingleParamModel())
    expected_group = object()

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def fake_all_gather_object(output, local_names, group):
        assert local_names == ("weight",)
        assert group is expected_group
        output[:] = [("z", "shared"), ("a", "shared")]

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)

    index = names.all_gather_index(expected_group)

    assert index.names == ("a", "shared", "z")
    assert dict(index) == {"a": 0, "shared": 1, "z": 2}
    assert names.local_index.names == ("weight",)
