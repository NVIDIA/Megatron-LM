# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.parameter_names import CanonicalParameterNameMap


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


class _PipelineChunk(torch.nn.Module):
    def __init__(self, layer_number):
        super().__init__()
        self.decoder = torch.nn.Module()
        self.decoder.layers = torch.nn.ModuleList([_Layer(layer_number)])


class _MTPModel(torch.nn.Module):
    def __init__(self, layer_number):
        super().__init__()
        self.mtp = torch.nn.Module()
        self.mtp.layers = torch.nn.ModuleList([_Layer(layer_number)])


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
        self.mlp.experts.linear_fc1.register_parameter(
            "bias1", torch.nn.Parameter(torch.zeros(1))
        )


class _SequentialExpertModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"num_moe_experts": 4})()
        self.mlp = torch.nn.Module()
        self.mlp.experts = torch.nn.Module()
        self.mlp.experts.local_experts = torch.nn.ModuleList(
            [torch.nn.Linear(1, 1, bias=False), torch.nn.Linear(1, 1, bias=False)]
        )


class _SharedExpertModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"num_moe_experts": 4})()
        self.mlp = torch.nn.Module()
        self.mlp.shared_experts = torch.nn.Linear(1, 1, bias=False)


class _SingleParamModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


class _TiedParamModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Linear(1, 1, bias=False)
        self.output = torch.nn.Linear(1, 1, bias=False)
        self.output.weight = self.embedding.weight


def test_canonical_parameter_names_use_global_layer_numbers():
    model = _PipelineModel()
    names = CanonicalParameterNameMap(model)

    assert names[model.decoder.layers[0].weight] == "decoder.layers.4.weight"
    assert names[model.decoder.layers[1].weight] == "decoder.layers.7.weight"


def test_canonical_parameter_names_disambiguate_pipeline_chunks_by_global_layer():
    first = _PipelineChunk(5)
    second = _PipelineChunk(8)
    names = CanonicalParameterNameMap([first, second])

    assert names[first.decoder.layers[0].weight] == "decoder.layers.4.weight"
    assert names[second.decoder.layers[0].weight] == "decoder.layers.7.weight"


def test_canonical_parameter_names_use_global_mtp_layer_numbers():
    model = _MTPModel(layer_number=5)
    names = CanonicalParameterNameMap(model)

    assert names[model.mtp.layers[0].weight] == "mtp.layers.4.weight"


def test_canonical_parameter_names_keep_one_site_for_repeated_mtp_layer():
    model = _MTPModel(layer_number=1)
    model.config = type("Config", (), {"mtp_num_layers": 2, "mtp_use_repeated_layer": True})()
    names = CanonicalParameterNameMap(model)

    assert tuple(names.values()) == ("mtp.layers.0.weight",)


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


def test_canonical_parameter_names_do_not_offset_shared_expert_names():
    model = _SharedExpertModel()
    names = CanonicalParameterNameMap(model, expert_parallel_rank=1, expert_parallel_size=2)

    assert tuple(names.values()) == ("mlp.shared_experts.weight",)


def test_canonical_parameter_names_deduplicate_tied_parameter():
    model = _TiedParamModel()
    names = CanonicalParameterNameMap(model)

    assert len(names) == 1
    assert names[model.embedding.weight] == "embedding.weight"


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
