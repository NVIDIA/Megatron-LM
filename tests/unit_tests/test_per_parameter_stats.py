# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core import per_parameter_stats as pps
from megatron.core.per_parameter_stats import (
    NamedTensorBucket,
    PerParameterStatRegistry,
    get_or_create_per_parameter_stat_registry,
    reduce_raw_moments_by_param,
)


class TwoParamModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(1))
        self.b = torch.nn.Parameter(torch.zeros(1))


class SequentialExpertModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"num_moe_experts": 4})()
        self.mlp = torch.nn.Module()
        self.mlp.experts = torch.nn.Module()
        self.mlp.experts.local_experts = torch.nn.ModuleList(
            [
                torch.nn.Linear(1, 1, bias=False),
                torch.nn.Linear(1, 1, bias=False),
            ]
        )


class _FakeCudaTensor:
    device = torch.device("cuda:0")
    dtype = torch.float32

    def is_contiguous(self) -> bool:
        return True


class _FakeExpertGroup:
    def __init__(self, rank=0, size=1):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


def test_reduce_raw_moments_by_param_on_cpu():
    registry = PerParameterStatRegistry(TwoParamModel())

    values, aggregate_moments = reduce_raw_moments_by_param(
        registry,
        [
            NamedTensorBucket(
                names=["a", "a", "b"],
                tensors=[torch.tensor([1.0, 2.0]), torch.tensor([2.0, 4.0]), torch.tensor([3.0])],
            )
        ],
    )

    assert dict(values) == {
        "a": {
            "count": pytest.approx(4.0),
            "sum_1": pytest.approx(9.0),
            "sum_2": pytest.approx(21.0),
            "sum_3": pytest.approx(81.0),
            "sum_4": pytest.approx(321.0),
        },
        "b": {
            "count": pytest.approx(1.0),
            "sum_1": pytest.approx(3.0),
            "sum_2": pytest.approx(9.0),
            "sum_3": pytest.approx(27.0),
            "sum_4": pytest.approx(81.0),
        },
    }
    assert aggregate_moments == {
        "count": pytest.approx(5.0),
        "sum_1": pytest.approx(12.0),
        "sum_2": pytest.approx(30.0),
        "sum_3": pytest.approx(108.0),
        "sum_4": pytest.approx(402.0),
    }


def test_reduce_raw_moments_by_param_rejects_mismatched_names_and_tensors():
    registry = PerParameterStatRegistry(TwoParamModel())

    with pytest.raises(ValueError, match="names but"):
        reduce_raw_moments_by_param(
            registry,
            [NamedTensorBucket(names=["a"], tensors=[torch.tensor([1.0]), torch.tensor([2.0])])],
        )


def test_registry_cache_is_per_model_identity():
    first_model = TwoParamModel()
    second_model = TwoParamModel()

    first_registry = get_or_create_per_parameter_stat_registry(first_model)
    assert get_or_create_per_parameter_stat_registry(first_model) is first_registry
    assert get_or_create_per_parameter_stat_registry(second_model) is not first_registry


def test_registry_cache_includes_expert_group_identity():
    model = TwoParamModel()
    expert_group = _FakeExpertGroup(rank=0, size=2)

    first_registry = get_or_create_per_parameter_stat_registry(model)
    expert_registry = get_or_create_per_parameter_stat_registry(
        model, expert_model_parallel_group=expert_group
    )

    assert expert_registry is not first_registry
    assert get_or_create_per_parameter_stat_registry(
        model, expert_model_parallel_group=expert_group
    ) is expert_registry


def test_registry_uses_explicit_expert_group_for_expert_names(monkeypatch):
    expert_group = _FakeExpertGroup(rank=1, size=2)

    monkeypatch.setattr(pps, "get_pg_rank", lambda group: 1 if group is expert_group else 0)
    monkeypatch.setattr(pps, "get_pg_size", lambda group: 2 if group is expert_group else 1)

    registry = PerParameterStatRegistry(
        SequentialExpertModel(), expert_model_parallel_group=expert_group
    )

    assert set(registry.param_to_name.values()) == {
        "mlp.experts.local_experts.2.weight",
        "mlp.experts.local_experts.3.weight",
    }


def test_local_raw_moments_multi_tensor_path_preserves_order(monkeypatch):
    calls = []

    def fake_multi_tensor_applier(op, noop_flag_buffer, tensor_lists):
        calls.append([tensor.dtype for tensor in tensor_lists[0]])
        return op(0, noop_flag_buffer, tensor_lists)

    def fake_multi_tensor_raw_moments(_, __, tensor_lists):
        return torch.stack([pps._torch_raw_moment_row(tensor) for tensor in tensor_lists[0]])

    monkeypatch.setattr(pps, "multi_tensor_applier", fake_multi_tensor_applier)
    monkeypatch.setattr(pps, "multi_tensor_raw_moments", fake_multi_tensor_raw_moments)
    monkeypatch.setattr(pps, "_can_use_multi_tensor_raw_moments", lambda tensors, device: True)

    tensors = [
        torch.tensor([1.0, 2.0], dtype=torch.float32),
        torch.tensor([3.0], dtype=torch.bfloat16),
        torch.tensor([4.0, 5.0], dtype=torch.float32),
    ]
    rows = pps._local_raw_moments(tensors, torch.device("cpu"))
    expected = torch.stack([pps._torch_raw_moment_row(tensor) for tensor in tensors])

    torch.testing.assert_close(rows, expected)
    assert calls == [[torch.float32, torch.float32], [torch.bfloat16]]


def test_local_raw_moments_multi_tensor_path_splits_oversized_tensors(monkeypatch):
    calls = []

    def fake_multi_tensor_applier(op, noop_flag_buffer, tensor_lists):
        calls.append([tensor.numel() for tensor in tensor_lists[0]])
        return op(0, noop_flag_buffer, tensor_lists)

    def fake_multi_tensor_raw_moments(_, __, tensor_lists):
        return torch.stack([pps._torch_raw_moment_row(tensor) for tensor in tensor_lists[0]])

    monkeypatch.setattr(pps, "multi_tensor_applier", fake_multi_tensor_applier)
    monkeypatch.setattr(pps, "multi_tensor_raw_moments", fake_multi_tensor_raw_moments)
    monkeypatch.setattr(pps, "_can_use_multi_tensor_raw_moments", lambda tensors, device: True)
    monkeypatch.setattr(pps, "_MAX_MULTI_TENSOR_RAW_MOMENTS_NUMEL", 4)

    tensors = [torch.arange(1.0, 11.0), torch.tensor([11.0, 12.0, 13.0])]
    rows = pps._local_raw_moments(tensors, torch.device("cpu"))
    expected = torch.stack([pps._torch_raw_moment_row(tensor) for tensor in tensors])

    torch.testing.assert_close(rows, expected)
    assert calls == [[4, 4, 2, 3]]


def test_multi_tensor_raw_moments_env_guard_disables_fast_path(monkeypatch):
    tensor = _FakeCudaTensor()
    device = torch.device("cuda:0")

    monkeypatch.setattr(pps, "multi_tensor_applier", object())
    monkeypatch.setattr(pps, "multi_tensor_raw_moments", object())
    monkeypatch.delenv("MEGATRON_DISABLE_MULTI_TENSOR_RAW_MOMENTS", raising=False)
    assert pps._can_use_multi_tensor_raw_moments([tensor], device)

    monkeypatch.setenv("MEGATRON_DISABLE_MULTI_TENSOR_RAW_MOMENTS", "1")
    assert not pps._can_use_multi_tensor_raw_moments([tensor], device)
