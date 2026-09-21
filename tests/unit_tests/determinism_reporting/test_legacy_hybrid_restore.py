# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU contract for the historical dp-zero conversion loader's owner refresh."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rank", [0, 1])
def test_legacy_loader_restores_full_precision_owner_before_refresh(dtype, rank, monkeypatch):
    path = Path(__file__).resolve().parents[3] / "megatron/core/optimizer/distrib_optimizer.py"
    owner = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "DistributedOptimizer"
    )
    methods = [
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in ("_update_legacy_world_tensors", "load_parameter_state_from_dp_zero_legacy")
    ]

    class Hybrid:
        pass

    namespace = {"torch": torch, "HybridDeviceOptimizer": Hybrid}
    module = ast.Module(
        body=[ast.ClassDef(name="Loader", bases=[], keywords=[], body=methods, decorator_list=[])],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    parameter = torch.zeros(4, dtype=dtype)
    optimizer = Hybrid()
    optimizer.param_groups = [{"params": [parameter]}]
    optimizer.state = {
        parameter: {key: torch.zeros(4) for key in ("master_param", "exp_avg", "exp_avg_sq")}
    }
    loader = namespace["Loader"]()
    loader.optimizer = optimizer
    loader.optimizer_state_keys = ("exp_avg", "exp_avg_sq")
    loader.data_parallel_group_gloo = SimpleNamespace(size=lambda: 2, rank=lambda: rank)
    loader.gbuf_ranges = [
        {
            torch.float32: [
                {"param_map": {parameter: {"gbuf_local": SimpleNamespace(start=0, end=4)}}}
            ]
        }
    ]
    loader.model_param_group_index_map = {parameter: (0, 0)}
    loader.buffers = [
        SimpleNamespace(
            numel_unpadded=8, buckets=[SimpleNamespace(numel_unpadded=8, grad_data=torch.zeros(8))]
        )
    ]
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _: [0, 1])
    original = torch.tensor([1.001, 2.002, 3.003, 4.004])
    state = {
        0: {
            torch.float32: {
                key: [original.repeat(2) * scale]
                for key, scale in (("param", 1), ("exp_avg", 2), ("exp_avg_sq", 3))
            }
        }
    }
    incoming = iter(original * scale for scale in (1, 2, 3))

    def scatter(output, values, *args):
        expected = next(incoming)
        if rank == 0:
            assert len(values) == 2 and all(torch.equal(value, expected) for value in values)
        else:
            assert values is None
        output.copy_(expected)

    monkeypatch.setattr(torch.distributed, "scatter", scatter)

    def refresh():
        assert torch.equal(optimizer.state[parameter]["master_param"], original)
        assert torch.equal(optimizer.state[parameter]["exp_avg"], original * 2)
        assert torch.equal(optimizer.state[parameter]["exp_avg_sq"], original * 3)

    optimizer._sync_hdo_state_to_sub_optimizers = Mock(side_effect=refresh)
    loader.load_parameter_state_from_dp_zero_legacy(state if rank == 0 else None)
    assert torch.equal(parameter, original.to(dtype))
    optimizer._sync_hdo_state_to_sub_optimizers.assert_called_once_with()
