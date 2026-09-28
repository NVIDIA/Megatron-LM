# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
from torch import nn

from megatron.core.context_parallel.shared_state import gather_state, redistribute_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.state_boundary import TensorField, TensorSchema
from megatron.core.transformer.stateful_module import StatefulGraphs, StatefulModule
from tests.unit_tests.determinism.kernels.harness import (
    assert_module_replays_bit_exact,
    assert_replays_bit_exact,
    deterministic_algorithms,
    seeded,
)
from tests.unit_tests.test_utilities import Utils


class _Region(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(16, device="cuda"))

    def forward(self, hidden, state):
        memory = hidden * self.weight + state["memory"]
        return memory.square(), {"memory": memory}


class _Replay(nn.Module):
    def __init__(self, mode, inputs):
        super().__init__()
        fields = (TensorField("memory", (64, 2, 16), torch.float32, "contiguous", True),)
        self.region = StatefulModule(_Region(), fields, fields)
        self.mode = mode
        self.graphs = (
            StatefulGraphs(self.region, inputs[0], {"memory": inputs[1]}, backend=mode)
            if mode in ("torch", "transformer_engine")
            else None
        )

    def forward(self, hidden, memory):
        if self.graphs is None:
            output, state = self.region.run(
                hidden, {"memory": memory}, recompute=self.mode == "checkpoint"
            )
        else:
            output, state = self.graphs.run(hidden, {"memory": memory})
        # The module replay harness backpropagates through its first output.
        # Combine both branches so every captured differentiable output is active.
        return output + state["memory"]


@pytest.mark.parametrize("mode", ["eager", "checkpoint", "torch", "transformer_engine"])
def test_shared_region_replays_forward_and_all_gradients(mode):
    Utils.initialize_model_parallel(1, 1)
    seeded()
    inputs = tuple(torch.randn(64, 2, 16, device="cuda", requires_grad=True) for _ in range(2))
    module = _Replay(mode, inputs)
    try:
        with deterministic_algorithms(True):
            assert_module_replays_bit_exact(module, inputs)
    finally:
        if module.graphs is not None:
            module.graphs.close()
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("operation", ["gather", "layout"])
def test_shared_cp_state_replays_forward_and_gradients(operation):
    if Utils.world_size < 2 or Utils.world_size % 2:
        pytest.skip("Requires an even number of workers")
    Utils.initialize_model_parallel(1, 1, context_parallel_size=2)
    group = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["cp"]).cp
    seeded()
    schema = TensorSchema((TensorField("memory", (16, 4), torch.float32, "contiguous", True),))

    def run(value):
        if operation == "gather":
            result, _ = gather_state({"memory": value}, schema, group)
        else:
            result, _ = redistribute_state({"memory": value}, schema, "zigzag", group)
        return result["memory"]

    try:
        with deterministic_algorithms(True):
            assert_replays_bit_exact(run, (torch.randn(16, 4, device="cuda", requires_grad=True),))
    finally:
        Utils.destroy_model_parallel()
