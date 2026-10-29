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
from tests.unit_tests.transformer.test_stateful_module import (  # pylint: disable=unused-import
    te_rng_tracker,
)


class _Region(nn.Module):
    def __init__(self, width=16, dtype=torch.float32, fp8=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(width, device="cuda", dtype=dtype))
        self.projection = None
        if fp8:
            from transformer_engine.pytorch import Linear

            self.projection = Linear(width, width, bias=False, params_dtype=dtype)

    def forward(self, hidden, state):
        if self.projection is not None:
            from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

            assert FP8GlobalStateManager.is_fp8_enabled()
            hidden = self.projection(hidden)
        memory = hidden * self.weight + state["memory"]
        return memory.square(), {"memory": memory}


class _Replay(nn.Module):
    def __init__(self, mode, inputs):
        super().__init__()
        sample = inputs[0]
        fields = (TensorField("memory", tuple(sample.shape), sample.dtype, "contiguous", True),)
        self.region = StatefulModule(
            _Region(sample.shape[-1], sample.dtype, fp8=mode == "mxfp8"), fields, fields
        )
        self.mode = mode
        options = {}
        if mode == "mxfp8":
            from transformer_engine.common.recipe import MXFP8BlockScaling

            mode = "transformer_engine"
            options = dict(fp8_enabled=True, fp8_recipe=MXFP8BlockScaling())
        self.graphs = (
            StatefulGraphs(
                self.region, inputs[0], {"memory": inputs[1]}, backend=mode, capture_kwargs=options
            )
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


@pytest.mark.usefixtures("te_rng_tracker")
@pytest.mark.parametrize("mode", ["eager", "checkpoint", "torch", "transformer_engine", "mxfp8"])
def test_shared_region_replays_forward_and_all_gradients(mode):
    if mode == "mxfp8" and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 requires SM100")
    Utils.initialize_model_parallel(1, 1)
    seeded()
    width, dtype = (32, torch.bfloat16) if mode == "mxfp8" else (16, torch.float32)
    inputs = tuple(
        torch.randn(64, 2, width, device="cuda", dtype=dtype, requires_grad=True) for _ in range(2)
    )
    module = None
    try:
        module = _Replay(mode, inputs)
        with deterministic_algorithms(True):
            assert_module_replays_bit_exact(module, inputs)
    finally:
        if module is not None and module.graphs is not None:
            module.graphs.close()
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("operation", ["gather", "layout", "zigzag_gather"])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_shared_cp_state_replays_forward_and_gradients(operation, sequence_parallel):
    if Utils.world_size < 2 or Utils.world_size % 2:
        pytest.skip("Requires an even number of workers")
    if sequence_parallel and (Utils.world_size < 4 or Utils.world_size % 4):
        pytest.skip("Requires TP2 x CP2")
    Utils.initialize_model_parallel(2 if sequence_parallel else 1, 1, context_parallel_size=2)
    groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["cp", "tp", "tp_cp"])
    group = groups.cp
    options = dict(
        sequence_parallel=sequence_parallel, tp_group=groups.tp, tp_cp_group=groups.tp_cp
    )
    seeded(1234 + Utils.rank)
    schema = TensorSchema((TensorField("memory", (16, 4), torch.float32, "contiguous", True),))

    def run(value):
        if operation == "gather":
            result, _ = gather_state({"memory": value}, schema, group, **options)
        else:
            result, layout = redistribute_state(
                {"memory": value}, schema, "zigzag", group, **options
            )
            if operation == "zigzag_gather":
                result, _ = gather_state(result, layout, group, **options)
        return result["memory"]

    try:
        with deterministic_algorithms(True):
            assert_replays_bit_exact(run, (torch.randn(16, 4, device="cuda", requires_grad=True),))
    finally:
        Utils.destroy_model_parallel()
