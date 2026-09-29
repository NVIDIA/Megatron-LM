# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real context-parallel shared-state redistribution, gradients and graph capture."""

import gc

import pytest
import torch
from torch import nn

from megatron.core.context_parallel.shared_state import gather_state, redistribute_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.state_boundary import TensorField, TensorSchema
from megatron.core.transformer.stateful_module import StatefulGraphs, StatefulModule
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def cp():
    if Utils.world_size < 2 or Utils.world_size % 2:
        pytest.skip("Requires an even number of GPU workers")
    Utils.initialize_model_parallel(1, 1, context_parallel_size=2)
    yield ProcessGroupCollection.use_mpu_process_groups(required_pgs=["cp"]).cp
    gc.collect()
    Utils.destroy_model_parallel()


def _state(group):
    values = torch.arange(group.rank() * 16, (group.rank() + 1) * 16, device="cuda").float()
    shared = values.reshape(8, 2).requires_grad_()
    ids = torch.arange(group.rank() * 8, (group.rank() + 1) * 8, device="cuda")
    constant = torch.ones((), device="cuda")
    state = {"shared": shared, "ids": ids, "constant": constant}
    schema = TensorSchema(
        (
            TensorField("shared", (8, 2), shared.dtype, "contiguous", True),
            TensorField("ids", (8,), ids.dtype, "contiguous", False),
            TensorField("constant", (), constant.dtype, "replicated", False),
        )
    )
    return state, schema


def test_layout_roundtrip_preserves_shared_state_and_owner_gradients(cp):
    state, schema = _state(cp)
    shuffled, layout = redistribute_state(state, schema, "zigzag", cp)
    restored, original = redistribute_state(shuffled, layout, "contiguous", cp)
    assert original == schema
    for key in state:
        torch.testing.assert_close(restored[key], state[key], rtol=0, atol=0)
    assert restored["constant"] is state["constant"]
    restored["shared"].square().sum().backward()
    torch.testing.assert_close(state["shared"].grad, 2 * state["shared"], rtol=0, atol=0)


def test_gather_adds_all_consumer_gradients_and_preserves_integer_state(cp):
    state, schema = _state(cp)
    gathered, global_schema = gather_state(state, schema, cp)
    torch.testing.assert_close(
        gathered["shared"], torch.arange(32, device="cuda").float().reshape(16, 2)
    )
    torch.testing.assert_close(gathered["ids"], torch.arange(16, device="cuda"))
    assert global_schema.fields[0].shape == (16, 2)
    assert global_schema.fields[0].layout == "replicated"
    (gathered["shared"].sum() * (cp.rank() + 1)).backward()
    torch.testing.assert_close(state["shared"].grad, torch.full_like(state["shared"], 3))


def test_gather_then_local_slice_does_not_double_reduce(cp):
    state, schema = _state(cp)
    gathered, _ = gather_state(state, schema, cp)
    local = gathered["shared"].narrow(0, cp.rank() * 8, 8)
    local.square().sum().backward()
    torch.testing.assert_close(state["shared"].grad, 2 * state["shared"])


class _CrossRankRegion(nn.Module):
    def __init__(self, group):
        super().__init__()
        self.group = group
        self.weight = nn.Parameter(torch.tensor(0.25, device="cuda"))
        self.schema = TensorSchema(
            (TensorField("shared", (8, 2), torch.float32, "contiguous", True),)
        )

    def forward(self, hidden, state):
        gathered, _ = gather_state(state, self.schema, self.group)
        return hidden + gathered["shared"].sum(0) * self.weight, {}


@pytest.mark.usefixtures("te_rng_tracker")
@pytest.mark.parametrize("mode", ["recompute", "torch", "transformer_engine"])
def test_cp_collective_region_matches_eager_in_recompute_and_graphs(cp, mode):
    module = _CrossRankRegion(cp)
    actual = StatefulModule(module, module.schema.fields)
    # Copy the parameters, retaining the same explicit process group.
    expected = StatefulModule(_CrossRankRegion(cp), actual.inputs.fields)
    expected.load_state_dict(actual.state_dict())
    state, _ = _state(cp)
    x = torch.randn(8, 2, device="cuda", requires_grad=True)
    y = x.detach().clone().requires_grad_()
    reference_state = {"shared": state["shared"].detach().clone().requires_grad_()}
    if mode == "recompute":
        output, _ = actual.run(x, state, recompute=True)
    else:
        graphs = StatefulGraphs(actual, x, state, backend=mode)
        output, _ = graphs.run(x, state)
    reference, _ = expected.run(y, reference_state)
    torch.testing.assert_close(output, reference)
    output.square().sum().backward()
    reference.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad)
    torch.testing.assert_close(state["shared"].grad, reference_state["shared"].grad)
    torch.testing.assert_close(actual.module.weight.grad, expected.module.weight.grad)
    if mode != "recompute":
        del output, reference
        graphs.close()
