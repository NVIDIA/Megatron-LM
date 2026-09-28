# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Generic shared-projection training with CP2, PP2/VPP and real execution boundaries."""

import gc

import pytest
import torch
from torch import nn

from megatron.core.context_parallel.shared_state import gather_state
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.pipeline_payload import TensorStatePayload
from megatron.core.pipeline_parallel.schedules import (
    forward_backward_pipelining_with_interleaving,
    forward_backward_pipelining_without_interleaving,
)
from megatron.core.pipeline_parallel.typed_p2p_communication import create_pipeline_control_group
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.state_boundary import TensorField, TensorSchema
from megatron.core.transformer.stateful_module import StatefulGraphs, StatefulModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

FIELDS = (
    TensorField("memory", (8, 2), torch.float32, "contiguous", True),
    TensorField("ids", (8,), torch.int64, "contiguous", False),
)


class _Region(nn.Module):
    def __init__(self, index, cp):
        super().__init__()
        self.index, self.cp = index, cp
        self.weight = nn.Parameter(torch.tensor(0.1 * (index + 1), device="cuda"))

    def forward(self, hidden, state):
        if self.index == 0:
            memory = hidden * self.weight
            state = {
                "memory": memory,
                "ids": torch.arange(8, device=hidden.device) + self.cp.rank() * 8,
            }
            return memory + hidden.sin() * 0.1, state
        gathered, _ = gather_state(state, TensorSchema(FIELDS), self.cp)
        selected = gathered["memory"].index_select(0, state["ids"])
        output = hidden * self.weight + (selected + gathered["memory"].mean(0)) * self.weight
        return output, state


class _Stage(nn.Module):
    model_type = ModelType.encoder_or_decoder
    pipeline_payload_factory = TensorStatePayload

    def __init__(self, config, index, total, cp, mode, microbatches):
        super().__init__()
        self.config, self.index, self.total = config, index, total
        self.mode, self.microbatches = mode, microbatches
        self.region = StatefulModule(
            _Region(index, cp), () if index == 0 else FIELDS, () if index == total - 1 else FIELDS
        )
        self.input_tensor, self.iteration = None, 0
        self.graphs = None
        if mode in ("torch", "transformer_engine"):
            sample = torch.zeros(8, 2, device="cuda", requires_grad=True)
            state = (
                {}
                if index == 0
                else {
                    "memory": torch.ones_like(sample, requires_grad=True),
                    "ids": torch.arange(8, device="cuda") + cp.rank() * 8,
                }
            )
            self.graphs = StatefulGraphs(
                self.region, sample, state, slots=microbatches, backend=mode
            )

    def set_input_tensor(self, tensors):
        self.input_tensor = tensors[0] if isinstance(tensors, list) else tensors

    def forward(self, data):
        payload, self.input_tensor = self.input_tensor, None
        hidden, state = (data, {}) if payload is None else payload.restore()
        if self.graphs is None:
            hidden, state = self.region.run(hidden, state, recompute=self.mode == "recompute")
        else:
            hidden, state = self.graphs.run(hidden, state, slot=self.iteration % self.microbatches)
        self.iteration += 1
        if self.index == self.total - 1:
            return hidden.square().mean()
        return TensorStatePayload.from_state(
            hidden, state, self.region.outputs, boundary_id=f"after:{self.index}"
        )


@pytest.fixture(scope="module")
def control_group():
    if Utils.world_size < 4 or Utils.world_size % 4:
        pytest.skip("Requires a multiple of four GPU workers for PP2 x CP2")
    Utils.initialize_model_parallel(1, 2, context_parallel_size=2)
    pp = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["pp"]).pp
    group = create_pipeline_control_group(pp, world_group=torch.distributed.group.WORLD)
    yield group
    torch.distributed.destroy_process_group(group)
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("vp_size", [1, 2])
@pytest.mark.parametrize("mode", ["eager", "recompute", "torch", "transformer_engine"])
def test_cp_pp_state_matches_unpartitioned_layers(vp_size, mode, control_group):
    Utils.initialize_model_parallel(
        1,
        2,
        virtual_pipeline_model_parallel_size=vp_size if vp_size > 1 else None,
        context_parallel_size=2,
    )
    groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["pp", "tp", "cp"])
    control = control_group
    count, total = 4, 2 * vp_size
    config = TransformerConfig(
        num_layers=total,
        hidden_size=2,
        num_attention_heads=1,
        pipeline_model_parallel_size=2,
        context_parallel_size=2,
        virtual_pipeline_model_parallel_size=vp_size if vp_size > 1 else None,
        microbatch_group_size_per_vp_stage=2,
        pipeline_dtype=torch.float32,
        batch_p2p_comm=False,
        overlap_p2p_comm=False,
        deallocate_pipeline_outputs=True,
    )
    models = []
    try:
        for vp in range(vp_size):
            models.append(_Stage(config, groups.pp.rank() + 2 * vp, total, groups.cp, mode, count))
            models[-1].pipeline_control_group = control
        reference = [_Stage(config, i, total, groups.cp, "eager", count) for i in range(total)]
        batches = [
            (
                torch.arange(16, device="cuda").float().reshape(8, 2) / 16 + groups.cp.rank() + b
            ).requires_grad_()
            for b in range(count)
        ]
        expected = []
        for batch in batches:
            value = None
            for stage in reference:
                stage.set_input_tensor(value)
                value = stage(batch.detach().clone().requires_grad_())
            expected.append(value.detach().clone())
            # Match MCore's legacy two-value loss reduction: CP / microbatches.
            (value * groups.cp.size() / count).backward()
        actual = []

        def forward(iterator, stage):
            output = stage(next(iterator))

            def loss(value):
                actual.append(value.detach().clone())
                return value, {"loss": value.detach().clone()}

            return output, loss

        schedule = (
            forward_backward_pipelining_with_interleaving
            if vp_size > 1
            else forward_backward_pipelining_without_interleaving
        )
        schedule(
            forward_step_func=forward,
            data_iterator=[iter(batches) for _ in models] if vp_size > 1 else iter(batches),
            model=models if vp_size > 1 else models[0],
            num_microbatches=count,
            seq_length=16,
            micro_batch_size=1,
            p2p_communicator=P2PCommunicator(groups.pp, config),
            pg_collection=groups,
        )
        if groups.pp.rank() == groups.pp.size() - 1:
            torch.testing.assert_close(torch.stack(actual), torch.stack(expected))
        for model in models:
            torch.testing.assert_close(
                model.region.module.weight.grad,
                reference[model.index].region.module.weight.grad,
                rtol=1e-5,
                atol=1e-6,
            )
            assert model.input_tensor is None
    finally:
        for model in models:
            if model.graphs is not None:
                model.graphs.close()
        gc.collect()
        Utils.destroy_model_parallel()
