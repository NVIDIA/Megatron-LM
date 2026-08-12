# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pipeline-parallel integration tests for experimental Megatron-FSDP v2."""

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint

from megatron.core import ModelParallelConfig, parallel_state
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils


class PipelineStage(nn.Module):
    """One pipeline stage with the input-tensor protocol expected by MCore."""

    def __init__(
        self,
        hidden_size: int,
        config: ModelParallelConfig,
        vp_stage: int | None = None,
        recompute: bool = False,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.config = config
        self.model_type = ModelType.encoder_or_decoder
        self.vp_stage = vp_stage
        self.recompute = recompute
        self._pipeline_input: torch.Tensor | None = None

    def set_input_tensor(self, input_tensors: list[torch.Tensor | None]) -> None:
        assert len(input_tensors) == 1
        self._pipeline_input = input_tensors[0]

    def forward(self, input_tensor: torch.Tensor | None) -> torch.Tensor:
        if self._pipeline_input is not None:
            input_tensor = self._pipeline_input
        assert input_tensor is not None
        if self.recompute:
            return checkpoint(self.linear, input_tensor, use_reentrant=False)
        return self.linear(input_tensor)


def _flat_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


@pytest.mark.parametrize("virtual_pipeline_parallel_size", [None, 2], ids=["pp", "pp-vpp"])
@pytest.mark.parametrize("recompute", [False, True], ids=["no-recompute", "recompute"])
def test_pipeline_parallel_schedule_trains_for_two_iterations(
    distributed_setup, virtual_pipeline_parallel_size, recompute
):
    """Exercise PP send/recv, FSDP collectives, accumulation, and optimizer refresh."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 4 or world_size % 2:
        pytest.skip("This test requires an even world size of at least four ranks.")

    pipeline_parallel_size = 2
    hidden_size = 8
    sequence_length = 4
    micro_batch_size = 2
    num_microbatches = 4

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pipeline_parallel_size,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallel_size,
    )
    try:
        dp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        mesh = DeviceMesh.from_group(dp_group, device.type)
        config = ModelParallelConfig(
            pipeline_model_parallel_size=pipeline_parallel_size,
            pipeline_dtype=torch.float32,
            virtual_pipeline_model_parallel_size=virtual_pipeline_parallel_size,
        )
        config.hidden_size = hidden_size
        # Experimental FSDP v2 reduces gradients from its autograd hooks and
        # fences the current stream in a final autograd callback. The legacy
        # MCore finalizer expects the DDP/Megatron-FSDP-v1 wrapper contract.
        config.finalize_model_grads_func = None

        torch.manual_seed(1234)
        models = [
            PipelineStage(hidden_size, config, vp_stage=vp_stage, recompute=recompute).to(device)
            for vp_stage in range(virtual_pipeline_parallel_size or 1)
        ]
        # All chunks share one FSDP context, mirroring the training-loop wrap
        # of a multi-chunk (VPP) model.
        with fully_shard_context(device=device):
            for model in models:
                fully_shard(model, mesh=mesh, placements=_flat_placements())
        optimizer = torch.optim.SGD(
            (parameter for model in models for parameter in model.parameters()), lr=0.05
        )
        fully_shard_optimizer(optimizer)

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pp_group = pg_collection.pp
        p2p_communicator = P2PCommunicator(pp_group, config)
        forward_backward_func = get_forward_backward_func(
            pp_size=pipeline_parallel_size, vp_size=virtual_pipeline_parallel_size
        )

        def forward_step_func(data_iterator, stage):
            input_tensor = (
                next(data_iterator)
                if parallel_state.is_pipeline_first_stage(
                    ignore_virtual=False, vp_stage=stage.vp_stage
                )
                else None
            )

            def loss_func(output_tensor):
                loss = output_tensor.float().square().mean()
                return loss, {"loss": loss.detach()}

            return stage(input_tensor), loss_func

        for iteration in range(2):
            optimizer.zero_grad(set_to_none=True)
            if parallel_state.is_pipeline_first_stage():
                inputs = [
                    torch.full(
                        (sequence_length, micro_batch_size, hidden_size),
                        fill_value=float(iteration + microbatch + 1),
                        device=device,
                    )
                    for microbatch in range(num_microbatches)
                ]
                first_stage_iterator = iter(inputs)
            else:
                first_stage_iterator = None
            data_iterator = (
                [first_stage_iterator] + [None] * (len(models) - 1)
                if virtual_pipeline_parallel_size is not None
                else first_stage_iterator
            )

            losses = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=models,
                num_microbatches=num_microbatches,
                seq_length=sequence_length,
                micro_batch_size=micro_batch_size,
                forward_only=False,
                pg_collection=pg_collection,
                p2p_communicator=p2p_communicator,
            )

            parameters = [parameter for model in models for parameter in model.parameters()]
            assert parameters
            for parameter in parameters:
                assert isinstance(parameter, DTensor)
                assert isinstance(parameter.grad, DTensor)
                assert torch.isfinite(parameter.grad.to_local()).all()

            params_before_step = [parameter.to_local().detach().clone() for parameter in parameters]
            optimizer.step()
            assert any(
                not torch.equal(before, parameter.to_local())
                for before, parameter in zip(params_before_step, parameters)
            )

            expected_loss_count = num_microbatches if parallel_state.is_pipeline_last_stage() else 0
            assert len(losses) == expected_loss_count

        dist.barrier()
    finally:
        Utils.destroy_model_parallel()
