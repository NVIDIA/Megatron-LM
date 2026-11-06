# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Tests for the gradient scaling of expert parameters in Megatron-FSDP.

Expert gradients are reduced over the expert data parallel group, which is narrower than the
data parallel group the loss is averaged over. Besides the reduction over their own group,
they therefore need a residual scale of the expert data parallel size over the data parallel
size, whether the reduction sums or averages.
"""

import pytest
import torch
from torch.nn.functional import mse_loss

from megatron.core.distributed.fsdp.src.megatron_fsdp.distributed_data_parallel_config import (
    DistributedDataParallelConfig,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import (
    fully_shard_model,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    gradient_reduce_preprocessing,
)
from tests.unit_tests.distributed.mfsdp_v1.test_mfsdp_nonuniform_sharding import (
    DIM_SIZE,
    DP_SHARD,
    LEARNING_RATE,
    NUM_STEPS,
    OPTIM_GRADS,
    OPTIM_GRADS_PARAMS,
    TP,
    destroy_device_mesh,
)
from tests.unit_tests.test_utilities import Utils

EP = "ep"
NO_SHARD = "no_shard"
OPTIM = "optim"

EXPERT_PARALLEL_SIZE = 2
GRAD = [1.0, 2.0, 3.0, 4.0]


class TestGradientReducePreprocessing:
    """The in-place scaling and reduce operation chosen ahead of a gradient reduction."""

    @staticmethod
    def preprocess(scaling_factor, average_in_collective, group_size):
        """Preprocess a fixed gradient and return it with the chosen reduce operation."""
        grad = torch.tensor(GRAD)
        ddp_config = DistributedDataParallelConfig(
            average_in_collective=average_in_collective,
            # Keep the scale out of a premul-sum reduce operation, which hides it from view.
            gradient_reduce_div_fusion=False,
        )
        reduce_op = gradient_reduce_preprocessing(
            grad, scaling_factor, ddp_config, group_size=group_size
        )
        return grad, reduce_op

    @pytest.mark.parametrize("group_size", [1, 4])
    def test_average_in_collective_applies_the_residual_scale(self, group_size):
        """
        AVG divides by the reduction group's size and nothing else, so the residual scale
        that expert gradients carry has to be applied before the reduction. Dropping it
        leaves every expert gradient too large by the expert parallel size.
        """
        grad, reduce_op = self.preprocess(0.5, average_in_collective=True, group_size=group_size)
        assert reduce_op == torch.distributed.ReduceOp.AVG
        torch.testing.assert_close(grad, 0.5 * torch.tensor(GRAD), rtol=0, atol=0)

    def test_average_in_collective_leaves_a_unit_scale_to_the_reduction(self):
        """Non-expert gradients carry no residual scale and reach the reduction untouched."""
        grad, reduce_op = self.preprocess(1.0, average_in_collective=True, group_size=4)
        assert reduce_op == torch.distributed.ReduceOp.AVG
        torch.testing.assert_close(grad, torch.tensor(GRAD), rtol=0, atol=0)

    @pytest.mark.parametrize("group_size", [1, 4])
    def test_sum_reduction_applies_the_whole_scale(self, group_size):
        """Without average_in_collective, the whole scale is applied before a SUM reduction."""
        grad, reduce_op = self.preprocess(0.25, average_in_collective=False, group_size=group_size)
        assert reduce_op == torch.distributed.ReduceOp.SUM
        torch.testing.assert_close(grad, 0.25 * torch.tensor(GRAD), rtol=0, atol=0)

    @pytest.mark.parametrize("average_in_collective", [False, True])
    def test_no_scaling_factor_sums_unscaled_gradients(self, average_in_collective):
        """Per-token loss normalization disables the scaling, so gradients are summed as is."""
        grad, reduce_op = self.preprocess(None, average_in_collective, group_size=4)
        assert reduce_op == torch.distributed.ReduceOp.SUM
        torch.testing.assert_close(grad, torch.tensor(GRAD), rtol=0, atol=0)


class ToyMoELayer(torch.nn.Module):
    """A layer of experts that every token is routed to, so that every expert gets a gradient."""

    def __init__(self, dim: int = DIM_SIZE, num_experts: int = 2):
        super().__init__()
        self.experts = torch.nn.ModuleList([torch.nn.Linear(dim, dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum(expert(x) for expert in self.experts) / len(self.experts)


class ToyExpertParallelModel(torch.nn.Module):
    """
    Toy model with a dense layer followed by a layer of experts.

    Megatron-FSDP classifies a parameter as an expert parameter when ".experts." appears in
    its name, so the experts are nested in a submodule to put that infix in their names.
    """

    def __init__(self, dim: int = DIM_SIZE):
        super().__init__()
        self.dense = torch.nn.Linear(dim, dim)
        self.mlp = ToyMoELayer(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.dense(x))


def build_expert_parallel_meshes():
    """
    Build a DP-Shard device mesh spanning the world and an expert device mesh whose DP-Shard
    dimension only spans the ranks that share an expert parallel rank.
    """
    from torch.distributed.device_mesh import init_device_mesh

    world_size = torch.distributed.get_world_size()
    if world_size % EXPERT_PARALLEL_SIZE != 0:
        pytest.skip(f"Requires a world size divisible by {EXPERT_PARALLEL_SIZE}.")

    device_mesh = init_device_mesh(
        "cuda", mesh_shape=(world_size, 1), mesh_dim_names=(DP_SHARD, TP)
    )
    expt_device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(EXPERT_PARALLEL_SIZE, world_size // EXPERT_PARALLEL_SIZE, 1),
        mesh_dim_names=(EP, DP_SHARD, TP),
    )
    return device_mesh, expt_device_mesh


def assert_loss_curve_matches_reference(fsdp_model, fsdp_optimizer, reference_model, expt_dp_group):
    """
    Train the sharded model against a replicated reference and compare their loss curves.

    The reference is an unwrapped replica whose gradients are reduced by hand the way MCore
    DDP reduces them: the loss is averaged over the data parallel group, so non-expert
    gradients are averaged over the world, while expert gradients are summed over the expert
    data parallel group and that sum is divided by the world size. SGD applies the gradient
    unnormalized, so a wrongly scaled gradient shows up in the loss of the following step.
    """
    world_size = torch.distributed.get_world_size()
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=LEARNING_RATE)

    # Give every rank its own data so that the gradient reductions are not no-ops.
    data_generator = torch.Generator(device="cuda").manual_seed(
        91011 + torch.distributed.get_rank()
    )
    model_input = torch.randn(DIM_SIZE, DIM_SIZE, device="cuda", generator=data_generator)
    target = torch.randn(DIM_SIZE, DIM_SIZE, device="cuda", generator=data_generator)

    reference_losses = []
    losses = []
    for _ in range(NUM_STEPS):
        reference_loss = mse_loss(reference_model(model_input), target)
        loss = mse_loss(fsdp_model(model_input), target)
        reference_losses.append(reference_loss.detach().clone())
        losses.append(loss.detach().clone())

        reference_loss.backward()
        loss.backward()
        for name, param in reference_model.named_parameters():
            group = expt_dp_group if ".experts." in name else None
            torch.distributed.all_reduce(param.grad, group=group)
            param.grad /= world_size

        reference_optimizer.step()
        fsdp_optimizer.step()
        reference_optimizer.zero_grad()
        fsdp_optimizer.zero_grad()

    torch.testing.assert_close(torch.stack(losses), torch.stack(reference_losses))


class TestExpertGradientScaling:
    """Expert gradient scaling under expert parallelism, which narrows the expert DP group."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("average_in_collective", [False, True])
    @pytest.mark.parametrize(
        "sharding_strategy, overlap_grad_reduce",
        [
            # Reduced bucket by bucket during the backward pass.
            (OPTIM_GRADS_PARAMS, True),
            (OPTIM_GRADS, True),
            (NO_SHARD, True),
            # Reduced in one bulk all-reduce or reduce-scatter before the optimizer step.
            (NO_SHARD, False),
            (OPTIM, False),
        ],
    )
    def test_loss_curve_matches_reference(
        self, sharding_strategy, overlap_grad_reduce, average_in_collective
    ):
        """
        Expert gradients have to be averaged over the data parallel group either way.

        An AVG reduction over the expert data parallel group only divides by that group's
        size, so with average_in_collective the residual scale between it and the data
        parallel size still has to be applied, or every expert gradient comes out too large
        by the expert parallel size.
        """
        device_mesh, expt_device_mesh = build_expert_parallel_meshes()
        torch.manual_seed(1234)
        reference_model = ToyExpertParallelModel().cuda()
        model = ToyExpertParallelModel().cuda()
        model.load_state_dict(reference_model.state_dict())

        fsdp_model = fully_shard_model(
            module=model,
            device_mesh=device_mesh,
            dp_shard_dim=DP_SHARD,
            tp_dim=TP,
            expt_device_mesh=expt_device_mesh,
            fsdp_unit_modules=[torch.nn.Linear],
            zero_dp_strategy=sharding_strategy,
            overlap_grad_reduce=overlap_grad_reduce,
            average_in_collective=average_in_collective,
        )
        expt_dp_group = expt_device_mesh[DP_SHARD].get_group()
        # The experts have to be reduced over the narrower expert group for the scale to matter.
        expert_groups = [
            group
            for group in fsdp_model.param_and_grad_buffer.parameter_groups
            if group.is_expert_param
        ]
        assert expert_groups, "No parameter was classified as an expert parameter."
        for group in expert_groups:
            assert group.main_grad_buffer.data_parallel_group.size() == expt_dp_group.size()

        optimizer = fully_shard_optimizer(
            torch.optim.SGD(fsdp_model.parameters(), lr=LEARNING_RATE)
        )
        assert_loss_curve_matches_reference(fsdp_model, optimizer, reference_model, expt_dp_group)
        destroy_device_mesh(device_mesh)
        destroy_device_mesh(expt_device_mesh)
