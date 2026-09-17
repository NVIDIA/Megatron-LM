# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Tests for non-uniform DP-Shard sharding, where expert and non-expert parameters are
sharded by different ZeRO strategies.
"""

import logging

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
from megatron.core.distributed.fsdp.src.megatron_fsdp.utils import (
    all_sharding_strategies_in,
    any_sharding_strategy_in,
    get_sharding_strategies_in_use,
    get_sharding_strategy,
)
from tests.unit_tests.test_utilities import Utils

logger = logging.getLogger(__name__)

DP_SHARD = "dp_shard"
TP = "tp"
OPTIM_GRADS = "optim_grads"
OPTIM_GRADS_PARAMS = "optim_grads_params"

DIM_SIZE = 8
NUM_EXPERTS = 2
NUM_STEPS = 3
LEARNING_RATE = 0.1


class ToyMoEModel(torch.nn.Module):
    """
    Toy model owning both expert and non-expert parameters.

    Megatron-FSDP classifies a parameter as an expert parameter when ".experts." appears in
    its name, which is what places the two classes in separate parameter groups and lets
    them be sharded by different strategies.
    """

    def __init__(self, dim: int = DIM_SIZE, num_experts: int = NUM_EXPERTS):
        super().__init__()
        self.dense = torch.nn.Linear(dim, dim)
        self.experts = torch.nn.ModuleList([torch.nn.Linear(dim, dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        # Route to every expert so that all expert parameters receive a gradient.
        return sum(expert(x) for expert in self.experts) / len(self.experts)


def destroy_device_mesh(device_mesh):
    """Release the global DeviceMesh state so the next test can build its own mesh."""
    del device_mesh
    try:
        from torch.distributed.device_mesh import _mesh_resources

        _mesh_resources.child_to_root_mapping.clear()
        _mesh_resources.root_to_flatten_mapping.clear()
        _mesh_resources.mesh_stack.clear()
        _mesh_resources.mesh_dim_group_options.clear()
        _mesh_resources.flatten_name_to_root_dims.clear()
    except Exception as e:
        # Global _MeshEnv is on a convoluted deprecation path.
        logger.warning(f"Did not clean the deprecated DeviceMesh global state. Skipping...\n{e}")


def build_flat_mesh():
    """Build a DP-Shard-only device mesh spanning the world."""
    from torch.distributed.device_mesh import init_device_mesh

    # Megatron-FSDP requires a TP dimension, trivial here, to place its DTensor shards.
    return init_device_mesh(
        "cuda", mesh_shape=(torch.distributed.get_world_size(), 1), mesh_dim_names=(DP_SHARD, TP)
    )


def build_replicated_models(seed: int = 1234):
    """Build identically initialized reference and test models."""
    torch.manual_seed(seed)
    reference_model = ToyMoEModel().cuda()
    model = ToyMoEModel().cuda()
    model.load_state_dict(reference_model.state_dict())
    return reference_model, model


def assert_loss_curve_matches_reference(fsdp_model, fsdp_optimizer, reference_model):
    """
    Train the sharded model against a replicated reference and compare their loss curves.

    The reference is an unwrapped replica whose gradients are averaged by hand, so any
    sharding scheme that assembles the wrong weights or applies the wrong gradient scaling
    diverges from it on the step after the error is introduced.
    """
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=LEARNING_RATE)

    # Give every rank its own data so that the gradient average over the data parallel group
    # is not a no-op, which is what makes gradient scaling errors observable in the loss.
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
        # The reference model is not wrapped, so average its rank-local gradients by hand to
        # match the data parallel gradient reduction that Megatron-FSDP performs.
        for param in reference_model.parameters():
            torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.AVG)

        reference_optimizer.step()
        fsdp_optimizer.step()
        reference_optimizer.zero_grad()
        fsdp_optimizer.zero_grad()

    torch.testing.assert_close(torch.stack(losses), torch.stack(reference_losses))


class TestShardingStrategyResolution:
    """Resolution of the DP-Shard sharding strategy that applies to a class of parameters."""

    @staticmethod
    def make_config(strategy, experts_strategy=None):
        """Build a DDP config carrying the given pair of DP-Shard strategies."""
        return DistributedDataParallelConfig(
            data_parallel_sharding_strategy=strategy,
            expert_data_parallel_sharding_strategy=experts_strategy,
        )

    def test_experts_strategy_applies_to_expert_parameters_only(self):
        config = self.make_config(OPTIM_GRADS, experts_strategy=OPTIM_GRADS_PARAMS)
        assert get_sharding_strategy(config, is_expert_param=False) == OPTIM_GRADS
        assert get_sharding_strategy(config, is_expert_param=True) == OPTIM_GRADS_PARAMS

    def test_unset_experts_strategy_falls_back_to_the_common_strategy(self):
        config = self.make_config(OPTIM_GRADS)
        assert get_sharding_strategy(config, is_expert_param=False) == OPTIM_GRADS
        assert get_sharding_strategy(config, is_expert_param=True) == OPTIM_GRADS

    def test_strategies_in_use_reports_every_configured_strategy(self):
        uniform = self.make_config(OPTIM_GRADS)
        assert get_sharding_strategies_in_use(uniform) == (OPTIM_GRADS,)

        non_uniform = self.make_config(OPTIM_GRADS, experts_strategy=OPTIM_GRADS_PARAMS)
        assert set(get_sharding_strategies_in_use(non_uniform)) == {OPTIM_GRADS, OPTIM_GRADS_PARAMS}
        # Machinery needed by either class of parameters has to be enabled for the model.
        assert any_sharding_strategy_in(non_uniform, [OPTIM_GRADS_PARAMS])
        assert not all_sharding_strategies_in(non_uniform, [OPTIM_GRADS_PARAMS])
        assert all_sharding_strategies_in(non_uniform, [OPTIM_GRADS, OPTIM_GRADS_PARAMS])

    @pytest.mark.parametrize(
        "field_name", ["data_parallel_sharding_strategy", "expert_data_parallel_sharding_strategy"]
    )
    def test_invalid_strategy_is_rejected(self, field_name):
        with pytest.raises(ValueError, match=f"Invalid {field_name}"):
            DistributedDataParallelConfig(**{field_name: "zero_3"})


class TestNonUniformSharding:
    """DP-Shard strategies that differ between expert and non-expert parameters."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "dense_strategy, experts_strategy, expect_release_hooks",
        [
            (OPTIM_GRADS_PARAMS, OPTIM_GRADS_PARAMS, True),
            (OPTIM_GRADS, OPTIM_GRADS_PARAMS, True),
            (OPTIM_GRADS_PARAMS, OPTIM_GRADS, True),
            (OPTIM_GRADS, OPTIM_GRADS, False),
        ],
    )
    def test_weight_release_hook_follows_either_parameter_class(
        self, dense_strategy, experts_strategy, expect_release_hooks
    ):
        """
        The post-forward weight release hook is needed if either parameter class is fully
        sharded.

        Only 'optim_grads_params' gathers weights for each microbatch, so only it needs them
        released afterwards. An FSDP unit module can own both classes at once, so keying the
        hook off the non-expert strategy alone would leave fully sharded expert weights
        gathered for the whole step, silently raising peak memory. Which parameters actually
        get released is then decided per parameter group inside the hook.
        """
        device_mesh = build_flat_mesh()
        _, model = build_replicated_models()

        fsdp_model = fully_shard_model(
            module=model,
            device_mesh=device_mesh,
            dp_shard_dim=DP_SHARD,
            tp_dim=TP,
            expt_device_mesh=device_mesh,
            fsdp_unit_modules=[torch.nn.Linear],
            zero_dp_strategy=dense_strategy,
            expert_zero_dp_strategy=experts_strategy,
        )

        release_hooks = [
            name for name in fsdp_model.forward_hooks if name.startswith("release module")
        ]
        assert bool(release_hooks) == expect_release_hooks, (
            f"dense={dense_strategy} experts={experts_strategy} registered "
            f"{len(release_hooks)} weight release hooks"
        )
        destroy_device_mesh(device_mesh)

    @pytest.mark.parametrize("experts_strategy", [OPTIM_GRADS, OPTIM_GRADS_PARAMS])
    @pytest.mark.parametrize("dense_strategy", [OPTIM_GRADS, OPTIM_GRADS_PARAMS])
    def test_loss_curve_matches_reference(self, dense_strategy, experts_strategy):
        """Sharding the two parameter classes differently must not alter the loss curve."""
        device_mesh = build_flat_mesh()
        reference_model, model = build_replicated_models()

        fsdp_model = fully_shard_model(
            module=model,
            device_mesh=device_mesh,
            dp_shard_dim=DP_SHARD,
            tp_dim=TP,
            # Expert parameters are data parallel over the same group here (no expert
            # parallelism), which isolates the sharding strategy as the only difference.
            expt_device_mesh=device_mesh,
            fsdp_unit_modules=[torch.nn.Linear],
            zero_dp_strategy=dense_strategy,
            expert_zero_dp_strategy=experts_strategy,
        )
        optimizer = fully_shard_optimizer(
            torch.optim.SGD(fsdp_model.parameters(), lr=LEARNING_RATE)
        )

        assert_loss_curve_matches_reference(fsdp_model, optimizer, reference_model)
        destroy_device_mesh(device_mesh)

    @pytest.mark.parametrize("experts_strategy", [OPTIM_GRADS, OPTIM_GRADS_PARAMS])
    @pytest.mark.parametrize("dense_strategy", [OPTIM_GRADS, OPTIM_GRADS_PARAMS])
    def test_buffers_are_distributed_per_parameter_class(self, dense_strategy, experts_strategy):
        """
        Whether a buffer is sharded is a property of its parameter group, not of the model.

        Only 'optim_grads_params' shards the model weights, so a model that mixes it with a
        strategy that does not must end up with both sharded and unsharded weight buffers.
        """
        device_mesh = build_flat_mesh()
        _, model = build_replicated_models()

        fsdp_model = fully_shard_model(
            module=model,
            device_mesh=device_mesh,
            dp_shard_dim=DP_SHARD,
            tp_dim=TP,
            expt_device_mesh=device_mesh,
            fsdp_unit_modules=[torch.nn.Linear],
            zero_dp_strategy=dense_strategy,
            expert_zero_dp_strategy=experts_strategy,
        )

        seen_weights_sharded = {}
        for group in fsdp_model.param_and_grad_buffer.parameter_groups:
            if group.model_weight_buffer is None:
                continue
            expected_strategy = experts_strategy if group.is_expert_param else dense_strategy
            assert group.sharding_strategy == expected_strategy
            seen_weights_sharded.setdefault(group.is_expert_param, set()).add(
                group.model_weight_buffer.is_data_distributed
            )

        for is_expert_param, weights_sharded in seen_weights_sharded.items():
            strategy = experts_strategy if is_expert_param else dense_strategy
            assert weights_sharded == {strategy == OPTIM_GRADS_PARAMS}, (
                f"{'expert' if is_expert_param else 'non-expert'} weight buffers "
                f"{weights_sharded} do not match strategy {strategy}"
            )

        destroy_device_mesh(device_mesh)

    def test_main_weight_shards_tile_the_parameter(self):
        """
        Every rank must report the location of its own main weight shard in the parameter.

        Quantized parameter formats place their scaling blocks using this offset, and the
        compute weight buffer that reports it is itself unsharded under 'optim_grads', where
        asking it to locate the whole item would have every rank claim that its shard starts
        at the beginning of the parameter.
        """
        device_mesh = build_flat_mesh()
        _, model = build_replicated_models()

        fsdp_model = fully_shard_model(
            module=model,
            device_mesh=device_mesh,
            dp_shard_dim=DP_SHARD,
            tp_dim=TP,
            expt_device_mesh=device_mesh,
            fsdp_unit_modules=[torch.nn.Linear],
            # Shards the main weights while leaving the compute weights replicated, which is
            # the configuration in which the two buffers disagree about the shard location.
            zero_dp_strategy=OPTIM_GRADS,
        )

        checked_any_parameter = False
        for group in fsdp_model.param_and_grad_buffer.parameter_groups:
            weight_buffer = group.model_weight_buffer
            main_weight_buffer = group.main_weight_buffer
            if weight_buffer is None or main_weight_buffer is None:
                continue
            assert not weight_buffer.is_data_distributed
            if not main_weight_buffer.is_data_distributed:
                continue
            data_parallel_group = main_weight_buffer.data_parallel_group

            for param in group.params:
                item_id = main_weight_buffer.param_idx[param]
                item_size = weight_buffer.item_index_map[item_id].size
                shard_start, shard_end = weight_buffer.locate_item_shard_in_global_item(item_id)

                # The reported extent has to describe the shard that is handed out.
                assert (
                    shard_end - shard_start
                    == weight_buffer.get_item(item_id, only_shard=True).numel()
                )

                extents = [None] * data_parallel_group.size()
                torch.distributed.all_gather_object(
                    extents, (shard_start, shard_end), group=data_parallel_group
                )
                # The shards tile the parameter exactly: sorted, contiguous, and complete.
                covered = 0
                for start, end in sorted(extent for extent in extents if extent[0] != extent[1]):
                    assert start == covered, f"gap or overlap at {start}, expected {covered}"
                    covered = end
                assert covered == item_size, f"shards cover {covered} of {item_size} elements"
                checked_any_parameter = True

        assert checked_any_parameter, "No sharded main weight buffer was checked."
        destroy_device_mesh(device_mesh)
