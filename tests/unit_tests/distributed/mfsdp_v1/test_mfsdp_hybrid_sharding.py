# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Tests for Hybrid FSDP, where the optimizer state is sharded over DP-Outer on top of a
DP-Shard strategy that may keep the model weights replicated. The toy model and the
loss curve reference are shared with the non-uniform DP-Shard sharding tests.
"""

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import (
    fully_shard_model,
    fully_shard_optimizer,
)
from tests.unit_tests.distributed.mfsdp_v1.test_mfsdp_nonuniform_sharding import (
    DP_SHARD,
    LEARNING_RATE,
    OPTIM_GRADS,
    OPTIM_GRADS_PARAMS,
    TP,
    assert_loss_curve_matches_reference,
    build_replicated_models,
    destroy_device_mesh,
)
from tests.unit_tests.test_utilities import Utils

DP_OUTER = "dp_outer"
HSDP = "hsdp"
NO_SHARD = "no_shard"
OPTIM = "optim"

# (DP_OUTER, DP_SHARD) factorizations. Each case is skipped unless it spans the world, so
# that the same test file covers the 8-GPU CI world and smaller local worlds. DP_SHARD of
# one is included because it makes the DP-Shard reduction collective degenerate.
HYBRID_MESHES = [(2, 2), (4, 1), (2, 4), (4, 2), (8, 1)]


def build_hybrid_mesh(dp_outer: int, dp_shard: int):
    """Build a (DP-Outer, DP-Shard) device mesh, skipping if it does not span the world."""
    from torch.distributed.device_mesh import init_device_mesh

    world_size = torch.distributed.get_world_size()
    if dp_outer * dp_shard != world_size:
        pytest.skip(
            f"Mesh (dp_outer={dp_outer}, dp_shard={dp_shard}) does not span the "
            f"{world_size}-rank world."
        )

    device_mesh = init_device_mesh(
        "cuda", mesh_shape=(dp_outer, dp_shard, 1), mesh_dim_names=(DP_OUTER, DP_SHARD, TP)
    )
    # Hybrid FSDP shards the optimizer state over the flattened DP group.
    device_mesh[(DP_OUTER, DP_SHARD)]._flatten(HSDP)
    return device_mesh


def assert_reduction_schedule_matches_group_width(fsdp_model):
    """
    A bucket may reduce on every backward pass only if its DP-Shard group is wider than one rank.

    A one-rank reduction cannot carry a premul-sum multiplier, so reducing such a bucket per
    microbatch rescales its accumulating buffer repeatedly instead of once per cycle.
    """
    checked = 0
    # Walk the buffer's own mapping rather than fsdp_model.parameters(): once sharded, the
    # module exposes DTensors, which are not the keys this mapping was built from.
    for param, bucket_id in fsdp_model.param_and_grad_buffer.param_to_param_group.items():
        buffer = fsdp_model.grad_reduce_pipeline.get_fsdp_buffer(bucket_id)
        group_width = buffer.data_parallel_group.size()
        reduces_every_backward = fsdp_model._reduces_grad_every_backward(param)
        if group_width == 1:
            assert not reduces_every_backward, (
                f"bucket {bucket_id} has a single-rank DP-Shard group but is scheduled to "
                f"reduce on every backward pass, which rescales its accumulating buffer once "
                f"per microbatch"
            )
            checked += 1
    assert checked, "no single-rank DP-Shard bucket was exercised, so nothing was verified"


class TestHybridFsdpOverReplicatedWeights:
    """DP-Outer optimizer state sharding on top of DP-Shard strategies of either kind."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @staticmethod
    def hybrid_shard_model(model, device_mesh, dp_shard_strategy, experts_strategy=None):
        """Fully shard `model` with DP-Outer optimizer state sharding enabled."""
        hybrid_group = device_mesh[HSDP].get_group()
        return fully_shard_model(
            module=model,
            device_mesh=device_mesh,
            dp_shard_dim=DP_SHARD,
            tp_dim=TP,
            dp_outer_dim=DP_OUTER,
            hybrid_fsdp_group=hybrid_group,
            expt_device_mesh=device_mesh,
            hybrid_fsdp_expt_group=hybrid_group,
            fsdp_unit_modules=[torch.nn.Linear],
            zero_dp_strategy=dp_shard_strategy,
            expert_zero_dp_strategy=experts_strategy,
            outer_dp_sharding_strategy=OPTIM,
        )

    @pytest.mark.parametrize("mesh_shape", HYBRID_MESHES)
    @pytest.mark.parametrize("dp_shard_strategy", [OPTIM_GRADS, OPTIM_GRADS_PARAMS])
    def test_loss_curve_matches_reference(self, dp_shard_strategy, mesh_shape):
        """
        Sharding the optimizer state over DP-Outer must not alter the loss curve.

        DP-Outer sharding indexes every DP-wide shard by logical hybrid rank rather than by
        the rank ordering of the hybrid process group, so the weights and gradients it
        communicates have to be reassembled DP-Outer first and DP-Shard second. Doing it on
        the hybrid group directly permutes them, and a DP-Shard strategy that leaves the
        weights replicated is the case where nothing else forces the correct ordering.
        """
        device_mesh = build_hybrid_mesh(*mesh_shape)
        reference_model, model = build_replicated_models()

        fsdp_model = self.hybrid_shard_model(model, device_mesh, dp_shard_strategy)
        optimizer = fully_shard_optimizer(
            torch.optim.SGD(fsdp_model.parameters(), lr=LEARNING_RATE)
        )

        assert_loss_curve_matches_reference(fsdp_model, optimizer, reference_model)
        destroy_device_mesh(device_mesh)

    @pytest.mark.parametrize("mesh_shape", HYBRID_MESHES)
    def test_non_uniform_loss_curve_matches_reference(self, mesh_shape):
        """DP-Outer sharding must also hold when the two parameter classes differ."""
        device_mesh = build_hybrid_mesh(*mesh_shape)
        reference_model, model = build_replicated_models()

        fsdp_model = self.hybrid_shard_model(
            model, device_mesh, OPTIM_GRADS, experts_strategy=OPTIM_GRADS_PARAMS
        )
        optimizer = fully_shard_optimizer(
            torch.optim.SGD(fsdp_model.parameters(), lr=LEARNING_RATE)
        )

        assert_loss_curve_matches_reference(fsdp_model, optimizer, reference_model)
        destroy_device_mesh(device_mesh)

    # A DP-Shard width of one is the case under test, and it is reached both by sharding
    # everything over DP-Outer and by an expert group whose parallelism spans DP-Shard.
    @pytest.mark.parametrize("mesh_shape", [(4, 1), (8, 1)])
    def test_single_rank_dp_shard_group_reduces_once_per_cycle(self, mesh_shape):
        """A single-rank DP-Shard bucket must not be reduced on every backward pass."""
        device_mesh = build_hybrid_mesh(*mesh_shape)
        _, model = build_replicated_models()

        fsdp_model = self.hybrid_shard_model(
            model, device_mesh, OPTIM_GRADS, experts_strategy=OPTIM_GRADS_PARAMS
        )
        assert_reduction_schedule_matches_group_width(fsdp_model)
        destroy_device_mesh(device_mesh)

    @pytest.mark.parametrize("replicating_strategy", [NO_SHARD, OPTIM])
    @pytest.mark.parametrize("for_experts", [False, True])
    def test_gradient_replicating_dp_shard_strategy_is_rejected(
        self, replicating_strategy, for_experts
    ):
        """
        DP-Outer reduction consumes the DP-Shard gradient shard, so DP-Shard has to produce
        one. The error has to name the offending strategy whichever class configured it.
        """
        world_size = torch.distributed.get_world_size()
        if world_size < 2:
            pytest.skip("Requires at least 2 ranks to build a hybrid device mesh.")
        device_mesh = build_hybrid_mesh(2, world_size // 2)
        _, model = build_replicated_models()

        with pytest.raises(ValueError, match=replicating_strategy):
            self.hybrid_shard_model(
                model,
                device_mesh,
                OPTIM_GRADS_PARAMS if for_experts else replicating_strategy,
                experts_strategy=replicating_strategy if for_experts else None,
            )

        destroy_device_mesh(device_mesh)
