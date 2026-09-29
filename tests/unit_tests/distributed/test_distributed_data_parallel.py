# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
from packaging import version
from torch import testing

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


# Test model for testing DDP
class TestModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim * 4)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(input_dim * 4, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class TestDistributedDataParallel:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.3.0'),
        reason="Device mesh feature requires PyTorch 2.3 or later",
    )
    @pytest.mark.parametrize("dp_size", [2, 8])  # Test with 2 or 8 GPUs
    def test_ddp_with_dp_process_groups(self, dp_size):
        """Test that DDP works correctly with dp pgs from parallel state and user defined pgs."""

        # Skip test if we don't have enough GPUs
        world_size = torch.distributed.get_world_size()
        if world_size != dp_size:
            pytest.skip(f"This test requires {dp_size} GPUs, but only {world_size} are available")

        # Simple model config
        input_dim = 13
        output_dim = 17

        # Setup DDP config
        ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True, bucket_size=10000)

        # Create two identical models
        model1 = TestModel(input_dim=input_dim, output_dim=output_dim).cuda()
        model2 = TestModel(input_dim=input_dim, output_dim=output_dim).cuda()

        # Ensure identical weights
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p2.data.copy_(p1.data)

        # Wrap first model with default process groups
        transformer_config = TransformerConfig(
            num_attention_heads=1, num_layers=1, context_parallel_size=1
        )

        ddp_model1 = DistributedDataParallel(
            transformer_config, ddp_config=ddp_config, module=model1
        )

        # Initialize torch.distributed if not already initialized
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')

        # Create HyperCommGrid with dimension ep, pp, dp (reversed from device mesh order)
        grid = HyperCommGrid([1, 1, 1, 1, dp_size], ["tp", "cp", "ep", "pp", "dp"])

        # Create process groups config with ONLY dp group
        pg_collection = ProcessGroupCollection()

        pg_collection.dp = grid.create_pg("dp")
        pg_collection.dp_cp = grid.create_pg(["dp", "cp"])
        pg_collection.pp = grid.create_pg("pp")
        pg_collection.tp = grid.create_pg("tp")
        pg_collection.ep = grid.create_pg("ep")

        # Wrap second model with minimal process groups (only dp)
        ddp_model2 = DistributedDataParallel(
            transformer_config, ddp_config=ddp_config, module=model2, pg_collection=pg_collection
        )

        # Create identical inputs with integer values
        batch_size = 2
        input_data = torch.randint(0, 10, (batch_size, input_dim), device='cuda', dtype=torch.long)
        input_data = input_data.float()  # Convert to float for model compatibility

        # Forward pass
        out1 = ddp_model1(input_data)
        out2 = ddp_model2(input_data)

        testing.assert_close(out1, out2, rtol=0, atol=0)

        # Loss and backward
        loss1 = out1.sum()
        loss2 = out2.sum()

        loss1.backward()
        loss2.backward()

        # Check gradients are identical using torch.testing
        for p1, p2 in zip(ddp_model1.parameters(), ddp_model2.parameters()):
            if hasattr(p1, 'main_grad') and hasattr(p2, 'main_grad'):
                testing.assert_close(p1.main_grad, p2.main_grad, rtol=0, atol=0)


class TestExpertGradientScalingFactorGTPRemat:
    """Regression coverage for expert_gradient_scaling_factor's GTP_remat/EGTP_remat
    correction, folded directly into DDP's one-time prescale computation (Case 2,
    average_in_collective=False -- the only case reachable when GTP_remat is active).

    dp_cp_group.size() is deliberately shrunk by gtp_weight_remat_size (see the comment in
    DistributedDataParallel.__init__); the egtp/gtp correction is what keeps expert params on
    the same 1/dp_size target as dense params despite that shrink. Applied unconditionally,
    for any gtp vs egtp relationship -- both gtp>egtp and gtp<egtp need it.
    """

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _scaling_factors(self, gtp, egtp):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            gtp_remat_size=gtp,
            expert_gtp_remat_size=egtp,
        )

        class _TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dense = torch.nn.Linear(4, 4, bias=False)
                self.expert = torch.nn.Linear(4, 4, bias=False)
                self.expert.weight.allreduce = False

        # gtp_weight_remat_size / expert_gtp_weight_remat_size are DERIVED fields (resolved in
        # ModelParallelConfig.__post_init__ from the user-facing tensor_parallel_num_weight_shards
        # / expert_tensor_parallel_num_weight_shards knobs) -- passing them directly is a no-op;
        # this is also how production configs set them (core_transformer_config_from_args copies
        # the user-facing knob from CLI args), so exercising the same path here is intentional,
        # not just a workaround.
        config = TransformerConfig(
            num_attention_heads=1,
            num_layers=1,
            hidden_size=4,
            tensor_parallel_num_weight_shards=gtp,
            expert_tensor_parallel_num_weight_shards=egtp,
        )
        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=False, overlap_grad_reduce=False
        )
        ddp_model = DistributedDataParallel(config, ddp_config, _TinyModel().cuda())
        assert len(ddp_model.expert_parallel_buffers) == 1
        assert len(ddp_model.buffers) == 1
        return (
            ddp_model.buffers[0].gradient_scaling_factor,
            ddp_model.expert_parallel_buffers[0].gradient_scaling_factor,
            ddp_model.dp_cp_group.size(),
        )

    @pytest.mark.parametrize(
        "gtp,egtp,expected_correction",
        [
            (2, 1, 0.5),  # gtp > egtp: dense-only GTP, replicated experts
            (4, 1, 0.25),  # gtp > egtp: wider gap
            (4, 2, 0.5),  # gtp > egtp: both axes active, still gtp != egtp
            (2, 2, 1.0),  # matched (gtp == egtp): correction collapses to 1.0
            (1, 1, 1.0),  # GTP_remat inactive entirely
            (1, 2, 2.0),  # gtp < egtp: EGTP-sharded experts, GTP_remat inactive
            (2, 4, 2.0),  # gtp < egtp: wider gap
        ],
    )
    def test_expert_scaling_factor_gets_the_egtp_over_gtp_correction(
        self, gtp, egtp, expected_correction
    ):
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires 4 CUDA devices")
        dense_factor, expert_factor, dp_cp_size = self._scaling_factors(gtp, egtp)
        assert dense_factor == 1.0 / dp_cp_size
        assert expert_factor == (1.0 / dp_cp_size) * expected_correction, (
            f"gtp={gtp} egtp={egtp}: expert_gradient_scaling_factor={expert_factor} != "
            f"(1/dp_cp_group.size()={1.0/dp_cp_size}) * (egtp/gtp={expected_correction})"
        )
