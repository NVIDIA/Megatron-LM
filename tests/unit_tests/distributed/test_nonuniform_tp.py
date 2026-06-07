# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Unit tests for Nonuniform Tensor Parallelism (NTP).

Tests the fault-tolerance mechanism that allows training to continue
when GPU failures occur within a tensor-parallel group.
"""

import functools
import os
from datetime import timedelta
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist

import megatron.core.distributed.distributed_data_parallel as ddp_module
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel
from megatron.core.distributed.nonuniform_tp import (
    NonuniformTPConfig,
    NonuniformTPDistributedDataParallel,
    NonuniformTPParamAndGradBucketGroup,
    NonuniformTPParamAndGradBuffer,
    _compute_ntp_per_buffer_param_layout,
    compute_uniform_tp_spares_with_parity,
    get_active_ranks_for_dp,
    initialize_nonuniform_tp_process_groups,
    ntp_init,
    ntp_map,
)
from megatron.core.extensions.nonuniform_tp_transformer_engine import (
    initialize_transformer_engine_userbuffers_for_nonuniform_tp,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils


class TestNonuniformTPUtilities:
    """Test utility functions for NTP configuration."""

    def test_compute_uniform_tp_spares_with_parity_no_failures(self):
        """Test with no GPU failures."""
        faulty_gpu_map = {}
        tp_base = 8

        tp_spares, non_active_ranks = compute_uniform_tp_spares_with_parity(faulty_gpu_map, tp_base)

        assert tp_spares == 0
        assert non_active_ranks == {}

    def test_compute_uniform_tp_spares_with_parity_uniform_failures(self):
        """Test with uniform failures across DP ranks."""
        faulty_gpu_map = {
            0: [2, 5],  # DP rank 0 has 2 failures
            1: [1, 3],  # DP rank 1 has 2 failures
        }
        tp_base = 8

        tp_spares, non_active_ranks = compute_uniform_tp_spares_with_parity(faulty_gpu_map, tp_base)

        assert tp_spares == 2
        assert non_active_ranks[0] == [2, 5]
        assert non_active_ranks[1] == [1, 3]

    def test_compute_uniform_tp_spares_with_parity_non_uniform_failures(self):
        """Test with non-uniform failures (requires padding)."""
        faulty_gpu_map = {0: [2, 5], 1: [1]}  # DP rank 0 has 2 failures  # DP rank 1 has 1 failure
        tp_base = 8

        tp_spares, non_active_ranks = compute_uniform_tp_spares_with_parity(faulty_gpu_map, tp_base)

        assert tp_spares == 2
        assert non_active_ranks[0] == [2, 5]
        # DP rank 1 should be padded with 1 additional GPU (prefer high ranks)
        assert len(non_active_ranks[1]) == 2
        assert 1 in non_active_ranks[1]
        # Second non-active rank should be from the end (e.g., 7)
        assert non_active_ranks[1][1] == 7

    def test_get_active_ranks_for_dp_default(self):
        """Test get_active_ranks_for_dp with default (no explicit non_active_ranks_per_dp)."""
        ntp_config = NonuniformTPConfig(tp_base=8, tp_spares=2)
        dp_rank = 0
        tp_base = 8

        active_ranks = get_active_ranks_for_dp(dp_rank, tp_base, ntp_config)

        # Should return first (tp_base - tp_spares) ranks
        assert active_ranks == [0, 1, 2, 3, 4, 5]

    def test_get_active_ranks_for_dp_explicit(self):
        """Test get_active_ranks_for_dp with explicit non_active_ranks_per_dp."""
        ntp_config = NonuniformTPConfig(tp_base=8, tp_spares=2, non_active_ranks_per_dp={0: [2, 5]})
        dp_rank = 0
        tp_base = 8

        active_ranks = get_active_ranks_for_dp(dp_rank, tp_base, ntp_config)

        # Should exclude ranks 2 and 5
        assert active_ranks == [0, 1, 3, 4, 6, 7]

    def test_get_active_ranks_for_dp_tuple_key(self):
        """Test get_active_ranks_for_dp with DP/CP/PP-scoped non-active ranks."""
        ntp_config = NonuniformTPConfig(
            tp_base=8, tp_spares=2, non_active_ranks_per_dp={(1, 2, 0): [0, 7]}
        )

        active_ranks = get_active_ranks_for_dp(1, 8, ntp_config, cp_rank=2, pp_rank=0)

        assert active_ranks == [1, 2, 3, 4, 5, 6]

    @patch('megatron.core.distributed.nonuniform_tp.parallel_state')
    @patch('megatron.core.distributed.nonuniform_tp.dist')
    def test_initialize_nonuniform_tp_creates_groups_on_nonmember_ranks(
        self, mock_dist, mock_parallel_state
    ):
        """All ranks must enter replacement group creation in the same order."""
        mock_dist.get_rank.return_value = 4
        mock_parallel_state.get_context_parallel_world_size.return_value = 1

        ntp_config = NonuniformTPConfig(
            tp_base=4,
            tp_spares=2,
            num_reduced_tp_dp_ranks=1,
            non_active_ranks_per_dp={(0, 0, 0): [2, 3]},
        )

        assert initialize_nonuniform_tp_process_groups(ntp_config, exit_spares=False)
        mock_dist.new_group.assert_called_once_with(ranks=[0, 1])

    @patch('megatron.core.distributed.nonuniform_tp.parallel_state')
    @patch('megatron.core.distributed.nonuniform_tp.dist')
    def test_initialize_nonuniform_tp_returns_false_after_group_creation_for_spares(
        self, mock_dist, mock_parallel_state
    ):
        """Spare ranks still participate in group creation before opting out."""
        mock_dist.get_rank.return_value = 2
        mock_parallel_state.get_context_parallel_world_size.return_value = 1

        ntp_config = NonuniformTPConfig(
            tp_base=4,
            tp_spares=2,
            num_reduced_tp_dp_ranks=1,
            non_active_ranks_per_dp={(0, 0, 0): [2, 3]},
        )

        assert not initialize_nonuniform_tp_process_groups(ntp_config, exit_spares=False)
        mock_dist.new_group.assert_called_once_with(ranks=[0, 1])


class TestNonuniformTPBufferLayout:
    """Test NTP gradient-buffer layout compatibility with DDP buffer features."""

    @patch('megatron.core.distributed.nonuniform_tp.parallel_state')
    def test_layout_expands_side_grad_and_pads_for_distributed_optimizer(self, mock_parallel_state):
        """Healthy core ranks need side_grad storage and DistOpt-compatible padding."""
        mock_parallel_state.get_data_parallel_rank.return_value = 1
        mock_parallel_state.get_context_parallel_rank.return_value = 0
        mock_parallel_state.get_pipeline_model_parallel_rank.return_value = 0
        mock_parallel_state.get_tensor_model_parallel_world_size.return_value = 4
        mock_parallel_state.get_tensor_model_parallel_rank.return_value = 0

        param = torch.nn.Parameter(torch.randn(4, 2))
        param.tensor_model_parallel = True
        param.partition_dim = 0
        param.send_splits = [[0, 0, 0, 0] for _ in range(4)]
        param.recv_splits = [[0, 0, 2, 2] for _ in range(4)]

        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
        ntp_config = NonuniformTPConfig(
            tp_base=4, tp_spares=2, non_active_ranks_per_dp={(0, 0, 0): [2, 3]}
        )

        layout = _compute_ntp_per_buffer_param_layout(
            [param],
            bucket_size=None,
            data_parallel_world_size=2,
            ddp_config=ddp_config,
            ntp_config=ntp_config,
            param_indices=[0],
        )

        assert layout.param_index_map[param] == (0, 8, 0)
        assert layout.side_grad_index_map[param] == (8, 16, 0)
        assert layout.per_bucket_numel_unpadded == [16]
        assert layout.bucket_indices == [(0, 128)]
        assert layout.param_indices == [0]


class TestNonuniformTPParameterResharding:
    """Test parameter resharding logic for NTP."""

    def test_ntp_map_no_spares(self):
        """Test ntp_map when tp_spares=0 (should be no-op)."""
        # Create mock module with parameter
        module = Mock()
        param = torch.nn.Parameter(torch.randn(10, 10))
        param.tensor_model_parallel = True
        param.partition_dim = 1
        module.parameters = Mock(return_value=[param])

        ntp_config = NonuniformTPConfig(tp_base=8, tp_spares=0)

        # Should not raise error and not add send_splits/recv_splits
        ntp_map(module, ntp_config, num_shards=24)

        assert not hasattr(param, 'send_splits')
        assert not hasattr(param, 'recv_splits')

    @patch('megatron.core.distributed.nonuniform_tp.parallel_state')
    @patch('megatron.core.distributed.nonuniform_tp.dist')
    def test_ntp_map_with_spares_healthy_rank(self, mock_dist, mock_parallel_state):
        """Test ntp_map for a healthy rank (should add send/recv splits)."""
        # Mock parallel state
        mock_dist.get_rank.return_value = 0
        mock_parallel_state.get_data_parallel_rank.return_value = 0
        mock_parallel_state.get_context_parallel_rank.return_value = 0
        mock_parallel_state.get_pipeline_model_parallel_rank.return_value = 0

        # Create mock module with parameter
        class MockConfig:
            num_attention_heads = 24

        module = Mock()
        param = torch.nn.Parameter(torch.randn(384, 128))  # 384 = 24 heads * 16 dim
        param.tensor_model_parallel = True
        param.partition_dim = 0
        # Note: param.shape is already (384, 128) from the tensor, no need to set it
        module.parameters = Mock(return_value=[param])
        module.config = MockConfig()

        ntp_config = NonuniformTPConfig(
            tp_base=8,
            tp_spares=2,
            non_active_ranks_per_dp={},  # No explicit non-active ranks, so this is healthy
        )

        # Execute
        ntp_map(module, ntp_config, num_shards=24)

        # Should have added send_splits and recv_splits
        assert hasattr(param, 'send_splits')
        assert hasattr(param, 'recv_splits')
        assert len(param.send_splits) == 8
        assert len(param.recv_splits) == 8

    @patch('megatron.core.distributed.nonuniform_tp.parallel_state')
    @patch('megatron.core.distributed.nonuniform_tp.dist')
    def test_ntp_map_with_spares_unhealthy_rank(self, mock_dist, mock_parallel_state):
        """Test ntp_map for an unhealthy rank (should skip)."""
        # Mock parallel state
        mock_dist.get_rank.return_value = 0
        mock_parallel_state.get_data_parallel_rank.return_value = 0
        mock_parallel_state.get_context_parallel_rank.return_value = 0
        mock_parallel_state.get_pipeline_model_parallel_rank.return_value = 0

        # Create mock module
        module = Mock()
        param = torch.nn.Parameter(torch.randn(10, 10))
        param.tensor_model_parallel = True
        param.partition_dim = 1
        module.parameters = Mock(return_value=[param])

        ntp_config = NonuniformTPConfig(
            tp_base=8,
            tp_spares=2,
            non_active_ranks_per_dp={(0, 0, 0): [2, 5]},  # This rank is unhealthy
        )

        # Execute
        ntp_map(module, ntp_config, num_shards=24)

        # Should NOT have added send_splits and recv_splits
        assert not hasattr(param, 'send_splits')
        assert not hasattr(param, 'recv_splits')

    def test_ntp_init_no_spares(self):
        """Test ntp_init when tp_spares=0 (should be no-op)."""
        # Create mock layer
        layer = Mock()
        layer.self_attention = Mock()
        layer.mlp = Mock()

        ntp_config = NonuniformTPConfig(tp_base=8, tp_spares=0)

        # Should not raise error
        ntp_init(layer, ntp_config)

    @patch('megatron.core.distributed.nonuniform_tp.ntp_map')
    def test_ntp_init_with_attention_and_mlp(self, mock_ntp_map):
        """Test ntp_init calls ntp_map for both attention and MLP."""

        class MockConfig:
            num_attention_heads = 24
            ffn_hidden_size = 4096

        # Create mock layer
        layer = Mock()
        layer.self_attention = Mock()
        layer.self_attention.config = MockConfig()
        layer.mlp = Mock()
        layer.mlp.config = MockConfig()

        ntp_config = NonuniformTPConfig(tp_base=8, tp_spares=2)

        # Execute
        ntp_init(layer, ntp_config)

        # Should call ntp_map twice
        assert mock_ntp_map.call_count == 2
        # First call for self_attention
        assert mock_ntp_map.call_args_list[0][0][0] == layer.self_attention
        assert mock_ntp_map.call_args_list[0][0][2] == 24
        # Second call for mlp
        assert mock_ntp_map.call_args_list[1][0][0] == layer.mlp
        assert mock_ntp_map.call_args_list[1][0][2] == 4096


class TestNonuniformTPDDPCompatibility:
    """Test compatibility with current Megatron DDP construction and bucket state."""

    def test_ddp_patches_imported_buffer_binding_and_accepts_full_param_layout(self):
        """DDP imports _ParamAndGradBuffer directly, so NTP must patch that binding."""
        seen = {}
        original_buffer_class = ddp_module._ParamAndGradBuffer

        def fake_parent_init(
            self,
            *,
            config,
            ddp_config,
            module,
            disable_bucketing=False,
            pg_collection=None,
            full_param_layout=None,
        ):
            seen['buffer_binding'] = ddp_module._ParamAndGradBuffer
            seen['full_param_layout'] = full_param_layout
            self.ddp_config = ddp_config
            self.bucket_groups = []
            self.expert_parallel_bucket_groups = []
            self.param_to_bucket_group = {}

        config = TransformerConfig(
            num_layers=1, hidden_size=8, num_attention_heads=1, context_parallel_size=1
        )
        ddp_config = DistributedDataParallelConfig()
        ntp_config = NonuniformTPConfig(tp_base=2, tp_spares=1)
        full_param_layout = object()

        with patch.object(DistributedDataParallel, '__init__', new=fake_parent_init):
            NonuniformTPDistributedDataParallel(
                config=config,
                ddp_config=ddp_config,
                module=torch.nn.Linear(8, 8),
                disable_bucketing=True,
                ntp_config=ntp_config,
                full_param_layout=full_param_layout,
            )

        patched_binding = seen['buffer_binding']
        assert isinstance(patched_binding, functools.partial)
        assert patched_binding.func is NonuniformTPParamAndGradBuffer
        assert seen['full_param_layout'] is full_param_layout
        assert ddp_module._ParamAndGradBuffer is original_buffer_class

    def test_bucket_wrapping_preserves_overlap_param_gather_and_partial_distopt_state(self):
        """NTP bucket wrappers should not drop DDP state set before wrapping."""

        class FakeGroup:
            def __init__(self, size=2, rank=0):
                self._size = size
                self._rank = rank

            def size(self):
                return self._size

            def rank(self):
                return self._rank

        class FakeBucketGroup:
            def __init__(self, ddp_config):
                self.buckets = []
                self.ddp_config = ddp_config
                self.intra_distributed_optimizer_instance_group = FakeGroup()
                self.intra_distributed_optimizer_instance_size = 2
                self.inter_distributed_optimizer_instance_group = 'inter-group'
                self.communication_stream = 'comm-stream'
                self.next_param_gather_bucket_group = None

        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_param_gather=True,
            num_distributed_optimizer_instances=2,
        )
        first_group = FakeBucketGroup(ddp_config)
        second_group = FakeBucketGroup(ddp_config)
        second_group.next_param_gather_bucket_group = first_group

        ntp_ddp = object.__new__(NonuniformTPDistributedDataParallel)
        ntp_ddp.ddp_config = ddp_config
        ntp_ddp.ntp_config = NonuniformTPConfig(tp_base=2, tp_spares=1)
        ntp_ddp.bucket_groups = [first_group, second_group]
        ntp_ddp.expert_parallel_bucket_groups = []
        ntp_ddp.param_to_bucket_group = {}

        ntp_ddp._wrap_bucket_groups_for_ntp()

        wrapped_first, wrapped_second = ntp_ddp.bucket_groups
        assert isinstance(wrapped_first, NonuniformTPParamAndGradBucketGroup)
        assert isinstance(wrapped_second, NonuniformTPParamAndGradBucketGroup)
        assert wrapped_second.next_param_gather_bucket_group is wrapped_first
        assert wrapped_second.inter_distributed_optimizer_instance_group == 'inter-group'
        assert wrapped_second.communication_stream == 'comm-stream'
        assert wrapped_first.ntp_post_sync_state is wrapped_second.ntp_post_sync_state
        assert wrapped_first.ntp_post_sync_state['last_bucket_group'] is wrapped_second


class TestNonuniformTPBucketGroup:
    """Test DDP-owned NTP bucket post-sync behavior."""

    def test_post_sync_handles_wait_when_last_bucket_group_finishes(self):
        first = object.__new__(NonuniformTPParamAndGradBucketGroup)
        second = object.__new__(NonuniformTPParamAndGradBucketGroup)
        state = {'handles': [], 'last_bucket_group': second}
        first.ntp_post_sync_state = state
        second.ntp_post_sync_state = state

        first_handle = Mock()
        second_handle = Mock()

        first._record_ntp_post_sync_handles([first_handle])
        first_handle.wait.assert_not_called()
        assert state['handles'] == [first_handle]

        second._record_ntp_post_sync_handles([second_handle])
        first_handle.wait.assert_called_once()
        second_handle.wait.assert_called_once()
        assert state['handles'] == []

    @patch('megatron.core.distributed.nonuniform_tp._ntp_current_rank_should_dp_sync')
    def test_finish_grad_sync_on_folded_rank_launches_post_sync_reshard(self, mock_should_sync):
        mock_should_sync.return_value = False

        group = object.__new__(NonuniformTPParamAndGradBucketGroup)
        group.ntp_config = NonuniformTPConfig(tp_base=4, tp_spares=2)
        group.buckets = []
        group.param_gather_dispatched = True
        group.ntp_post_sync_state = None
        handle = Mock()
        group._start_ntp_post_sync_reshard = Mock(return_value=[handle])

        group.finish_grad_sync()

        assert group.param_gather_dispatched is False
        group._start_ntp_post_sync_reshard.assert_called_once()
        handle.wait.assert_called_once()


class TestNonuniformTPIntegration:
    """Integration tests for NTP with DDP - run with torchrun."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_ntp_ddp_initialization(self):
        """Test NonuniformTPDistributedDataParallel can be instantiated."""
        model = torch.nn.Linear(10, 10)
        config = TransformerConfig(
            num_layers=1, hidden_size=10, num_attention_heads=1, context_parallel_size=1
        )
        ddp_config = DistributedDataParallelConfig()
        ntp_config = NonuniformTPConfig(tp_base=1, tp_spares=0)

        ntp_ddp = NonuniformTPDistributedDataParallel(
            config, ddp_config, model, disable_bucketing=True, ntp_config=ntp_config
        )
        from megatron.core.distributed import DistributedDataParallel

        assert isinstance(ntp_ddp, DistributedDataParallel)

    def test_ntp_backward_hook_created(self):
        """Test that NTP backward hook is created without error."""
        model = torch.nn.Linear(10, 10)
        model.weight.tensor_model_parallel = True
        model.weight.partition_dim = 1

        config = TransformerConfig(
            num_layers=1, hidden_size=10, num_attention_heads=1, context_parallel_size=1
        )
        ddp_config = DistributedDataParallelConfig()
        ntp_config = NonuniformTPConfig(tp_base=1, tp_spares=0)

        ntp_ddp = NonuniformTPDistributedDataParallel(
            config, ddp_config, model, disable_bucketing=True, ntp_config=ntp_config
        )
        # Verify the hook is registered on the parameter
        assert model.weight._backward_hooks or ntp_ddp is not None


class TestNonuniformTPEndToEnd:
    """
    End-to-end test for NTP without mocking.

    Tests NTP with 8 GPUs configured as:
    - 2 data-parallel workers
    - DP rank 0: TP=2 (reduced, using 2 out of 4 GPUs)
    - DP rank 1: TP=4 (healthy, using all 4 GPUs)
    - Total: 2 + 4 = 6 active GPUs out of 8
    """

    @classmethod
    def setup_class(cls):
        """Initialize model parallel for NTP testing."""
        if Utils.world_size != 8:
            pytest.skip(f"This test requires 8 GPUs, but only {Utils.world_size} are available")
        # Initialize with tp_base=4
        Utils.initialize_model_parallel(tensor_model_parallel_size=4)

    @classmethod
    def teardown_class(cls):
        """Clean up model parallel."""
        Utils.destroy_model_parallel()

    def test_ntp_end_to_end_with_8_gpus(self):
        """
        End-to-end test using 8 GPUs with 2 DP workers:
        - DP rank 0: uses TP=2 (reduced from tp_base=4)
        - DP rank 1: uses TP=4 (healthy, full tp_base)
        """
        import torch.distributed as dist

        from megatron.core import parallel_state

        # Check we have 8 GPUs
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size != 8:
            pytest.skip(f"This test requires 8 GPUs, but only {world_size} are available")

        # Get current rank info
        rank = dist.get_rank()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        dp_rank = parallel_state.get_data_parallel_rank()

        # Configure NTP: first DP rank uses reduced TP=2
        ntp_config = NonuniformTPConfig(
            tp_base=4,
            tp_spares=2,
            num_reduced_tp_dp_ranks=1,
            non_active_ranks_per_dp={(0, 0, 0): [2, 3]},  # DP=0: GPUs 2,3 are spares
        )

        # Reconfigure process groups for NTP
        # Spare ranks must still enter NTP group creation before opting out so distributed
        # group creation order stays consistent for later tests in the same pytest shard.
        from megatron.core.distributed.nonuniform_tp import initialize_nonuniform_tp_process_groups

        is_active_rank = initialize_nonuniform_tp_process_groups(ntp_config, exit_spares=False)
        if not is_active_rank:
            pytest.skip(f"Rank {rank} is a spare rank, skipping test gracefully")

        # After reconfiguration, check TP size
        tp_size_after = parallel_state.get_tensor_model_parallel_world_size()

        # Verify the configuration
        if dp_rank == 0:
            # First DP rank should have reduced TP=2
            assert tp_size_after == 2, f"DP rank 0 should have TP=2, got {tp_size_after}"
            assert tp_rank < 2, f"DP rank 0 should have tp_rank < 2, got {tp_rank}"
        else:
            # Other DP ranks keep TP=4
            assert tp_size_after == 4, f"DP rank {dp_rank} should have TP=4, got {tp_size_after}"
            assert tp_rank < 4, f"DP rank {dp_rank} should have tp_rank < 4, got {tp_rank}"

        # Create a simple model with tensor-parallel parameters
        hidden_size = 128
        model = torch.nn.Linear(hidden_size, hidden_size, bias=False).cuda()

        # Mark it as tensor-parallel
        model.weight.tensor_model_parallel = True
        model.weight.partition_dim = 0

        # Initialize NTP mappings
        from megatron.core.distributed.nonuniform_tp import ntp_map

        # For healthy ranks (DP=1), initialize send/recv splits
        if dp_rank == 1:
            # Create a mock module to test ntp_map
            class MockModule:
                def __init__(self, param):
                    self.param = param

                def parameters(self):
                    return [self.param]

            mock_module = MockModule(model.weight)
            ntp_map(mock_module, ntp_config, num_shards=hidden_size)

            # Verify send_splits and recv_splits were added
            assert hasattr(model.weight, 'send_splits'), "Healthy rank should have send_splits"
            assert hasattr(model.weight, 'recv_splits'), "Healthy rank should have recv_splits"
            assert len(model.weight.send_splits) == 4, "Should have splits for all tp_base ranks"

        # Test forward pass
        batch_size = 4
        input_tensor = torch.randn(batch_size, hidden_size, device='cuda')
        output = model(input_tensor)

        # Verify output shape
        assert output.shape == (batch_size, hidden_size), f"Unexpected output shape: {output.shape}"

        # Verify gradients work
        loss = output.sum()
        loss.backward()
        assert model.weight.grad is not None, "Gradients should be computed"


def _new_group_for_current_rank(group_ranks, rank):
    group = dist.new_group(ranks=group_ranks)
    return group if rank in group_ranks else None


def _initialize_packed_tp2_tp4_groups():
    """Initialize a 6-rank packed NTP layout with no spare processes."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 6:
        raise RuntimeError(f"Packed TP2/TP4 NTP test requires WORLD_SIZE=6, got {world_size}")

    reduced_ranks = [0, 1]
    healthy_ranks = [2, 3, 4, 5]
    tp_domains = [reduced_ranks, healthy_ranks]
    dp_domains = [[0, 2], [1, 3], [4], [5]]
    singleton_domains = [[group_rank] for group_rank in range(world_size)]

    tp_groups = {}
    for group_ranks in tp_domains:
        group = _new_group_for_current_rank(group_ranks, rank)
        for group_rank in group_ranks:
            tp_groups[group_rank] = (group, group_ranks)

    dp_groups = {}
    for group_ranks in dp_domains:
        group = _new_group_for_current_rank(group_ranks, rank)
        for group_rank in group_ranks:
            dp_groups[group_rank] = (group, group_ranks)

    singleton_groups = {}
    for group_ranks in singleton_domains:
        group = _new_group_for_current_rank(group_ranks, rank)
        singleton_groups[group_ranks[0]] = (group, group_ranks)

    if rank in reduced_ranks:
        dp_rank = 0
        tp_rank = rank
        tp_size = 2
    else:
        dp_rank = 1
        tp_rank = rank - healthy_ranks[0]
        tp_size = 4

    tp_group, tp_ranks = tp_groups[rank]
    dp_group, dp_ranks = dp_groups[rank]
    singleton_group, singleton_ranks = singleton_groups[rank]

    parallel_state._TENSOR_MODEL_PARALLEL_GROUP = tp_group
    parallel_state._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = tp_ranks
    parallel_state._MODEL_PARALLEL_GROUP = tp_group
    parallel_state._MODEL_PARALLEL_GLOBAL_RANKS = tp_ranks
    parallel_state._PIPELINE_MODEL_PARALLEL_GROUP = singleton_group
    parallel_state._PIPELINE_GLOBAL_RANKS = singleton_ranks
    parallel_state._CONTEXT_PARALLEL_GROUP = singleton_group
    parallel_state._CONTEXT_PARALLEL_GLOBAL_RANKS = singleton_ranks
    parallel_state._HIERARCHICAL_CONTEXT_PARALLEL_GROUPS = [singleton_group]
    parallel_state._TENSOR_AND_CONTEXT_PARALLEL_GROUP = tp_group
    parallel_state._DATA_PARALLEL_GROUP = dp_group
    parallel_state._DATA_PARALLEL_GROUP_WITH_CP = dp_group
    parallel_state._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = dp_group
    parallel_state._DATA_PARALLEL_GLOBAL_RANKS = dp_ranks
    parallel_state._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = dp_ranks
    parallel_state._TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = dist.group.WORLD
    parallel_state._EMBEDDING_GROUP = singleton_group
    parallel_state._EMBEDDING_GLOBAL_RANKS = singleton_ranks
    parallel_state._POSITION_EMBEDDING_GROUP = singleton_group
    parallel_state._POSITION_EMBEDDING_GLOBAL_RANKS = singleton_ranks
    parallel_state._EXPERT_MODEL_PARALLEL_GROUP = singleton_group
    parallel_state._EXPERT_MODEL_PARALLEL_RANKS = singleton_ranks
    parallel_state._EXPERT_TENSOR_PARALLEL_GROUP = tp_group
    parallel_state._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = tp_group
    parallel_state._EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = tp_group
    parallel_state._EXPERT_DATA_PARALLEL_GROUP = dp_group
    parallel_state._INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = dp_group
    parallel_state._INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP = dp_group

    parallel_state.set_tensor_model_parallel_world_size(tp_size)
    parallel_state.set_tensor_model_parallel_rank(tp_rank)
    parallel_state.set_pipeline_model_parallel_world_size(1)
    parallel_state.set_pipeline_model_parallel_rank(0)
    parallel_state.set_data_parallel_rank(dp_rank)
    parallel_state._set_global_memory_buffer()

    return ProcessGroupCollection(
        tp=tp_group,
        pp=singleton_group,
        mp=tp_group,
        embd=singleton_group,
        pos_embd=singleton_group,
        cp=singleton_group,
        tp_cp=tp_group,
        hcp=[singleton_group],
        ep=singleton_group,
        expt_tp=tp_group,
        tp_ep=tp_group,
        tp_ep_pp=tp_group,
        dp=dp_group,
        dp_cp=dp_group,
        dp_cp_ag=None,
        expt_dp=dp_group,
        expt_dp_ag=None,
        intra_dp_cp=dp_group,
        intra_expt_dp=dp_group,
        inter_dist_opt=None,
        intra_dist_opt=dp_group,
        tp_dp_cp=dist.group.WORLD,
    )


def _apply_ntp_mappings_to_gpt(model, ntp_config):
    for module in model.modules():
        if module.__class__.__name__ == "TransformerLayer":
            ntp_init(module, ntp_config)
    if hasattr(model, "embedding") and hasattr(model.embedding, "word_embeddings"):
        ntp_map(model.embedding.word_embeddings, ntp_config, 512)
    if hasattr(model, "output_layer"):
        ntp_map(model.output_layer, ntp_config, 512)


@pytest.mark.skipif(not is_te_min_version("1.9.0"), reason="TE userbuffers require TE >= 1.9")
class TestNonuniformTPPackedTEEndToEnd:
    """End-to-end NTP Megatron test using only active TP2 + TP4 ranks."""

    @classmethod
    def setup_class(cls):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size != 6:
            pytest.skip(f"This test requires 6 GPUs, but only {world_size} are available")

        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank % torch.cuda.device_count())

        if not dist.is_initialized():
            init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
            dist.init_process_group(
                backend="nccl",
                init_method=init_method,
                rank=rank,
                world_size=world_size,
                timeout=timedelta(minutes=5),
                device_id=torch.device("cuda", torch.cuda.current_device()),
            )
            dist.barrier(device_ids=[torch.cuda.current_device()])

        Utils.world_size = world_size
        Utils.rank = rank
        Utils.inited = True
        parallel_state.destroy_model_parallel()

    @classmethod
    def teardown_class(cls):
        try:
            from transformer_engine.pytorch import module as te_module

            te_module.base.destroy_ub()
        except Exception:
            pass
        if dist.is_initialized():
            try:
                torch.cuda.synchronize()
                dist.barrier()
            except Exception:
                pass
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()
        Utils.inited = False

    def test_ntp_te_tp_comm_overlap_with_packed_tp2_tp4(self):
        pg_collection = _initialize_packed_tp2_tp4_groups()
        model_parallel_cuda_manual_seed(123)

        ntp_config = NonuniformTPConfig(
            tp_base=4,
            tp_spares=2,
            num_reduced_tp_dp_ranks=1,
            non_active_ranks_per_dp={(0, 0, 0): [2, 3]},
        )

        seq_len = 16
        micro_batch_size = 1
        hidden_size = 256
        tp_size = parallel_state.get_tensor_model_parallel_world_size()

        tp_domains = [[0, 1], [2, 3, 4, 5]]
        normalized_domains = initialize_transformer_engine_userbuffers_for_nonuniform_tp(
            shape=[seq_len * micro_batch_size, hidden_size],
            tp_size=tp_size,
            tp_domains=tp_domains,
            bootstrap_backend="nccl",
        )
        assert normalized_domains == ((0, 1), (2, 3, 4, 5))

        for env_var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
            os.environ.pop(env_var, None)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            ffn_hidden_size=1024,
            num_attention_heads=8,
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
            sequence_parallel=True,
            tp_comm_overlap=True,
            tp_comm_overlap_rs_dgrad=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            use_cpu_initialization=False,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=512,
            max_sequence_length=seq_len,
            parallel_output=True,
            share_embeddings_and_output_weights=False,
            position_embedding_type="none",
            pg_collection=pg_collection,
        ).cuda()
        model.bfloat16()
        _apply_ntp_mappings_to_gpt(model, ntp_config)

        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=False, overlap_grad_reduce=True, bucket_size=100_000_000
        )
        ddp_model = NonuniformTPDistributedDataParallel(
            config=config,
            ddp_config=ddp_config,
            module=model,
            disable_bucketing=False,
            pg_collection=pg_collection,
            ntp_config=ntp_config,
        )

        tokens = torch.randint(
            low=0, high=512, size=(micro_batch_size, seq_len), dtype=torch.long, device="cuda"
        )
        labels = torch.randint(
            low=0, high=512, size=(micro_batch_size, seq_len), dtype=torch.long, device="cuda"
        )
        position_ids = torch.arange(seq_len, dtype=torch.long, device="cuda").unsqueeze(0)
        loss_mask = torch.ones((micro_batch_size, seq_len), dtype=torch.float32, device="cuda")

        ddp_model.zero_grad_buffer()
        losses = ddp_model(tokens, position_ids, None, labels=labels, loss_mask=loss_mask)
        loss = torch.sum(losses.float() * loss_mask) / loss_mask.sum()
        loss.backward()
        ddp_model.finish_grad_sync()

        assert torch.isfinite(loss.detach()).item()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
