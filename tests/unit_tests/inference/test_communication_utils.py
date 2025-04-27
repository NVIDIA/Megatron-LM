import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.inference.communication_utils import (
    broadcast_from_last_pipeline_stage,
    recv_from_prev_pipeline_rank_,
    send_to_next_pipeline_rank,
)
from megatron.core.utils import is_torch_min_version
from tests.unit_tests.test_utilities import Utils


class TestCommunicationWithCustomPPGroup:
    """Test suite comparing communication with and without custom pp_group."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test parameters."""
        self.size = [16, 8]
        self.dtype = torch.float32

    @pytest.mark.skipif(
        not is_torch_min_version("2.4.0"),
        reason="torch.distributed.init_device_mesh requires torch >= 2.4.0",
    )
    @pytest.mark.parametrize("tp_size,pp_size", [(1, 8), (2, 4), (4, 2)])
    def test_broadcast_comparison(self, tp_size, pp_size):
        """Test broadcast with different parallel configurations."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=pp_size
        )

        rank = dist.get_rank()

        device = torch.device(f"cuda:{rank}")

        # Set a random seed based on rank for reproducibility but different values
        torch.manual_seed(rank)

        local_tensor = torch.randn(self.size, dtype=self.dtype, device=device)

        # Broadcast using global state
        tensor_received_global = broadcast_from_last_pipeline_stage(
            size=self.size, dtype=self.dtype, tensor=local_tensor
        )

        # Align with mcore minor-to-major order: tp-cp-dp-pp
        # Note init_device_mesh uses major-to-minor order, reverse the order of mcore
        mesh = dist.init_device_mesh("cuda", (pp_size, tp_size), mesh_dim_names=["pp", "tp"])
        pp_group = mesh.get_group(mesh_dim="pp")

        # Broadcast using custom pp_group
        tensor_received_custom = broadcast_from_last_pipeline_stage(
            size=self.size, dtype=self.dtype, tensor=local_tensor, pp_group=pp_group
        )

        # Synchronize before test
        dist.barrier()
        assert torch.allclose(
            tensor_received_global, tensor_received_custom
        ), "broadcast_from_last_pipeline_stage should be the same with or without custom pp_group"
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        not is_torch_min_version("2.4.0"),
        reason="torch.distributed.init_device_mesh requires torch >= 2.4.0",
    )
    @pytest.mark.parametrize("tp_size,pp_size", [(1, 8), (2, 4), (4, 2)])
    def test_send_recv(self, tp_size, pp_size):
        """Test send/recv in a ring pattern with different configs."""
        # Initialize model parallel for this test
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=pp_size
        )

        # Get rank info
        rank = dist.get_rank()

        # Set a random seed based on rank for reproducibility but different values
        torch.manual_seed(rank)

        # Create unique random data for this rank
        device = torch.device(f"cuda:{rank}")
        local_send_data = torch.randn(self.size, dtype=self.dtype, device=device)

        # Synchronize before test
        dist.barrier()

        # Send/recv using global state
        if not parallel_state.is_pipeline_first_stage():
            local_recv_buffer_global = torch.zeros(self.size, dtype=self.dtype, device=device)
            recv_from_prev_pipeline_rank_(recv_buffer=local_recv_buffer_global)
        else:
            local_recv_buffer_global = torch.zeros(self.size, dtype=self.dtype, device=device)

        if not parallel_state.is_pipeline_last_stage():
            send_to_next_pipeline_rank(tensor=local_send_data)

        dist.barrier()

        # Align with mcore minor-to-major order: tp-cp-dp-pp
        # Note init_device_mesh uses major-to-minor order, reverse the order of mcore
        mesh = dist.init_device_mesh("cuda", (pp_size, tp_size), mesh_dim_names=["pp", "tp"])
        pp_group = mesh.get_group(mesh_dim="pp")

        # Send/recv using custom pp_group
        if pp_group.rank() != 0:
            local_recv_buffer_custom = torch.zeros(self.size, dtype=self.dtype, device=device)
            recv_from_prev_pipeline_rank_(recv_buffer=local_recv_buffer_custom, pp_group=pp_group)
        else:
            local_recv_buffer_custom = torch.zeros(self.size, dtype=self.dtype, device=device)

        if pp_group.rank() != pp_group.size() - 1:
            send_to_next_pipeline_rank(tensor=local_send_data, pp_group=pp_group)

        dist.barrier()
        assert torch.allclose(
            local_recv_buffer_global, local_recv_buffer_custom
        ), "Custom and global recv buffers should be the same."
        Utils.destroy_model_parallel()
