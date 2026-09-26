# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.inference.communication_utils import (
    broadcast_float_list,
    broadcast_from_last_pipeline_stage,
    broadcast_int_list,
    broadcast_list,
    broadcast_tensor,
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

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

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

        # Initialize torch.distributed if not already initialized
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')

        # Note: HyperCommGrid uses minor-to-major order (tp, pp), which is reverse of device mesh
        grid = HyperCommGrid([tp_size, pp_size], ["tp", "pp"])
        pp_group = grid.create_pg("pp")

        # Broadcast using custom pp_group
        tensor_received_custom = broadcast_from_last_pipeline_stage(
            size=self.size, dtype=self.dtype, tensor=local_tensor, pp_group=pp_group
        )

        # Synchronize before test
        dist.barrier()
        assert torch.allclose(
            tensor_received_global, tensor_received_custom
        ), "broadcast_from_last_pipeline_stage should be the same with or without custom pp_group"

        grid.destroy()

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

        # Initialize torch.distributed if not already initialized
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')

        # Note: HyperCommGrid uses minor-to-major order (tp, pp), which is reverse of device mesh
        grid = HyperCommGrid([tp_size, pp_size], ["tp", "pp"])
        pp_group = grid.create_pg("pp")

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

        grid.destroy()


class TestBroadcastWithCustomGroup:
    """Compare broadcast helpers with MPU fallback vs an explicit process group."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test parameters."""
        self.size = [16, 8]
        self.dtype = torch.float32

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _create_mp_group(self, tp_size, pp_size):
        """Build a HyperCommGrid MP group matching initialize_model_parallel(tp, pp)."""
        # Note: HyperCommGrid uses minor-to-major order (tp, pp), which is reverse of device mesh
        grid = HyperCommGrid([tp_size, pp_size], ["tp", "pp"])
        return grid, grid.create_pg(["tp", "pp"])

    @pytest.mark.skipif(
        not is_torch_min_version("2.4.0"),
        reason="torch.distributed.init_device_mesh requires torch >= 2.4.0",
    )
    @pytest.mark.parametrize("tp_size,pp_size", [(1, 8), (2, 4), (4, 2)])
    def test_broadcast_tensor_comparison(self, tp_size, pp_size):
        """broadcast_tensor with an explicit MP group matches the MPU fallback."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=pp_size
        )

        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")
        torch.manual_seed(rank)

        src_rank = parallel_state.get_model_parallel_src_rank()
        local_tensor = (
            torch.randn(self.size, dtype=self.dtype, device=device)
            if rank == src_rank
            else None
        )

        tensor_received_global = broadcast_tensor(
            self.size, self.dtype, tensor=local_tensor, data_parallel=True
        )

        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')

        grid, mp_group = self._create_mp_group(tp_size, pp_size)
        tensor_received_custom = broadcast_tensor(
            self.size, self.dtype, tensor=local_tensor, data_parallel=True, group=mp_group
        )

        dist.barrier()
        assert torch.allclose(
            tensor_received_global, tensor_received_custom
        ), "broadcast_tensor should be the same with or without an explicit group"

        grid.destroy()

    @pytest.mark.skipif(
        not is_torch_min_version("2.4.0"),
        reason="torch.distributed.init_device_mesh requires torch >= 2.4.0",
    )
    @pytest.mark.parametrize("tp_size,pp_size", [(1, 8), (2, 4), (4, 2)])
    def test_broadcast_list_comparison(self, tp_size, pp_size):
        """broadcast_list / int / float helpers match MPU fallback with an explicit group."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=pp_size
        )

        src_rank = parallel_state.get_model_parallel_src_rank()
        int_values = [1, 2, 3, 4] if dist.get_rank() == src_rank else None
        float_values = [1.5, 2.5, 3.5, 4.5] if dist.get_rank() == src_rank else None

        int_global = broadcast_int_list(4, int_list=int_values, data_parallel=True)
        float_global = broadcast_float_list(4, float_list=float_values, data_parallel=True)
        list_global = broadcast_list(
            4, torch.int64, list_values=int_values, data_parallel=True
        )

        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')

        grid, mp_group = self._create_mp_group(tp_size, pp_size)
        int_custom = broadcast_int_list(
            4, int_list=int_values, data_parallel=True, group=mp_group
        )
        float_custom = broadcast_float_list(
            4, float_list=float_values, data_parallel=True, group=mp_group
        )
        list_custom = broadcast_list(
            4, torch.int64, list_values=int_values, data_parallel=True, group=mp_group
        )

        dist.barrier()
        assert torch.equal(
            int_global, int_custom
        ), "broadcast_int_list should be the same with or without an explicit group"
        assert torch.allclose(
            float_global, float_custom
        ), "broadcast_float_list should be the same with or without an explicit group"
        assert torch.equal(
            list_global, list_custom
        ), "broadcast_list should be the same with or without an explicit group"

        grid.destroy()

    @pytest.mark.skipif(
        not is_torch_min_version("2.4.0"),
        reason="torch.distributed.init_device_mesh requires torch >= 2.4.0",
    )
    @pytest.mark.parametrize("tp_size,pp_size", [(1, 8), (2, 4), (4, 2)])
    def test_broadcast_tensor_explicit_group_without_data_parallel(self, tp_size, pp_size):
        """Passing group with data_parallel=False uses that group and the given rank."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=pp_size
        )

        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")
        torch.manual_seed(rank)

        local_tensor = (
            torch.randn(self.size, dtype=self.dtype, device=device) if rank == 0 else None
        )

        tensor_received_global = broadcast_tensor(
            self.size, self.dtype, tensor=local_tensor, rank=0, data_parallel=False
        )

        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')

        grid, mp_group = self._create_mp_group(tp_size, pp_size)
        # tp*pp == world size in these configs, so the MP group is the world group.
        tensor_received_custom = broadcast_tensor(
            self.size,
            self.dtype,
            tensor=local_tensor,
            rank=0,
            data_parallel=False,
            group=mp_group,
        )

        dist.barrier()
        assert torch.allclose(
            tensor_received_global, tensor_received_custom
        ), "broadcast_tensor with an explicit world-equivalent group should match default"

        grid.destroy()
