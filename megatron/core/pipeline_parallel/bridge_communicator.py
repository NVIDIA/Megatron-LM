# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid


@dataclass
class SendOp:
    """Describes a single send operation for a single rank."""

    destination_rank: int
    batch_slice: slice
    send_shape: Tuple[int, ...]


@dataclass
class RecvOp:
    """Describes a single receive operation for a single rank."""

    source_rank: int
    recv_shape: Tuple[int, ...]


@dataclass
class RankCommInfo:
    """Explicit communication plan for a single rank."""

    role: Literal['SENDER', 'RECEIVER', 'NOOP'] = 'NOOP'

    sends: List[SendOp] = field(default_factory=list)
    receives: List[RecvOp] = field(default_factory=list)


class BridgeCommunicator:
    """Facilitates communication of activations and gradients between two modules with its own parallel mapping (TP/DP/PP/CP).

    The role of BridgeCommunicator:
    - Initializes the communicator between a pair of source and destination grids
    - Builds a communication schedule for each rank
    - Provides public methods: send_forward, recv_forward, send_forward_recv_backward,
      send_backward_recv_forward to be used by the pipeline schedule.
    """

    def __init__(
        self,
        src_grid: HyperCommGrid,
        dest_grid: HyperCommGrid,
        dim_mapping: Optional[Dict[str, int]] = None,
    ):
        """Initialize the bridge communicator between source and destination grids.

        Args:
            src_grid: Source HyperCommGrid
            dest_grid: Destination HyperCommGrid
            dim_mapping: Dictionary mapping parallelism types to tensor dimensions
        """
        self.src_grid = src_grid
        self.dest_grid = dest_grid
        self.current_rank = dist.get_rank()
        self.comm_map: Dict[int, RankCommInfo] = {}
        if dim_mapping is None:
            self.dim_mapping = {'s': 1, 'b': 0, 'h': 2}
        else:
            self.dim_mapping = dim_mapping

        self.activation_gather_ranks = self._create_activation_gather_scatter_pg(
            self.src_grid, is_src=True
        )
        self.activation_scatter_ranks = self._create_activation_gather_scatter_pg(
            self.dest_grid, is_src=False
        )

        self.src_tp_leaders, self.src_local_leader_rank = self.get_leader_rank(
            self.src_grid, is_src=True
        )
        self.dest_tp_leaders, self.dest_local_leader_rank = self.get_leader_rank(
            self.dest_grid, is_src=False
        )

        if self.activation_gather_ranks:
            print(f"Current rank {self.current_rank} generating scatter ranks in dest grid")
            self.activation_scatter_ranks = self.get_boundary_pp_stage_ranks(
                self.dest_grid, is_src=False
            )
        if self.activation_scatter_ranks:
            print(f"Current rank {self.current_rank} generating gather ranks in src grid")
            self.activation_gather_ranks = self.get_boundary_pp_stage_ranks(
                self.src_grid, is_src=True
            )

        self.activation_gather_pg = dist.new_group(ranks=self.activation_gather_ranks)
        self.activation_scatter_pg = dist.new_group(ranks=self.activation_scatter_ranks)

        log_msg = (
            f"[Rank {self.current_rank}] "
            f"srcLeader={self.src_local_leader_rank} "
            f"destLeader={self.dest_local_leader_rank} "
            f"gatherGrpRanks={self.activation_gather_ranks} "
            f"scatterGrpRanks={self.activation_scatter_ranks}"
        )
        print(log_msg, flush=True)
        dist.barrier()

        self.build_comm_schedule(self.src_tp_leaders, self.dest_tp_leaders)

        if self.current_rank == self.src_local_leader_rank:
            print(f"comm map  {self.current_rank} is {self.comm_map}")
        dist.barrier()

    def get_leader_rank(self, grid: HyperCommGrid, is_src: bool) -> List[int]:
        """Get the leader rank for a given grid and direction."""
        leader_ranks = []
        local_leader_rank = None
        # grid.gen_rank_enum(["tp", "cp", "pp"]) # vary tp & cp, freeze dp
        # returns a list of sublists, each sublist is a group of ranks that have different tp & cp & pp, same dp
        per_dp_replica_ranks = grid._gen_rank_enum([x for x in grid.dim_names if x != "dp"])
        if is_src:
            # Add rank from last pp stage
            ranks = []
            for group in per_dp_replica_ranks:
                if self.current_rank in group:
                    assert (
                        local_leader_rank is None
                    ), "only one local leader rank is allowed per dp replica"
                    local_leader_rank = group[-1]
                ranks.append(group[-1])
            leader_ranks.extend(ranks)
        else:
            # Add rank from first pp stage
            ranks = []
            for group in per_dp_replica_ranks:
                if self.current_rank in group:
                    assert (
                        local_leader_rank is None
                    ), "only one local leader rank is allowed per dp replica"
                    local_leader_rank = group[0]
                ranks.append(group[0])
            leader_ranks.extend(ranks)
        return leader_ranks, local_leader_rank

    def get_boundary_pp_stage_ranks(self, grid: HyperCommGrid, is_src: bool):
        """Get ranks of tp-cp corresponding to last stage of pp for the current grid."""
        tpcp_ranks = grid._gen_rank_enum(['tp', 'cp'])
        pp_dim = grid.shape[grid.dim_names.index('pp')]
        boundary_pp_stage_ranks = []
        for group in tpcp_ranks:
            for rank in group:
                rank_coords = []
                temp_rank = rank - grid.rank_offset
                for dim_size in reversed(grid.shape):
                    rank_coords.append(temp_rank % dim_size)
                    temp_rank //= dim_size
                rank_coords.reverse()
                pp_coord = rank_coords[grid.dim_names.index('pp')]
                if is_src:
                    if pp_coord == pp_dim - 1:
                        boundary_pp_stage_ranks.append(rank)
                else:
                    if pp_coord == 0:
                        boundary_pp_stage_ranks.append(rank)
        return boundary_pp_stage_ranks

    def _create_activation_gather_scatter_pg(self, grid: HyperCommGrid, is_src: bool):

        if (
            self.current_rank < grid.rank_offset
            or self.current_rank >= grid.rank_offset + grid.size
        ):
            return []

        pp_group_ranks = dist.get_process_group_ranks(grid.get_pg(['pp']))
        activation_comm_ranks = []

        # on the src grid, all tp-cp ranks of last pp stage and dp replica that current rank belongs to
        # participtes in the activation gather
        # on the dest grid, all tp-cp ranks of first pp stage and dp replica that current rank belongs to
        # participtes in the activation scatter

        if is_src and not self.current_rank == pp_group_ranks[-1]:
            # if current rank belongs to src grid. If not belongs to last pp stage, return empty ranks and pg
            return activation_comm_ranks

        if not is_src and not self.current_rank == pp_group_ranks[0]:
            # if current rank belongs to dest grid. If not belongs to first pp stage, return empty ranks and pg
            return activation_comm_ranks

        all_tpcp_group_ranks = grid._gen_rank_enum(['tp', 'cp'])
        for each_group_ranks in all_tpcp_group_ranks:
            if self.current_rank in each_group_ranks:
                activation_comm_ranks = each_group_ranks
                break

        return activation_comm_ranks

    def is_current_rank_in_grid(self, grid: HyperCommGrid) -> bool:
        """Check if the current rank is in the grid."""
        return grid.rank_offset <= self.current_rank < grid.rank_offset + grid.size

    def build_comm_schedule(self, src_tp_leaders: List[int], dest_tp_leaders: List[int]):
        """Get src/dest tp leaders and populate comm_map for each rank.

        This method analyzes the source and destination grids to determine
        which ranks need to send/receive data and builds the communication
        schedule accordingly.
        """
        # Ensure that the number of leaders can be evenly divided
        src_count = len(src_tp_leaders)
        dest_count = len(dest_tp_leaders)

        if src_count % dest_count != 0 and dest_count % src_count != 0:
            raise ValueError(
                f"Source TP leaders count ({src_count}) and destination TP leaders count "
                f"({dest_count}) must be evenly divisible. One must be a multiple of the other."
            )
        # Get all ranks in source and destination grids
        src_all_ranks = list(
            range(self.src_grid.rank_offset, self.src_grid.rank_offset + self.src_grid.size)
        )
        dest_all_ranks = list(
            range(self.dest_grid.rank_offset, self.dest_grid.rank_offset + self.dest_grid.size)
        )

        all_ranks = src_all_ranks + dest_all_ranks

        # Initialize all ranks as NOOP by default
        for rank in all_ranks:
            self.comm_map[rank] = RankCommInfo(role='NOOP')

        scale_factor = src_count / dest_count
        if scale_factor > 1:
            # Fan-in: multiple source leaders send to fewer destination leaders
            scale_factor = int(scale_factor)
            for i, dest_rank in enumerate(dest_tp_leaders):
                # Each destination rank receives from scale_factor source ranks
                src_ranks = src_tp_leaders[i * scale_factor : (i + 1) * scale_factor]

                # Set up senders
                for src_rank in src_ranks:
                    self.comm_map[src_rank] = RankCommInfo(
                        role='SENDER',
                        sends=[
                            SendOp(
                                destination_rank=dest_rank,
                                batch_slice=slice(None),
                                send_shape=(1,),  # placeholder
                            )
                        ],
                    )

                # Set up receiver
                self.comm_map[dest_rank] = RankCommInfo(
                    role='RECEIVER',
                    receives=[
                        RecvOp(source_rank=src_rank, recv_shape=(1,))  # placeholder
                        for src_rank in src_ranks
                    ],
                )
        else:
            # Fan-out: fewer source leaders send to more destination leaders
            scale_factor = int(dest_count / src_count)
            for i, src_rank in enumerate(src_tp_leaders):
                # Each source rank sends to scale_factor destination ranks
                dest_ranks = dest_tp_leaders[i * scale_factor : (i + 1) * scale_factor]

                # Set up sender
                self.comm_map[src_rank] = RankCommInfo(
                    role='SENDER',
                    sends=[
                        SendOp(
                            destination_rank=dest_rank,
                            batch_slice=slice(None),
                            send_shape=(1,),  # placeholder
                        )
                        for dest_rank in dest_ranks
                    ],
                )

                # Set up receivers
                for dest_rank in dest_ranks:
                    self.comm_map[dest_rank] = RankCommInfo(
                        role='RECEIVER',
                        receives=[RecvOp(source_rank=src_rank, recv_shape=(1,))],  # placeholder
                    )

    def send_forward(self, tensor_to_send: torch.Tensor):
        """Send forward activation tensor.

        Args:
            tensor_to_send: The tensor to send to the destination grid
        """
        if not self.is_current_rank_in_grid(self.src_grid):
            raise ValueError(
                f"Rank {self.current_rank} is not in the source grid. Send forward is only allowed on src grid"
            )

        rank_info = self.comm_map.get(self.current_rank)

        assert rank_info is not None, f"Rank {self.current_rank} is not in the comm map"

        if rank_info.role == 'SENDER':
            # Current rank is a sender - gather tensors from all ranks in activation_gather_group
            assert (
                self.current_rank == self.src_local_leader_rank
            ), f"Rank {self.current_rank} is not the leader rank"
            gathered_tensors = [
                torch.zeros_like(tensor_to_send) for _ in range(len(self.activation_gather_ranks))
            ]
            dist.gather(
                tensor_to_send,
                gather_list=gathered_tensors,
                dst=self.src_local_leader_rank,
                group=self.activation_gather_pg,
            )

            aggregated_tensor = self._reconstruct_tensor_from_gathered(
                gathered_tensors,
                self.src_grid,
                self.activation_gather_pg,
                self.activation_gather_ranks,
            )
            print(f"rank {self.current_rank} gathered tensor shape {aggregated_tensor.shape}")
            self._communicate_shapes(tensor_to_send_next=aggregated_tensor)
            # Send splits to destination ranks
            num_sends = len(rank_info.sends)
            if num_sends > 0:
                batch_size = aggregated_tensor.size(0)
                split_size = batch_size // num_sends
                for i, send_op in enumerate(rank_info.sends):
                    start_idx = i * split_size
                    end_idx = (i + 1) * split_size if i < num_sends - 1 else batch_size
                    tensor_split = aggregated_tensor[start_idx:end_idx]
                    # Send the tensor split to the destination rank
                    print(
                        f"rank {self.current_rank} sending tensor to dst rank {send_op.destination_rank} shape {tensor_split.shape} sum {tensor_split.sum()}"
                    )
                    dist.send(tensor_split, dst=send_op.destination_rank)

        elif rank_info.role == 'NOOP' and self.current_rank in self.activation_gather_ranks:
            dist.gather(
                tensor_to_send,
                gather_list=None,
                dst=self.src_local_leader_rank,
                group=self.activation_gather_pg,
            )

    def receive_forward(
        self, tensor_shape: Optional[Tuple[int, ...]] = None, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Receive forward activation tensor.

        Args:
            tensor_shape: Expected tensor shape (None if using shape communication)
            dtype: Expected tensor dtype

        Returns:
            torch.Tensor: The received activation tensor
        """
        # receive forward only gets called on the dest grid
        if not self.is_current_rank_in_grid(self.dest_grid):
            raise ValueError(
                f"Rank {self.current_rank} is not in the destination grid. Receive forward is only allowed on dest grid"
            )

        rank_info = self.comm_map.get(self.current_rank)
        assert rank_info is not None, f"Rank {self.current_rank} is not in the comm map"

        if rank_info.role == 'RECEIVER':
            assert (
                self.current_rank == self.dest_local_leader_rank
            ), f"Rank {self.current_rank} is not the leader rank"
            # p2p call to receive the tensor
            recv_forward_shapes, recv_grad_shapes = self._communicate_shapes(recv_prev=True)
            print(
                f"rank {self.current_rank} received forward shapes {recv_forward_shapes} and grad shapes {recv_grad_shapes}"
            )
            received_tensors_list = []
            for i, each_receive_op in enumerate(rank_info.receives):
                tensor_to_recv = torch.empty(
                    recv_forward_shapes[i], device=torch.cuda.current_device(), dtype=dtype
                )
                dist.recv(tensor_to_recv, src=each_receive_op.source_rank)
                print(
                    f"rank {self.current_rank} received tensor from src rank {each_receive_op.source_rank} shape {tensor_to_recv.shape} sum {tensor_to_recv.sum()}"
                )
                received_tensors_list.append(tensor_to_recv)

            aggregated_tensor = torch.cat(received_tensors_list, dim=0)
            print(
                f"rank {self.current_rank} aggregated tensor shape {aggregated_tensor.shape} sum {aggregated_tensor.sum()}"
            )

            tensor_dict = self._decompose_tensor_by_grid_dims(
                aggregated_tensor, self.dest_grid, self.activation_scatter_ranks
            )

            # received_tensor = torch.empty_like(scatter_list[0])
            print('*' * 100)
            print(
                f"rank {self.current_rank} scatter list shape {[x.shape for x in tensor_dict.values()]} doing the scatter in rank group {dist.get_process_group_ranks(self.activation_scatter_pg)}"
            )
            print('*' * 100)
            # Send each tensor in scatter_list to corresponding ranks in activation_scatter_pg
            scatter_ranks = dist.get_process_group_ranks(self.activation_scatter_pg)

            # Prepare scatter list ordered by process group ranks
            scatter_list = [tensor_dict[rank] for rank in scatter_ranks]

            # Create tensor to receive the scattered data
            received_tensor = torch.empty_like(tensor_dict[self.current_rank])

            # Scatter the tensors to all ranks in the group
            dist.scatter(
                received_tensor,
                scatter_list=scatter_list,
                src=self.current_rank,
                group=self.activation_scatter_pg,
            )

            print(f"rank {self.current_rank} scattered tensor chunks to all ranks in group")
            return received_tensor

        elif rank_info.role == 'NOOP' and self.current_rank in self.activation_scatter_ranks:
            # Non-leader rank - participate in scatter operation
            received_tensor = torch.empty(
                tensor_shape, device=torch.cuda.current_device(), dtype=dtype
            )
            dist.scatter(
                received_tensor,
                scatter_list=None,
                src=self.dest_local_leader_rank,
                group=self.activation_scatter_pg,
            )
            print(
                f"rank {self.current_rank} received tensor from scatter operation, shape {received_tensor.shape}"
            )
            return received_tensor

    def send_backward(self, grad_tensor: torch.Tensor):
        """Send backward gradient tensor.

        Note: Gradient senders are activation 'RECEIVERS'

        Args:
            grad_tensor: The gradient tensor to send back
        """
        if not self.is_current_rank_in_grid(self.dest_grid):
            raise ValueError(
                f"Rank {self.current_rank} is not in the destination grid. Send backward is only allowed on dest grid"
            )

        rank_info = self.comm_map.get(self.current_rank)
        assert rank_info is not None, f"Rank {self.current_rank} is not in the comm map"

        if rank_info.role == 'RECEIVER':
            # Current rank is a receiver - gather gradients from all ranks in activation_scatter_ranks
            assert (
                self.current_rank == self.dest_local_leader_rank
            ), f"Rank {self.current_rank} is not the leader rank"
            print(
                f"rank {self.current_rank} is a receiver rank. Running gather on {self.activation_scatter_ranks}"
            )
            gathered_tensors = [
                torch.zeros_like(grad_tensor) for _ in range(len(self.activation_scatter_ranks))
            ]
            dist.gather(
                grad_tensor,
                gather_list=gathered_tensors,
                dst=self.dest_local_leader_rank,
                group=self.activation_scatter_pg,
            )

            aggregated_tensor = self._reconstruct_tensor_from_gathered(
                gathered_tensors,
                self.dest_grid,
                self.activation_scatter_pg,
                self.activation_scatter_ranks,
            )
            print(
                f"rank {self.current_rank} gathered gradient tensor shape {aggregated_tensor.shape}"
            )
            self._communicate_shapes(tensor_to_send_prev=aggregated_tensor)

            # Send gradients back to source ranks
            num_receives = len(rank_info.receives)
            if num_receives > 0:
                batch_size = aggregated_tensor.size(0)
                split_size = batch_size // num_receives
                for i, recv_op in enumerate(rank_info.receives):
                    start_idx = i * split_size
                    end_idx = (i + 1) * split_size if i < num_receives - 1 else batch_size
                    tensor_split = aggregated_tensor[start_idx:end_idx]
                    # Send the gradient split back to the source rank
                    print(
                        f"rank {self.current_rank} sending gradient to src rank {recv_op.source_rank} shape {tensor_split.shape} sum {tensor_split.sum()}"
                    )
                    dist.send(tensor_split, dst=recv_op.source_rank)

        elif rank_info.role == 'NOOP' and self.current_rank in self.activation_scatter_ranks:
            # Non-leader rank - participate in gather
            print(f"rank {self.current_rank} is in activation scatter ranks")
            dist.gather(
                grad_tensor,
                gather_list=None,
                dst=self.dest_local_leader_rank,
                group=self.activation_scatter_pg,
            )

    def receive_backward(
        self, tensor_shape: Optional[Tuple[int, ...]] = None, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Receive backward gradient tensor.

        Note: Gradient receivers are activation 'SENDERS'

        Args:
            tensor_shape: Expected gradient tensor shape

        Returns:
            torch.Tensor: The received gradient tensor
        """
        # receive backward only gets called on the src grid
        if not self.is_current_rank_in_grid(self.src_grid):
            raise ValueError(
                f"Rank {self.current_rank} is not in the source grid. Receive backward is only allowed on src grid"
            )

        rank_info = self.comm_map.get(self.current_rank)
        assert rank_info is not None, f"Rank {self.current_rank} is not in the comm map"

        if rank_info.role == 'SENDER':
            assert (
                self.current_rank == self.src_local_leader_rank
            ), f"Rank {self.current_rank} is not the leader rank"
            recv_forward_shapes, recv_grad_shapes = self._communicate_shapes(recv_next=True)
            print(
                f"rank {self.current_rank} received forward shapes {recv_forward_shapes} and grad shapes {recv_grad_shapes}"
            )
            # Receive gradient tensors from destination ranks
            received_gradients_list = []
            for i, send_op in enumerate(rank_info.sends):
                # The destination rank that we sent to will send us gradients back
                grad_tensor = torch.empty(
                    recv_grad_shapes[i], device=torch.cuda.current_device(), dtype=dtype
                )
                dist.recv(grad_tensor, src=send_op.destination_rank)
                print(
                    f"rank {self.current_rank} received gradient from dest rank {send_op.destination_rank} shape {grad_tensor.shape} sum {grad_tensor.sum()}"
                )
                received_gradients_list.append(grad_tensor)

            # Concatenate received gradients
            aggregated_gradient = torch.cat(received_gradients_list, dim=0)
            print(
                f"rank {self.current_rank} aggregated gradient shape {aggregated_gradient.shape} sum {aggregated_gradient.sum()}"
            )

            # Decompose and scatter to ranks in activation_gather_pg
            tensor_dict = self._decompose_tensor_by_grid_dims(
                aggregated_gradient, self.src_grid, self.activation_gather_ranks
            )

            print('*' * 100)
            print(
                f"rank {self.current_rank} scatter list shape {[x.shape for x in tensor_dict.values()]} doing the scatter in rank group {dist.get_process_group_ranks(self.activation_gather_pg)}"
            )
            print('*' * 100)

            # Send each tensor in tensor_dict to corresponding ranks in activation_gather_pg
            scatter_ranks = dist.get_process_group_ranks(self.activation_gather_pg)

            # Prepare scatter list ordered by process group ranks
            scatter_list = [tensor_dict[rank] for rank in scatter_ranks]

            # Create tensor to receive the scattered data
            received_gradient = torch.empty_like(tensor_dict[self.current_rank])

            # Scatter the tensors to all ranks in the group
            dist.scatter(
                received_gradient,
                scatter_list=scatter_list,
                src=self.current_rank,
                group=self.activation_gather_pg,
            )

            print(f"rank {self.current_rank} scattered gradient chunks to all ranks in group")
            return received_gradient

        elif rank_info.role == 'NOOP' and self.current_rank in self.activation_gather_ranks:
            # Non-leader rank - participate in scatter operation
            received_gradient = torch.empty(
                tensor_shape, device=torch.cuda.current_device(), dtype=dtype
            )
            dist.scatter(
                received_gradient,
                scatter_list=None,
                src=self.src_local_leader_rank,
                group=self.activation_gather_pg,
            )
            print(
                f"rank {self.current_rank} received gradient from scatter operation, shape {received_gradient.shape}"
            )
            return received_gradient

    def send_forward_recv_backward(
        self,
        input_tensor: torch.Tensor,
        grad_shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Combined operation: send forward activation and receive backward gradient.

        Args:
            input_tensor: The tensor to send forward
            grad_shape: Expected gradient tensor shape
            dtype: Expected tensor dtype

        Returns:
            torch.Tensor: The received gradient tensor
        """
        if not self.is_current_rank_in_grid(self.src_grid):
            raise ValueError(
                f"Rank {self.current_rank} is not in the source grid. send_forward_recv_backward is only allowed on src grid"
            )

        rank_info = self.comm_map.get(self.current_rank)
        assert rank_info is not None, f"Rank {self.current_rank} is not in the comm map"

        if rank_info.role == 'SENDER':
            # Current rank is a sender - gather tensors from all ranks in activation_gather_group
            assert (
                self.current_rank == self.src_local_leader_rank
            ), f"Rank {self.current_rank} is not the leader rank"
            print(
                f"rank {self.current_rank} is a sender rank. Running gather on {self.activation_gather_ranks}"
            )
            # Gather activations from all ranks in activation_gather_group
            gathered_tensors = [
                torch.zeros_like(input_tensor) for _ in range(len(self.activation_gather_ranks))
            ]
            dist.gather(
                input_tensor,
                gather_list=gathered_tensors,
                dst=self.src_local_leader_rank,
                group=self.activation_gather_pg,
            )

            # Reconstruct tensor from gathered activations
            aggregated_tensor = self._reconstruct_tensor_from_gathered(
                gathered_tensors,
                self.src_grid,
                self.activation_gather_pg,
                self.activation_gather_ranks,
            )
            print(f"rank {self.current_rank} gathered tensor shape {aggregated_tensor.shape}")

            # Communicate shapes for both directions (send forward, receive backward)
            recv_forward_shapes, recv_grad_shapes = self._communicate_shapes(
                tensor_to_send_next=aggregated_tensor, recv_next=True
            )
            print(
                f"rank {self.current_rank} received forward shapes {recv_forward_shapes} and grad shapes {recv_grad_shapes}"
            )

            # Prepare simultaneous send/receive operations
            num_sends = len(rank_info.sends)
            if num_sends > 0:
                # Prepare activation splits for sending
                batch_size = aggregated_tensor.size(0)
                split_size = batch_size // num_sends
                activation_splits = []
                for i in range(num_sends):
                    start_idx = i * split_size
                    end_idx = (i + 1) * split_size if i < num_sends - 1 else batch_size
                    tensor_split = aggregated_tensor[start_idx:end_idx]
                    activation_splits.append(tensor_split)

                # Prepare gradient receive tensors
                received_gradients_list = []
                for i, recv_grad_shape in enumerate(recv_grad_shapes):
                    grad_tensor = torch.empty(
                        recv_grad_shape, device=torch.cuda.current_device(), dtype=dtype
                    )
                    received_gradients_list.append(grad_tensor)

                # Create batch P2P operations for simultaneous send/receive
                ops = []
                for i, send_op in enumerate(rank_info.sends):
                    # Send activation
                    ops.append(
                        torch.distributed.P2POp(
                            torch.distributed.isend, activation_splits[i], send_op.destination_rank
                        )
                    )
                    print(
                        f"rank {self.current_rank} prepared send activation to dst rank {send_op.destination_rank} shape {activation_splits[i].shape}"
                    )

                    # Receive gradient
                    ops.append(
                        torch.distributed.P2POp(
                            torch.distributed.irecv,
                            received_gradients_list[i],
                            send_op.destination_rank,
                        )
                    )
                    print(
                        f"rank {self.current_rank} prepared receive gradient from dst rank {send_op.destination_rank} shape {received_gradients_list[i].shape}"
                    )

                # Execute all operations simultaneously
                print(f"rank {self.current_rank} executing {len(ops)} simultaneous P2P operations")
                reqs = torch.distributed.batch_isend_irecv(ops)
                for req in reqs:
                    req.wait()

                # To protect against race condition when using batch_isend_irecv()
                torch.cuda.synchronize()

                # Concatenate received gradients
                aggregated_gradient = torch.cat(received_gradients_list, dim=0)
                print(
                    f"rank {self.current_rank} aggregated gradient shape {aggregated_gradient.shape} sum {aggregated_gradient.sum()}"
                )

                # Decompose and scatter to ranks in activation_gather_pg
                tensor_dict = self._decompose_tensor_by_grid_dims(
                    aggregated_gradient, self.src_grid, self.activation_gather_ranks
                )

                print('*' * 100)
                print(
                    f"rank {self.current_rank} scatter list shape {[x.shape for x in tensor_dict.values()]} doing the scatter in rank group {dist.get_process_group_ranks(self.activation_gather_pg)}"
                )
                print('*' * 100)

                # Send each tensor in tensor_dict to corresponding ranks in activation_gather_pg
                scatter_ranks = dist.get_process_group_ranks(self.activation_gather_pg)

                # Prepare scatter list ordered by process group ranks
                scatter_list = [tensor_dict[rank] for rank in scatter_ranks]

                # Create tensor to receive the scattered data
                received_gradient = torch.empty_like(tensor_dict[self.current_rank])

                # Scatter the tensors to all ranks in the group
                dist.scatter(
                    received_gradient,
                    scatter_list=scatter_list,
                    src=self.current_rank,
                    group=self.activation_gather_pg,
                )

                print(f"rank {self.current_rank} scattered gradient chunks to all ranks in group")
                return received_gradient

        elif rank_info.role == 'NOOP' and self.current_rank in self.activation_gather_ranks:
            # Non-leader rank - participate in both gather (for activations) and receive (for gradients)
            print(
                f"rank {self.current_rank} is a noop rank. Running gather on {self.activation_gather_ranks}"
            )
            # Participate in activation gather
            dist.gather(
                input_tensor,
                gather_list=None,
                dst=self.src_local_leader_rank,
                group=self.activation_gather_pg,
            )
            # Receive gradient from leader using scatter operation
            received_gradient = torch.empty(
                grad_shape, device=torch.cuda.current_device(), dtype=dtype
            )
            print(
                f"rank {self.current_rank} is a noop rank. Waiting for gradient from leader rank {self.src_local_leader_rank}"
            )
            dist.scatter(
                received_gradient,
                scatter_list=None,
                src=self.src_local_leader_rank,
                group=self.activation_gather_pg,
            )
            print(
                f"rank {self.current_rank} received gradient from scatter operation, shape {received_gradient.shape}"
            )
            return received_gradient

        else:
            raise ValueError(
                f"Rank {self.current_rank} with role {rank_info.role} cannot participate in send_forward_recv_backward"
            )

    def send_backward_recv_forward(
        self,
        grad_tensor: torch.Tensor,
        forward_shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Combined operation: send backward gradient and receive forward activation.

        Args:
            grad_tensor: The gradient tensor to send backward
            forward_shape: Expected forward tensor shape
            dtype: Expected tensor dtype

        Returns:
            torch.Tensor: The received activation tensor
        """
        if not self.is_current_rank_in_grid(self.dest_grid):
            raise ValueError(
                f"Rank {self.current_rank} is not in the destination grid. send_backward_recv_forward is only allowed on dest grid"
            )

        rank_info = self.comm_map.get(self.current_rank)
        assert rank_info is not None, f"Rank {self.current_rank} is not in the comm map"

        if rank_info.role == 'RECEIVER':
            # Current rank is a receiver - gather gradients from all ranks in activation_scatter_ranks
            assert (
                self.current_rank == self.dest_local_leader_rank
            ), f"Rank {self.current_rank} is not the leader rank"
            print(
                f"rank {self.current_rank} is a receiver rank. Running gather on {self.activation_scatter_ranks}"
            )
            # Gather gradients from all ranks in activation_scatter_ranks
            gathered_tensors = [
                torch.zeros_like(grad_tensor) for _ in range(len(self.activation_scatter_ranks))
            ]
            dist.gather(
                grad_tensor,
                gather_list=gathered_tensors,
                dst=self.dest_local_leader_rank,
                group=self.activation_scatter_pg,
            )

            # Reconstruct gradient tensor from gathered gradients
            aggregated_gradient = self._reconstruct_tensor_from_gathered(
                gathered_tensors,
                self.dest_grid,
                self.activation_scatter_pg,
                self.activation_scatter_ranks,
            )
            print(
                f"rank {self.current_rank} gathered gradient tensor shape {aggregated_gradient.shape}"
            )

            # Communicate shapes for both directions (send backward, receive forward)
            recv_forward_shapes, recv_grad_shapes = self._communicate_shapes(
                tensor_to_send_prev=aggregated_gradient, recv_prev=True
            )
            print(
                f"rank {self.current_rank} received forward shapes {recv_forward_shapes} and grad shapes {recv_grad_shapes}"
            )

            # Prepare simultaneous send/receive operations
            num_receives = len(rank_info.receives)
            if num_receives > 0:
                # Prepare gradient splits for sending
                batch_size = aggregated_gradient.size(0)
                split_size = batch_size // num_receives
                gradient_splits = []
                for i in range(num_receives):
                    start_idx = i * split_size
                    end_idx = (i + 1) * split_size if i < num_receives - 1 else batch_size
                    tensor_split = aggregated_gradient[start_idx:end_idx]
                    gradient_splits.append(tensor_split)

                # Prepare activation receive tensors
                received_activations_list = []
                for i, recv_forward_shape in enumerate(recv_forward_shapes):
                    activation_tensor = torch.empty(
                        recv_forward_shape, device=torch.cuda.current_device(), dtype=dtype
                    )
                    received_activations_list.append(activation_tensor)

                # Create batch P2P operations for simultaneous send/receive
                ops = []
                for i, recv_op in enumerate(rank_info.receives):
                    # Send gradient
                    ops.append(
                        torch.distributed.P2POp(
                            torch.distributed.isend, gradient_splits[i], recv_op.source_rank
                        )
                    )
                    print(
                        f"rank {self.current_rank} prepared send gradient to src rank {recv_op.source_rank} shape {gradient_splits[i].shape}"
                    )

                    # Receive activation
                    ops.append(
                        torch.distributed.P2POp(
                            torch.distributed.irecv,
                            received_activations_list[i],
                            recv_op.source_rank,
                        )
                    )
                    print(
                        f"rank {self.current_rank} prepared receive activation from src rank {recv_op.source_rank} shape {received_activations_list[i].shape}"
                    )

                # Execute all operations simultaneously
                print(f"rank {self.current_rank} executing {len(ops)} simultaneous P2P operations")
                reqs = torch.distributed.batch_isend_irecv(ops)
                for req in reqs:
                    req.wait()

                # To protect against race condition when using batch_isend_irecv()
                torch.cuda.synchronize()

                # Concatenate received activations
                aggregated_activation = torch.cat(received_activations_list, dim=0)
                print(
                    f"rank {self.current_rank} aggregated activation shape {aggregated_activation.shape} sum {aggregated_activation.sum()}"
                )

                # Decompose and scatter to ranks in activation_scatter_pg
                tensor_dict = self._decompose_tensor_by_grid_dims(
                    aggregated_activation, self.dest_grid, self.activation_scatter_ranks
                )

                print('*' * 100)
                print(
                    f"rank {self.current_rank} scatter list shape {[x.shape for x in tensor_dict.values()]} doing the scatter in rank group {dist.get_process_group_ranks(self.activation_scatter_pg)}"
                )
                print('*' * 100)

                # Send each tensor in tensor_dict to corresponding ranks in activation_scatter_pg
                scatter_ranks = dist.get_process_group_ranks(self.activation_scatter_pg)

                # Prepare scatter list ordered by process group ranks
                scatter_list = [tensor_dict[rank] for rank in scatter_ranks]

                # Create tensor to receive the scattered data
                received_activation = torch.empty_like(tensor_dict[self.current_rank])

                # Scatter the tensors to all ranks in the group
                dist.scatter(
                    received_activation,
                    scatter_list=scatter_list,
                    src=self.current_rank,
                    group=self.activation_scatter_pg,
                )

                print(f"rank {self.current_rank} scattered activation chunks to all ranks in group")
                return received_activation

        elif rank_info.role == 'NOOP' and self.current_rank in self.activation_scatter_ranks:
            # Non-leader rank - participate in both gather (for gradients) and receive (for activations)
            print(
                f"rank {self.current_rank} is a noop rank. Running gather on {self.activation_scatter_ranks}"
            )
            # Participate in gradient gather
            dist.gather(
                grad_tensor,
                gather_list=None,
                dst=self.dest_local_leader_rank,
                group=self.activation_scatter_pg,
            )
            print(
                f"rank {self.current_rank} is a noop rank. Waiting for activation from leader rank {self.dest_local_leader_rank}"
            )
            # Receive activation from leader using scatter operation
            received_activation = torch.empty(
                forward_shape, device=torch.cuda.current_device(), dtype=dtype
            )
            dist.scatter(
                received_activation,
                scatter_list=None,
                src=self.dest_local_leader_rank,
                group=self.activation_scatter_pg,
            )
            print(
                f"rank {self.current_rank} received activation from scatter operation, shape {received_activation.shape}"
            )
            return received_activation

        else:
            raise ValueError(
                f"Rank {self.current_rank} with role {rank_info.role} cannot participate in send_backward_recv_forward"
            )

    def _reconstruct_tensor_from_gathered(
        self,
        gathered_tensors: List[torch.Tensor],
        grid: HyperCommGrid,
        curr_pg: dist.ProcessGroup,
        enum_group: List[int],
    ) -> torch.Tensor:
        """Reconstruct tensor using the grid's native rank enumeration logic."""
        # Get all non-DP dimensions that were split
        non_dp_dims = [dim for dim in grid.dim_names if dim != "dp"]
        print("*" * 100)
        print("Starting to reconstruct tensor from gathered tensors")
        print(f"non_dp_dims: {non_dp_dims}")
        print(f"enum_group: {enum_group}")
        if not non_dp_dims:
            return gathered_tensors[0]

        curr_pg_ranks = dist.get_process_group_ranks(curr_pg)
        print(f"curr_pg: {curr_pg_ranks}")
        # Create mapping from rank to tensor and order by enumeration
        # Tensors are gathered in the order of curr_pg_ranks
        rank_to_tensor = dict(zip(curr_pg_ranks, gathered_tensors))
        ordered_tensors = [rank_to_tensor[rank] for rank in enum_group if rank in rank_to_tensor]

        if not ordered_tensors:
            raise ValueError("No tensors found for the given ranks")

        # Simple concatenation approach based on grid dimensions
        return self._concatenate_by_grid_dims(ordered_tensors, grid, non_dp_dims)

    def _concatenate_by_grid_dims(
        self, tensors: List[torch.Tensor], grid: HyperCommGrid, non_dp_dims: List[str]
    ) -> torch.Tensor:
        """Concatenate tensors based on grid dimensions using a simpler approach."""
        if len(tensors) == 1:
            return tensors[0]

        # Map parallelism types to tensor dimensions
        dim_mapping = {
            'tp': self.dim_mapping['h'],
            'cp': self.dim_mapping['s'],
            'ep': self.dim_mapping['h'],
        }  # TP/EP: hidden dim, CP: sequence dim

        # Get grid shape for reconstruction
        grid_shape = [grid.shape[grid.dim_names.index(dim)] for dim in non_dp_dims]
        print(f"grid_shape: {grid_shape}")
        # Reshape tensor list to match grid structure
        current_tensors = tensors

        # Process each grid dimension
        for dim_name, dim_size in zip(non_dp_dims, grid_shape):
            print(f"Processing dim_name: {dim_name}, dim_size: {dim_size}")
            if dim_name in dim_mapping:
                tensor_dim = dim_mapping[dim_name]
                print(f"tensor_dim: {tensor_dim}")
                # Group tensors for this dimension and concatenate
                grouped_tensors = []
                group_size = len(current_tensors) // dim_size
                print(f"group_size: {group_size}")
                for group_idx in range(group_size):
                    group_start = group_idx * dim_size
                    group_end = group_start + dim_size
                    print(f"group_start: {group_start}, group_end: {group_end}")
                    group = current_tensors[group_start:group_end]
                    grouped_tensors.append(torch.cat(group, dim=tensor_dim))

                current_tensors = grouped_tensors
                print(f"current_tensors shape: {current_tensors[0].shape}")
        print("*" * 100)
        return (
            current_tensors[0] if len(current_tensors) == 1 else torch.cat(current_tensors, dim=0)
        )

    def _communicate_shapes(
        self,
        tensor_to_send_next: Optional[torch.Tensor] = None,
        recv_next: bool = False,
        recv_prev: bool = False,
        tensor_to_send_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]:
        """Communicate tensor shapes between sender and receiver ranks in the bridge.

        This is used to communicate tensor shapes before actual tensor communication
        when dealing with variable sequence lengths or dynamic shapes.

        Args:
            tensor_to_send_next: The tensor to send to the next rank (None if not sending)
            tensor_to_send_prev: The tensor to send to the previous rank (None if not sending)
            recv_next: Whether to receive from the next rank (None if not receiving)
            recv_prev: Whether to receive from the previous rank (None if not receiving)

        Returns:
            Tuple containing:
            - List of forward shapes that will be received (empty if not a receiver)
            - List of gradient shapes that will be received (empty if not expecting gradients)
        """
        rank_info = self.comm_map.get(self.current_rank)
        if not rank_info or rank_info.role == 'NOOP':
            return [], []

        recv_forward_shapes = []
        recv_grad_shapes = []
        print(
            f"rank {self.current_rank} is a {rank_info.role} and is running the shape communication"
        )
        # Collect all P2P operations for batch execution
        ops = []
        recv_forward_shape_tensors = []
        recv_grad_shape_tensors = []

        if rank_info.role == 'SENDER':
            # Prepare send operations for forward shapes
            if tensor_to_send_next is not None:
                send_shape = tensor_to_send_next.shape
                send_shape_tensor = torch.tensor(
                    send_shape, device=torch.cuda.current_device(), dtype=torch.int64
                )
                print(
                    f"rank {self.current_rank} is a sender and is sending forward shapes {send_shape_tensor}"
                )
                # Add send operations for each destination
                for send_op in rank_info.sends:
                    print(
                        f"rank {self.current_rank} is a sender and is sending forward shapes to rank {send_op.destination_rank}"
                    )
                    ops.append(
                        torch.distributed.P2POp(
                            torch.distributed.isend, send_shape_tensor, send_op.destination_rank
                        )
                    )

            # If expecting gradients back, prepare receive operations
            if recv_next:
                for send_op in rank_info.sends:
                    print(
                        f"rank {self.current_rank} is a sender and is receiving gradient shapes from rank {send_op.destination_rank}"
                    )
                    grad_shape_tensor = torch.empty(
                        (3), device=torch.cuda.current_device(), dtype=torch.int64
                    )
                    recv_grad_shape_tensors.append(grad_shape_tensor)
                    ops.append(
                        torch.distributed.P2POp(
                            torch.distributed.irecv, grad_shape_tensor, send_op.destination_rank
                        )
                    )

        elif rank_info.role == 'RECEIVER':
            # Prepare receive operations for forward shapes
            if recv_prev:
                for recv_op in rank_info.receives:
                    print(
                        f"rank {self.current_rank} is a receiver and is receiving forward shapes from rank {recv_op.source_rank}"
                    )
                    forward_shape_tensor = torch.empty(
                        (3), device=torch.cuda.current_device(), dtype=torch.int64
                    )
                    recv_forward_shape_tensors.append(forward_shape_tensor)
                    ops.append(
                        torch.distributed.P2POp(
                            torch.distributed.irecv, forward_shape_tensor, recv_op.source_rank
                        )
                    )

            # If we need to send gradient shapes back, prepare send operations
            if tensor_to_send_prev is not None:

                grad_shape = tensor_to_send_prev.shape
                grad_shape_tensor = torch.tensor(
                    grad_shape, device=torch.cuda.current_device(), dtype=torch.int64
                )

                for recv_op in rank_info.receives:
                    print(
                        f"rank {self.current_rank} is a receiver and is sending gradient shapes to rank {recv_op.source_rank}"
                    )
                    ops.append(
                        torch.distributed.P2POp(
                            torch.distributed.isend, grad_shape_tensor, recv_op.source_rank
                        )
                    )

        # Execute all operations in a single batch
        print(
            f"rank {self.current_rank} is a {rank_info.role} and is executing {len(ops)} operations"
        )
        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # To protect against race condition when using batch_isend_irecv()
        # Following the pattern from the original p2p communication code
        torch.cuda.synchronize()

        # Extract shapes from received tensors
        for forward_shape_tensor in recv_forward_shape_tensors:
            shape = forward_shape_tensor.tolist()
            recv_forward_shapes.append(tuple(shape))

        for grad_shape_tensor in recv_grad_shape_tensors:
            shape = grad_shape_tensor.tolist()
            recv_grad_shapes.append(tuple(shape))

        return recv_forward_shapes, recv_grad_shapes

    def _decompose_tensor_by_grid_dims(
        self, aggregated_tensor: torch.Tensor, grid: HyperCommGrid, rank_enum: List[int]
    ) -> Dict[int, torch.Tensor]:
        """Decompose an aggregated tensor into smaller tensors based on grid dimensions.

        This is the inverse operation of _concatenate_by_grid_dims.

        Args:
            aggregated_tensor: The tensor to decompose
            grid: The HyperCommGrid defining the parallelism structure

        Returns:
            List of tensors split according to the grid dimensions.
            The tensors are returned in the same order as grid._gen_rank_enum(non_dp_dims).
        """
        print("*" * 100)
        print("Starting to decompose tensor by grid dimensions")
        # Get all non-DP dimensions that were split
        non_dp_dims = [dim for dim in grid.dim_names if dim != "dp"]
        print(f"non_dp_dims: {non_dp_dims}")
        if not non_dp_dims:
            return [aggregated_tensor]

        # Get the rank enumeration to determine the exact order we need
        print(f"rank_enum: {rank_enum}")

        # Map parallelism types to tensor dimensions
        dim_mapping = {
            'tp': self.dim_mapping['h'],
            'cp': self.dim_mapping['s'],
            'ep': self.dim_mapping['h'],
        }  # TP/EP: hidden dim, CP: sequence dim

        # Get grid shape for decomposition
        grid_shape = [grid.shape[grid.dim_names.index(dim)] for dim in non_dp_dims]
        print(f"grid_shape: {grid_shape}")

        # Start with the aggregated tensor
        current_tensors = [aggregated_tensor]
        print(f"aggregated_tensor shape: {aggregated_tensor.shape}")

        # Process each grid dimension in reverse order (to undo concatenation)
        for dim_name, dim_size in reversed(list(zip(non_dp_dims, grid_shape))):
            print(f"Processing dim_name: {dim_name}, dim_size: {dim_size}")
            if dim_name in dim_mapping:
                tensor_dim = dim_mapping[dim_name]
                print(f"tensor_dim: {tensor_dim}")
                new_tensors = []
                for tensor in current_tensors:
                    print(f"Processing tensor of shape: {tensor.shape}")
                    # Calculate split size for this dimension
                    total_size = tensor.size(tensor_dim)
                    print(f"total_size: {total_size}")
                    split_size = total_size // dim_size
                    print(f"split_size: {split_size}")
                    # Create split sizes list
                    split_sizes = [split_size] * dim_size
                    # Handle remainder if total_size is not evenly divisible
                    remainder = total_size % dim_size
                    for i in range(remainder):
                        split_sizes[i] += 1

                    # Split the tensor
                    splits = torch.split(tensor, split_sizes, dim=tensor_dim)
                    print(f"splits length: {len(splits)}")
                    print(f"splits shapes: {[x.shape for x in splits]}")
                    new_tensors.extend(splits)

                current_tensors = new_tensors
                for i, tensor in enumerate(current_tensors):
                    print(f"split {i} shape: {tensor.shape}")

        # Tensors are in the order of rank_enum
        # Return as a dictionary to ensure the correct tensor is selected for each rank
        enum_order_tensor_dict = dict(zip(rank_enum, current_tensors))
        print(f"enum_order_tensor_dict keys: {enum_order_tensor_dict.keys()}")
        print("*" * 100)
        return enum_order_tensor_dict
