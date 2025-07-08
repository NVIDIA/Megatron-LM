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

    def __init__(self, src_grid: HyperCommGrid, dest_grid: HyperCommGrid):
        """Initialize the bridge communicator between source and destination grids.

        Args:
            src_grid: Source HyperCommGrid
            dest_grid: Destination HyperCommGrid
        """
        self.src_grid = src_grid
        self.dest_grid = dest_grid
        self.current_rank = dist.get_rank()
        self.comm_map: Dict[int, RankCommInfo] = {}

        self.activation_gather_ranks, self.activation_gather_pg = (
            self._create_activation_gather_scatter_pg(self.src_grid, is_src=True)
        )
        self.activation_scatter_ranks, self.activation_scatter_pg = (
            self._create_activation_gather_scatter_pg(self.dest_grid, is_src=False)
        )

        self.src_tp_leaders, self.src_local_leader_rank = self.get_leader_rank(
            self.src_grid, is_src=True
        )
        self.dest_tp_leaders, self.dest_local_leader_rank = self.get_leader_rank(
            self.dest_grid, is_src=False
        )

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

    def _create_activation_gather_scatter_pg(self, grid: HyperCommGrid, is_src: bool):

        if (
            self.current_rank < grid.rank_offset
            or self.current_rank >= grid.rank_offset + grid.size
        ):
            return [], None

        pp_group_ranks = dist.get_process_group_ranks(grid.get_pg(['pp']))

        activation_comm_ranks = []
        activation_comm_pg = None

        # on the src grid, all tp-cp ranks of last pp stage and dp replica that current rank belongs to
        # participtes in the activation gather
        # on the dest grid, all tp-cp ranks of first pp stage and dp replica that current rank belongs to
        # participtes in the activation scatter

        if is_src and not self.current_rank == pp_group_ranks[-1]:
            # if current rank belongs to src grid. If not belongs to last pp stage, return empty ranks and pg
            return activation_comm_ranks, activation_comm_pg

        if not is_src and not self.current_rank == pp_group_ranks[0]:
            # if current rank belongs to dest grid. If not belongs to first pp stage, return empty ranks and pg
            return activation_comm_ranks, activation_comm_pg

        all_tpcp_group_ranks = grid._gen_rank_enum(['tp', 'cp'])
        for each_group_ranks in all_tpcp_group_ranks:
            if self.current_rank in each_group_ranks:
                activation_comm_ranks = each_group_ranks
                break

        activation_comm_pg = dist.new_group(ranks=activation_comm_ranks)
        # activation_comm_pg = grid.get_pg(['tp', 'cp'])

        return activation_comm_ranks, activation_comm_pg

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

        # self._communicate_shapes(tensor_to_send_next=tensor_to_send)
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

            # TODO: ykarnati -  naive concat for now - FIXME
            aggregated_tensor = torch.cat(gathered_tensors, dim=0)
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

            received_tensors_list = []
            for each_receive_op in rank_info.receives:
                tensor_to_recv = torch.empty(
                    tensor_shape, device=torch.cuda.current_device(), dtype=dtype
                )
                dist.recv(tensor_to_recv, src=each_receive_op.source_rank)
                print(
                    f"rank {self.current_rank} received tensor from src rank {each_receive_op.source_rank} shape {tensor_to_recv.shape} sum {tensor_to_recv.sum()}"
                )
                received_tensors_list.append(tensor_to_recv)

            # TODO: ykarnati - naive concat for now - FIXME
            aggregated_tensor = torch.cat(received_tensors_list, dim=0)
            print(
                f"rank {self.current_rank} aggregated tensor shape {aggregated_tensor.shape} sum {aggregated_tensor.sum()}"
            )

            scatter_list = list(
                torch.chunk(aggregated_tensor, chunks=len(self.activation_scatter_ranks), dim=0)
            )
            received_tensor = torch.empty_like(scatter_list[0])
            print('*' * 100)
            print(
                f"rank {self.current_rank} scatter list shape {[x.shape for x in scatter_list]} doing the scatter in rank group {dist.get_process_group_ranks(self.activation_scatter_pg)}"
            )
            print('*' * 100)
            # Send each tensor in scatter_list to corresponding ranks in activation_scatter_pg
            scatter_ranks = dist.get_process_group_ranks(self.activation_scatter_pg)

            # Collect all send requests for parallel execution
            send_requests = []

            for i, tensor_chunk in enumerate(scatter_list):
                target_rank = scatter_ranks[i]
                if target_rank != self.current_rank:
                    # Use asynchronous send for parallel execution
                    req = dist.isend(
                        tensor_chunk, dst=target_rank, group=self.activation_scatter_pg
                    )
                    send_requests.append(req)
                    print(
                        f"rank {self.current_rank} initiated send of tensor chunk {i} to rank {target_rank} shape {tensor_chunk.shape}"
                    )
                else:
                    # If sending to self, just copy the tensor
                    received_tensor = tensor_chunk.clone()
                    print(
                        f"rank {self.current_rank} kept tensor chunk {i} for self, shape {tensor_chunk.shape}"
                    )

            # Wait for all sends to complete
            for req in send_requests:
                req.wait()

        elif rank_info.role == 'NOOP' and self.current_rank in self.activation_scatter_ranks:
            # TODO: ykarnati - naive scatter for now - FIXME
            # we dont always evenly divide the tensor into chunks.
            # This is WRONG - just to test the comms
            scatter_shape = list(tensor_shape)
            scatter_shape[0] //= len(self.activation_scatter_ranks)

            received_tensor = torch.empty(
                tuple(scatter_shape), device=torch.cuda.current_device(), dtype=dtype
            )
            dist.recv(
                received_tensor, src=self.dest_local_leader_rank, group=self.activation_scatter_pg
            )
            print(
                f"rank {self.current_rank} received tensor from leader rank {self.dest_local_leader_rank} shape {received_tensor.shape}"
            )

    def send_backward(self, grad_tensor: torch.Tensor, variable_seq_lengths: bool = False):
        """Send backward gradient tensor.

        Note: Gradient senders are activation 'RECEIVERS'

        Args:
            grad_tensor: The gradient tensor to send back
        """
        pass

    def receive_backward(
        self, tensor_shape: Optional[Tuple[int, ...]] = None, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Receive backward gradient tensor.

        Note: Gradient receivers are activation 'SENDERS'

        Args:
            tensor_shape: Expected gradient tensor shape
            dtype: Expected tensor dtype

        Returns:
            torch.Tensor: The received gradient tensor
        """
        pass

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
        pass

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
        pass

    def _reconstruct_tensor_from_gathered(
        self, gathered_tensors: List[torch.Tensor], grid: HyperCommGrid
    ) -> torch.Tensor:
        """Reconstruct tensor using the grid's native rank enumeration logic."""
        # Get all non-DP dimensions that were split
        non_dp_dims = [dim for dim in grid.dim_names if dim != "dp"]

        if not non_dp_dims:
            return gathered_tensors[0]

        # Create rank enumeration for non-DP dimensions
        rank_enum = grid._gen_rank_enum(non_dp_dims)
        dp_group_ranks = dist.get_process_group_ranks(self.dp_pg)
        # Find which enumeration group our DP group belongs to
        dp_group_set = set(dp_group_ranks)
        matching_enum_group = None
        for enum_group in rank_enum:
            if dp_group_set.issubset(set(enum_group)):
                matching_enum_group = enum_group
                break

        if not matching_enum_group:
            raise ValueError("No matching enumeration group found")

        # Create mapping from rank to tensor and order by enumeration
        rank_to_tensor = dict(zip(dp_group_ranks, gathered_tensors))
        ordered_tensors = [
            rank_to_tensor[rank] for rank in matching_enum_group if rank in rank_to_tensor
        ]

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
        # TODO: should we infer the dims order from the grid?
        dim_mapping = {'tp': -1, 'cp': 1, 'ep': -1}  # TP/EP: hidden dim, CP: sequence dim

        # Get grid shape for reconstruction
        grid_shape = [grid.shape[grid.dim_names.index(dim)] for dim in non_dp_dims]

        # Reshape tensor list to match grid structure
        current_tensors = tensors

        # Process each grid dimension
        for dim_name, dim_size in zip(non_dp_dims, grid_shape):
            if dim_name in dim_mapping:
                tensor_dim = dim_mapping[dim_name]
                # Group tensors for this dimension and concatenate
                grouped_tensors = []
                group_size = len(current_tensors) // dim_size

                for group_idx in range(group_size):
                    group_start = group_idx * dim_size
                    group_end = group_start + dim_size
                    group = current_tensors[group_start:group_end]
                    grouped_tensors.append(torch.cat(group, dim=tensor_dim))

                current_tensors = grouped_tensors

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

                # Add send operations for each destination
                for send_op in rank_info.sends:
                    ops.append(
                        torch.distributed.P2POp(
                            torch.distributed.isend, send_shape_tensor, send_op.destination_rank
                        )
                    )

                # If expecting gradients back, prepare receive operations
                if recv_next is not None:
                    for send_op in rank_info.sends:
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
            if recv_prev is not None:
                for recv_op in rank_info.receives:
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
                    ops.append(
                        torch.distributed.P2POp(
                            torch.distributed.isend, grad_shape_tensor, recv_op.source_rank
                        )
                    )

        # Execute all operations in a single batch
        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # To protect against race condition when using batch_isend_irecv()
        # Following the pattern from the original p2p communication code
        torch.cuda.synchronize()

        # Extract shapes from received tensors
        if rank_info.role == 'RECEIVER':
            for forward_shape_tensor in recv_forward_shape_tensors:
                shape = forward_shape_tensor.tolist()
                recv_forward_shapes.append(tuple(shape))

        if rank_info.role == 'SENDER' and tensor_to_send_prev is not None:
            for grad_shape_tensor in recv_grad_shape_tensors:
                shape = grad_shape_tensor.tolist()
                recv_grad_shapes.append(tuple(shape))

        return recv_forward_shapes, recv_grad_shapes


if __name__ == "__main__":
    import os

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    def create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=1):
        """Create a HyperCommGrid with tensor parallelism=2, context parallelism=2, and data parallelism=2.

        Returns:
            HyperCommGrid: A grid configured with tp=2, cp=2, dp=2 (total size = 8).
        """
        # Set up environment for world size 8 if not already set
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = "8"

        grid = HyperCommGrid(
            shape=[tp, cp, pp, dp],
            dim_names=["tp", "cp", "pp", "dp"],
            rank_offset=offset,
            backend="nccl",
        )
        # print(grid)
        _ = grid.create_pg(["tp"])
        _ = grid.create_pg(["cp"])
        _ = grid.create_pg(["pp"])
        _ = grid.create_pg(["dp"])
        return grid

    grid1 = create_hypercomm_grid(offset=0, tp=2, cp=2, pp=1, dp=1)
    grid2 = create_hypercomm_grid(offset=4, tp=2, cp=2, pp=1, dp=1)
    bridge_communicator = BridgeCommunicator(grid1, grid2)
    assert bridge_communicator.src_grid == grid1
    assert bridge_communicator.dest_grid == grid2
    assert bridge_communicator.current_rank == dist.get_rank()
    # assert bridge_communicator.comm_map is not None

    random_hidden_state = torch.randn(16, 256, 1024).cuda()  # (batch_size, seq_len, hidden_size)
    current_rank = dist.get_rank()
    if bridge_communicator.is_current_rank_in_grid(bridge_communicator.src_grid):
        bridge_communicator.send_forward(random_hidden_state)
    else:
        bridge_communicator.receive_forward(
            tensor_shape=(64, 256, 1024), dtype=random_hidden_state.dtype
        )
        # recv fwd once we implement

    # kill distributed
    dist.destroy_process_group()
