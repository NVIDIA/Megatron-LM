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
        
        # self.build_comm_schedule()

        self.activation_gather_ranks, self.activation_gather_pg = self._create_activation_gather_scatter_pg(self.src_grid, is_src=True)
        # self.activation_scatter_ranks, self.activation_scatter_pg = self._create_activation_gather_scatter_pg(self.dest_grid, is_src=False)
        

    def get_leader_rank(self, grid: HyperCommGrid, is_src: bool) -> List[int]:
        """Get the leader rank for a given grid and direction."""
        leader_ranks = []
        # grid.gen_rank_enum(["tp", "cp", "pp"]) # vary tp & cp, freeze dp
        # returns a list of sublists, each sublist is a group of ranks that have different tp & cp & pp, same dp
        per_dp_replica_ranks = grid._gen_rank_enum([x for x in grid.dim_names if x != "dp"])
        if is_src:
            # Add rank from last pp stage
            leader_ranks.extend(group[-1] for group in per_dp_replica_ranks)
        else:
            # Add rank from first pp stage
            leader_ranks.extend(group[0] for group in per_dp_replica_ranks)
        return leader_ranks

    def _create_activation_gather_scatter_pg(self, grid: HyperCommGrid, is_src: bool):
   
        current_rank  = dist.get_rank()
        pp_group_ranks = dist.get_process_group_ranks(grid.get_pg(['pp']))

        activation_comm_ranks = []
        activation_comm_pg = None

        # on the src grid, all tp-cp ranks of last pp stage and dp replica that current rank belongs to
        # participtes in the activation gather
        # on the dest grid, all tp-cp ranks of first pp stage and dp replica that current rank belongs to
        # participtes in the activation scatter

        if is_src and not current_rank == pp_group_ranks[-1]:
            # if current rank belongs to src grid. If not belongs to last pp stage, return empty ranks and pg
            return activation_comm_ranks, activation_comm_pg

        if not is_src and not current_rank == pp_group_ranks[0]:
            # if current rank belongs to dest grid. If not belongs to first pp stage, return empty ranks and pg
            return activation_comm_ranks, activation_comm_pg  

        all_tpcp_group_ranks = grid._gen_rank_enum(['tp', 'cp'])
        for each_group_ranks in all_tpcp_group_ranks:
            if current_rank in each_group_ranks:
                activation_comm_ranks = each_group_ranks
                break

        activation_comm_pg = dist.new_group(ranks=activation_comm_ranks)
        # activation_comm_pg = grid.get_pg(['tp', 'cp'])

        if current_rank == 0:
            breakpoint()
        dist.barrier()

        return activation_comm_ranks, activation_comm_pg



    

    def build_comm_schedule(self):
        """Get src/dest tp leaders and populate comm_map for each rank.
        
        This method analyzes the source and destination grids to determine
        which ranks need to send/receive data and builds the communication
        schedule accordingly.
        """
        src_tp_leaders = self.get_leader_rank(self.src_grid, is_src=True)
        dest_tp_leaders = self.get_leader_rank(self.dest_grid, is_src=False)
        # Ensure that the number of leaders can be evenly divided
        src_count = len(src_tp_leaders)
        dest_count = len(dest_tp_leaders)
        
        if src_count % dest_count != 0 and dest_count % src_count != 0:
            raise ValueError(
                f"Source TP leaders count ({src_count}) and destination TP leaders count "
                f"({dest_count}) must be evenly divisible. One must be a multiple of the other."
            )
        # Get all ranks in source and destination grids
        src_all_ranks = list(range(self.src_grid.rank_offset, 
                                  self.src_grid.rank_offset + self.src_grid.size))
        dest_all_ranks = list(range(self.dest_grid.rank_offset, 
                                   self.dest_grid.rank_offset + self.dest_grid.size))
       
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
                src_ranks = src_tp_leaders[i * scale_factor:(i + 1) * scale_factor]
                
                # Set up senders
                for src_rank in src_ranks:
                    self.comm_map[src_rank] = RankCommInfo(
                        role='SENDER',
                        sends=[SendOp(
                            destination_rank=dest_rank,
                            batch_slice=slice(None),
                            send_shape=(1,)  # placeholder
                        )]
                    )
                
                # Set up receiver
                self.comm_map[dest_rank] = RankCommInfo(
                    role='RECEIVER',
                    receives=[RecvOp(
                        source_rank=src_rank,
                        recv_shape=(1,)  # placeholder
                    ) for src_rank in src_ranks]
                )
        else:
            # Fan-out: fewer source leaders send to more destination leaders
            scale_factor = int(dest_count / src_count)
            for i, src_rank in enumerate(src_tp_leaders):
                # Each source rank sends to scale_factor destination ranks
                dest_ranks = dest_tp_leaders[i * scale_factor:(i + 1) * scale_factor]
                
                # Set up sender
                self.comm_map[src_rank] = RankCommInfo(
                    role='SENDER',
                    sends=[SendOp(
                        destination_rank=dest_rank,
                        batch_slice=slice(None),
                        send_shape=(1,)  # placeholder
                    ) for dest_rank in dest_ranks]
                )
                
                # Set up receivers
                for dest_rank in dest_ranks:
                    self.comm_map[dest_rank] = RankCommInfo(
                        role='RECEIVER',
                        receives=[RecvOp(
                            source_rank=src_rank,
                            recv_shape=(1,)  # placeholder
                        )]
                    )
        # Get the local rank (within the DP group) of the leader rank.
        # dp_group_global_ranks = dist.get_process_group_ranks(self.dp_pg)
        # for local_rank, global_rank in enumerate(dp_group_global_ranks):
        #     if global_rank in self.comm_map and self.comm_map[global_rank].role == 'SENDER':
        #         self.dp_leader_local_rank = local_rank

        if dist.get_rank() == 0:
            breakpoint()
        dist.barrier()

    def send_forward(self, tensor_to_send: torch.Tensor):
        """Send forward activation tensor.
        
        Args:
            tensor_to_send: The tensor to send to the destination grid
        """
        # Get current rank's communication info
        rank_info = self.comm_map.get(self.current_rank)
        if not rank_info:
            return
        
        self._communicate_shapes(tensor_to_send_next=tensor_to_send)
            
        # Get DP group information
        dp_size = dist.get_world_size(self.dp_pg)
        
        if rank_info.role == 'SENDER':
            # Current rank is a sender - gather tensors from all ranks in DP group
            gathered_tensors = [torch.zeros_like(tensor_to_send) for _ in range(dp_size)]
            dist.gather(tensor_to_send, gather_list=gathered_tensors, dst=self.dp_leader_local_rank, group=self.dp_pg)          
            # Determine which grid this rank belongs to
            current_grid = self.src_grid if self.current_rank in range(self.src_grid.rank_offset, 
                                                                      self.src_grid.rank_offset + self.src_grid.size) else self.dest_grid
            
            # Reconstruct tensor properly handling TP/CP dimensions
            aggregated_tensor = self._reconstruct_tensor_from_gathered(gathered_tensors, current_grid)
            
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
                    dist.send(tensor_split, dst=send_op.destination_rank)
                    
        elif rank_info.role == 'NOOP':
            dist.gather(tensor_to_send, gather_list=None, dst=self.dp_leader_local_rank, group=self.dp_pg)
                
    def receive_forward(self, tensor_shape: Optional[Tuple[int, ...]] = None, 
                       dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Receive forward activation tensor.
        
        Args:
            tensor_shape: Expected tensor shape (None if using shape communication)
            dtype: Expected tensor dtype
            
        Returns:
            torch.Tensor: The received activation tensor
        """
        pass

    def send_backward(self, grad_tensor: torch.Tensor, variable_seq_lengths: bool = False):
        """Send backward gradient tensor.
        
        Note: Gradient senders are activation 'RECEIVERS'
        
        Args:
            grad_tensor: The gradient tensor to send back
        """
        pass

    def receive_backward(self, tensor_shape: Optional[Tuple[int, ...]] = None,
                        dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Receive backward gradient tensor.
        
        Note: Gradient receivers are activation 'SENDERS'
        
        Args:
            tensor_shape: Expected gradient tensor shape
            dtype: Expected tensor dtype
            
        Returns:
            torch.Tensor: The received gradient tensor
        """
        pass

    def send_forward_recv_backward(self, input_tensor: torch.Tensor,
                                  grad_shape: Optional[Tuple[int, ...]] = None,
                                  dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Combined operation: send forward activation and receive backward gradient.
        
        Args:
            input_tensor: The tensor to send forward
            grad_shape: Expected gradient tensor shape
            dtype: Expected tensor dtype
            
        Returns:
            torch.Tensor: The received gradient tensor
        """
        pass

    def send_backward_recv_forward(self, grad_tensor: torch.Tensor,
                                  forward_shape: Optional[Tuple[int, ...]] = None,
                                  dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Combined operation: send backward gradient and receive forward activation.
        
        Args:
            grad_tensor: The gradient tensor to send backward
            forward_shape: Expected forward tensor shape
            dtype: Expected tensor dtype
            
        Returns:
            torch.Tensor: The received activation tensor
        """
        pass

    def _reconstruct_tensor_from_gathered(self, gathered_tensors: List[torch.Tensor], 
                                         grid: HyperCommGrid) -> torch.Tensor:
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
        ordered_tensors = [rank_to_tensor[rank] for rank in matching_enum_group if rank in rank_to_tensor]
        
        if not ordered_tensors:
            raise ValueError("No tensors found for the given ranks")
        
        # Simple concatenation approach based on grid dimensions
        return self._concatenate_by_grid_dims(ordered_tensors, grid, non_dp_dims)

    def _concatenate_by_grid_dims(self, tensors: List[torch.Tensor], 
                                 grid: HyperCommGrid, 
                                 non_dp_dims: List[str]) -> torch.Tensor:
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
        for (dim_name, dim_size) in zip(non_dp_dims, grid_shape):
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
        
        return current_tensors[0] if len(current_tensors) == 1 else torch.cat(current_tensors, dim=0)

    def _communicate_shapes(self, 
                           tensor_to_send_next: Optional[torch.Tensor] = None,
                           recv_next: bool = False,
                           recv_prev: bool = False,
                           tensor_to_send_prev: Optional[torch.Tensor] = None) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]:
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
                    ops.append(torch.distributed.P2POp(
                        torch.distributed.isend,
                        send_shape_tensor,
                        send_op.destination_rank
                    ))
                
                # If expecting gradients back, prepare receive operations
                if recv_next is not None:
                    for send_op in rank_info.sends:
                        grad_shape_tensor = torch.empty(
                            (3), device=torch.cuda.current_device(), dtype=torch.int64
                        )
                        recv_grad_shape_tensors.append(grad_shape_tensor)
                        ops.append(torch.distributed.P2POp(
                            torch.distributed.irecv,
                            grad_shape_tensor,
                            send_op.destination_rank
                        ))
                        
        elif rank_info.role == 'RECEIVER':
            # Prepare receive operations for forward shapes
            if recv_prev is not None:
                for recv_op in rank_info.receives:
                    forward_shape_tensor = torch.empty(
                        (3), device=torch.cuda.current_device(), dtype=torch.int64
                    )
                recv_forward_shape_tensors.append(forward_shape_tensor)
                ops.append(torch.distributed.P2POp(
                    torch.distributed.irecv,
                    forward_shape_tensor,
                    recv_op.source_rank
                ))
            
            # If we need to send gradient shapes back, prepare send operations
            if tensor_to_send_prev is not None:
                grad_shape = tensor_to_send_prev.shape
                grad_shape_tensor = torch.tensor(
                    grad_shape, device=torch.cuda.current_device(), dtype=torch.int64
                )
                
                for recv_op in rank_info.receives:
                    ops.append(torch.distributed.P2POp(
                        torch.distributed.isend,
                        grad_shape_tensor,
                        recv_op.source_rank
                    ))
        
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