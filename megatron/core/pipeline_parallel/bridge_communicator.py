# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

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
    """
    Facilitates communication of activations and gradients between two modules 
    with its own parallel mapping (TP/DP/PP/CP).
    
    The role of BridgeCommunicator:
    - Initializes the communicator between a pair of source and destination grids
    - Builds a communication schedule for each rank
    - Provides public methods: send_forward, recv_forward, send_forward_recv_backward, 
      send_backward_recv_forward to be used by the pipeline schedule.
    """
    
    def __init__(self, src_grid: HyperCommGrid, dest_grid: HyperCommGrid):
        """
        Initialize the bridge communicator between source and destination grids.
        
        Args:
            src_grid: Source HyperCommGrid
            dest_grid: Destination HyperCommGrid
        """
        self.src_grid = src_grid
        self.dest_grid = dest_grid
        self.comm_map: Dict[int, RankCommInfo] = {}
        
        self.build_comm_schedule()

    def get_leader_rank(self, grid: HyperCommGrid, is_src: bool) -> List[int]:
        """
        Get the leader rank for a given grid and direction.
        """
        leader_ranks = []
        dp_groups = grid._gen_rank_enum([x for x in grid.dim_names if x != "dp"])
        if is_src:
            # Add rank from last pp stage
            leader_ranks.extend(group[-1] for group in dp_groups)
        else:
            # Add rank from first pp stage
            leader_ranks.extend(group[0] for group in dp_groups)
        return leader_ranks

    def build_comm_schedule(self):
        """
        Get src/dest tp leaders and populate comm_map for each rank.
        
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
        # Combine all ranks from both grids
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

    def send_forward(self, tensor_to_send: torch.Tensor):
        """
        Send forward activation tensor.
        
        Args:
            tensor_to_send: The tensor to send to the destination grid
        """
        pass

    def receive_forward(self) -> torch.Tensor:
        """
        Receive forward activation tensor.
        
        Returns:
            torch.Tensor: The received activation tensor
        """
        pass

    def send_backward(self, grad_tensor: torch.Tensor):
        """
        Send backward gradient tensor.
        
        Note: Gradient senders are activation 'RECEIVERS'
        
        Args:
            grad_tensor: The gradient tensor to send back
        """
        pass

    def receive_backward(self) -> torch.Tensor:
        """
        Receive backward gradient tensor.
        
        Note: Gradient receivers are activation 'SENDERS'
        
        Returns:
            torch.Tensor: The received gradient tensor
        """
        pass

    def send_forward_recv_backward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Combined operation: send forward activation and receive backward gradient.
        
        Args:
            input_tensor: The tensor to send forward
            
        Returns:
            torch.Tensor: The received gradient tensor
        """
        pass

    def send_backward_recv_forward(self, grad_tensor: torch.Tensor) -> torch.Tensor:
        """
        Combined operation: send backward gradient and receive forward activation.
        
        Args:
            grad_tensor: The gradient tensor to send backward
            
        Returns:
            torch.Tensor: The received activation tensor
        """
        pass