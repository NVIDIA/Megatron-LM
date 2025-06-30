# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

import torch

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

    def build_comm_schedule(self):
        """
        Get src/dest tp leaders and populate comm_map for each rank.
        
        This method analyzes the source and destination grids to determine
        which ranks need to send/receive data and builds the communication
        schedule accordingly.
        """
        
        # Initialize all ranks as NOOP by default
        
        # TODO: Implement the actual logic to determine src/dest tp leaders
        # and populate the communication schedule based on the grid configurations
        
        # Example placeholder logic - this needs to be implemented based on 
        # the actual grid topology and parallelism mappings
        
        # Example: if current rank is a source leader
        # src_leader_rank = self._get_src_leader_rank()
        # dest_leader_rank = self._get_dest_leader_rank()
        
        # if current_rank == src_leader_rank:
        #     self.comm_map[current_rank] = RankCommInfo(
        #         role='SENDER',
        #         sends=[SendOp(
        #             destination_rank=dest_leader_rank, 
        #             batch_slice=slice(None), 
        #             send_shape=(1,)  # placeholder
        #         )]
        #     )
        
        # elif current_rank == dest_leader_rank:
        #     self.comm_map[current_rank] = RankCommInfo(
        #         role='RECEIVER',
        #         receives=[RecvOp(
        #             source_rank=src_leader_rank, 
        #             recv_shape=(1,)  # placeholder
        #         )]
        #     )

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