# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator

# Types
Shape = Union[List[int], torch.Size]


@dataclass
class RankModuleInfo:
    """Information about a rank in a module.

    Attributes:
        pp_rank: The stage index of the current rank within the module's pipeline.
        pp_size: The total number of pipeline stages (ranks) in the module.
        p2p_communicator: Intra-module point-to-point communicator.
        bridge_comms_as_src_module: Bridge communicators for outgoing connections
            from this module to downstream modules. One module may have multiple
            bridge communicators if it has multiple outgoing connections.
        bridge_comms_as_dest_module: Bridge communicators for incoming connections
            to this module from upstream modules. One module may have multiple
            bridge communicators if it has multiple incoming connections.
        is_source_stage: True if this rank is at the absolute first stage in the
            overall model (no incoming connections).
        is_terminal_stage: True if this rank is at the absolute last stage in the
            overall model (no outgoing connections).
    """

    pp_rank: int
    pp_size: int
    p2p_communicator: Optional[P2PCommunicator]
    bridge_comms_as_src_module: Optional[List[BridgeCommunicator]]
    bridge_comms_as_dest_module: Optional[List[BridgeCommunicator]]
    is_source_stage: Optional[bool] = True
    is_terminal_stage: Optional[bool] = True


class MultiModulePipelineCommunicator:
    """Communicator for a multi-module pipeline."""

    def __init__(
        self,
        module_to_grid_map: Dict[str, HyperCommGrid],
        topology: Dict[str, List[str]],
        config: ModelParallelConfig,
        dim_mapping: Dict[str, List[int]] = None,
    ):
        """
        Initialize the MultiModulePipelineCommunicator.

        Args:
            module_to_grid_map (dict): A dictionary mapping module names to HyperCommGrids.
                Example:
                    module_to_grid_map = {
                        'image_encoder': image_encoder_grid,
                        'audio_encoder': audio_encoder_grid,
                        'llm': llm_grid,
                        'generator': generator_grid
                    }
            topology (dict): A dictionary mapping module names to lists of outgoing modules.
                Example:
                    topology = {
                        'image_encoder': ['llm'],
                        'audio_encoder': ['llm'],
                        'llm': ['generator'],
                        'generator': []
                    }
            config (ModelParallelConfig): A ModelParallelConfig object.
            dim_mapping (Dict[str, List[int]]): Dimension mapping for sequence, batch, hidden.
                Example:
                    dim_mapping = {'s': 0, 'h': 2, 'b': 1}
                Default: None
        """
        self.module_to_grid_map = module_to_grid_map
        self.topology = topology
        self.config = config
        self.dim_mapping = dim_mapping
        self.current_rank = dist.get_rank()

        # Build bridge communicators for all modules
        self.bridge_comms = []
        self._build_bridge_comms()

        self.rank_module_map = {}
        self._build_rank_module_info_map()

    def _build_bridge_comms(self):
        """Construct and store BridgeCommunicator objects that describe the outgoing
        communication relationships for all of the modules.
        """
        for src_module_name, src_grid in self.module_to_grid_map.items():
            for dest_module_name in self.topology[src_module_name]:
                dest_grid = self.module_to_grid_map[dest_module_name]
                bridge_comm = BridgeCommunicator(
                    src_grid=src_grid,
                    dest_grid=dest_grid,
                    dim_mapping=self.dim_mapping,
                    comm_dtype=self.config.pipeline_dtype,
                    src_module_name=src_module_name,
                    dest_module_name=dest_module_name,
                )
                self.bridge_comms.append(bridge_comm)

    @property
    def is_pp_first_stage(self):
        """Return True if the current rank has the absolute first stage in the overall model.

        The absolute first stage is defined as:
        1. The current rank must be in the first PP stage (pp_rank == 0) of some module
        2. That module must be a source module (no incoming connections in topology)
        """
        for module_name, rank_module_info in self.rank_module_map.items():
            # Check if this rank is at the first PP stage of this module
            if rank_module_info.pp_rank == 0:
                # Check if this module is a source module (no incoming connections)
                if self._is_source_module(module_name):
                    return True
        return False

    @property
    def is_pp_last_stage(self):
        """Return True if the current rank has the absolute last stage in the overall model.

        The absolute last stage is defined as:
        1. The current rank must be in the last PP stage of some module
        2. That module must be a sink module (no outgoing connections in topology)
        """
        for module_name, rank_module_info in self.rank_module_map.items():
            # Check if this rank is at the last PP stage of this module
            if rank_module_info.pp_rank == rank_module_info.pp_size - 1:
                # Check if this module is a sink module (no outgoing connections)
                if self._is_sink_module(module_name):
                    return True
        return False

    def _is_source_module(self, module_name: str) -> bool:
        """Check if a module is a source module (has no incoming connections)."""
        # A module is a source if no other module lists it as a destination
        for src_module, dest_modules in self.topology.items():
            if module_name in dest_modules:
                return False
        return True

    def _is_sink_module(self, module_name: str) -> bool:
        """Check if a module is a sink module (has no outgoing connections)."""
        return len(self.topology.get(module_name, [])) == 0

    def is_current_rank_in_grid(self, grid: HyperCommGrid) -> bool:
        """Check if the current rank is in the grid."""
        return grid.rank_offset <= self.current_rank < grid.rank_offset + grid.size

    @property
    def num_warmup_microbatches(self):
        """Calculate the number of warmup microbatches for the current rank.

        Uses the same simple logic as P2PCommunicator:
        total_pipeline_stages - current_rank_stage - 1

        Returns:
            int: Number of warmup microbatches for this rank
        """
        # Get total pipeline depth across all modules
        total_stages = self.compute_total_pipeline_stages(self.topology, self.module_to_grid_map)

        # Get current rank's position in the overall pipeline (0-indexed)
        # Use compute_total_pipeline_stages with current rank to get cumulative position
        if self.rank_module_map:
            # Take the first module this rank belongs to
            # TODO: ykarnati - improve this logic.
            module_name = next(iter(self.rank_module_map.keys()))
            current_stage = (
                self.compute_total_pipeline_stages(
                    self.topology,
                    self.module_to_grid_map,
                    rank=self.current_rank,
                    module_name=module_name,
                )
                - 1
            )  # Convert from 1-indexed to 0-indexed
        else:
            current_stage = 0

        assert (
            current_stage <= total_stages
        ), f"current_stage: {current_stage} is greater than total_stages: {total_stages}"
        logging.debug(
            f"[Rank {dist.get_rank()} ][MultiModulePipelineCommunicator] "
            f"current_stage: {current_stage} total_stages: {total_stages} "
            f"num_warmup_microbatches: {total_stages - current_stage - 1}"
        )
        return total_stages - current_stage - 1

    def _build_rank_module_info_map(self):
        """For each module in the current rank, initialize the P2P communicator
        and build the bridge communicator info for the module.
        Each rank may hold multiple modules when colocated.
        """
        for module_name, module_grid in self.module_to_grid_map.items():
            if self.is_current_rank_in_grid(module_grid):
                # Initialize P2P communicator
                pp_group = module_grid.get_pg('pp')
                p2p_comm = P2PCommunicator(pp_group, self.config)
                pp_size = dist.get_world_size(pp_group)
                rank_in_pp_group = dist.get_group_rank(pp_group, self.current_rank)
                pp_rank = rank_in_pp_group % pp_size

                bridge_comms_as_dest_module = []
                bridge_comms_as_src_module = []
                # If first stage, check if the module has any incoming modules
                # If so, initialize bridge communicator
                if pp_rank == 0:
                    for bridge_comm in self.bridge_comms:
                        if (
                            bridge_comm.is_current_rank_in_grid(bridge_comm.dest_grid)
                            and bridge_comm.dest_module_name == module_name
                        ):
                            bridge_comms_as_dest_module.append(bridge_comm)
                # If last stage, check if the module has any outgoing modules
                # If so, initialize bridge communicator
                if pp_rank == pp_size - 1:
                    for bridge_comm in self.bridge_comms:
                        if (
                            bridge_comm.is_current_rank_in_grid(bridge_comm.src_grid)
                            and bridge_comm.src_module_name == module_name
                        ):
                            bridge_comms_as_src_module.append(bridge_comm)
                # Build RankModuleInfo for the module
                rank_module_info = RankModuleInfo(
                    pp_rank=pp_rank,
                    pp_size=pp_size,
                    p2p_communicator=p2p_comm,
                    bridge_comms_as_dest_module=bridge_comms_as_dest_module,
                    bridge_comms_as_src_module=bridge_comms_as_src_module,
                )
                self.rank_module_map[module_name] = rank_module_info

    def recv_forward(
        self, tensor_shape: Optional[Shape] = None, is_first_stage: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Receive forward activation tensor.

        Args:
            tensor_shape: Expected activation tensor shape

        Returns:
            A dictionary mapping module names to tensors.
        """
        logging.debug(
            f"[Rank {dist.get_rank()} ][MultiModulePipelineCommunicator] "
            f"[receive_forward] tensors_shape: {tensor_shape}, is_first_stage: {is_first_stage}"
        )
        input_dict = {}
        for module_name, rank_module_info in self.rank_module_map.items():

            if rank_module_info.pp_rank == 0:
                # If first stage, and has incoming modules, receive forward activation
                # from incoming modules.
                for bridge_comm in rank_module_info.bridge_comms_as_dest_module:
                    input_dict[bridge_comm.src_module_name] = bridge_comm.recv_forward()
            else:
                # If not first stage, receive forward activation tensor from P2P communicator.
                input_dict[module_name] = rank_module_info.p2p_communicator.recv_forward(
                    tensor_shapes=tensor_shape, is_first_stage=False
                )
        return input_dict

    def send_forward(self, output_dict: Dict[str, torch.Tensor], is_last_stage: bool = False):
        """Send forward activation tensor.

        Args:
            output_dict: A dictionary mapping module names to tensors.
        """
        logging.debug(
            f"[Rank {dist.get_rank()} ][MultiModulePipelineCommunicator] "
            f"[send_forward] output_dict keys: {output_dict.keys()}, is_last_stage: {is_last_stage}"
        )
        for module_name, rank_module_info in self.rank_module_map.items():
            if rank_module_info.pp_rank == rank_module_info.pp_size - 1:
                # If last stage, and has outgoing modules, send forward activation
                # by using bridge communicator.
                for bridge_comm in rank_module_info.bridge_comms_as_src_module:
                    bridge_comm.send_forward(output_dict[module_name])
            else:
                # If not last stage, send forward activation by using P2P communicator.
                rank_module_info.p2p_communicator.send_forward(
                    output_dict[module_name], is_last_stage=False
                )

    def send_forward_recv_backward(
        self,
        output_dict: Dict[str, torch.Tensor],
        tensor_shape: Optional[Shape] = None,
        is_last_stage: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Send forward activation tensor and receive backward activation tensor.

        Args:
            output_dict: A dictionary mapping module names to tensors.
            tensor_shape: Expected gradient tensor shape

        Returns:
            A dictionary mapping module names to tensors.
        """
        logging.debug(
            f"[Rank {dist.get_rank()} ][MultiModulePipelineCommunicator] "
            f"[send_forward_recv_backward] output_dict keys: {output_dict.keys()}, "
            f"tensor_shape: {tensor_shape}, is_last_stage: {is_last_stage}"
        )
        grad_dict = {}
        for module_name, rank_module_info in self.rank_module_map.items():
            if rank_module_info.pp_rank == rank_module_info.pp_size - 1:
                # If last stage, and has outgoing modules, send forward activation and
                # receive backward gradient by using bridge communicator.
                for bridge_comm in rank_module_info.bridge_comms_as_src_module:
                    grad_dict[bridge_comm.src_module_name] = bridge_comm.send_forward_recv_backward(
                        output_dict[module_name]
                    )
            else:
                # If not last stage, send forward activation and receive backward gradient
                # by using P2P communicator.
                grad_dict[module_name] = (
                    rank_module_info.p2p_communicator.send_forward_recv_backward(
                        output_dict[module_name], tensor_shapes=tensor_shape, is_last_stage=False
                    )
                )
        return grad_dict

    def send_backward_recv_forward(
        self,
        grad_dict: Dict[str, torch.Tensor],
        tensor_shape: Optional[Shape] = None,
        is_first_stage: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Send backward activation tensor and receive forward activation tensor.

        Args:
            grad_dict: A dictionary mapping module names to tensors.
            tensor_shape: Expected gradient tensor shape

        Returns:
            A dictionary mapping module names to tensors.
        """
        logging.debug(
            f"[Rank {dist.get_rank()} ][MultiModulePipelineCommunicator] "
            f"[send_backward_recv_forward] grad_dict keys: {grad_dict.keys()}, "
            f"tensor_shape: {tensor_shape}, is_first_stage: {is_first_stage}"
        )
        input_dict = {}
        for module_name, rank_module_info in self.rank_module_map.items():
            if rank_module_info.pp_rank == 0:
                for bridge_comm in rank_module_info.bridge_comms_as_dest_module:
                    # If first stage, and has incoming modules, send backward gradient and
                    # receive forward activation by using bridge communicator.
                    input_dict[bridge_comm.src_module_name] = (
                        bridge_comm.send_backward_recv_forward(
                            grad_dict[bridge_comm.src_module_name]
                        )
                    )
            else:
                # If not first stage, send backward gradient and receive forward activation
                # by using P2P communicator.
                input_dict[module_name] = (
                    rank_module_info.p2p_communicator.send_backward_recv_forward(
                        grad_dict[module_name], tensor_shapes=tensor_shape, is_first_stage=False
                    )
                )
        return input_dict

    def recv_backward(
        self, tensor_shape: Optional[Shape] = None, is_last_stage: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Receive backward activation tensor.

        Args:
            tensor_shape: Expected gradient tensor shape

        Returns:
            A dictionary mapping module names to tensors.
        """
        logging.debug(
            f"[Rank {dist.get_rank()} ][MultiModulePipelineCommunicator] "
            f"[recv_backward] tensor_shape: {tensor_shape}, is_last_stage: {is_last_stage}"
        )
        grad_dict = {}
        for module_name, rank_module_info in self.rank_module_map.items():
            if rank_module_info.pp_rank == rank_module_info.pp_size - 1:
                # If last stage, and has incoming modules, receive backward gradient
                # by using bridge communicator.
                for bridge_comm in rank_module_info.bridge_comms_as_src_module:
                    grad_dict[bridge_comm.src_module_name] = bridge_comm.recv_backward()
            else:
                # If not last stage, receive backward gradient by using P2P communicator.
                grad_dict[module_name] = rank_module_info.p2p_communicator.recv_backward(
                    tensor_shapes=tensor_shape, is_last_stage=False
                )
        return grad_dict

    def send_backward(self, grad_dict: Dict[str, torch.Tensor], is_first_stage: bool = False):
        """Send backward activation tensor.

        Args:
            grad_dict: A dictionary mapping module names to tensors.
        """
        logging.debug(
            f"[Rank {dist.get_rank()} ][MultiModulePipelineCommunicator] "
            f"[send_backward] grad_dict keys: {grad_dict.keys()}, is_first_stage: {is_first_stage}"
        )
        for module_name, rank_module_info in self.rank_module_map.items():
            if rank_module_info.pp_rank == 0:
                # If first stage, and has incoming modules, send backward activation
                # by using bridge communicator.
                for bridge_comm in rank_module_info.bridge_comms_as_dest_module:
                    bridge_comm.send_backward(grad_dict[bridge_comm.src_module_name])
            else:
                # If not first stage, send backward activation by using P2P communicator.
                rank_module_info.p2p_communicator.send_backward(
                    grad_dict[module_name], is_first_stage=False
                )

    @staticmethod
    def compute_total_pipeline_stages(
        topology: Dict[str, List[str]],
        module_to_grid_map: Dict[str, HyperCommGrid],
        rank: Optional[int] = None,
        module_name: Optional[str] = None,
    ) -> int:
        """Compute the total number of pipeline stages across a multi-module chain.

        Interprets ``topology`` as a directed acyclic graph (DAG) where nodes are modules
        and edges indicate forward data flow from source to destination modules. Each node
        is assigned a weight equal to its pipeline parallel size (number of PP stages).

        The total number of stages is defined as the length of the longest path in this DAG
        under node weights.

        If ``rank`` is None (default), returns the maximum over all terminal (sink) modules of
        the sum of PP sizes along a path ending at that terminal. For example, given:

            image_encoder ->\
                              -> llm -> generator
            audio_encoder  ->/

        the total is: max(pp(image_encoder), pp(audio_encoder)) + pp(llm) + pp(generator).

        If ``rank`` is provided, the result is the total number of pipeline stages up to (and
        including) the PP stage that ``rank`` occupies inside its module. In this case, the
        weight of the target module equals (pp_rank_index(rank) + 1) instead of the module's
        full PP size; other modules still contribute their full PP sizes. If the rank belongs to
        multiple modules (colocation), pass ``module_name`` to disambiguate; otherwise the
        maximum across all candidate modules containing the rank is returned.

        Args:
            topology: Mapping from a module to its list of outgoing modules.
            module_to_grid_map: Mapping from module name to its ``HyperCommGrid``.

        Returns:
            The total number of pipeline stages along the longest path given the constraints.

        Raises:
            ValueError: If the topology contains cycles; or has no terminal nodes when
                ``rank`` is None
        """
        nodes = set(module_to_grid_map.keys())
        # Build adjacency and reverse-adjacency (predecessors).
        adj: Dict[str, List[str]] = {node: list(topology.get(node, [])) for node in nodes}
        preds: Dict[str, List[str]] = {node: [] for node in nodes}
        for src, outs in adj.items():
            for dst in outs:
                preds[dst].append(src)

        # Identify terminal nodes (no outgoing edges) for the rank=None case.
        sinks = [node for node, outs in adj.items() if not outs]
        if rank is None and not sinks:
            raise ValueError(
                "Topology must be a DAG with at least one terminal (no outgoing) module."
            )

        def pp_size(name: str) -> int:
            grid = module_to_grid_map[name]
            pp_dim_index = grid.dim_names.index('pp')
            return grid.shape[pp_dim_index]

        def partial_weight_for_target(target: str) -> Optional[int]:
            if rank is None:
                return None
            grid = module_to_grid_map.get(target)
            rank_groups = grid._gen_rank_enum(['pp'])
            stage_index: Optional[int] = None
            for group in rank_groups:
                if rank in group:
                    stage_index = group.index(rank)
                    break
            return stage_index + 1

        def longest_path_to(target: str) -> int:
            visiting = set()
            partial = partial_weight_for_target(target)

            def weight(name: str) -> int:
                if partial is not None and name == target:
                    return partial
                return pp_size(name)

            def dfs(node: str) -> int:
                if node in visiting:
                    raise ValueError("Topology contains cycles; expected a DAG.")
                visiting.add(node)
                best = 0
                for p in preds.get(node, []):
                    val = dfs(p)
                    if val > best:
                        best = val
                visiting.remove(node)
                return weight(node) + best

            return dfs(target)

        if rank is None:
            return max(longest_path_to(sink) for sink in sinks)

        return longest_path_to(module_name)
