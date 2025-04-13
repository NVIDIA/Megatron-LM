# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Dataclasses for organizing model parallelism and gradient communication process groups."""

from dataclasses import dataclass, field
import logging
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

class WrappedProcessGroup(object):
    """ Wrapped torch.distirbuted.ProcessGroup and rank groups for xla compatibility"""

    def __init__(self, process_group: torch.distributed.ProcessGroup=torch.distributed.group.WORLD, 
                 rank_groups: Optional[List[List[int]]]=None):
        self.__process_group = process_group
        self.__rank_groups = rank_groups if rank_groups else self.__all_rank_groups()

    def __all_rank_groups(self) -> List[List[int]]:
        """Gets all the ranks of all processes in the group.

        Returns:
            range: A List[List[int]] of ranks in the process group.
        """
        try:
            world_size = torch.distributed.get_world_size()
            if self.__process_group == torch.distributed.group.WORLD:
                return [range(world_size)]
            
            ranks = torch.distributed.get_process_group_ranks(self.__process_group)
            num_sub_groups = world_size // len(ranks)
            all_ranks = [ [ int(lr + len(ranks)*sg) for lr in range(len(ranks)) ] for sg in range(num_sub_groups)]
            assert ranks in all_ranks, f"{ranks} not in {all_ranks}"
            logger.info(f"process_group: {self.__process_group}, all_ranks: {all_ranks}")
            return all_ranks
        except Exception as e:
            logger.warning(str(e))
            return None
    
    @property
    def rank_groups(self):
        return self.__rank_groups
    
    @property
    def process_group(self):
        return self.__process_group
    
    def __getattr__(self, name):
        if name == 'process_group':
            return self.__process_group
        
        if name == 'rank_groups':
            return self.rank_groups
        
        return getattr(self.__process_group, name)

@dataclass
class ModelCommProcessGroups:
    """Process groups for transformer model parallelism.

    Fields use init=False and must be set after instance creation.

    Args:
        tp: Tensor parallel process group
        pp: Pipeline parallel process group
        mp: Model parallel group (tensor + pipeline)
        embd: Embedding process group
        pos_embd: Position embedding process group
        cp: Context parallel process group
        tp_cp: Tensor and context parallel group
        hcp: Hierarchical context parallel groups
        ep: Expert model parallel group
        expt_tp: Expert tensor parallel group
        tp_ep: Tensor and expert parallel group
        tp_ep_pp: Tensor, expert, and pipeline parallel group

    Example:
        # Create instance and set needed process groups
        model_pgs = ModelCommProcessGroups()
        model_pgs.tp = tp_group
        model_pgs.pp = pp_group

        # Pass to model components
        model = TransformerModel(..., process_groups=model_pgs)
    """

    # _TENSOR_MODEL_PARALLEL_GROUP
    tp: WrappedProcessGroup = field(init=False)

    # _PIPELINE_MODEL_PARALLEL_GROUP
    pp: WrappedProcessGroup = field(init=False)

    # _MODEL_PARALLEL_GROUP
    mp: torch.distributed.ProcessGroup = field(init=False)

    # _EMBEDDING_GROUP
    embd: WrappedProcessGroup = field(init=False)

    # _POSITION_EMBEDDING_GROUP
    pos_embd: WrappedProcessGroup = field(init=False)

    # _CONTEXT_PARALLEL_GROUP
    cp: WrappedProcessGroup = field(init=False)

    # _TENSOR_AND_CONTEXT_PARALLEL_GROUP
    tp_cp: WrappedProcessGroup = field(init=False)

    # _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS
    hcp: List[WrappedProcessGroup] = field(init=False)

    # _EXPERT_MODEL_PARALLEL_GROUP
    ep: WrappedProcessGroup = field(init=False)

    # _EXPERT_TENSOR_PARALLEL_GROUP
    expt_tp: WrappedProcessGroup = field(init=False)

    # _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP
    tp_ep: WrappedProcessGroup = field(init=False)

    # _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP
    tp_ep_pp: WrappedProcessGroup = field(init=False)


@dataclass
class GradCommProcessGroups:
    """Process groups for gradient communication in distributed training.

    Fields use init=False and must be set after instance creation.

    Args:
        dp: Data parallel process group
        dp_cp: Data and context parallel group
        expt_dp: Expert data parallel group
        intra_dp_cp: Intra partial data parallel group
        inter_dp_cp: Inter partial data parallel group

    Example:
        # Create instance and set needed process groups
        grad_pgs = GradCommProcessGroups()
        grad_pgs.dp = dp_group

        # Pass to distributed data parallel wrapper
        ddp_model = DistributedDataParallel(..., process_groups=grad_pgs)
    """

    # _DATA_PARALLEL_GROUP
    dp: WrappedProcessGroup = field(init=False)

    # _DATA_PARALLEL_GROUP_WITH_CP
    dp_cp: WrappedProcessGroup = field(init=False)

    # _EXPERT_DATA_PARALLEL_GROUP
    expt_dp: WrappedProcessGroup = field(init=False)

    # _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP
    intra_dp_cp: WrappedProcessGroup = field(init=False)

    # _INTER_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP
    inter_dp_cp: WrappedProcessGroup = field(init=False)
