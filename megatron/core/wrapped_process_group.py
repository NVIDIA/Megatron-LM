import logging
import traceback
import torch


from typing import List, Optional

from megatron.core.device_utils import get_xla_model

logger = logging.getLogger(__name__)
xm = get_xla_model()

class WrappedProcessGroup(object):
    """ Wrapped torch.distirbuted.ProcessGroup and rank groups for xla compatibility"""

    def __init__(self, process_group: torch.distributed.ProcessGroup=torch.distributed.group.WORLD,
                 rank_groups: Optional[List[List[int]]]=None):
        self.__process_group = process_group 
        self.__rank_groups = rank_groups if rank_groups else self.__all_rank_groups() if xm else None

    def __all_rank_groups(self) -> List[List[int]]:
        """Gets all the ranks of all processes in the group.

        Returns:
            range: A List[List[int]] of ranks in the process group.
        """
        
        try:
            if isinstance(self.__process_group, list):
                self.__rank_groups = None
                return None
            
            world_size = torch.distributed.get_world_size()
            world_ranks = [ r  for r in range(world_size)]
            if self.process_group is None or self.__process_group == torch.distributed.group.WORLD:
                return [world_ranks]

            group_ranks = torch.distributed.get_process_group_ranks(self.__process_group)
            group_ranks = [ r - group_ranks[0] for r in group_ranks]
            group_size = len(group_ranks)
            num_rank_groups = world_size // group_size
            strides = [ group_ranks[i+1] - group_ranks[i] for i in range(len(group_ranks) - 1)]
            all_ranks = []
            for _ in range(num_rank_groups):
                _ranks = [world_ranks[0]]
                for j in range(1,group_size,1):
                    _ranks.append(world_ranks[j-1]+ strides[j-1])
                all_ranks.append(_ranks)
                world_ranks = [ r for r in world_ranks if r not in _ranks]

            assert group_ranks in all_ranks
            return all_ranks
        except Exception as e:
            traceback.print_exc()
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