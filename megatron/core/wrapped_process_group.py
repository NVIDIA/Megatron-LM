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
                self.__rank_groups = self.__process_group
                return
            
            world_size = torch.distributed.get_world_size()
            if self.__process_group is None or self.__process_group == torch.distributed.group.WORLD:
                return [range(world_size)]

            ranks = torch.distributed.get_process_group_ranks(self.__process_group)
            num_sub_groups = world_size // len(ranks)
            all_ranks = [ [ int(lr + len(ranks)*sg) for lr in range(len(ranks)) ] for sg in range(num_sub_groups)]
            assert ranks in all_ranks, f"{ranks} not in {all_ranks}"
            logger.info(f"process_group: {self.__process_group}, all_ranks: {all_ranks}")
            return all_ranks
        except Exception as e:
            logger.warning(f"process_group: {self.__process_group}")
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