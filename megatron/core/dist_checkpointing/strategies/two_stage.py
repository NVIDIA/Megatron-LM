# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" 2-stage checkpoint loading. """
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import partial, wraps
from itertools import chain
from logging import getLogger
from operator import attrgetter, itemgetter
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch

from ..dict_utils import dict_list_map_inplace, map_reduce, nested_values
from ..mapping import ShardedStateDict, ShardedTensor
from .base import LoadShardedStrategy
from .tensorstore import _load_from_array, open_ts_array
from .zarr import flatten_range, load_zarr_based_sharded_metadata

_import_trigger = None


timers = defaultdict(list)

logger = getLogger(__name__)
logger.warning(
    'megatron.core.dist_checkpointing.two_stage module is deprecated'
    ' and will be removed in Megatron-Core v0.12. Please use'
    ' FullyParallelLoadStrategyWrapper to accomplish a parallelized checkpoint load.'
)


def timed(verbose=True):
    """Timing decorator."""

    def timed_dec(fn):
        name = fn.__name__

        @wraps(fn)
        def wrapped(*args, **kwargs):
            if verbose:
                logger.debug(f'{name} init')
            start = time.time()
            ret = fn(*args, **kwargs)
            took = time.time() - start
            if verbose:
                logger.debug(f'{name} took {took}s')
            timers[name].append(took)
            return ret

        return wrapped

    return timed_dec


@dataclass
class _ShardedTensorMetadata:
    global_rank: int
    sharded_tensor_no_data: ShardedTensor
    dist_group_rank: Tuple[int]  # id of distributed group
    dist_group_ranks: Tuple[int]  # id of distributed group
    data_size: Optional[int] = None  # bytes


def sharded_tensor_chunk_id(sharded_tensor: ShardedTensor):
    """Id of a sharded tensor."""
    return (sharded_tensor.key, sharded_tensor.global_offset)


class TwoStageDataParallelLoadShardedStrategy(LoadShardedStrategy):
    """Loads one checkpoint replica from storage and broadcasts to other nodes.

    This strategy loads checkpoint from storage on minimal set of nodes
    and distributes the checkpoint to other nodes with torch.distributed.
    Loading is performed with tensorstore.

    Steps:
    0. (optional) create Gloo distributed groups
    1. Exchange ShardedTensors metadata between all nodes
    2. Align needed tensors within DP groups
    3. For each globally unique tensor:
    3.a) on one of the ranks load it from storage to CPU and move to CUDA
    3.b) allocate CUDA tensor on other ranks
    3.c) broadcast within DP group
    3.d) copy tensor content to the model param location
    3.e) free tensor buffers from a) and b)

    Notes:
    1. Loading and broadcasting is done sequentially to avoid both host and device OOMs
    2. There is a lot of overlap potential between all three steps done for each tensor:
    2.a) loading from storage to numpy
    2.b) moving CPU tensors to CUDA
    2.c) broadcast
    """

    def __init__(self, data_parallel_group, cpu_transfer=True):
        super().__init__()

        self.cpu_transfer = cpu_transfer
        self.data_parallel_group_orig = data_parallel_group
        self.data_parallel_group = None if cpu_transfer else data_parallel_group
        self.dp_group_ranks = tuple(
            sorted(torch.distributed.get_process_group_ranks(data_parallel_group))
        )
        self.dp_group_rank = self.data_parallel_group_orig.rank()
        self.global_rank = torch.distributed.get_rank()

    def load(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path):
        """Main load method."""
        self.maybe_init_gloo_group()
        all_tensors_sorted = self._build_load_plan(sharded_state_dict)
        self._exchange_loaded_tensors(all_tensors_sorted, sharded_state_dict, checkpoint_dir)
        # TODO: fix hang in summarize_load_times
        # self.summarize_load_times()
        return sharded_state_dict

    def summarize_load_times(self):
        """Summarize load times."""
        torch.distributed.barrier()
        logger.info('Checkpoint loading finished. Summary:')
        # TODO: `timers` keys are not guaranteed to be the same across ranks which causes hangs
        for key, times in sorted(timers.items()):
            times_sum = sum(times)
            max_times = torch.tensor([times_sum], device='cuda')
            avg_times = torch.tensor([times_sum], device='cuda')
            torch.distributed.all_reduce(max_times, op=torch.distributed.ReduceOp.MAX)
            torch.distributed.all_reduce(avg_times, op=torch.distributed.ReduceOp.SUM)
            avg_times /= torch.distributed.get_world_size()
            if torch.distributed.get_rank() == 0:
                logger.info(f'{key}: max {max_times[0]}, avg {avg_times[0]}')

    @timed(verbose=False)
    def load_tensor_from_storage(self, checkpoint_dir, ten_meta: _ShardedTensorMetadata):
        """Load tensor from storage."""
        logger.debug(f'_load_from_array({ten_meta.sharded_tensor_no_data.key}) init')
        ret = _load_from_array(
            ten_meta.sharded_tensor_no_data,
            checkpoint_dir,
            load_directly_on_device=False,
            apply_flattened_range=False,
        )
        logger.debug(f'_load_from_array({ten_meta.sharded_tensor_no_data.key}) DONE')
        return ret

    @timed()
    def maybe_init_gloo_group(self):
        """Create Gloo groups."""
        if not self.cpu_transfer:
            return
        all_groups = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(all_groups, self.dp_group_ranks)
        all_groups = set(tuple(sorted(gr)) for gr in all_groups)
        for group_ranks in sorted(all_groups):
            # "two_stage" module will be deprecated, so not replace new_group()
            # with ...parallel_state.create_group() func setting group_desc here.
            gloo_pg = torch.distributed.new_group(ranks=group_ranks, backend='gloo')
            if self.global_rank in group_ranks:
                self.data_parallel_group = gloo_pg
                assert self.dp_group_rank == self.data_parallel_group.rank()

    def check_backend_compatibility(self, loaded_version):
        pass  # TODO

    def check_version_compatibility(self, loaded_version):
        pass  # TODO

    @timed()
    def _build_load_plan(
        self, sharded_state_dict: ShardedStateDict
    ) -> List[_ShardedTensorMetadata]:
        local_meta = [
            _ShardedTensorMetadata(
                self.global_rank,
                sharded_ten.without_data(),
                self.dp_group_rank,
                self.dp_group_ranks,
            )
            for sharded_ten in nested_values(sharded_state_dict)
        ]
        all_meta = [None] * self.data_parallel_group.size()
        torch.distributed.all_gather_object(all_meta, local_meta, group=self.data_parallel_group)
        all_meta = list(chain.from_iterable(all_meta))
        all_tensors_sorted = self.deduplicate_chunks(all_meta)
        return all_tensors_sorted

    @timed()
    def deduplicate_chunks(self, ten_metas: List[_ShardedTensorMetadata]):
        """Group tensors by chunk and then pick the tensor with the lowest rank.

        NOTE: with proper loading overlap, loading from randomized ranks
         (instead of the smallest one) could be beneficial here.
        """
        ten_metas = map_reduce(
            ten_metas,
            key_fn=lambda meta: sharded_tensor_chunk_id(meta.sharded_tensor_no_data),
            reduce_fn=partial(min, key=attrgetter('dist_group_rank')),
        )
        all_metas_sorted = list(map(itemgetter(1), sorted(ten_metas.items())))
        return all_metas_sorted

    @timed()
    def _exchange_loaded_tensors(
        self, ten_metas: List[_ShardedTensorMetadata], sharded_state_dict, checkpoint_dir
    ):
        logger.debug(f'_exchange_loaded_tensors, num ten_metas: {len(ten_metas)}')
        for ten_meta in ten_metas:

            src_rank = torch.distributed.get_global_rank(
                self.data_parallel_group, ten_meta.dist_group_rank
            )

            if self.dp_group_rank == ten_meta.dist_group_rank:
                exchange_tensor = self.load_tensor_from_storage(checkpoint_dir, ten_meta)
                if not self.cpu_transfer:
                    exchange_tensor = exchange_tensor.cuda()
            else:
                # TODO: for non-flattened ranges we could reuse the buffer from the start here
                exchange_tensor = torch.empty(
                    ten_meta.sharded_tensor_no_data.local_shape,
                    device='cpu' if self.cpu_transfer else 'cuda',
                    dtype=ten_meta.sharded_tensor_no_data.dtype,
                )

            logger.debug(
                f'exchange {ten_meta.sharded_tensor_no_data.key}, {exchange_tensor.shape}\
({exchange_tensor.numel()}), broadcast({src_rank} -> {self.dp_group_ranks})'
            )
            torch.distributed.broadcast(
                exchange_tensor, group=self.data_parallel_group, src=src_rank
            )
            self._distribute_data_to_state_dict(ten_meta, exchange_tensor, sharded_state_dict)
            logger.debug(f'exchange {ten_meta.sharded_tensor_no_data.key} done')

            # free buffer memory
            exchange_tensor = None

    @timed(verbose=False)
    def _distribute_data_to_state_dict(
        self,
        ten_meta: _ShardedTensorMetadata,
        loaded_ten: torch.Tensor,
        sharded_state_dict: ShardedStateDict,
    ):
        tensor_key = sharded_tensor_chunk_id(ten_meta.sharded_tensor_no_data)

        def _fill_in_data(t: Union[ShardedTensor, torch.Tensor]):
            if not isinstance(t, ShardedTensor) or sharded_tensor_chunk_id(t) != tensor_key:
                # already filled-in or key not matching
                return t
            sharded_tensor: ShardedTensor = t
            x = loaded_ten
            if sharded_tensor.flattened_range is not None:
                x = flatten_range(sharded_tensor, x)

            # Reuse existing buffer
            sharded_tensor.data.data.copy_(x)
            return sharded_tensor.data

        dict_list_map_inplace(_fill_in_data, sharded_state_dict)

    def load_tensors_metadata(self, checkpoint_dir: Path):
        def get_ts_shape_dtype(path):
            arr = open_ts_array(path)
            return arr.shape, arr.dtype.numpy_dtype

        return load_zarr_based_sharded_metadata(checkpoint_dir, get_ts_shape_dtype)
