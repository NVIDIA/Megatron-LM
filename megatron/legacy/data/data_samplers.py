# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Dataloaders."""


import random
import torch
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import Dataset
from megatron.training import get_args
from megatron.core import mpu
from megatron.core.datasets.utils import Split


def build_pretraining_data_loader(dataset, consumed_samples):
    """Build dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args()
    
    if hasattr(dataset,'split'):
        split = dataset.split
    elif hasattr(dataset,'index_split'):
        split = dataset.index_split
    else:
        split = None

    if split == Split.valid and args.full_validation:
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=0,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size())
    elif args.dataloader_type == 'single':
        if args.sft_sequence_packing:
            if args.async_hybrid_context_parallel_scheduler:
                assert args.hybrid_context_parallel_scheduler == "only_packing_no_scheduling"
                batch_sampler = MegatronSFTPrefetchDPBalancedSampler(
                    dataset=dataset,
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=args.micro_batch_size,
                    global_batch_size=args.global_batch_size,
                    data_parallel_rank=mpu.get_data_parallel_rank(),
                    data_parallel_size=mpu.get_data_parallel_world_size())
            else:
                batch_sampler = MegatronSFTSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=args.micro_batch_size,
                    global_batch_size=args.global_batch_size,
                    data_parallel_rank=mpu.get_data_parallel_rank(),
                    data_parallel_size=mpu.get_data_parallel_world_size())
        else:
            # Megatron sampler
            batch_sampler = MegatronPretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=args.micro_batch_size,
                data_parallel_rank=mpu.get_data_parallel_rank(),
                data_parallel_size=mpu.get_data_parallel_world_size())
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding)
    elif args.dataloader_type == "external":
        # External dataloaders are passed through. User is expected to provide a
        # torch-compatible dataloader and define samplers, if needed.
        return dataset
    else:
        raise Exception('{} dataloader type is not supported.'.format(
                args.dataloader_type))

    # Torch dataloader.
    if args.sft_sequence_packing:
        extra_kwargs = {"collate_fn": lambda x: x,}
    else:
        extra_kwargs = {}
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True,
                                       persistent_workers=True if args.num_workers > 0 else False,
                                       **extra_kwargs,
                                       )

class MegatronPretrainingSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]

class MegatronSFTSampler(MegatronPretrainingSampler):
    """
    Data sampler for hybrid context parallel (Hybrid CP) format.
    This data sampler pulls in the entire global batch at once across all data parallel ranks.
    This helps provide the Hybrid CP Dataloader Wrapper to schedule and load balance sub-samples
    of the entire global batch.
    """

    def __init__(self, total_samples, consumed_samples, micro_batch_size, global_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        super().__init__(total_samples, consumed_samples, micro_batch_size, data_parallel_rank, data_parallel_size, drop_last)
        self.global_batch_size = global_batch_size
        self.data_parallel_size = data_parallel_size
        self.num_micro_batches = self.global_batch_size // self.micro_batch_times_data_parallel_size

    def __len__(self):
        return self.total_samples

    def get_start_end_idx_global_batch(self):
        start_idx = [self.data_parallel_rank * self.micro_batch_size + i * self.micro_batch_size * self.data_parallel_size for i in range(self.num_micro_batches)]
        end_idx = [start_idx[i] + self.micro_batch_size for i in range(self.num_micro_batches)]
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size * self.num_micro_batches:
                start_idx, end_idx = self.get_start_end_idx_global_batch()
                global_batch_idx = []
                for i in range(self.num_micro_batches):
                    global_batch_idx.extend(batch[start_idx[i]:end_idx[i]])
                yield global_batch_idx
                # if torch.distributed.get_rank() == 0:
                #     print(f"rank={torch.distributed.get_rank()}, {batch=}")
                # yield batch
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx_global_batch()
            global_batch_idx = []
            for i in range(self.num_micro_batches):
                global_batch_idx.extend(batch[start_idx[i]:end_idx[i]])
            yield global_batch_idx


class MegatronSFTPrefetchDPBalancedSampler(MegatronPretrainingSampler):
    """
    Data sampler for hybrid context parallel (Hybrid CP) format.
    This data sampler pulls in the entire global batch at once across all data parallel ranks.
    This helps provide the Hybrid CP Dataloader Wrapper to schedule and load balance sub-samples
    of the entire global batch.
    """

    def __init__(self, dataset, total_samples, consumed_samples, micro_batch_size, global_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        super().__init__(total_samples, consumed_samples, micro_batch_size, data_parallel_rank, data_parallel_size, drop_last)
        self.dataset = dataset
        self.global_batch_size = global_batch_size
        self.data_parallel_size = data_parallel_size
        self.num_micro_batches = self.global_batch_size // self.micro_batch_times_data_parallel_size
        
        from megatron.training.yaml_arguments import core_transformer_config_from_yaml
        from megatron.training.arguments import core_transformer_config_from_args
        args = get_args()
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)
        
        self.config = config
        from megatron.core.pipeline_parallel.data_schedule import PipelineAwareBalancedHybridCPscheduler
        self.data_scheduler = PipelineAwareBalancedHybridCPscheduler(self.config)

        ctx = mp.get_context('fork')
        self._queue1 = ctx.Queue()
        self._queue2 = ctx.Queue()
        self._prefetch_process = ctx.Process(target=self.prefetch_batch,
                                         args=(self._queue1, self._queue2),
                                         name=f'prefetch_batch', daemon=False)
        self._prefetch_process.start()

    def __len__(self):
        return self.total_samples

    # def get_start_end_idx_global_batch(self):
    #     start_idx = [self.data_parallel_rank * self.micro_batch_size + i * self.micro_batch_size * self.data_parallel_size for i in range(self.num_micro_batches)]
    #     end_idx = [start_idx[i] + self.micro_batch_size for i in range(self.num_micro_batches)]
    #     return start_idx, end_idx

    def get_shape(self, idx):
        data = self.dataset[idx]
        shape = data["tokens"].shape
        return shape

    def get_numel(self, idx):
        data = self.dataset[idx]
        numel = data["tokens"].numel()
        return [idx, numel]

    def prepare_info(self, batch, batch_numel):
        pass

    def prefetch_batch(self, queue1, queue2):
        torch.multiprocessing._set_thread_name("pt_prefetch_batch")
        # global_store = DistKVStore(world_size=torch.distributed.get_world_size(), rank=torch.distributed.get_rank(), group_name=global_group_name)
        # within_node_store = DistKVStore(world_size=8, rank=torch.distributed.get_rank(), group_name=within_node_group_name)
        # assert torch.distributed.get_world_size() % 8 == 0, f"world_size should be divisible by 8" # 单机8卡
        # if torch.distributed.get_rank() % 8 == 0: # 每个节点的0号rank
        #     cross_node_store = DistKVStore(world_size=torch.distributed.get_world_size() // 8, rank=torch.distributed.get_rank(), group_name=cross_node_group_name)
        # else:
        #     cross_node_store = None

        while True:
            full_batch = queue1.get()
            # print(f"GET queue1, {full_batch=}")
            if full_batch == None:
                return
            batch_data = self.prepare_batch(full_batch)
            queue2.put(batch_data)
            # print(f"PUT queue2, {batch_data=}")

    def prepare_batch(self, batch):
        torch.multiprocessing._set_thread_name("pt_prefetch_batch")
        batch_numel = [self.get_numel(idx) for idx in batch]
        # TODO: use distributed `get_numel` to reduce io pressure.
        
        groups, sample_id_groups, cp_sizes = self.data_scheduler.get_groups_and_subsamples(batch_numel, self.config, return_cp_sizes=True)
        return groups, sample_id_groups, cp_sizes

    def __iter__(self):
        # batch = []
        # Last batch will be dropped if drop_last is not set False
        batch = list(range(self.consumed_samples, min(self.consumed_samples + self.global_batch_size, self.total_samples)))
        # print(f"PUT queue1, {batch=}")
        self._queue1.put(batch)
        while self.consumed_samples < self.total_samples:
        # for idx in range(self.consumed_samples, self.total_samples):
        #     batch.append(idx)
            # if len(batch) == self.global_batch_size:
                # groups, sample_id_groups, cp_sizes = self.prepare_batch(batch)
            batch_data = self._queue2.get(timeout=3000)
            groups, sample_id_groups, cp_sizes = batch_data
            # print(f"GET queue2, {groups=}")

            consumed_samples_before = self.consumed_samples
            next_full_batch = list(range(consumed_samples_before + self.global_batch_size, min(consumed_samples_before + 2*self.global_batch_size, self.total_samples)))
            # print(f"PUT queue1, {next_full_batch=}")
            self._queue1.put(next_full_batch)
            
            # global_batch_idx = []
            for microbatch_idx in range(len(sample_id_groups)):
                microbatch = sample_id_groups[microbatch_idx][self.data_parallel_rank]
                microbatch_cp_sizes = cp_sizes[microbatch_idx][self.data_parallel_rank]
                num_microbatch_left = [len(sample_id_groups)-microbatch_idx-1] * len(microbatch)
                # print(f"{groups=}\n{sample_id_groups=}")
                yield list(zip(microbatch, num_microbatch_left, microbatch_cp_sizes))
                # global_batch_idx.extend(microbatch)

            # yield global_batch_idx
            # batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            groups, sample_id_groups, cp_sizes = self.prepare_batch(batch)
            global_batch_idx = []
            for microbatch_idx in range(len(sample_id_groups)):
                microbatch = sample_id_groups[microbatch_idx][self.data_parallel_rank]
                microbatch_cp_sizes = cp_sizes[microbatch_idx][self.data_parallel_rank]
                num_microbatch_left = [len(sample_id_groups)-microbatch_idx-1] * len(microbatch)
                assert len(microbatch) == len(microbatch_cp_sizes)
                global_batch_idx.extend(list(zip(microbatch, num_microbatch_left, microbatch_cp_sizes)))
            yield global_batch_idx


class RandomSeedDataset(Dataset):

    def __init__(self, dataset):
        args = get_args()
        self.base_seed = args.seed
        self.curr_seed = args.seed
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.curr_seed = self.base_seed + epoch

    def __getitem__(self, idx):
        seed = idx + self.curr_seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return self.dataset[idx]


class MegatronPretrainingRandomSampler:

    def __init__(self, dataset, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, data_sharding):
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.data_sharding = data_sharding
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)

        # data sharding and random sampling
        if self.data_sharding:
            bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                           * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.data_parallel_size
            start_idx = self.data_parallel_rank * bucket_size

            g = torch.Generator()
            g.manual_seed(self.epoch)
            random_idx = torch.randperm(bucket_size, generator=g).tolist()
            idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) \
                                * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_range_total = \
                torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[self.data_parallel_rank::self.data_parallel_size]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []
