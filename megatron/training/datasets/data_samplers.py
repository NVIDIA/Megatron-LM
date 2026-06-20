# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Dataloaders."""


import random

import numpy as np
import torch
from torch.utils.data import Dataset

from megatron.core import mpu
from megatron.core.datasets.utils import Split
from megatron.training import get_args
from megatron.training.dist_signal_handler import DistributedSignalHandler


def build_pretraining_data_loader(dataset, consumed_samples, cfg=None):
    """Build dataloader given an input dataset.

    This is the entry point used by the live training loop. The data-loader
    configuration (dataloader type, workers, sharding, pin-memory, persistent
    workers) and batch sizes are sourced from ``cfg`` (a ``PretrainConfigContainer``)
    when it carries a populated ``cfg.dataset``; otherwise they fall back to the
    global ``args`` so existing flows (e.g. RL, or callers that have not yet
    populated ``cfg.dataset``) behave exactly as before. The common sampler types
    ('single', 'cyclic', 'batch', 'external') are delegated to the single,
    globals-free implementation in :mod:`megatron.training.datasets.data_loaders`;
    only the Megatron-LM-specific samplers (full-validation and hybrid context
    parallel) are built here, since they are not part of the consolidated core.

    Args:
        dataset: The dataset to wrap (or None).
        consumed_samples: Samples already consumed (for resumption).
        cfg: Optional ``PretrainConfigContainer``. When ``cfg.dataset`` is set, the
            data-loader knobs and batch sizes come from the container.
    """

    if dataset is None:
        return None
    args = get_args()

    # Prefer the config container's dataset config (the "container approach"); fall
    # back to global args when it is absent so nothing regresses.
    dataset_cfg = getattr(cfg, "dataset", None) if cfg is not None else None

    if hasattr(dataset, 'split'):
        split = dataset.split
    elif hasattr(dataset, 'index_split'):
        split = dataset.index_split
    else:
        split = None

    is_eval = split in (Split.valid, Split.test)

    # Resolve data-loader knobs + batch sizes from the container or from args.
    if dataset_cfg is not None:
        dataloader_type = dataset_cfg.dataloader_type if dataset_cfg.dataloader_type is not None else args.dataloader_type
        num_workers = dataset_cfg.num_workers
        data_sharding = dataset_cfg.data_sharding
        pin_memory = dataset_cfg.pin_memory
        persistent_workers = dataset_cfg.persistent_workers
        if is_eval:
            micro_batch_size = cfg.validation.eval_micro_batch_size or cfg.train.micro_batch_size
            global_batch_size = cfg.validation.eval_global_batch_size or cfg.train.global_batch_size
        else:
            micro_batch_size = cfg.train.micro_batch_size
            global_batch_size = cfg.train.global_batch_size
    else:
        dataloader_type = args.dataloader_type
        num_workers = args.num_workers
        data_sharding = args.data_sharding
        pin_memory = True
        persistent_workers = args.num_workers > 0
        micro_batch_size = getattr(args, 'eval_micro_batch_size', args.micro_batch_size) if is_eval else args.micro_batch_size
        global_batch_size = getattr(args, 'eval_global_batch_size', args.global_batch_size) if is_eval else args.global_batch_size

    if dataloader_type == "external":
        # External dataloaders are passed through. User is expected to provide a
        # torch-compatible dataloader and define samplers, if needed.
        return dataset

    def worker_init_fn(_):
        import os

        # Defensively close GPU device FDs in worker processes so workers do not
        # keep references into NVIDIA memory space. This helps ensure GPU memory
        # can be reclaimed even if a dataloader worker is delayed or fails to exit.
        def close_nvidia_fds():
            for fd in os.listdir("/proc/self/fd"):
                try:
                    path = os.readlink(f"/proc/self/fd/{fd}")
                    if path.startswith("/dev/nvidia"):
                        os.close(int(fd))
                except OSError:
                    pass

        close_nvidia_fds()
        if args.exit_signal_handler:
            DistributedSignalHandler(args.exit_signal).__enter__()

    maybe_worker_init_fn = worker_init_fn if num_workers > 0 else None

    # Megatron-LM-specific samplers not present in the consolidated core: build
    # the DataLoader here. (full_validation / hybrid_context_parallel are global
    # parallelism settings, so they are read from args regardless of source.)
    batch_sampler = None
    extra_kwargs = {}
    if split == Split.valid and args.full_validation:
        batch_sampler = MegatronFullValidationSampler(
            total_samples=len(dataset),
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size())
    elif dataloader_type == 'single' and args.hybrid_context_parallel:
        batch_sampler = HybridCPMegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size())
        extra_kwargs = {"collate_fn": lambda x: x}

    if batch_sampler is not None:
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            worker_init_fn=maybe_worker_init_fn,
            **extra_kwargs,
        )

    # Common path ('single' / 'cyclic' / 'batch'): delegate to the single
    # globals-free implementation so the sampler-selection logic lives in one
    # place. Imported lazily to avoid a circular import (data_loaders imports the
    # sampler classes from this module).
    from megatron.training.datasets.data_loaders import (
        build_pretraining_data_loader as _build_pretraining_data_loader,
    )

    return _build_pretraining_data_loader(
        dataset,
        consumed_samples,
        dataloader_type,
        micro_batch_size,
        num_workers,
        data_sharding,
        data_parallel_rank=mpu.get_data_parallel_rank(),
        data_parallel_size=mpu.get_data_parallel_world_size(),
        worker_init_fn=maybe_worker_init_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        global_batch_size=global_batch_size,
    )

class MegatronPretrainingSampler:
    """
    Sampler for Megatron pretraining dataloaders that divides data samples across
    data parallel workers. Each worker receives a contiguous chunk of data determined by
    its rank and the micro batch size. Supports dropping the last incomplete batch if
    specified, and keeps track of total and consumed samples. Designed to work with
    distributed training using Megatron's data parallelism.
    """

    def __init__(
        self,
        total_samples,
        consumed_samples,
        micro_batch_size,
        data_parallel_rank,
        data_parallel_size,
        drop_last=True,
    ):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, 'no sample to consume: {}'.format(self.total_samples)
        assert (
            self.consumed_samples < self.total_samples
        ), 'no samples left to consume: {}, {}'.format(self.consumed_samples, self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert (
            self.data_parallel_rank < data_parallel_size
        ), 'data_parallel_rank should be smaller than data size: {}, ' '{}'.format(
            self.data_parallel_rank, data_parallel_size
        )

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        """
        Calculate the start and end indices for the current data parallel worker's
        chunk within a batch.

        Returns:
            tuple: (start_idx, end_idx) indicating the slice of the batch for this worker.
        """
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

class HybridCPMegatronPretrainingSampler(MegatronPretrainingSampler):
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
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx_global_batch()
            global_batch_idx = []
            for i in range(self.num_micro_batches):
                global_batch_idx.extend(batch[start_idx[i]:end_idx[i]])
            yield global_batch_idx


class MegatronFullValidationSampler:
    """Sampler for full validation that handles small datasets gracefully.

    This sampler is designed for validation datasets that may be smaller than
    data_parallel_size * micro_batch_size. It uses micro_batch_size=1 to minimize
    the samples needed per batch and properly handles partial batches where some
    ranks may not have data.
    """

    def __init__(self, total_samples, data_parallel_rank, data_parallel_size):
        self.total_samples = total_samples
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.micro_batch_size = 1  # Always use 1 for small dataset support

        # Sanity checks
        assert self.total_samples > 0, f'no sample to consume: {self.total_samples}'
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            f'data_parallel_rank should be smaller than data size: {self.data_parallel_rank}, {data_parallel_size}'

    def __len__(self):
        """Returns the number of batches this rank will yield."""
        # Each batch takes data_parallel_size samples (1 per rank)
        # This rank gets samples at indices: data_parallel_rank, data_parallel_rank + data_parallel_size, ...
        num_batches = 0
        for batch_idx in range(0, self.total_samples, self.data_parallel_size):
            # Check if this rank has data in this batch
            sample_idx = batch_idx + self.data_parallel_rank
            if sample_idx < self.total_samples:
                num_batches += 1
        return num_batches

    def __iter__(self):
        """Yield batches for this data parallel rank."""
        for batch_idx in range(0, self.total_samples, self.data_parallel_size):
            # Check if this rank has data in this batch
            sample_idx = batch_idx + self.data_parallel_rank
            if sample_idx < self.total_samples:
                # Yield a batch with a single sample index for this rank
                yield [sample_idx]


class RandomSeedDataset(Dataset):
    """
    A dataset wrapper that resets the random seed before each sample.

    This ensures deterministic behavior per sample by setting the RNG state
    for torch, numpy, and random before accessing each underlying data sample.
    The base seed is retrieved from training arguments, and can be varied per epoch
    using the set_epoch method to ensure different shuffling or augmentation each epoch.

    Args:
        dataset: The underlying dataset to wrap.

    Methods:
        set_epoch(epoch): Change the seed offset so each epoch produces different randomization.
        __getitem__(idx): Sets the seed based on the sample index and current epoch.
    """

    def __init__(self, dataset, seed):
        self.base_seed = seed
        self.curr_seed = seed
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        """
        Change the seed offset so each epoch produces different randomization.

        Args:
            epoch: The epoch number to use as the seed offset.
        """
        self.curr_seed = self.base_seed + epoch

    def __getitem__(self, idx):
        seed = idx + self.curr_seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return self.dataset[idx]


class MegatronPretrainingRandomSampler:
    """
    Sampler for Megatron pretraining dataloaders that performs random sampling
    across data parallel workers. Supports data sharding to divide the dataset
    into buckets and shuffle within each bucket. Designed to work with distributed
    training using Megatron's data parallelism.
    """

    def __init__(
        self,
        dataset,
        total_samples,
        consumed_samples,
        micro_batch_size,
        data_parallel_rank,
        data_parallel_size,
        data_sharding,
    ):
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.data_sharding = data_sharding
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * data_parallel_size
        self.last_batch_size = self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, 'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert (
            self.data_parallel_rank < data_parallel_size
        ), 'data_parallel_rank should be smaller than data size: {}, ' '{}'.format(
            self.data_parallel_rank, data_parallel_size
        )

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
            bucket_size = (
                self.total_samples // self.micro_batch_times_data_parallel_size
            ) * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.data_parallel_size
            start_idx = self.data_parallel_rank * bucket_size

            g = torch.Generator()
            g.manual_seed(self.epoch)
            random_idx = torch.randperm(bucket_size, generator=g).tolist()
            idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_range_total = torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[self.data_parallel_rank :: self.data_parallel_size]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []


class MegatronGlobalBatchSampler:
    """Batch sampler that yields an entire global batch's worth of indices at once.

    Unlike ``MegatronPretrainingSampler`` (which yields one micro-batch per step),
    this sampler accumulates a full global batch, distributes the indices across
    data-parallel ranks in an interleaved fashion, and yields *all* of this rank's
    indices for the global batch in a single list. This is required for
    variable-length fine-tuning, where the collate function must see the whole
    global batch to compute a common padding length before the training loop
    splits it back into micro-batches.

    Args:
        total_samples: Total number of samples in the dataset.
        consumed_samples: Number of samples already consumed (for resuming).
        micro_batch_size: Batch size per GPU.
        global_batch_size: Total batch size across all data-parallel ranks.
        data_parallel_rank: Rank of the current GPU in the data-parallel group.
        data_parallel_size: Total number of GPUs in the data-parallel group.
        drop_last: If True, drops the last incomplete global batch.
        pad_samples_to_global_batch_size: If True, pads the trailing partial batch
            with -1 indices so every rank yields a full slice.
    """

    def __init__(
        self,
        total_samples,
        consumed_samples,
        micro_batch_size,
        global_batch_size,
        data_parallel_rank,
        data_parallel_size,
        drop_last=True,
        pad_samples_to_global_batch_size=False,
    ):
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.drop_last = drop_last
        self.pad_samples_to_global_batch_size = pad_samples_to_global_batch_size
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * data_parallel_size

        assert self.total_samples > 0, 'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0, f'micro_batch_size must be > 0, but {self.micro_batch_size}'
        assert data_parallel_size > 0, f'data parallel size must be > 0, but {data_parallel_size}'
        assert (
            self.data_parallel_rank < data_parallel_size
        ), 'data_parallel_rank should be smaller than data size: {}, {}'.format(
            self.data_parallel_rank, data_parallel_size
        )

        self._global_batch_size = global_batch_size
        if self._global_batch_size % self.micro_batch_times_data_parallel_size != 0:
            raise RuntimeError(
                f"`global_batch_size` ({self._global_batch_size}) is not divisible by "
                f"`micro_batch_size ({self.micro_batch_size}) x data_parallel_size "
                f"({self.data_parallel_size})`"
            )
        self._num_micro_batches = self._global_batch_size // self.micro_batch_times_data_parallel_size
        self._global_batch_size_on_this_data_parallel_rank = (
            self._num_micro_batches * self.micro_batch_size
        )

    def __len__(self):
        num_available_samples = self.total_samples - self.consumed_samples % self.total_samples
        if self.drop_last:
            return num_available_samples // self._global_batch_size
        return (num_available_samples + self._global_batch_size - 1) // self._global_batch_size

    def __iter__(self):
        batch = []
        for idx in range(self.consumed_samples % self.total_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self._global_batch_size:
                all_indices = [
                    batch[i]
                    for i in range(
                        self.data_parallel_rank, self._global_batch_size, self.data_parallel_size
                    )
                ]
                assert len(all_indices) == self._global_batch_size_on_this_data_parallel_rank
                yield all_indices
                batch = []

        # Trailing partial batch.
        if len(batch) > 0 and not self.drop_last:
            all_indices = [
                batch[i] for i in range(self.data_parallel_rank, len(batch), self.data_parallel_size)
            ]
            if self.pad_samples_to_global_batch_size:
                num_pad = self._global_batch_size // self.data_parallel_size - len(all_indices)
                all_indices = all_indices + [-1] * num_pad
            yield all_indices
