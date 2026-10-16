# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Dataloaders."""

import logging
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from megatron.core import mpu
from megatron.core.datasets.utils import Split
from megatron.core.num_microbatches_calculator import (
    get_current_running_global_batch_size,
    get_global_batch_size_upper_bound,
    get_num_microbatches,
)
from megatron.core.utils import log_single_rank
from megatron.training import get_args
from megatron.training.dist_signal_handler import DistributedSignalHandler

logger = logging.getLogger(__name__)


def get_dataloader_prefetch_factor(prefetch_factor, num_workers, num_microbatches_per_step=None):
    """Resolve --dataloader-prefetch-factor into the DataLoader prefetch_factor.

    torch keeps ``prefetch_factor * num_workers`` batches outstanding and dispatches
    one new index each time a batch is consumed. A training step pulls
    ``num_microbatches_per_step`` batches (one microbatch each) from the loader, so a
    window of at least that many batches has every batch of step k+1 dispatched while
    step k still runs. 'auto' resolves to ``ceil(num_microbatches_per_step /
    num_workers) + 1``; the extra batch per worker keeps the window above one step.

    Args:
        prefetch_factor: None (keep torch's default), a positive int, or 'auto'.
        num_workers: Number of DataLoader worker processes.
        num_microbatches_per_step: Batches one step pulls from this loader. When None,
            'auto' keeps torch's default.

    Returns:
        The prefetch_factor to pass to the DataLoader, or None to leave it unset.
    """
    if prefetch_factor is None:
        return None
    if prefetch_factor != 'auto' and (
        not isinstance(prefetch_factor, int)
        or isinstance(prefetch_factor, bool)
        or prefetch_factor <= 0
    ):
        raise ValueError(
            f"--dataloader-prefetch-factor must be a positive integer or 'auto', "
            f"got {prefetch_factor!r}"
        )
    if num_workers == 0:
        if prefetch_factor == 'auto':
            # Without worker processes nothing is prefetched, so there is nothing to size.
            return None
        raise ValueError(
            f"--dataloader-prefetch-factor {prefetch_factor} requires --num-workers > 0: "
            "the DataLoader only prefetches in worker processes."
        )
    if prefetch_factor != 'auto':
        return prefetch_factor
    if num_microbatches_per_step is None:
        return None
    return -(-num_microbatches_per_step // num_workers) + 1


def _get_train_num_microbatches_per_step():
    """Return the most batches one training step pulls from this rank's loader."""
    # The packing schedulers (get_batch_and_global_seqlens) and the ordinary schedules
    # pull get_num_microbatches() batches per step. A step batch-size schedule grows
    # that after the loader is built, so scale to the largest global batch of the run.
    return -(
        -get_num_microbatches()
        * get_global_batch_size_upper_bound()
        // get_current_running_global_batch_size()
    )


def build_pretraining_data_loader(dataset, consumed_samples):
    """Build dataloader given an input dataset."""

    if dataset is None:
        return None
    # Empty split (e.g. valid/test when --eval-iters 0): return null loader
    try:
        if len(dataset) == 0:
            return None
    except TypeError:
        pass
    args = get_args()

    if hasattr(dataset, 'split'):
        split = dataset.split
    elif hasattr(dataset, 'index_split'):
        split = dataset.index_split
    else:
        split = None

    if args.dataloader_type == "external":
        # External dataloaders are passed through. User is expected to provide a
        # torch-compatible dataloader and define samplers, if needed.
        return dataset

    # Use eval-specific batch sizes for validation/test splits
    is_eval = split in (Split.valid, Split.test)
    micro_batch_size = (
        getattr(args, 'eval_micro_batch_size', args.micro_batch_size)
        if is_eval
        else args.micro_batch_size
    )
    if split == Split.valid and args.full_validation:
        batch_sampler = MegatronFullValidationSampler(
            total_samples=len(dataset),
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
        )
    elif args.dataloader_type == 'single':
        # Packing schedulers consume one microbatch at a time and form
        # global/DCP batches themselves.
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
        )
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding,
        )
    else:
        raise Exception('{} dataloader type is not supported.'.format(args.dataloader_type))

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

    maybe_worker_init_fn = worker_init_fn if args.num_workers > 0 else None
    # Identity collate for VarlenDataset and packing-scheduler paths;
    # they emit one variable-length dict per sample, not stack-able by
    # the default collate. --varlen-sbhd-validation is excluded: it bypasses
    # packing and emits fixed-length [seq_length] samples that the default
    # collate stacks normally.
    if (
        (args.use_varlen_dataset and not args.varlen_sbhd_validation)
        or args.dynamic_context_parallel
        or args.sequence_packing_scheduler is not None
        or getattr(args, "use_vanilla_collate_fn", False)
    ):
        extra_kwargs = {"collate_fn": lambda x: x}
    else:
        extra_kwargs = {}
    # Own generator: otherwise _BaseDataLoaderIter draws _base_seed from the default CPU generator
    # on ITERATOR creation, after load_checkpoint has restored it. initial_seed() reads the seed
    # without consuming a draw.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(torch.initial_seed())
    # Only the training loader gets a prefetch window: persistent validation/test workers
    # would fill it when their iterators are created and hold it between evaluations.
    flag = getattr(args, 'dataloader_prefetch_factor', None)
    prefetch_factor = None
    if not is_eval and flag is not None:
        # An explicit --dataloader-prefetch-factor overrides the packing default below.
        num_microbatches_per_step = (
            _get_train_num_microbatches_per_step()
            if flag == 'auto' and args.num_workers > 0
            else None
        )
        prefetch_factor = get_dataloader_prefetch_factor(
            flag, args.num_workers, num_microbatches_per_step
        )
        source = f"--dataloader-prefetch-factor {flag}"
    elif not is_eval and args.sequence_packing_scheduler is not None and args.num_workers > 0:
        # Packing schedulers draw a whole global batch, global_batch_size /
        # (micro_batch_size * DP) items on every DP rank, at the start of each
        # step. Keeping that many items in flight lets the workers and the
        # pin-memory thread build the next step's batch while the current step
        # trains, at the cost of host memory for one more global batch.
        items_per_step = args.global_batch_size // (
            micro_batch_size * mpu.get_data_parallel_world_size()
        )
        prefetch_factor = max(2, -(-items_per_step // args.num_workers))
        source = f"sequence-packing scheduler default, {items_per_step} items per step"
    if prefetch_factor is not None:
        extra_kwargs["prefetch_factor"] = prefetch_factor
        log_single_rank(
            logger,
            logging.INFO,
            f"> dataloader prefetch_factor {prefetch_factor} ({source}; "
            f"num_workers {args.num_workers}, split {split})",
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        worker_init_fn=maybe_worker_init_fn,
        generator=loader_generator,
        **extra_kwargs,
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
        assert (
            self.data_parallel_rank < data_parallel_size
        ), f'data_parallel_rank should be smaller than data size: {self.data_parallel_rank}, {data_parallel_size}'

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
