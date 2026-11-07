# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Inspect sample keys and time the Energon multimodal data pipeline without training a model."""

import faulthandler
import os
import time

import numpy as np
import torch
import tqdm
from data_loading.task_encoder import MultiModalTaskEncoder
from multimodal_args import add_multimodal_extra_args

from megatron.energon import FileStoreCachePool, WorkerConfig, get_savable_loader, get_train_dataset
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron


def main():
    """Initialize CPU-side data loading and report sample keys and iteration timings."""
    faulthandler.enable()
    print(f"PID: {os.getpid()}")

    torch.distributed.init_process_group(backend='gloo')

    initialize_megatron(
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_multimodal_extra_args,
        allow_no_cuda=True,
        skip_mpu_initialization=True,
    )

    # No model-parallel groups are initialized; shard data across all launched ranks.
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    data_parallel_group = torch.distributed.group.WORLD

    args = get_args()

    dname = args.data_path[0] if type(args.data_path) is list else args.data_path

    worker_config = WorkerConfig(
        rank=rank,
        world_size=world_size,
        num_workers=args.num_workers,
        data_parallel_group=data_parallel_group,
    )

    print(f"worker_config: {worker_config}")

    train_dataset = get_train_dataset(
        dname,
        batch_size=1,
        task_encoder=MultiModalTaskEncoder(),
        worker_config=worker_config,
        packing_buffer_size=args.packing_buffer_size,
        shuffle_buffer_size=None,
        max_samples_per_sequence=None,
        repeat=True,
    )
    train_dataloader = get_savable_loader(
        train_dataset,
        gc_collect_every_n_steps=100000,
        cache_pool=FileStoreCachePool(num_workers=8, max_cache_size_gbytes=8, method="raw"),
        watchdog_timeout_seconds=120,
    )

    max_iter = 200

    times = np.zeros(max_iter, dtype=np.float32)

    total_samples = 0

    step = -1
    start = time.time()
    try:
        with tqdm.tqdm(total=len(train_dataloader), position=rank) as pbar:
            # Iterate over the train dataloader
            for step, batch in enumerate(train_dataloader):
                times[step] = time.time() - start
                if step == times.shape[0] - 1:
                    break
                pbar.update(batch['samples_seen'].item())
                total_samples += batch['samples_seen'].item()

                print(f"Step {step} {batch['__key__']}")

                start = time.time()
                if total_samples >= len(train_dataloader):
                    break
    finally:
        if step > 2:
            print("sec/iter so far:", times[: step - 1].mean())


if __name__ == "__main__":
    main()
