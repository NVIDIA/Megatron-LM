# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import torch
from dataset_helpers import TaskEncoder, print_error_handler

from megatron.core import mpu
from megatron.energon import (
    LimitDataset,
    RepeatDataset,
    WorkerConfig,
    get_loader,
    get_savable_loader,
    get_train_dataset,
    get_val_datasets,
)
from megatron.training import get_args, get_num_microbatches, print_rank_0
from megatron.training.checkpointing import get_checkpoint_name


def datasets_provider(worker_config=None):
    """Create multimodal train, validation and test datasets."""
    args = get_args()
    dname = args.data_path[0] if type(args.data_path) is list else args.data_path
    train_dataset = get_train_dataset(
        dname,
        batch_size=args.micro_batch_size,
        task_encoder=TaskEncoder(),
        worker_config=worker_config,
        virtual_epoch_length=1000,
        max_samples_per_sequence=100,
        shuffle_buffer_size=100,
        handler=print_error_handler,
        image_decode="pil",
    )

    val_datasets = get_val_datasets(
        dname,
        batch_size=args.micro_batch_size,
        # This is the total number over all workers
        # limit=args.eval_iters * get_num_microbatches(),
        task_encoder=TaskEncoder(),
        worker_config=worker_config,
        handler=print_error_handler,
        image_decode="pil",
    )
    val_datasets_without_source_datasets = [
        # Limit the dataset to eval_iters * num_microbatches
        LimitDataset(
            # Repeat the inner dataset in case it's too short
            RepeatDataset(val_ds, worker_config=worker_config),
            length=args.eval_iters * get_num_microbatches(),
            worker_config=worker_config,
            reset_after_epoch=True,
        )
        for val_ds, _src_ds in val_datasets
    ]

    return train_dataset, val_datasets_without_source_datasets, None


def train_valid_test_dataloaders_provider(train_val_test_num_samples):
    """Build multimodal train, validation and test dataloaders."""
    args = get_args()

    worker_debug_path = None
    worker_log_level = 0

    rank = mpu.get_data_parallel_rank()
    world_size = mpu.get_data_parallel_world_size()
    data_parallel_group = mpu.get_data_parallel_group()

    worker_config = WorkerConfig(
        rank=rank,
        world_size=world_size,
        num_workers=args.num_workers,
        data_parallel_group=data_parallel_group,
        worker_debug_path=worker_debug_path,
        worker_log_level=worker_log_level,
    )
    train_ds, valid_ds1, test_ds = datasets_provider(worker_config)

    train_dataloader = get_savable_loader(train_ds, worker_config=worker_config)
    if args.load is not None:
        if hasattr(args, "dataloader_path"):
            dp_rank = (
                mpu.get_data_parallel_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            data_save_name = get_checkpoint_name(
                args.dataloader_path,
                args.iteration,
                save_basename=f"train_dataloader_dprank{dp_rank:03d}.pt",
            )
            try:
                dataset_state_dict = torch.load(
                    data_save_name, map_location="cpu"
                )
                if (
                    "dataset_state_dict" in dataset_state_dict.keys()
                    and dataset_state_dict["train_data_path"]
                    != args.train_data_path
                ):
                    print_rank_0(
                        f"Not restoring dataset state from {data_save_name}, path to dataset changed from {dataset_state_dict['train_data_path']} to {args.train_data_path}"
                    )
                else:
                    train_dataloader.restore_state_rank(
                        dataset_state_dict["dataloader_state_dict"]
                    )
                    print_rank_0(
                        f"restoring dataset state from {data_save_name}"
                    )
            except Exception as e:
                print_rank_0(
                    "loading dataloader checkpoint failed. Skipping. " + str(e)
                )

    valid_dataloader = [
        iter(cyclic_iter(get_loader(valid_ds, worker_config=worker_config)))
        for valid_ds in valid_ds1
    ]
    test_dataloader = None

    return iter(cyclic_iter(train_dataloader)), valid_dataloader, iter(cyclic_iter(test_dataloader))



def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x
