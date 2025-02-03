# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import os

import torch
from dataset_helpers import TaskEncoder, print_error_handler

from megatron.core import parallel_state
from megatron.energon import (
    LimitDataset,
    RepeatDataset,
    WorkerConfig,
    get_loader,
    get_savable_loader,
    get_train_dataset,
    get_val_datasets,
)
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.parallel_state import get_tensor_model_parallel_rank, get_pipeline_model_parallel_world_size, get_pipeline_model_parallel_rank
from megatron.training import get_args
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
        max_samples_per_sequence=None,
        shuffle_buffer_size=None,
        packing_buffer_size=args.packing_buffer_size,
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
        packing_buffer_size=args.packing_buffer_size,
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


def is_first_or_last_stage(pp_size, encoder_pipeline_model_parallel_size):
    """Check if the current pipeline parallel stage is the first or last stage."""
    if pp_size == 1:    # No pipeline parallelism.
        return True

    is_valid_rank = False
    pp_rank = get_pipeline_model_parallel_rank()
    if encoder_pipeline_model_parallel_size == 0:
        # No separate pipeline stage for the vision model. Run the dataloader on the first and last pipeline stage.
        is_valid_rank = pp_rank in (0, pp_size-1)
    elif encoder_pipeline_model_parallel_size == 1:
        # Separate pipeline stage for the vision model. Run the dataloader on the first vision and LM stage and last LM stage.
        is_valid_rank = pp_rank in (0, 1, pp_size-1)
    else:
        raise NotImplementedError("encoder-pipeline-model-parallel-size > 1 is not supported yet")

    return is_valid_rank


def is_dataloader_rank(encoder_pipeline_model_parallel_size):
    """Check if we should have the dataloader on this tensor and pipeline parallel rank."""
    # Run dataloader only on the first tensor parallel rank (will be broadcasted to others).
    is_first_rank = get_tensor_model_parallel_rank() == 0

    pp_size = get_pipeline_model_parallel_world_size()
    is_first_rank = is_first_rank and is_first_or_last_stage(pp_size, encoder_pipeline_model_parallel_size)

    return is_first_rank


def train_valid_test_dataloaders_provider(train_val_test_num_samples):
    """Build multimodal train, validation and test dataloaders."""
    args = get_args()

    # Dataloader is only on specific ranks.
    if not is_dataloader_rank(args.encoder_pipeline_model_parallel_size):
        return None, None, None

    worker_debug_path = None
    worker_log_level = 0

    rank = parallel_state.get_data_parallel_rank()
    world_size = parallel_state.get_data_parallel_world_size()
    data_parallel_group = parallel_state.get_data_parallel_group()

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
        if getattr(args, "dataloader_save", None):
            dp_rank = parallel_state.get_data_parallel_rank()
            data_save_name = get_checkpoint_name(
                args.dataloader_save,
                args.iteration,
                pipeline_rank=0,    # Only the first pipeline parallel rank stores the dataloader checkpoint.
                basename=f"train_dataloader_dprank{dp_rank:03d}.pt",
            )
            if os.path.exists(data_save_name):
                try:
                    dataset_state_dict = torch.load(data_save_name, map_location="cpu")
                    train_dataloader.restore_state_rank(dataset_state_dict["dataloader_state_dict"])
                    print(f"restored dataset state from {data_save_name}")
                except Exception as e:
                    print("loading dataset state failed. Skipping. " + str(e))
            else:
                print(f"dataset state {data_save_name} does not exist")

    valid_dataloader = [
        EnergonDataloader(get_loader(valid_ds, worker_config=worker_config))
        for valid_ds in valid_ds1
    ]
    test_dataloader = None

    return EnergonDataloader(train_dataloader), valid_dataloader, EnergonDataloader(test_dataloader)


class EnergonDataloader:
    """A wrapper to use Megatron Energon dataloader with the Megatron-LM training loop."""
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self._iter = iter(cyclic_iter(dataloader))

    def __next__(self):
        return self._iter.__next__()

    def __iter__(self):
        return self._iter.__iter__()

    def save_state(self):
        return self._dataloader.save_state_rank()


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x
