# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Finetune utilities."""

import torch
import torch.nn.functional as F
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import utils
from megatron.core import mpu
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.training import evaluate_and_print_results
from megatron.training import setup_model_and_optimizer
from megatron.training import train_step
from megatron.training import training_log
from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import average_losses_across_data_parallel_group, print_params_min_max_norm
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module, ModelType


def process_batch(batch):
    """Process batch and produce inputs for the model."""
    images = batch[0].cuda().contiguous()
    labels = batch[1].cuda().contiguous()
    return images, labels


def build_data_loader(dataset, micro_batch_size,
                      num_workers, drop_last, shuffle):
    """Data loader. Note that batch-size is the local (per GPU) batch-size."""

    # Sampler.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank,
        drop_last=drop_last, shuffle=shuffle
    )

    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
    )

    return data_loader


def _build_infinite_size_dataloader(dataloader):
    """Build a looped dataloader with infinite size."""

    iterator = dataloader.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = dataloader.__iter__()


def _build_train_valid_dataloaders(train_dataset, valid_dataset):
    """Traing and validation dataloaders."""
    args = get_args()

    print_rank_0('building train and validation dataloaders ...')
    # Training dataset.
    train_dataloader = build_data_loader(train_dataset, args.micro_batch_size,
                                         args.num_workers, False, True)
    # Set the training iterations.
    args.train_iters_per_epoch = len(train_dataloader)
    args.train_iters = args.epochs * args.train_iters_per_epoch
    # Validation dataset. For this dataset, we do not need to set up
    # shuffling so we can just use a simple infinite loop.
    valid_dataloader_ = build_data_loader(valid_dataset, args.micro_batch_size,
                                          args.num_workers, True,  False)
    valid_dataloader = _build_infinite_size_dataloader(valid_dataloader_)

    # Now that we've built the data loaders, set batch_size arguments
    # to the actual batch size the model will see for this dataset.
    # This is necessary so pipeline transfers know what size they are
    # and the LR schedule, which is based on samples seen, gets set
    # correctly.
    args.orig_micro_batch_size = args.micro_batch_size
    args.orig_global_batch_size = args.global_batch_size

    return train_dataloader, valid_dataloader


def _train(
    model,
    optimizer,
    opt_param_scheduler,
    forward_step,
    train_dataloader,
    valid_dataloader,
    end_of_epoch_callback,
    process_non_loss_data_func=None
):
    """Train the model."""
    args = get_args()
    timers = get_timers()

    # Turn on training mode which enables dropout.
    for m in model:
        m.train()

    # Tracking loss.
    losses_dict_sum = {}

    # Starting epoch and iteration
    start_epoch = args.iteration // args.train_iters_per_epoch
    start_iteration = args.iteration % args.train_iters_per_epoch
    iteration = args.iteration

    # Memory reporting flag.
    report_memory_flag = True

    # For each remaining epoch
    timers("interval-time", log_level=0).start(barrier=True)
    for epoch in range(start_epoch, args.epochs):
        print_rank_0("working on epoch {} ...".format(epoch + 1))

        # Set the data loader epoch to shuffle the index iterator.
        train_dataloader.sampler.set_epoch(args.seed + epoch)
        train_dataloader.dataset.set_epoch(epoch)

        # For all the batches in the dataset.
        for iteration_, batch in enumerate(train_dataloader):

            # Ignore the iterations before starting value
            if iteration_ < start_iteration:
                continue
            # Set to zero so the next epoch does not skip any batches.
            start_iteration = 0

            # Train for one step.
            losses_dict, skipped_iter, grad_norm, num_zeros_in_grad = train_step(
                forward_step, batch, model, optimizer, opt_param_scheduler
            )
            iteration += 1

            # Logging.
            params_norm = None

            report_memory_flag = training_log(
                losses_dict,
                losses_dict_sum,
                optimizer.param_groups[0]["lr"],
                iteration,
                optimizer.get_loss_scale().item(),
                report_memory_flag,
                skipped_iter,
                grad_norm,
                params_norm,
                num_zeros_in_grad
            )

            # Autoresume
            if args.adlr_autoresume and \
                    iteration % args.adlr_autoresume_interval == 0:
                check_adlr_autoresume_termination(iteration, model, optimizer,
                                                  opt_param_scheduler)

            # Checkpointing
            if args.save and args.save_interval and \
                    iteration % args.save_interval == 0:
                save_checkpoint(iteration, model, optimizer,
                                opt_param_scheduler)

            # Evaluation
            if args.eval_interval and iteration % args.eval_interval == 0:
                prefix = "iteration {}".format(iteration)
                evaluate_and_print_results(
                    prefix,
                    forward_step,
                    valid_dataloader,
                    model,
                    iteration,
                    process_non_loss_data_func,
                    False,
                )

        # Callback at the end of each epoch.
        if end_of_epoch_callback is not None:
            end_of_epoch_callback(model, epoch)


def finetune(
    train_valid_datasets_provider,
    model_provider,
    forward_step,
    model_type=ModelType.encoder_or_decoder,
    process_non_loss_data_func=None,
    end_of_epoch_callback_provider=None,
):
    """Main finetune function used across all tasks."""
    args = get_args()
    timers = get_timers()

    # Train and validation data loaders.
    timers("train/valid/test dataset/dataloder", log_level=0).start()
    if args.epochs > 0:
        train_dataset, valid_dataset = train_valid_datasets_provider()
        train_dataloader, valid_dataloader = _build_train_valid_dataloaders(
            train_dataset, valid_dataset
        )
    timers("train/valid/test dataset/dataloder").stop()

    # Build calback function.
    timers("callback function", log_level=0).start()
    end_of_epoch_callback = None
    if end_of_epoch_callback_provider is not None:
        end_of_epoch_callback = end_of_epoch_callback_provider()
    timers("callback function").stop()

    # Build model, optimizer and learning rate scheduler.
    timers("model and optimizer", log_level=0).start()
    model, optimizer, opt_param_scheduler = \
        setup_model_and_optimizer(
            model_provider,
            model_type,
            scale_lr_cond=lambda name, param: ".head." in name,
            lr_mult=args.head_lr_mult)
    timers("model and optimizer").stop()

    # If pretrained checkpoint is provided and we have not trained for
    # any iteration (i.e., iteration is zero), then load the pretrained
    # checkpoint.
    timers("pretrained checkpoint", log_level=0).start(barrier=True)
    if args.iteration == 0 and args.pretrained_checkpoint is not None:
        if args.pretrained_checkpoint_type == 'default':
            original_load = args.load
            args.load = args.pretrained_checkpoint
            _ = load_checkpoint(model, None, None, strict=False)
            args.load = original_load
        elif args.pretrained_checkpoint_type == 'external':
            unwrap_model = utils.unwrap_model(model)
            state_dict = torch.load(args.pretrained_checkpoint,
                                    map_location="cpu")
            unwrap_model[0].module.backbone.load_state_dict(state_dict,
                                                            strict=False)
        elif args.pretrained_checkpoint_type == 'constrastive':
            unwrap_model = utils.unwrap_model(model)
            state_dict = torch.load(args.pretrained_checkpoint,
                                    map_location="cpu")
            state_dict = state_dict["model"]
            state_dict = {k.replace("teacher.backbone.", ""): v
                          for k, v in state_dict.items()
                          if k.startswith("teacher.backbone.")}
            unwrap_model[0].module.backbone.load_state_dict(state_dict,
                                                            strict=False)
        else:
            raise Exception("pretrained checkpoint type {} not supported".format(args.pretrained_checkpoint_type))

        # This is critical when only model is loaded. We should make sure
        # master parameters are also updated.
        optimizer.reload_model_params()

    timers("pretrained checkpoint").stop()

    # Print setup timing.
    print_rank_0("done with setups ...")
    timers.log(
        [
            "train/valid/test dataset/dataloder",
            "callback function",
            "model and optimizer",
            "pretrained checkpoint",
        ]
    )
    print_rank_0("training ...")

    # Finetune the model.
    if args.epochs > 0:
        _train(
            model,
            optimizer,
            opt_param_scheduler,
            forward_step,
            train_dataloader,
            valid_dataloader,
            end_of_epoch_callback,
            process_non_loss_data_func,
        )
    # Or just evaluate.
    else:
        if end_of_epoch_callback is not None:
            print_rank_0("evaluation only mode, setting epoch to -1")
            end_of_epoch_callback(model, epoch=-1)

    print_rank_0("done :-)")

