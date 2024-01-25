# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""

import os
import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron.arguments import core_transformer_config_from_args

from axonn.intra_layer import drop
from axonn.intra_layer import optimize_communication
from axonn.intra_layer.communication import ForwardAllReduce
from axonn import axonn as ax
from contextlib import nullcontext

from custom_litgpt_dataloader.data_util import create_dataloaders

def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    args = get_args()

    print_rank_0('building GPT model ...')
    config = core_transformer_config_from_args(get_args())
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )

    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64
    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    
    if args.custom_dataloader:
        data = {"text": data}

    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    
    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids, # for this to work we need access to the tokenizer
        args.reset_attention_mask, # for this to work we need access to the tokenizer
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids

def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    #loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    loss_mask_sum = loss_mask.sum()
    loss = torch.sum(losses.view(-1) * loss_mask)

    group = ax.comm_handle.depth_intra_layer_parallel_group
    torch.distributed.all_reduce(loss_mask_sum, group=group)
    loss = ForwardAllReduce.apply(loss, group)
    loss = loss / loss_mask_sum
    # Check individual rank losses are not NaN prior to DP all-reduce.
    args = get_args()
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    tokens = drop(tokens, skip_channels=True)
    labels = drop(labels, skip_channels=True)
    loss_mask = drop(loss_mask, skip_channels=True)
    position_ids = drop(position_ids, skip_channels=True)
        #print(tokens.shape)
        #print(labels.shape)
        #print(loss_mask.shape)
        #print(attention_mask.shape)
        #print(position_ids.shape)
        #exit()
    
    if args.overlap_axonn_comm:
        ctx = partial(optimize_communication, 
                      overlap_all_reduce=True, 
                      overlap_reduce_scatter=args.overlap_axonn_reduce_scatter, 
                      overlap_all_gather=args.overlap_axonn_all_gather, 
                      model_object_for_overlapping_allgathers=model
                      )
    else:
        ctx = nullcontext
    with ctx():
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    if args.custom_dataloader:
        train_iterator, valid_iterator = create_dataloaders(
            batch_size= args.micro_batch_size,
            block_size= args.seq_length,
        )

        # these flags are set within megatron in 
        # the OG dataloader
        args.do_train = True
        args.do_valid = True
        args.do_test = False

        return train_iterator, valid_iterator
    else:
        print_rank_0('> building train, validation, and test datasets '
                     'for GPT ...')
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup),
            train_data_prefix=args.train_data_path,
            valid_data_prefix=args.valid_data_path,
            test_data_prefix=args.test_data_path,
            data_cache_path=args.data_cache_path)
        print_rank_0("> finished creating GPT datasets ...")

        return train_ds, valid_ds, test_ds


def set_device_and_init_torch_dist():
    from mpi4py import MPI
    import os
    MPI.Init()
    world_rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()

    # assign a unique GPU to each MPI process on a node    
    local_rank = world_rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    init_method = "tcp://"
    master_ip = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "6000")
    init_method += master_ip + ":" + master_port
   
    # create a process group across all processes 
    torch.distributed.init_process_group(
                init_method=init_method,
                backend="nccl",
                world_size=world_size,
                rank=world_rank
    )

    os.environ["RANK"] = str(world_rank)
    os.environ["WORLD_SIZE"] = str(world_size)


if __name__ == "__main__":
    set_device_and_init_torch_dist()
    #torch.cuda.set_per_process_memory_fraction(0.5) # 40GB
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
