# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Megatron initialization."""

import random
import os
import time

import numpy as np
import torch
from megatron import fused_kernels
from megatron import get_adlr_autoresume
from megatron import get_args
from megatron import get_tensorboard_writer
from megatron import mpu
from megatron.global_vars import set_global_variables
from megatron.mpu import (set_tensor_model_parallel_rank,
                          set_tensor_model_parallel_world_size)
from deepspeed.accelerator import get_accelerator
import deepspeed
import deepspeed.utils.groups as groups

def initialize_megatron(extra_args_provider=None, args_defaults={},
                        ignore_unknown_args=False, allow_no_cuda=False):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only 
    data processing. In general this arg should not be set unless you know 
    what you are doing.
    Returns a function to finalize distributed env initialization 
    (optionally, only when args.lazy_mpu_init == True)
    """
    if not allow_no_cuda:
        # Make sure cuda is available.
        assert get_accelerator().is_available(), 'Megatron requires accelerator.'

    # Parse args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(extra_args_provider=extra_args_provider,
                         args_defaults=args_defaults,
                         ignore_unknown_args=ignore_unknown_args)

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed()
        
        # Random seeds for reproducibility.
        if args.rank == 0:
            print('> setting random seeds to {} ...'.format(args.seed))
        _set_random_seed(args.seed)

    args = get_args()
    if  args.lazy_mpu_init:
        args.use_cpu_initialization=True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals    
        set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        set_tensor_model_parallel_rank(args.rank)    
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Initialize memory buffers.
        _initialize_mem_buffs()
        
        # Autoresume.
        _init_autoresume()

        # Compile dependencies.
        _compile_dependencies()

        # No continuation function
        return None


def _compile_dependencies():

    args = get_args()

    # =========================
    # Compile dataset C++ code.
    # =========================
    # TODO: move this to ninja
    if _is_rank_0():
        start_time = time.time()
        print('> compiling dataset index builder ...')
        from megatron.data.dataset_utils import compile_helper
        compile_helper()
        print('>>> done with dataset index builder. Compilation time: {:.3f} '
              'seconds'.format(time.time() - start_time), flush=True)
        
    if not get_accelerator().device_name() == 'cuda':
        print(">fused kernel is only supported in cuda, skip loading fused kernel")
        return 
    # ==================
    # Load fused kernels
    # ==================

    # Custom kernel constraints check.
    seq_len = args.seq_length
    attn_batch_size = \
        (args.num_attention_heads / args.tensor_model_parallel_size) * \
        args.micro_batch_size
    # Constraints on sequence length and attn_batch_size to enable warp based
    # optimization and upper triangular optimization (for causal mask)
    custom_kernel_constraint = seq_len > 16 and seq_len <=2048 and \
        seq_len % 4 == 0 and attn_batch_size % 4 == 0
    # Print a warning.
    if not ((args.fp16 or args.bf16) and
            custom_kernel_constraint and
            args.masked_softmax_fusion):
        if args.rank == 0:
            print('WARNING: constraints for invoking optimized'
                  ' fused softmax kernel are not met. We default'
                  ' back to unfused kernel invocations.', flush=True)
    
    # Always build on rank zero first.
    if _is_rank_0():
        start_time = time.time()
        print('> compiling and loading fused kernels ...', flush=True)
        if get_accelerator().device_count() > 0: # Skip when CPU-only
            fused_kernels.load(args)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        fused_kernels.load(args)
    # Simple barrier to make sure all ranks have passed the
    # compilation phase successfully before moving on to the
    # rest of the program. We think this might ensure that
    # the lock is released.
    torch.distributed.barrier()
    if _is_rank_0():
        print('>>> done with compiling and loading fused kernels. '
              'Compilation time: {:.3f} seconds'.format(
                  time.time() - start_time), flush=True)


def setup_deepspeed_random_and_activation_checkpointing(args):
    '''Optional DeepSpeed Activation Checkpointing features.
    Gives access to partition activations, contiguous memory optimizations
    and cpu checkpointing.
    Activation checkpoint requires keep track of the random states
    and setting the random seed for each MP process. Megatron uses
    mpu.get_cuda_rng_tracker and mpu.model_parallel_cuda_manual_seed
    for keeping track of the random states and setting the random seeds.
    Since they are used in places outside of activation checkpointing,
    we overwrite them to maintain consistency.
    This must be called before all the calls to mpu.model_parallel_cuda_manual_seed
    '''
    num_layers = args.num_layers // args.checkpoint_num_layers
    num_layers = num_layers if args.num_layers % args.checkpoint_num_layers == 0 else num_layers + 1
    if args.split_transformers:
        num_layers *= 2

    deepspeed.checkpointing.configure(
        mpu,
        partition_activations=args.partition_activations,
        contiguous_checkpointing=args.contigious_checkpointing,
        num_checkpoints=num_layers,
        checkpoint_in_cpu=args.checkpoint_in_cpu,
        synchronize=args.synchronize_each_layer,
        profile=args.profile_backward)

    mpu.checkpoint = deepspeed.checkpointing.checkpoint
    mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
    mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed


def _initialize_distributed():
    """Initialize torch.distributed and mpu."""
    args = get_args()
    device_count = get_accelerator().device_count()
    if torch.distributed.is_initialized():

        if args.rank == 0:
            print('torch distributed is already initialized, '
                  'skipping initialization ...', flush=True)
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    else:
        if args.rank == 0:
            print('> initializing torch distributed ...', flush=True)
        # Manually set the device ids.
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                assert args.local_rank == device, \
                    'expected local-rank to be the same as rank % device-count.'
            else:
                args.local_rank = device

            get_accelerator().set_device(device) # only do so when device_count > 0

        # Call the init process
        init_method = 'tcp://'
        master_ip = os.getenv('MASTER_ADDR', 'localhost')
        master_port = os.getenv('MASTER_PORT', '6000')
        init_method += master_ip + ':' + master_port

        if args.deepspeed or args.ds_inference:
            deepspeed.init_distributed()
        else:
            torch.distributed.init_process_group(
                backend=args.distributed_backend,
                world_size=args.world_size, rank=args.rank,
                init_method=init_method)
    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print('model parallel is already initialized')
        else:
            mpu.initialize_model_parallel(args.tensor_model_parallel_size,
                                          args.pipeline_model_parallel_size,
                                          args.virtual_pipeline_model_parallel_size)

    if args.deepspeed and args.deepspeed_activation_checkpointing:
        setup_deepspeed_random_and_activation_checkpointing(args)


def _init_autoresume():
    """Set autoresume start time."""
    autoresume = get_adlr_autoresume()
    if autoresume:
        torch.distributed.barrier()
        autoresume.init()
        torch.distributed.barrier()


def _set_random_seed(seed_):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        # No need to do so for CPU-only case.
        if get_accelerator().device_count() == 0:
            seed = seed_
        else:
            seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if get_accelerator().device_count() > 0:
            mpu.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError('Seed ({}) should be a positive integer.'.format(seed))


def write_args_to_tensorboard():
    """Write arguments to tensorboard."""
    args = get_args()
    writer = get_tensorboard_writer()
    if writer:
        for arg in vars(args):
            writer.add_text(arg, str(getattr(args, arg)),
                            global_step=args.iteration)


def _initialize_mem_buffs():
    """Initialize manually allocated static memory."""
    args = get_args()

    # Initialize memory for checkpointed activations.
    if args.distribute_checkpointed_activations:
        mpu.init_checkpointed_activations_memory_buffer()

def _is_rank_0():
    """Check whether it is rank 0. For AML, check if it is rank 0 of a node"""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0 or (
            'AZUREML_EXPERIMENT_ID' in os.environ and torch.distributed.get_rank() % get_accelerator().device_count() == 0
            ):
            return True
        else:
            return False
    else:
        return True
