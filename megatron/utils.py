# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

"""Utilities for logging and serialization"""

import os
import random
import time
import numpy as np
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

#from megatron.global_vars import get_args
#from megatron.global_vars import get_adlr_autoresume

from megatron import mpu
from megatron.fp16 import FP16_Module
from megatron.fp16 import FP16_Optimizer
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import get_params_for_weight_decay_optimization



def print_rank_0(message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def reduce_losses(losses):
    """Reduce a tensor of losses across all GPUs."""
    reduced_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(reduced_losses)
    reduced_losses = reduced_losses / torch.distributed.get_world_size()

    return reduced_losses


def check_adlr_autoresume_termination(iteration, model, optimizer,
                                      lr_scheduler, args):
    # Add barrier to ensure consistnecy.
    torch.distributed.barrier()
    if args.AutoResume.termination_requested():
        if args.save:
            save_checkpoint(iteration, model, optimizer, lr_scheduler, args)
        print_rank_0(">>> autoresume termination request found!")
        if torch.distributed.get_rank() == 0:
            args.AutoResume.request_resume()
        print_rank_0(">>> training terminated. Returning")
        exit(0)




def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i+1):, :(i+1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i+1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    return attention_mask, loss_mask, position_ids



def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, model-parallel,min, max, norm\n'
    optimizer_ = optimizer
    if isinstance(optimizer, FP16_Optimizer):
        optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = param.data.norm()
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.model_parallel))
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


def report_memory(name):
    """Simple GPU memory report."""

    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | cached: {}'.format(torch.cuda.memory_cached() / mega_bytes)
    string += ' | max cached: {}'.format(
        torch.cuda.max_memory_cached()/ mega_bytes)
    print_rank_0(string)


def vocab_size_with_padding(num_tokens, args):

    after = num_tokens
    multiple = args.make_vocab_size_divisible_by * \
               mpu.get_model_parallel_world_size()
    while (after % multiple) != 0:
        after += 1
    print_rank_0('> padded vocab (size: {}) with {} dummy '
                 'tokens (new size: {})'.format(
                     num_tokens, after - num_tokens, after))
    return after


def get_checkpoint_name(checkpoints_path, iteration, release=False,
                        mp_rank=None):
    if release:
        d = 'release'
    else:
        d = 'iter_{:07d}'.format(iteration)
    return os.path.join(checkpoints_path, d,
                        'mp_rank_{:02d}'.format(
                            mpu.get_model_parallel_rank() if mp_rank is None \
                            else mp_rank),
                        'model_optim_rng.pt')


def ensure_directory_exists(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_checkpoint_tracker_filename(checkpoints_path):
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')


def save_checkpoint(iteration, model, optimizer,
                    lr_scheduler, args):
    """Save a model checkpoint."""
    # Only rank zer0 of the data parallel writes to the disk.
    if isinstance(model, torchDDP):
        model = model.module
    if mpu.get_data_parallel_rank() == 0:
        checkpoint_name = get_checkpoint_name(args.save, iteration)
        print('global rank {} is saving checkpoint at iteration {:7d} to {}'.
              format(torch.distributed.get_rank(), iteration, checkpoint_name))

        sd = {}
        sd['iteration'] = iteration
        sd['model'] = model.state_dict_for_save_checkpoint()

        # Optimizer stuff.
        if not args.no_save_optim:
            if optimizer is not None:
                sd['optimizer'] = optimizer.state_dict()
            if lr_scheduler is not None:
                sd['lr_scheduler'] = lr_scheduler.state_dict()

        # rng states.
        if not args.no_save_rng:
            sd['random_rng_state'] = random.getstate()
            sd['np_rng_state'] = np.random.get_state()
            sd['torch_rng_state'] = torch.get_rng_state()
            sd['cuda_rng_state'] = torch.cuda.get_rng_state()
            sd['rng_tracker_states'] = mpu.get_cuda_rng_tracker().get_states()

        ensure_directory_exists(checkpoint_name)
        torch.save(sd, checkpoint_name)
        print('  successfully saved {}'.format(checkpoint_name))

    # Wait so everyone is done (necessary)
    torch.distributed.barrier()
    # And update the latest iteration
    if torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))
    # Wait so everyone is done (not necessary)
    torch.distributed.barrier()


def load_checkpoint(model, optimizer, lr_scheduler, args):
    """Load a model checkpoint."""
    if isinstance(model, torchDDP):
        model = model.module
    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(args.load)
    if not os.path.isfile(tracker_filename):
        print_rank_0('WARNING: could not find the metadata file {} '.format(
            tracker_filename))
        print_rank_0('    will not load any checkpoints and will start from '
                     'random')
        return 0
    iteration = 0
    release = False
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'
            if not release:
                print_rank_0('ERROR: Invalid metadata file {}. Exiting'.format(
                    tracker_filename))
                exit()

    assert iteration > 0 or release, 'error parsing metadata file {}'.format(
        tracker_filename)

    # Checkpoint.
    checkpoint_name = get_checkpoint_name(args.load, iteration, release)
    if mpu.get_data_parallel_rank() == 0:
        print('global rank {} is loading checkpoint {}'.format(
            torch.distributed.get_rank(), checkpoint_name))

    # Load the checkpoint.
    try:
        sd = torch.load(checkpoint_name, map_location='cpu')
    except ModuleNotFoundError:
        # For backward compatibility.
        print_rank_0(' > deserializing using the old code structure ...')
        import sys
        sys.modules['fp16.loss_scaler'] = sys.modules[
            'megatron.fp16.loss_scaler']
        sd = torch.load(checkpoint_name, map_location='cpu')
        sys.modules.pop('fp16.loss_scaler', None)
    except:
        print_rank_0('could not load the checkpoint')
        exit()

    # Iterations.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = sd['iteration']
        except KeyError:
            try: # Backward compatible with older checkpoints
                iteration = sd['total_iters']
            except KeyError:
                print_rank_0('A metadata file exists but Unable to load iteration '
                             ' from checkpoint {}, exiting'.format(checkpoint_name))
                exit()
    # Model.
    try:
        model.load_state_dict(sd['model'])
    except KeyError:
        print_rank_0('A metadata file exists but unable to load model '
                     'from checkpoint {}, exiting'.format(checkpoint_name))
        exit()

    # Optimizer.
    if not release and not args.finetune and not args.no_load_optim:
        try:
            if optimizer is not None:
                optimizer.load_state_dict(sd['optimizer'])
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(sd['lr_scheduler'])
        except KeyError:
            print_rank_0('Unable to load optimizer from checkpoint {}, exiting. '
                         'Specify --no-load-optim or --finetune to prevent '
                         'attempting to load the optimizer '
                         'state.'.format(checkpoint_name))
            exit()

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            random.setstate(sd['random_rng_state'])
            np.random.set_state(sd['np_rng_state'])
            torch.set_rng_state(sd['torch_rng_state'])
            torch.cuda.set_rng_state(sd['cuda_rng_state'])
            mpu.get_cuda_rng_tracker().set_states(sd['rng_tracker_states'])
        except KeyError:
            print_rank_0('Unable to load optimizer from checkpoint {}, exiting.'
                         'Specify --no-load-optim or --finetune to prevent '
                         'attempting to load the optimizer '
                         'state.'.format(checkpoint_name))
            exit()

    torch.distributed.barrier()
    if mpu.get_data_parallel_rank() == 0:
        print('  successfully loaded {}'.format(checkpoint_name))

    return iteration


def load_weights(src, dst, dst2src=False):
    """
    Loads weights from src to dst via in place copy.
    src is a huggingface gpt2model, while dst is one of our models.
    dst2src=True loads parameters from our models into huggingface's.
    ^dst2src is still untested
    """
    conv_layer = 'Conv1D' in  str(type(src))
    for n, p in src.named_parameters():
        if dst2src:
            data = dst._parameters[n].data
            load = p.data
        else:
            data = p.data
            load = dst._parameters[n].data
        if conv_layer and 'weight' in n:
            data = data.t().contiguous()
        load.copy_(data)
#        dst._parameters[n].data.copy_(data)

def load_mlp(our, oai, dst2src=False):
    load_weights(oai.c_fc, our.dense_h_to_4h, dst2src)
    load_weights(oai.c_proj, our.dense_4h_to_h, dst2src)

def load_attention(our, oai, dst2src=False):
    load_weights(oai.c_attn, our.query_key_value, dst2src)
    load_weights(oai.c_proj, our.dense, dst2src)

def load_transformer_layer(our, oai, dst2src=False):
    load_weights(oai.ln_1, our.input_layernorm, dst2src)
    load_weights(oai.ln_2, our.post_attention_layernorm, dst2src)
    load_mlp(our.mlp, oai.mlp, dst2src)
    load_attention(our.attention, oai.attn, dst2src)

def move_weights(our, oai, dst2src=False):
    """
    Loads weights from `oai` to `our` via in place copy.
    `oai` is a huggingface gpt2model, while `our` is one of our models.
    dst2src=True loads parameters from our models into huggingface's.
    ^dst2src=True is still untested
    """
#    while isinstance(our, (torchDDP, model.distributed.DistributedDataParallel, FP16_Module)):
#        our=our.module
    transformer_model = oai.transformer
    load_weights(transformer_model.ln_f, our.transformer.final_layernorm, dst2src)
    load_weights(transformer_model.wte, our.word_embeddings, dst2src)
    load_weights(transformer_model.wpe, our.position_embeddings, dst2src)

    for our_layer, oai_layer in zip(our.transformer.layers, oai.transformer.h):
        load_transformer_layer(our_layer, oai_layer, dst2src)


def merge_parallel_state_dicts(state_dicts):
    temp_sd = {}
    for sd in state_dicts:
        for k, v in sd.items():
            temp_sd[k].append()
    pass

def merge_parallel_checkpoints(checkpoint_dir, model_parallel_size):
    pass
