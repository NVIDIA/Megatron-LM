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

"""Split model (tensor and pipeline) parallel partitions."""

import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

import torch

from megatron import mpu
from megatron.checkpointing import load_checkpoint, save_checkpoint
from megatron.checkpointing import read_metadata
from megatron.checkpointing import get_checkpoint_names
from megatron.checkpointing import get_checkpoint_version
from megatron.checkpointing import get_checkpoint_tracker_filename
from megatron.global_vars import set_global_variables, get_args
from megatron.global_vars import rebuild_tokenizer


def split_into_partitions(tensor, num_partitions, partition_dim, stride):

    per_partition_size = mpu.utils.divide(tensor.size(partition_dim),
                                          num_partitions)
    per_partition_per_stride_size = mpu.utils.divide(per_partition_size, stride)

    partitions_list = torch.split(tensor,
                                  per_partition_per_stride_size,
                                  dim=partition_dim)

    partitions = []
    for i in range(num_partitions):
        partition = torch.cat(partitions_list[i::num_partitions],
                              dim=partition_dim)
        partitions.append(partition)

    return partitions


def get_model(model_type, pre_process=True, post_process=True):

    if model_type == 'BERT':
        from pretrain_bert import model_provider
    elif model_type == 'GPT':
        from pretrain_gpt import model_provider
    elif model_type == 'RACE':
        from tasks.race.finetune import model_provider
    elif model_type == ['MNLI', 'QQP']:
        num_classes = 2
        if model_type == 'MNLI':
            num_classes = 3
        from megatron.model.classification import Classification
        def model_provider(pre_process, post_process):
            return Classification(num_classes=num_classes, num_tokentypes=2)
    else:
        raise Exception('unrecognized model type: {}'.format(model_type))

    model = model_provider(pre_process=pre_process, post_process=post_process)
    model = model.half()

    return model


def get_parallel_checkpoint_name(path):
    tracker_filename = get_checkpoint_tracker_filename(path)
    iteration, release = read_metadata(tracker_filename)
    checkpoint_name = get_checkpoint_names(path, iteration, use_distributed_optimizer=False, release=release)
    if release:
        iteration = "release"
    return checkpoint_name, iteration


def get_mp_merge_args(parser):
    """Provide extra arguments required for merging."""
    group = parser.add_argument_group(title='mp merge')

    group.add_argument('--model-type', type=str, required=True,
                       choices=['BERT', 'GPT', 'RACE', 'MNLI', 'QQP'],
                       help='Type of the mdoel.')
    group.add_argument('--target-pipeline-model-parallel-size', type=int, default=1,
                       help='Degree of pipeline model parallelism in output model.')

    return parser


def main():

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    os.environ["WORLD_SIZE"] = f'{2**31}'

    # Args
    set_global_variables(extra_args_provider=get_mp_merge_args,
                         args_defaults = {'use_cpu_initialization': True,
                                          'micro_batch_size': 1,
                                          'no_load_optim': True,
                                          'no_load_rng': True,
                                          'no_save_optim': True,
                                          'no_save_rng': True,
                                          'save_interval': 1})
    args = get_args()
    model_type = args.model_type
    num_partitions = args.tensor_model_parallel_size
    orig_tensor_model_parallel_size = args.tensor_model_parallel_size
    orig_pipeline_model_parallel_size = args.pipeline_model_parallel_size
    args.tensor_model_parallel_size = 1
    args.pipeline_model_parallel_size = 1
    tokenizer = rebuild_tokenizer(args)

    print('\n splitting model parallel partitions ...')
    print(' > checkpoint path: {}'.format(args.load))
    print(' > model parameters:')
    print('    number of tokens ................ {} '.format(
        tokenizer.vocab_size))
    print('    number of layers ................ {}'.format(args.num_layers))
    print('    hidden size ..................... {}'.format(args.hidden_size))
    print('    number of attention heads ....... {}'.format(
        args.num_attention_heads))
    print('    maximum position embeddings ..... {}'.format(
        args.max_position_embeddings))

    # Full model.
    print('> building the full model ...')
    mpu.initialize.set_tensor_model_parallel_world_size(1)
    mpu.initialize.set_tensor_model_parallel_rank(0)
    mpu.initialize.set_pipeline_model_parallel_world_size(1)
    mpu.initialize.set_pipeline_model_parallel_rank(0)

    cp_model = get_model(model_type)
    checkpoint_name, iteration = get_parallel_checkpoint_name(args.load)
    print(f'> loading {checkpoint_name} ...')
    load_checkpoint(cp_model, None, None)
    print(f'> checkpoint version {get_checkpoint_version()}')

    args.tensor_model_parallel_size = orig_tensor_model_parallel_size
    args.pipeline_model_parallel_size = orig_pipeline_model_parallel_size
    tokenizer = rebuild_tokenizer(args)
    model = get_model(model_type)
    
    cp_params_gen = cp_model.parameters()
    model_params = model.parameters()

    for param, cp_param in zip(model_params, cp_params_gen):
        if not hasattr(param, 'tensor_model_parallel'):
            with torch.no_grad():
                param.data.copy_(cp_param.data)
        else:
            partition_dim = param.partition_dim
            if param.size(partition_dim) == cp_param.size(partition_dim):
                param.data.copy_(cp_param.data)
            else:
                param[:cp_param.size(partition_dim)].data.copy_(cp_param.data)

    model_params = {}
    for name, param in model.named_parameters():
        if not hasattr(param, 'tensor_model_parallel'):
            params = param
        else:
            partition_dim = param.partition_dim
            stride = param.partition_stride
            params = split_into_partitions(param, num_partitions, partition_dim, stride)
        model_params[name] = params

    

    # regex to parse out layer number from param name
    layer_re = re.compile('layers\.([0-9]+)')
    layers_per_part = args.num_layers // args.pipeline_model_parallel_size

    mpu.initialize.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
    mpu.initialize.set_pipeline_model_parallel_world_size(args.pipeline_model_parallel_size)
    for pp_rank in range(args.pipeline_model_parallel_size):
        mpu.initialize.set_pipeline_model_parallel_rank(pp_rank)
        def update_layer_num(m):
            # TODO! This assumes no interleaved pipeline execution
            layer = int(m.group(1))
            layer += pp_rank * layers_per_part
            return f'layers.{layer}'
        for tp_rank in range(args.tensor_model_parallel_size):
            mpu.initialize.set_tensor_model_parallel_rank(tp_rank)
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            partition_model = get_model(model_type, pre_process, post_process)
            for dst_name, partition_param in partition_model.named_parameters():
                if dst_name == "word_embeddings.weight":
                    # See comment in MegatronModule.initialize_word_embeddings()
                    src_name = "language_model.embedding.word_embeddings.weight"
                else:
                    # Translate destination layer number (0-N for each partition)
                    # to source layer number (single-model layer number)
                    src_name = re.sub(layer_re, update_layer_num, dst_name)
                print(f" > copying {src_name} to {dst_name} in rank tp-{tp_rank} and pp-{pp_rank}'s model")
                if not hasattr(partition_param, 'tensor_model_parallel'):
                    partition_param.data.copy_(model_params[src_name].data)
                else:
                    partition_param.data.copy_(model_params[src_name][tp_rank].data)
            save_checkpoint(iteration, partition_model, None, None)
            
    print('done :-)')


if __name__ == '__main__':

    main()
