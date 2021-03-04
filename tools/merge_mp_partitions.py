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

"""Merge model parallel partitions."""

import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

import torch

from megatron import mpu
from megatron.checkpointing import load_checkpoint, save_checkpoint
from megatron.checkpointing import ensure_directory_exists
from megatron.checkpointing import get_checkpoint_name
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


def merge_partitions(merged, partitions, partition_dim, stride):

    # Number and size of each partition.
    num_partitions = len(partitions)
    per_partition_size = None
    for partition in partitions:
        if per_partition_size is None:
            per_partition_size = partition.size(partition_dim)
        else:
            assert per_partition_size == partition.size(partition_dim)

    def concat_partitions(partitions_):
        with torch.no_grad():
            if (per_partition_size * num_partitions) == merged.size(
                    partition_dim):
                torch.cat(partitions_, dim=partition_dim, out=merged)
            else:
                print('     ***WARNING*** sizes do not match. Will cut '
                      'the merged partitions by {} along dimension {} '
                      'to reduce the size from {} to {} ...'.format(
                          (per_partition_size * num_partitions) - \
                          merged.size(partition_dim), partition_dim,
                          per_partition_size * num_partitions,
                          merged.size(partition_dim)))
                merged_ = torch.cat(partitions_, dim=partition_dim)
                merged_split = torch.split(merged_, merged.size(partition_dim),
                                           dim=partition_dim)
                merged_ = merged_split[0]
                assert merged_.size(partition_dim) == merged.size(partition_dim)
                merged.data.copy_(merged_.data)

    # If stride is 1, then do simple concatination.
    if stride == 1:
        concat_partitions(partitions)
        return

    # For none unity strides, first split based on stride and then group.
    per_partition_per_stride_size = mpu.utils.divide(per_partition_size, stride)
    # Chunk and build a list.
    chunks = None
    for i, partition in enumerate(partitions):
        chunk = torch.split(partition,
                            per_partition_per_stride_size,
                            dim=partition_dim)

        if chunks is None:
            chunks = [0]*(num_partitions*len(chunk))
        chunks[i::num_partitions] = chunk

    # Concatinate.
    concat_partitions(chunks)

    return


def get_model(model_type):

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
        def model_provider():
            return Classification(num_classes=num_classes, num_tokentypes=2)
    else:
        raise Exception('unrecognized model type: {}'.format(model_type))

    model = model_provider()
    model = model.half()

    return model


def get_parallel_checkpoint_name(path):

    tracker_filename = get_checkpoint_tracker_filename(path)
    iteration = 0
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        iteration = int(metastring)
    assert iteration > 0
    checkpoint_name = get_checkpoint_name(path, iteration)

    return checkpoint_name, iteration


def test_split_merge():

    print('testing split and merge ...')

    #[QKV.ROW-COL]
    tensor = torch.FloatTensor([[1.11, 1.12, 1.13, 1.14, 1.15],
                                [1.21, 1.22, 1.23, 1.24, 1.25],
                                [1.31, 1.32, 1.33, 1.34, 1.35],
                                [1.41, 1.42, 1.43, 1.44, 1.45],
                                [2.11, 2.12, 2.13, 2.14, 2.15],
                                [2.21, 2.22, 2.23, 2.24, 2.25],
                                [2.31, 2.32, 2.33, 2.34, 2.35],
                                [2.41, 2.42, 2.43, 2.44, 2.45],
                                [3.11, 3.12, 3.13, 3.14, 3.15],
                                [3.21, 3.22, 3.23, 3.24, 3.25],
                                [3.31, 3.32, 3.33, 3.34, 3.35],
                                [3.41, 3.42, 3.43, 3.44, 3.45]])

    num_partitions = 2
    partition_dim = 0
    stride = 3
    partitions = split_into_partitions(tensor, num_partitions,
                                       partition_dim, stride)

    merged = torch.zeros_like(tensor)
    merge_partitions(merged, partitions, partition_dim, stride)

    max_error = (merged - tensor).abs().max()
    print('  > max error (should be zero): {}'.format(max_error))


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

    if args.pipeline_model_parallel_size > 1:
        print("Checkpoints with pipeline model parallelism are not currently supported.")
        exit()

    model_type = args.model_type
    orig_tensor_model_parallel_size = args.tensor_model_parallel_size
    args.tensor_model_parallel_size = 1
    tokenizer = rebuild_tokenizer(args)

    print('\n merging model parallel partitions ...')
    print(' > number of partitions: {}'.format(orig_tensor_model_parallel_size))
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
    merged_model = get_model(model_type)

    # Build and load partitions.
    partitions = []
    iteration = 0
    args.tensor_model_parallel_size = orig_tensor_model_parallel_size
    tokenizer = rebuild_tokenizer(args)
    mpu.initialize.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
    for rank in range(args.tensor_model_parallel_size):
        # Reset these since load_checkpoint asserts they are 0, but we are loading
        # multiple checkpoints in the same process and they get set each time
        args.consumed_train_samples = 0
        args.consumed_valid_samples = 0

        mpu.initialize.set_tensor_model_parallel_rank(rank)
        checkpoint_name, iteration = get_parallel_checkpoint_name(args.load)
        model_ = get_model(model_type)
        print(f'> loading {checkpoint_name} ...')
        load_checkpoint(model_, None, None)
        print(f'> checkpoint version {get_checkpoint_version()}')
        partitions.append(model_)

    # Parameter generators so we can loop through them semiltaneouly.
    merged_params_gen = merged_model.named_parameters()
    partitions_params_gen = [partition.named_parameters()
                             for partition in partitions]
    while True:
        try:

            # Get the params and check names.
            name, merged_param = next(merged_params_gen)
            print(' > working on {} ...'.format(name))
            print('     merged         type: {}, size: {}'.format(
                merged_param.dtype, list(merged_param.size())))
            partitions_param = []
            for rank, partition_params_gen in enumerate(partitions_params_gen):
                partition_name, partition_param = next(partition_params_gen)
                assert partition_name == name
                partitions_param.append(partition_param)
                print('     partition {}    type: {}, size: {}'.format(
                    rank, partition_param.dtype, list(partition_param.size())))

            # For the non-parallel parameters, simply copy the rank 0 values.
            if not hasattr(merged_param, 'tensor_model_parallel'):
                print('     none-parallel parameter, simple copy from rank 0')
                with torch.no_grad():
                    merged_param.data.copy_(partitions_param[0].data)
            # For parallel parameters, merge the values
            else:
                dim = merged_param.partition_dim
                stride = merged_param.partition_stride
                print(f'     parallel parameter merge with stride {stride} along '
                      f'dimention {dim}')
                merge_partitions(merged_param,
                                 partitions_param,
                                 dim,
                                 stride)

        except StopIteration:
            break

    partitions = []
    args.tensor_model_parallel_size = 1
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size

    assert args.num_layers % args.pipeline_model_parallel_size == 0, \
        'num_layers must be divisible by target pipeline model parallel size'
    layers_per_part = args.num_layers // args.pipeline_model_parallel_size

    tokenizer = rebuild_tokenizer(args)
    mpu.initialize.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
    mpu.initialize.set_tensor_model_parallel_rank(0)
    mpu.initialize.set_pipeline_model_parallel_world_size(args.pipeline_model_parallel_size)

    # regex to parse out layer number from param name
    layer_re = re.compile('layers\.([0-9]+)')

    if args.pipeline_model_parallel_size > 1:
        merged_params = {}
        for name, merged_param in merged_model.named_parameters():
            merged_params[name] = merged_param

        for rank in range(args.pipeline_model_parallel_size):
            mpu.initialize.set_pipeline_model_parallel_rank(rank)
            model = get_model(model_type)
            def update_layer_num(m):
                # TODO! This assumes no interleaved pipeline execution
                layer = int(m.group(1))
                layer += rank * layers_per_part
                return f'layers.{layer}'

            for dst_name, partition_param in model.named_parameters():
                if dst_name == "word_embeddings.weight":
                    # See comment in MegatronModule.initialize_word_embeddings()
                    src_name = "language_model.embedding.word_embeddings.weight"
                else:
                    # Translate destination layer number (0-N for each partition)
                    # to source layer number (single-model layer number)
                    src_name = re.sub(layer_re, update_layer_num, dst_name)
                print(f" > copying {src_name} to {dst_name} in rank {rank}'s model")
                partition_param.data.copy_(merged_params[src_name].data)

            partitions.append(model)
    else:
        partitions = [merged_model]

    for rank, model in enumerate(partitions):
        mpu.initialize.set_pipeline_model_parallel_rank(rank)
        print(f"> saving rank {rank}'s model")
        save_checkpoint(iteration, model, None, None)

    print('done :-)')


if __name__ == '__main__':

    main()
