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

"""Race."""

from megatron import get_args
from megatron import get_tokenizer
from megatron import print_rank_0
from megatron.model.multiple_choice import MultipleChoice
from tasks.eval_utils import accuracy_func_provider
from tasks.finetune_utils import finetune
from tasks.race.data import RaceDataset


def train_valid_datasets_provider():
    """Provide train and validation datasets."""
    args = get_args()
    tokenizer = get_tokenizer()

    train_dataset = RaceDataset('training', args.train_data,
                                tokenizer, args.seq_length)
    valid_dataset = RaceDataset('validation', args.valid_data,
                                tokenizer, args.seq_length)

    return train_dataset, valid_dataset


def model_provider():
    """Build the model."""
    args = get_args()

    print_rank_0('building multichoice model for RACE ...')

    return MultipleChoice(
        num_layers=args.num_layers,
        vocab_size=args.padded_vocab_size,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        embedding_dropout_prob=args.hidden_dropout,
        attention_dropout_prob=args.attention_dropout,
        output_dropout_prob=args.hidden_dropout,
        max_sequence_length=args.max_position_embeddings,
        checkpoint_activations=args.checkpoint_activations)


def metrics_func_provider():
    """Privde metrics callback function."""
    args = get_args()
    tokenizer = get_tokenizer()

    def single_dataset_provider(datapath):
        name = datapath.split('RACE')[-1].strip('/').replace('/', '-')
        return RaceDataset(name, [datapath], tokenizer, args.seq_length)

    return accuracy_func_provider(single_dataset_provider)


def main():

    finetune(train_valid_datasets_provider, model_provider,
             end_of_epoch_callback_provider=metrics_func_provider)
