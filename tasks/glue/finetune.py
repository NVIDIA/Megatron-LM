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

"""GLUE finetuning/evaluation."""

from megatron.utils import print_rank_0
from megatron.model.classification import Classification
from tasks.eval_utils import accuracy_func_provider
from tasks.finetune_utils import finetune


def glue_classification(args, num_classes, Dataset,
                        name_from_datapath_func):

    def train_valid_datasets_provider(args):
        """Build train and validation dataset."""
        train_dataset = Dataset('training', args.train_data,
                                args.tokenizer, args.seq_length)
        valid_dataset = Dataset('validation', args.valid_data,
                                args.tokenizer, args.seq_length)
        return train_dataset, valid_dataset


    def model_provider(args):
        """Build the model."""
        print_rank_0('building classification model for {} ...'.format(
            args.task))
        return Classification(
            num_classes=num_classes,
            num_layers=args.num_layers,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            embedding_dropout_prob=args.hidden_dropout,
            attention_dropout_prob=args.attention_dropout,
            output_dropout_prob=args.hidden_dropout,
            max_sequence_length=args.max_position_embeddings,
            checkpoint_activations=args.checkpoint_activations)


    def metrics_func_provider(args):
        """Privde metrics callback function."""
        def single_dataset_provider(datapath, args):
            name = name_from_datapath_func(datapath)
            return Dataset(name, [datapath], args.tokenizer, args.seq_length)
        return accuracy_func_provider(args, single_dataset_provider)


    """Finetune/evaluate."""
    finetune(args, train_valid_datasets_provider, model_provider,
             end_of_epoch_callback_provider=metrics_func_provider)


def main(args):

    if args.task == 'MNLI':

        num_classes = 3
        from .mnli import MNLIDataset as Dataset
        def name_from_datapath(datapath):
            return datapath.split('MNLI')[-1].strip(
                '.tsv').strip('/').replace('_', '-')

    elif args.task == 'QQP':

        num_classes = 2
        from .qqp import QQPDataset as Dataset
        def name_from_datapath(datapath):
            return datapath.split('QQP')[-1].strip(
                '.tsv').strip('/').replace('_', '-')

    else:
        raise NotImplementedError('GLUE task {} is not implemented.'.format(
            args.task))

    glue_classification(args, num_classes, Dataset, name_from_datapath)
