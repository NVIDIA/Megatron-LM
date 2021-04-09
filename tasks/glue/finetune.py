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

"""GLUE finetuning/evaluation."""

from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import mpu
from megatron.model.classification import Classification
from tasks.eval_utils import accuracy_func_provider
from tasks.finetune_utils import finetune


def glue_classification(num_classes, Dataset,
                        name_from_datapath_func):

    def train_valid_datasets_provider():
        """Build train and validation dataset."""
        args = get_args()
        tokenizer = get_tokenizer()

        train_dataset = Dataset('training', args.train_data,
                                tokenizer, args.seq_length)
        valid_dataset = Dataset('validation', args.valid_data,
                                tokenizer, args.seq_length)

        return train_dataset, valid_dataset

    def model_provider(pre_process=True, post_process=True):
        """Build the model."""
        args = get_args()

        print_rank_0('building classification model for {} ...'.format(
            args.task))
        model = Classification(num_classes=num_classes, num_tokentypes=2,
                               pre_process=pre_process, post_process=post_process)

        return model

    def metrics_func_provider():
        """Privde metrics callback function."""
        def single_dataset_provider(datapath):
            args = get_args()
            tokenizer = get_tokenizer()

            name = name_from_datapath_func(datapath)
            return Dataset(name, [datapath], tokenizer, args.seq_length)
        return accuracy_func_provider(single_dataset_provider)

    """Finetune/evaluate."""
    finetune(train_valid_datasets_provider, model_provider,
             end_of_epoch_callback_provider=metrics_func_provider)


def main():
    args = get_args()

    if args.task == 'MNLI':

        num_classes = 3
        from tasks.glue.mnli import MNLIDataset as Dataset

        def name_from_datapath(datapath):
            return datapath.split('MNLI')[-1].strip(
                '.tsv').strip('/').replace('_', '-')

    elif args.task == 'QQP':

        num_classes = 2
        from tasks.glue.qqp import QQPDataset as Dataset

        def name_from_datapath(datapath):
            return datapath.split('QQP')[-1].strip(
                '.tsv').strip('/').replace('_', '-')

    else:
        raise NotImplementedError('GLUE task {} is not implemented.'.format(
            args.task))

    glue_classification(num_classes, Dataset, name_from_datapath)
