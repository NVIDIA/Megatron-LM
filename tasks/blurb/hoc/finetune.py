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
from megatron.model.classification_hoc import Classification_hoc
from tasks.blurb.hoc.eval_utils import accuracy_func_provider
from tasks.blurb.hoc.f1_utils import accuracy_f1_func_provider
from tasks.blurb.hoc.finetune_utils import finetune
from tasks.blurb.hoc.data import HOCDataset


def train_valid_datasets_provider():
    """Build train and validation dataset."""
    args = get_args()
    tokenizer = get_tokenizer()

    train_dataset = HOCDataset('train', args.train_data,
                            tokenizer, args.seq_length)
    valid_dataset = HOCDataset('dev', args.valid_data,
                            tokenizer, args.seq_length)

    return train_dataset, valid_dataset

def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    args = get_args()

    print_rank_0('building classification model for {} ...')
    model = Classification_hoc(num_classes=10, num_tokentypes=2,
                           pre_process=pre_process, post_process=post_process)

    return model

def metrics_func_provider():
    """Privde metrics callback function."""
    args = get_args()
    tokenizer = get_tokenizer()

    def single_dataset_provider(datapath):
        return HOCDataset('dev', [datapath], tokenizer, args.seq_length)
    return accuracy_func_provider(single_dataset_provider)

def test_metrics_func_provider():
    args = get_args()
    tokenizer = get_tokenizer()

    def single_dataset_provider(datapath):
        return HOCDataset('test', [datapath], tokenizer, args.seq_length)
    return accuracy_func_provider(single_dataset_provider)

def f1_func_provider():
    args = get_args()
    tokenizer = get_tokenizer()

    def single_dataset_provider(datapath):
        return HOCDataset('test', [datapath], tokenizer, args.seq_length)
    return accuracy_f1_func_provider(single_dataset_provider)

def main():
    finetune(train_valid_datasets_provider, model_provider,
             end_of_epoch_callback_provider=metrics_func_provider,
             end_of_training_eval_callback_provider=test_metrics_func_provider, 
             end_of_training_f1_callback_provider=f1_func_provider)
