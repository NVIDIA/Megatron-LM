# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Race."""

from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
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


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building multichoice model for RACE ...')
    model = MultipleChoice(num_tokentypes=2,
                           pre_process=pre_process,
                           post_process=post_process)

    return model


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
