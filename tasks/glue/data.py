# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""GLUE dataset."""

from abc import ABC
from abc import abstractmethod

from torch.utils.data import Dataset

from megatron import print_rank_0
from tasks.data_utils import build_sample
from tasks.data_utils import build_tokens_types_paddings_from_text


class GLUEAbstractDataset(ABC, Dataset):
    """GLUE base dataset class."""

    def __init__(self, task_name, dataset_name, datapaths,
                 tokenizer, max_seq_length):
        # Store inputs.
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        print_rank_0(' > building {} dataset for {}:'.format(self.task_name,
                                                             self.dataset_name))
        # Process the files.
        string = '  > paths:'
        for path in datapaths:
            string += ' ' + path
        print_rank_0(string)
        self.samples = []
        for datapath in datapaths:
            self.samples.extend(self.process_samples_from_single_path(datapath))
        print_rank_0('  >> total number of samples: {}'.format(
            len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        ids, types, paddings = build_tokens_types_paddings_from_text(
            raw_sample['text_a'], raw_sample['text_b'],
            self.tokenizer, self.max_seq_length)
        sample = build_sample(ids, types, paddings,
                              raw_sample['label'], raw_sample['uid'])
        return sample

    @abstractmethod
    def process_samples_from_single_path(self, datapath):
        """Abstract method that takes a single path / filename and
        returns a list of dataset samples, each sample being a dict of
            {'text_a': string, 'text_b': string, 'label': int, 'uid': int}
        """
        pass
