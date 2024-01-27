
import glob
import json
import os
import time

from torch.utils.data import Dataset

from megatron import print_rank_0
from tasks.data_utils import build_sample

LABELS = {'unclean': 0, 'clean': 1}

class FilterDataset(Dataset):

    def __init__(self, dataset_name, datapaths, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.dataset_name = dataset_name
        print_rank_0(' > building FILTER dataset for {}:'.format(
            self.dataset_name))

        string = '  > paths:'
        for path in datapaths:
            string += ' ' + path
        print_rank_0(string)

        self.samples = []
        for datapath in datapaths:
            self.samples.extend(process_single_datapath(datapath))

        print_rank_0('  >> total number of samples: {}'.format(
            len(self.samples)))

        # This indicates that each "sample" has multiple samples that
        # will collapse into batch dimension
        self.sample_multiplier = 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        text_ids = self.tokenizer.tokenize(raw_sample['text'])
        types = [1]*len(text_ids)
        paddings = [1]*len(text_ids)
        # truncate or pad TODO think about better batching and padding
        if len(text_ids) >= self.max_seq_length:
            text_ids = text_ids[:self.max_seq_length]
            types = types[:self.max_seq_length]
            paddings = paddings[:self.max_seq_length]
        else:
            pad_length = self.max_seq_length - len(text_ids)
            text_ids.extend([self.tokenizer.eos_id] * pad_length)
            types.extend([self.tokenizer.eos_id] * pad_length)
            paddings.extend([0] * pad_length)

        sample = build_sample(text_ids, types, paddings,
                              raw_sample['label'], raw_sample['uid'])
        return sample


def process_single_datapath(datapath):
    """Read in FILTER files, combine, clean-up, tokenize, and convert to
    samples."""

    print_rank_0('   > working on {}'.format(datapath))
    start_time = time.time()
    # open the file and load as a json object
    with open(datapath, 'r') as f:
        raw_samples = json.load(f)
    
    # build samples where each sample is a dict with keys 'text' and 'label', and 'uid'
    samples = []
    num_samples = 0
    uid = 0
    for raw_sample in raw_samples:
        neg_sample = raw_sample['input']
        pos_sample = raw_sample['output']

        samples.append(dict(text=neg_sample, label=LABELS['unclean'], uid=uid))
        uid+=1
        samples.append(dict(text=pos_sample, label=LABELS['clean'], uid=uid))
        uid+=1
        num_samples+=2

    elapsed_time = time.time() - start_time
    print_rank_0('    > processed {} samples'
                 ' in {:.2f} seconds'.format(num_samples, elapsed_time))

    return samples
