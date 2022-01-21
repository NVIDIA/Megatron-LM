
import glob
import os
import time

import torch
from torch.utils.data import Dataset
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import get_args
from tasks.data_utils import build_sample
from tasks.data_utils import build_sample_hoc
from tasks.data_utils import build_tokens_types_paddings_from_ids
from tasks.data_utils import build_tokens_types_paddings_from_text
from tasks.data_utils import clean_text

from transformers import AutoTokenizer
from transformers import BertTokenizerFast
from transformers import PreTrainedTokenizerFast
from pathlib import Path
import re
import numpy as np


class HOCDataset(Dataset):

    def __init__(self, dataset_name, datapaths, tokenizer, max_seq_length,ignore_index=-100, tasks=['I-PAR', 'I-INT', 'I-OUT']):
        args = get_args()
        
        self.dataset_name = dataset_name
        print_rank_0(' > building HOC dataset for {}:'.format(
            self.dataset_name))

        string = '  > paths:'
        for path in datapaths:
            string += ' ' + path
        print_rank_0(string)

        #HFTokenizer = BertTokenizerFast(args.vocab_file)
        MegatronTokenizer = tokenizer

        self.samples = []
        for datapath in datapaths:
            self.samples.extend(process_single_datapath(datapath, MegatronTokenizer, max_seq_length, self.dataset_name))

        print_rank_0('  >> total number of samples: {}'.format(
            len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def _read_hoc(file_path,dataset_name):
    fp = str(file_path)

    filenames = glob.glob(os.path.join(fp, '*.tsv'))


    data_x = []
    data_y = []
    abstract_ids = []
    for filename in filenames:
        fn_str = str(filename)
        if dataset_name == fn_str.split('/')[-1].split('.')[0]:
            with open(filename, 'r') as f:
                rowCounter = 0
                for row in f:
                    if rowCounter != 0:
                        labelSentenceIndex = row.split('\t')
                        data_x.append(labelSentenceIndex[1])
                        labels = labelSentenceIndex[0].split(',')
                        labels = [int(x.split('_')[1]) for x in labels]
                        data_y.append(labels)
                        abstract = labelSentenceIndex[-1].split('_')[0]
                        abstract_ids.append(abstract)

                    rowCounter += 1
        else:
            continue

    return data_x, data_y, abstract_ids

def process_single_datapath(datapath, MegatronTokenizer, max_seq_length, dataset_name):

    print_rank_0('   > working on {}'.format(datapath))
    start_time = time.time()
    data_x, data_y, abstract_ids = _read_hoc(datapath,dataset_name)

    samples = []
    num_samples = 0
    for i in range(len(data_x)):
        data_str = str(data_x[i])
        data_str.strip("[").strip("]")
        context = clean_text(data_str)
        no_context = None
        #Tokenize data
        ids, types, paddings = build_tokens_types_paddings_from_text(
            context, no_context, MegatronTokenizer,  max_seq_length)
        label = data_y[i]
        abstract_id = abstract_ids[i]
        samples.append(build_sample_hoc(ids,types,paddings,label,abstract_id))
        num_samples += 1
    
    elapsed_time = time.time() - start_time
    print_rank_0('    > processed {} samples'
                 ' in {:.2f} seconds'.format(num_samples, elapsed_time))
    return samples

