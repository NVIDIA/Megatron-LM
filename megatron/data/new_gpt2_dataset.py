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

"""GPT2 Style dataset."""

import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset

import helpers
#from bert_dataset import get_train_valid_test_split_


def print_rank_0(message):
    print(message)


def build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                    train_valid_test_num_samples,
                                    seq_length, seed, skip_warmup):

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(' > dataset split:')
    def print_split_stats(name, index):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))
    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], end=splits[index+1],
                                  step=1, dtype=np.int32)
            dataset = GPT2Dataset(name, data_prefix,
                                  documents, indexed_dataset,
                                  train_valid_test_num_samples[index],
                                  seq_length, seed)
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)


def get_indexed_dataset_(data_prefix, data_impl, skip_warmup):

    print_rank_0(' > building dataset index ...')

    start_time = time.time()
    indexed_dataset = make_indexed_dataset(data_prefix,
                                           data_impl,
                                           skip_warmup)
    print_rank_0(' > finished creating indexed dataset in {:4f} '
                 'seconds'.format(time.time() - start_time))

    print_rank_0(' > indexed dataset stats:')
    print_rank_0('    number of documents: {}'.format(
        indexed_dataset.sizes.shape[0]))

    return indexed_dataset


class GPT2Dataset(Dataset):

    def __init__(self, name, data_prefix,
                 documents, indexed_dataset,
                 num_samples, seq_length, seed):

        self.name = name
        self.data_prefix = data_prefix
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.seed = seed
        self.indexed_dataset = indexed_dataset

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        # Build index mappings.
        self.num_epochs, self.doc_idx, self.sample_idx, self.shuffle_idx \
            = _build_index_mappings(self.name, self.data_prefix, documents,
                                    self.indexed_dataset.sizes,
                                    self.num_samples, self.seq_length,
                                    self.seed)


    def __len__(self):
        return self.sample_idx.shape[0]


    def __getitem__(self, idx):
        # Get the shuffled index.
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx+1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx+1][1]
        # If we are within the same document, just extract the chunk.
        if doc_index_f == doc_index_l:
            sample = self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                              offset=offset_f,
                                              length=offset_l - offset_f + 1)
        else:
            # Otherwise, get the rest of the initial document.
            sample_list = [self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                                    offset=offset_f)]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f+1, doc_index_l):
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            sample_list.append(self.indexed_dataset.get(
                self.doc_idx[doc_index_l],
                length=offset_l+1))
            sample = np.concatenate(sample_list)

        return sample



def _build_index_mappings(name, data_prefix, documents, sizes,
                          num_samples, seq_length, seed):
    """doc-idx, sample-idx, and shuffle-idx."""
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples)
    # rng state
    np_rng = np.random.RandomState(seed=seed)
    
    # Filename of the index mappings.
    _filename = data_prefix
    _filename += '_{}_indexmap'.format(name)
    _filename += '_{}ns'.format(num_samples)
    _filename += '_{}sl'.format(seq_length)
    _filename += '_{}s'.format(seed)
    doc_idx_filename = _filename + '_doc_idx.npy'
    sample_idx_filename = _filename + '_sample_idx.npy'
    shuffle_idx_filename = _filename + '_shuffle_idx.npy'

    # Build the indexed mapping if not exist.
    if True: #torch.distributed.get_rank() == 0:
        if (not os.path.isfile(doc_idx_filename)) or \
           (not os.path.isfile(sample_idx_filename)) or \
           (not os.path.isfile(shuffle_idx_filename)):
            
            print_rank_0(' > WARNING: could not find index map files, building '
                         'the indices on rank 0 ...')
            # doc-idx.
            start_time = time.time()
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save doc-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # sample-idx.
            start_time = time.time()
            import helpers
            sample_idx = helpers.build_sample_idx(sizes, doc_idx, seq_length,
                                                  num_epochs, tokens_per_epoch)
            #sample_idx = _build_sample_idx(sizes, doc_idx, seq_length,
            #                               num_epochs, tokens_per_epoch)
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save sample-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # shuffle-idx.
            start_time = time.time()
            shuffle_idx = _build_shuffle_idx(sample_idx.shape[0], np_rng)
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save shuffle-idx mapping'
                         ' (seconds): {:4f}'.format(time.time() - start_time))

    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = torch.cuda.LongTensor([1])
    #torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    #assert counts[0].item() == torch.distributed.get_world_size(
    #    group=mpu.get_data_parallel_group())

    # Load mappings.
    start_time = time.time()
    print_rank_0(' > loading doc-idx mapping from {}'.format(
        doc_idx_filename))
    doc_idx = np.load(doc_idx_filename, allow_pickle=True)
    print_rank_0(' > loading sample-idx mapping from {}'.format(
        sample_idx_filename))
    sample_idx = np.load(sample_idx_filename, allow_pickle=True)
    print_rank_0(' > loading shuffle-idx mapping from {}'.format(
        shuffle_idx_filename))
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True)
    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))
    print_rank_0('    total number of samples: {}'.format(
        sample_idx.shape[0]))

    return num_epochs, doc_idx, sample_idx, shuffle_idx


def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def _num_epochs(tokens_per_epoch, seq_length, num_samples):
    """Based on number of samples and sequence lenght, calculate how many
    epochs will be needed."""
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((total_tokens - 1) // seq_length) >= num_samples:
            return num_epochs


def _build_doc_idx(documents, num_epochs, np_rng):
    """Build an array with length = number-of-epochs * number-of-dcuments.
    Each index is mapped to a corresponding document."""
    doc_idx = np.mgrid[0:num_epochs, 0:len(documents)][1]
    doc_idx[:] = documents
    doc_idx = doc_idx.reshape(-1)
    doc_idx = doc_idx.astype(np.int32)
    np_rng.shuffle(doc_idx)
    return doc_idx


def _build_sample_idx(sizes, doc_idx, seq_length,
                      num_epochs, tokens_per_epoch):
    """Sample index mapping is a 2D array with sizes
    [number-of-samples + 1, 2] where [..., 0] contains
    the index into `doc_idx` and [..., 0] is the
    starting offset in that document."""

    # Total number of samples. For -1 see comments in `_num_epochs`.
    num_samples = (num_epochs * tokens_per_epoch - 1) // seq_length
    sample_idx = np.zeros([num_samples + 1, 2], dtype=np.int32)

    # Index into sample_idx.
    sample_index = 0
    # Index into doc_idx.
    doc_idx_index = 0
    # Begining offset for each document.
    doc_offset = 0
    # Start with first document and no offset.
    sample_idx[sample_index][0] = doc_idx_index
    sample_idx[sample_index][1] = doc_offset
    sample_index += 1
    while sample_index <= num_samples:
        # Start with a fresh sequence.
        remaining_seq_length = seq_length + 1
        while remaining_seq_length != 0:
            # Get the document length.
            doc_id = doc_idx[doc_idx_index]
            doc_length = sizes[doc_id] - doc_offset
            # And add it to the current sequence.
            remaining_seq_length -= doc_length
            # If we have more than a full sequence, adjust offset and set
            # remaining length to zero so we return from the while loop.
            # Note that -1 here is for the same reason we have -1 in
            # `_num_epochs` calculations.
            if remaining_seq_length <= 0:
                doc_offset += (remaining_seq_length + doc_length - 1)
                remaining_seq_length = 0
            else:
                # Otherwise, start from the begining of the next document.
                doc_idx_index += 1
                doc_offset = 0
        # Record the sequence.
        sample_idx[sample_index][0] = doc_idx_index
        sample_idx[sample_index][1] = doc_offset
        sample_index += 1

    return sample_idx


def _build_shuffle_idx(size, np_rng):
    """Build the range [0, size) and shuffle."""
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    #np_rng.shuffle(shuffle_idx)
    return shuffle_idx



class IndexedDataset:

    def __init__(self, num_docs, min_doc_length, max_doc_length, seq_length):

        self.seq_length = seq_length
        assert min_doc_length > 0

        self.tokens = []
        self.sizes = np.zeros(num_docs, dtype=np.int32)
        for i in range(num_docs):
            size = np.random.randint(low=min_doc_length, high=max_doc_length,
                                     size=1, dtype=np.uint32)[0]
            tokens_ = np.random.randint(low=1, high=60000,
                                        size=size, dtype=np.uint32)
            tokens_[-1] = 0
            self.sizes[i] = size
            self.tokens.append(tokens_)

        self.tokens_flat = None

    def get(self, doc_idx, offset=None, length=None):
        if length is None:
            if offset is None:
                return self.tokens[doc_idx]
            else:
                return self.tokens[doc_idx][offset:]
        if offset is None:
            return self.tokens[doc_idx][0:length]
        return self.tokens[doc_idx][offset:(offset+length)]

    def get_sample(self, index):
        start = index * self.seq_length
        end = start + self.seq_length + 1
        return self.tokens_flat[start:end]

    def build_tokens_flat(self, doc_idx):
        self.tokens_flat = np.concatenate([self.tokens[i] for i in doc_idx])


def test(seed, data_prefix, seq_length, num_samples,
         num_docs, min_doc_length, max_doc_length):

    print('testing for seed: {}, seq-length: {}, num-samples: {}, '
          'num-docs: {}, min-doc-length: {}, max-doc-length: {}'.format(
              seed, seq_length, num_samples,
              num_docs, min_doc_length, max_doc_length))
    np.random.seed(seed)

    indexed_dataset = IndexedDataset(num_docs, min_doc_length,
                                     max_doc_length, seq_length)
    indices = np.random.randint(indexed_dataset.sizes.shape[0]-2, size=2)
    documents = np.arange(np.min(indices), np.max(indices)+1)
    dataset = GPT2Dataset('gpt2', data_prefix, documents, indexed_dataset,
                          num_samples, seq_length, seed)

    print(' > number of epochs:', dataset.num_epochs)
    indexed_dataset.build_tokens_flat(dataset.doc_idx)

    for idx in range(num_samples):
        a = dataset[idx]
        b = indexed_dataset.get_sample(idx)
        assert np.sum(a - b) == 0

    print('passed')
    

if __name__ == '__main__':

    print('gpt2 dataset ...')


    import random
    data_prefix = 'junk/'
    for seed in range(1234, 1245):
        random.seed(seed)
        num_docs = random.randint(1, 999)
        min_doc_length = random.randint(1, 99)
        max_doc_length = random.randint(100, 9999)
        num_samples = random.randint(num_docs, 100*num_docs)
        seq_length = random.randint(min_doc_length, max_doc_length)

        test(seed, data_prefix, seq_length, num_samples,
             num_docs, min_doc_length, max_doc_length)
    exit()

    '''

    num_docs = 5
    min_doc_length = 2
    max_doc_length = 10
    num_samples = 9
    seq_length = 4
    seed = 1234
    
    np.random.seed(seed)
    indexed_dataset = IndexedDataset(num_docs, min_doc_length,
                                     max_doc_length, seq_length)
    print('> indexed dataset:')
    for s in indexed_dataset.tokens:
        print('   {}'.format(s))

    documents = np.array([1,2,3], dtype=np.int32)

    dataset = GPT2Dataset('gpt2', documents, indexed_dataset,
                          num_samples, seq_length, seed)
    indexed_dataset.build_tokens_flat(dataset.doc_idx)

    print(indexed_dataset.get_sample(6))
    print(dataset[6])
    '''    
    '''
    myds = MyDataset(ds, num_samples, seq_length)
    num_docs = myds._num_docs()
    print('> number of document: {}'.format(num_docs))
    tokens_per_epoch = myds._num_tokens()
    print('> number of tokens: {}'.format(tokens_per_epoch))
    num_epochs = myds._num_epochs(tokens_per_epoch)
    print('> number of epochs: {}'.format(num_epochs))
    doc_idx = myds._build_doc_idx(num_docs, num_epochs)
    print('> doc_idx: {}'.format(doc_idx))

    ds.build_tokens_flat(doc_idx)
    sample_idx =myds._build_sample_idx(num_epochs, tokens_per_epoch, doc_idx)

    for s in sample_idx:
        print(s)
        
    print(ds.tokens_flat)
    print(myds.get_sample(8))
    print(ds.get_sample(8))
    '''
    

