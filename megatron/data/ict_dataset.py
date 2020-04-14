import random
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from megatron import get_tokenizer
from megatron import print_rank_0
from megatron import mpu
from megatron.data import helpers


class InverseClozeDataset(Dataset):
    """Dataset containing sentences and their blocks for an inverse cloze task."""
    def __init__(self, name, context_dataset, titles_dataset, data_prefix,
                 num_epochs, max_num_samples, max_seq_length,
                 short_seq_prob, seed):
        self.name = name
        self.seed = seed
        self.max_seq_length = max_seq_length
        self.context_dataset = context_dataset
        self.titles_dataset = titles_dataset
        self.short_seq_prob = short_seq_prob
        self.rng = random.Random(self.seed)

        self.samples_mapping = get_samples_mapping(self.context_dataset,
                                                   self.titles_dataset,
                                                   data_prefix,
                                                   num_epochs,
                                                   max_num_samples,
                                                   self.max_seq_length,
                                                   self.seed,
                                                   self.name)
        tokenizer = get_tokenizer()
        self.vocab_id_list = list(tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_list = tokenizer.inv_vocab
        self.cls_id = tokenizer.cls
        self.sep_id = tokenizer.sep
        self.mask_id = tokenizer.mask
        self.pad_id = tokenizer.pad

    def __len__(self):
        return self.samples_mapping.shape[0]

    def __getitem__(self, idx):
        start_index, end_index, _ = self.samples_mapping[idx]
        context = [self.indexed_dataset[i] for i in range(start_index, end_index)]
        assert len(context) > 1

        title = self.titles_dataset[idx]
        assert sum(len(c) for c in context) + len(title) <= self.max_seq_length - 3

        rand_sent_idx = self.rng.randint(0, len(context) - 1)
        if self.rng.random() < 0.1:
            input = list(context[rand_sent_idx])
        else:
            input = context.pop(rand_sent_idx)

        input_tokens, input_token_types, input_pad_mask = self.concat_and_pad_tokens(input)
        context_tokens, context_token_types, context_pad_mask = self.concat_and_pad_tokens(context, title)

        sample = {
            'input_text': np.array(input_tokens),
            'input_types': np.array(input_token_types),
            'input_pad_mask': np.array(input_pad_mask),
            'context_text': np.array(context_tokens),
            'context_types': np.array(context_token_types),
            'context_pad_mask': np.array(context_pad_mask)
        }

        return sample

    def concat_and_pad_tokens(self, tokens, title=None):
        """concat with special tokens and pad sequence to self.max_seq_length"""
        tokens = [self.cls_id] + tokens + [self.sep_id]
        if title is not None:
            tokens += title + [self.sep_id]
        assert len(tokens) <= self.max_seq_length

        num_pad = self.max_seq_length - len(tokens)
        pad_mask = [0] * len(tokens) + [1] * num_pad
        tokens += [self.pad_id] * num_pad
        token_types = [0] * self.max_seq_length
        return tokens, token_types, pad_mask


def get_samples_mapping(context_dataset,
                        titles_dataset,
                        data_prefix,
                        num_epochs,
                        max_num_samples,
                        max_seq_length,
                        seed,
                        name):
    if not num_epochs:
        if not max_num_samples:
            raise ValueError("Need to specify either max_num_samples "
                             "or num_epochs")
        num_epochs = np.iinfo(np.int32).max - 1
    if not max_num_samples:
        max_num_samples = np.iinfo(np.int64).max - 1

    # Filename of the index mapping
    indexmap_filename = data_prefix
    indexmap_filename += '_{}_indexmap'.format(name)
    if num_epochs != (np.iinfo(np.int32).max - 1):
        indexmap_filename += '_{}ep'.format(num_epochs)
    if max_num_samples != (np.iinfo(np.int64).max - 1):
        indexmap_filename += '_{}mns'.format(max_num_samples)
    indexmap_filename += '_{}msl'.format(max_seq_length)
    indexmap_filename += '_{}s'.format(seed)
    indexmap_filename += '.npy'

    # Build the indexed mapping if not exist.
    if torch.distributed.get_rank() == 0 and \
            not os.path.isfile(indexmap_filename):
        print(' > WARNING: could not find index map file {}, building '
              'the indices on rank 0 ...'.format(indexmap_filename))

        # Make sure the types match the helpers input types.
        assert context_dataset.doc_idx.dtype == np.int64
        assert context_dataset.sizes.dtype == np.int32

        # Build samples mapping
        verbose = torch.distributed.get_rank() == 0
        start_time = time.time()
        print_rank_0(' > building samples index mapping for {} ...'.format(
            name))
        samples_mapping = helpers.build_blocks_mapping(
            context_dataset.doc_idx,
            context_dataset.sizes,
            titles_dataset.sizes,
            num_epochs,
            max_num_samples,
            max_seq_length-3,  # account for added tokens
            seed,
            verbose)
        print_rank_0(' > done building samples index mapping')
        np.save(indexmap_filename, samples_mapping, allow_pickle=True)
        print_rank_0(' > saved the index mapping in {}'.format(
            indexmap_filename))
        # Make sure all the ranks have built the mapping
        print_rank_0(' > elapsed time to build and save samples mapping '
                     '(seconds): {:4f}'.format(
            time.time() - start_time))
    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    assert counts[0].item() == torch.distributed.get_world_size(
        group=mpu.get_data_parallel_group())

    # Load indexed dataset.
    print_rank_0(' > loading indexed mapping from {}'.format(
        indexmap_filename))
    start_time = time.time()
    samples_mapping = np.load(indexmap_filename, allow_pickle=True)
    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))
    print_rank_0('    total number of samples: {}'.format(
        samples_mapping.shape[0]))

    return samples_mapping

