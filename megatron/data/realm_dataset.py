import itertools
import os
import random
import time

import numpy as np
import spacy
import torch
from torch.utils.data import Dataset

from megatron import get_tokenizer, print_rank_0, mpu
from megatron.data.bert_dataset import BertDataset
from megatron.data.dataset_utils import create_masked_lm_predictions, pad_and_convert_to_numpy

#qa_nlp = spacy.load('en_core_web_lg')


def build_simple_training_sample(sample, target_seq_length, max_seq_length,
                                 vocab_id_list, vocab_id_to_token_dict,
                                 cls_id, sep_id, mask_id, pad_id,
                                 masked_lm_prob, np_rng):
    tokens = list(itertools.chain(*sample))[:max_seq_length - 2]
    tokens, tokentypes = create_single_tokens_and_tokentypes(tokens, cls_id, sep_id)

    max_predictions_per_seq = masked_lm_prob * max_seq_length
    (tokens, masked_positions, masked_labels, _) = create_masked_lm_predictions(
        tokens, vocab_id_list, vocab_id_to_token_dict, masked_lm_prob,
        cls_id, sep_id, mask_id, max_predictions_per_seq, np_rng)

    tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np \
        = pad_and_convert_to_numpy(tokens, tokentypes, masked_positions,
                                   masked_labels, pad_id, max_seq_length)

    train_sample = {
        'tokens': tokens_np,
        'labels': labels_np,
        'loss_mask': loss_mask_np,
        'pad_mask': padding_mask_np
    }
    return train_sample


class REALMDataset(Dataset):
    """Dataset containing simple masked sentences for masked language modeling.

    The dataset should yield sentences just like the regular BertDataset
    However, this dataset also needs to be able to return a set of blocks
    given their start and end indices.

    Presumably

    """
    def __init__(self, name, block_dataset, title_dataset, data_prefix,
                 num_epochs, max_num_samples, masked_lm_prob,
                 max_seq_length, short_seq_prob, seed):
        self.name = name
        self.seed = seed
        self.max_seq_length = max_seq_length
        self.masked_lm_prob = masked_lm_prob
        self.block_dataset = block_dataset
        self.title_dataset = title_dataset
        self.short_seq_prob = short_seq_prob
        self.rng = random.Random(self.seed)

        self.samples_mapping = self.get_samples_mapping(
            data_prefix, num_epochs, max_num_samples)
        self.tokenizer = get_tokenizer()
        self.vocab_id_list = list(self.tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_list = self.tokenizer.inv_vocab
        self.cls_id = self.tokenizer.cls
        self.sep_id = self.tokenizer.sep
        self.mask_id = self.tokenizer.mask
        self.pad_id = self.tokenizer.pad

    def __len__(self):
        return self.samples_mapping.shape[0]

    def __getitem__(self, idx):
        start_idx, end_idx, doc_idx, block_idx = self.samples_mapping[idx]
        seq_length = self.max_seq_length
        block = [list(self.block_dataset[i]) for i in range(start_idx, end_idx)]
        assert len(block) > 1
        np_rng = np.random.RandomState(seed=(self.seed + idx))

        sample = build_simple_training_sample(block, seq_length,
                                              self.max_seq_length,
                                              self.vocab_id_list,
                                              self.vocab_id_to_token_list,
                                              self.cls_id,
                                              self.sep_id,
                                              self.mask_id,
                                              self.pad_id,
                                              self.masked_lm_prob,
                                              np_rng)
        sample.update({'query_block_indices': np.array([block_idx]).astype(np.int64)})
        return sample

    def get_samples_mapping(self, data_prefix, num_epochs, max_num_samples):
        if not num_epochs:
            if not max_num_samples:
                raise ValueError("Need to specify either max_num_samples "
                                 "or num_epochs")
            num_epochs = np.iinfo(np.int32).max - 1
        if not max_num_samples:
            max_num_samples = np.iinfo(np.int64).max - 1

        # Filename of the index mapping
        indexmap_filename = data_prefix
        indexmap_filename += '_{}_indexmap'.format(self.name)
        if num_epochs != (np.iinfo(np.int32).max - 1):
            indexmap_filename += '_{}ep'.format(num_epochs)
        if max_num_samples != (np.iinfo(np.int64).max - 1):
            indexmap_filename += '_{}mns'.format(max_num_samples)
        indexmap_filename += '_{}msl'.format(self.max_seq_length)
        indexmap_filename += '_{}s'.format(self.seed)
        indexmap_filename += '.npy'

        # Build the indexed mapping if not exist.
        if torch.distributed.get_rank() == 0 and \
                not os.path.isfile(indexmap_filename):
            print(' > WARNING: could not find index map file {}, building '
                  'the indices on rank 0 ...'.format(indexmap_filename))

            # Make sure the types match the helpers input types.
            assert self.block_dataset.doc_idx.dtype == np.int64
            assert self.block_dataset.sizes.dtype == np.int32

            # Build samples mapping
            verbose = torch.distributed.get_rank() == 0
            start_time = time.time()
            print_rank_0(' > building samples index mapping for {} ...'.format(
                self.name))
            from megatron.data.dataset_utils import compile_helper
            compile_helper()
            from megatron.data import helpers
            samples_mapping = helpers.build_blocks_mapping(
                self.block_dataset.doc_idx,
                self.block_dataset.sizes,
                self.title_dataset.sizes,
                num_epochs,
                max_num_samples,
                self.max_seq_length-3,  # account for added tokens
                self.seed,
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


def create_single_tokens_and_tokentypes(_tokens, cls_id, sep_id):
    tokens = []
    tokens.append(cls_id)
    tokens.extend(list(_tokens))
    tokens.append(sep_id)
    tokentypes = [0] * len(tokens)
    return tokens, tokentypes


def spacy_ner(block_text):
    candidates = {}
    block = qa_nlp(block_text)
    starts = []
    answers = []
    for ent in block.ents:
        starts.append(int(ent.start_char))
        answers.append(str(ent.text))
    candidates['starts'] = starts
    candidates['answers'] = answers


class ICTDataset(Dataset):
    """Dataset containing sentences and their blocks for an inverse cloze task."""
    def __init__(self, name, block_dataset, title_dataset, data_prefix,
                 num_epochs, max_num_samples, max_seq_length,
                 short_seq_prob, seed):
        self.name = name
        self.seed = seed
        self.max_seq_length = max_seq_length
        self.block_dataset = block_dataset
        self.title_dataset = title_dataset
        self.short_seq_prob = short_seq_prob
        self.rng = random.Random(self.seed)

        self.samples_mapping = self.get_samples_mapping(
            data_prefix, num_epochs, max_num_samples)
        self.tokenizer = get_tokenizer()
        self.vocab_id_list = list(self.tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_list = self.tokenizer.inv_vocab
        self.cls_id = self.tokenizer.cls
        self.sep_id = self.tokenizer.sep
        self.mask_id = self.tokenizer.mask
        self.pad_id = self.tokenizer.pad

    def __len__(self):
        return self.samples_mapping.shape[0]

    def __getitem__(self, idx):
        start_idx, end_idx, doc_idx, block_idx = self.samples_mapping[idx]
        title = list(self.title_dataset[int(doc_idx)])
        block = [list(self.block_dataset[i]) for i in range(start_idx, end_idx)]
        assert len(block) > 1

        # avoid selecting the first or last sentence to be the query.
        if len(block) == 2:
            rand_sent_idx = int(self.rng.random() > 0.5)
        else:
            rand_sent_idx = self.rng.randint(1, len(block) - 2)

        # keep the query in the context 10% of the time.
        if self.rng.random() < 1:
            query = block[rand_sent_idx].copy()
        else:
            query = block.pop(rand_sent_idx)

        # still need to truncate because blocks are concluded when
        # the sentence lengths have exceeded max_seq_length.
        query = query[:self.max_seq_length - 2]
        block = list(itertools.chain(*block))[:self.max_seq_length - (3 + len(title))]

        query_tokens, query_pad_mask = self.concat_and_pad_tokens(query)
        block_tokens, block_pad_mask = self.concat_and_pad_tokens(block, title)

        sample = {
            'query_tokens': np.array(query_tokens),
            'query_pad_mask': np.array(query_pad_mask),
            'block_tokens': np.array(block_tokens),
            'block_pad_mask': np.array(block_pad_mask),
            'block_data': np.array([start_idx, end_idx, doc_idx, block_idx]).astype(np.int64)
        }

        return sample

    def encode_text(self, text):
        return self.tokenizer.tokenize(text)

    def decode_tokens(self, token_ids):
        tokens = self.tokenizer.tokenizer.convert_ids_to_tokens(token_ids)
        return ' '.join(token for token in tokens if token != '[PAD]')

    def get_block(self, start_idx, end_idx, doc_idx):
        """Get the IDs for an evidence block plus the title of the corresponding document"""
        block = [list(self.block_dataset[i]) for i in range(start_idx, end_idx)]
        title = list(self.title_dataset[int(doc_idx)])

        block = list(itertools.chain(*block))[:self.max_seq_length - (3 + len(title))]
        block_tokens, block_pad_mask = self.concat_and_pad_tokens(block, title)

        return (block_tokens, block_pad_mask)

    def get_null_block(self):
        block, title = [], []
        block_tokens, block_pad_mask = self.concat_and_pad_tokens(block, title)

        return (block_tokens, block_pad_mask)

    def concat_and_pad_tokens(self, tokens, title=None):
        """concat with special tokens and pad sequence to self.max_seq_length"""
        tokens = [self.cls_id] + tokens + [self.sep_id]
        if title is not None:
            tokens += title + [self.sep_id]
        assert len(tokens) <= self.max_seq_length, len(tokens)

        num_pad = self.max_seq_length - len(tokens)
        pad_mask = [1] * len(tokens) + [0] * num_pad
        tokens += [self.pad_id] * num_pad
        return tokens, pad_mask

    def get_samples_mapping(self, data_prefix, num_epochs, max_num_samples):
        if not num_epochs:
            if not max_num_samples:
                raise ValueError("Need to specify either max_num_samples "
                                 "or num_epochs")
            num_epochs = np.iinfo(np.int32).max - 1
        if not max_num_samples:
            max_num_samples = np.iinfo(np.int64).max - 1

        # Filename of the index mapping
        indexmap_filename = data_prefix
        indexmap_filename += '_{}_indexmap'.format(self.name)
        if num_epochs != (np.iinfo(np.int32).max - 1):
            indexmap_filename += '_{}ep'.format(num_epochs)
        if max_num_samples != (np.iinfo(np.int64).max - 1):
            indexmap_filename += '_{}mns'.format(max_num_samples)
        indexmap_filename += '_{}msl'.format(self.max_seq_length)
        indexmap_filename += '_{}s'.format(self.seed)
        indexmap_filename += '.npy'

        # Build the indexed mapping if not exist.
        if torch.distributed.get_rank() == 0 and \
                not os.path.isfile(indexmap_filename):
            print(' > WARNING: could not find index map file {}, building '
                  'the indices on rank 0 ...'.format(indexmap_filename))

            # Make sure the types match the helpers input types.
            assert self.block_dataset.doc_idx.dtype == np.int64
            assert self.block_dataset.sizes.dtype == np.int32

            # Build samples mapping
            verbose = torch.distributed.get_rank() == 0
            start_time = time.time()
            print_rank_0(' > building samples index mapping for {} ...'.format(
                self.name))
            from megatron.data.dataset_utils import compile_helper
            compile_helper()
            from megatron.data import helpers
            samples_mapping = helpers.build_blocks_mapping(
                self.block_dataset.doc_idx,
                self.block_dataset.sizes,
                self.title_dataset.sizes,
                num_epochs,
                max_num_samples,
                self.max_seq_length-3,  # account for added tokens
                self.seed,
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
