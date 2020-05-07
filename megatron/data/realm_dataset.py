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
from megatron.data.dataset_utils import create_masked_lm_predictions, pad_and_convert_to_numpy, is_start_piece


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


qa_nlp = spacy.load('en_core_web_lg')


def salient_span_mask(tokens, vocab_id_list, vocab_id_to_token_dict,
                      cls_id, sep_id, mask_id, np_rng,
                      do_permutation=False):
    """Creates the predictions for the masked LM objective.
    Note: Tokens here are vocab ids and not text tokens."""

    cand_indexes = []
    # Note(mingdachen): We create a list for recording if the piece is
    # the starting piece of current token, where 1 means true, so that
    # on-the-fly whole word masking is possible.
    token_boundary = [0] * len(tokens)

    for (i, token) in enumerate(tokens):
        if token == cls_id or token == sep_id:
            token_boundary[i] = 1
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if len(cand_indexes) >= 1 and not is_start_piece(vocab_id_to_token_dict[token]):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
            if is_start_piece(vocab_id_to_token_dict[token]):
                token_boundary[i] = 1

    output_tokens = list(tokens)

    masked_lm_positions = []
    masked_lm_labels = []

    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx:idx + n])
        ngram_indexes.append(ngram_index)

    np_rng.shuffle(ngram_indexes)

    masked_lms = []
    covered_indexes = set()
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Note(mingdachen):
        # Skip current piece if they are covered in lm masking or previous ngrams.
        for index_set in cand_index_set[0]:
            for index in index_set:
                if index in covered_indexes:
                    continue

        n = np_rng.choice(ngrams[:len(cand_index_set)],
                          p=pvals[:len(cand_index_set)] /
                            pvals[:len(cand_index_set)].sum(keepdims=True))
        index_set = sum(cand_index_set[n - 1], [])
        n -= 1
        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        while len(masked_lms) + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if np_rng.random() < 0.8:
                masked_token = mask_id
            else:
                # 10% of the time, keep original
                if np_rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_id_list[np_rng.randint(0, len(vocab_id_list))]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict

    np_rng.shuffle(ngram_indexes)

    select_indexes = set()
    if do_permutation:
        for cand_index_set in ngram_indexes:
            if len(select_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            # Note(mingdachen):
            # Skip current piece if they are covered in lm masking or previous ngrams.
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes or index in select_indexes:
                        continue

            n = np.random.choice(ngrams[:len(cand_index_set)],
                                 p=pvals[:len(cand_index_set)] /
                                   pvals[:len(cand_index_set)].sum(keepdims=True))
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1

            while len(select_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(select_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes or index in select_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                select_indexes.add(index)
        assert len(select_indexes) <= num_to_predict

        select_indexes = sorted(select_indexes)
        permute_indexes = list(select_indexes)
        np_rng.shuffle(permute_indexes)
        orig_token = list(output_tokens)

        for src_i, tgt_i in zip(select_indexes, permute_indexes):
            output_tokens[src_i] = orig_token[tgt_i]
            masked_lms.append(MaskedLmInstance(index=src_i, label=orig_token[src_i]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels, token_boundary)


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
                 short_seq_prob, seed, use_titles=True):
        self.name = name
        self.seed = seed
        self.max_seq_length = max_seq_length
        self.block_dataset = block_dataset
        self.title_dataset = title_dataset
        self.short_seq_prob = short_seq_prob
        self.rng = random.Random(self.seed)
        self.use_titles = use_titles

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
        if self.use_titles:
            title = list(self.title_dataset[int(doc_idx)])
            title_pad_offset = 3 + len(title)
        else:
            title = None
            title_pad_offset = 2
        block = [list(self.block_dataset[i]) for i in range(start_idx, end_idx)]
        assert len(block) > 1

        rand_sent_idx = self.rng.randint(0, len(block) - 1)

        # keep the query in the context 10% of the time.
        if self.rng.random() < 1:
            query = block[rand_sent_idx].copy()
        else:
            query = block.pop(rand_sent_idx)

        # still need to truncate because blocks are concluded when
        # the sentence lengths have exceeded max_seq_length.
        query = query[:self.max_seq_length - 2]
        block = list(itertools.chain(*block))[:self.max_seq_length - title_pad_offset]

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
        if title is None:
            tokens = [self.cls_id] + tokens + [self.sep_id]
        else:
            tokens = [self.cls_id] + title + [self.sep_id] + tokens + [self.sep_id]
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
