# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""T5 Style dataset."""

import collections

import numpy as np
import torch

from megatron import get_tokenizer
from megatron.data.dataset_utils import (
    create_masked_lm_predictions,
    get_samples_mapping
)
from megatron.data.gpt_dataset import build_index_mappings_full_docs


class LengthExceededError(ValueError):
    def __init__(self, msg=None):
        if msg is None:
            msg = (
                'The sequence input became too long. '
                'Try to increase `--seq-length` or `--encoder-seq-length`.'
            )
        super().__init__(msg)


class DecoderLengthExceededError(ValueError):
    def __init__(self, msg=None):
        if msg is None:
            msg = (
                'The sequence input for the decoder became too long. '
                'Try to increase `--decoder-seq-length`.'
            )
        super().__init__(msg)


class T5Dataset(torch.utils.data.Dataset):

    def __init__(self, name, indexed_dataset, data_prefix,
                 splits_string, num_epochs, max_num_samples, masked_lm_prob,
                 max_seq_length, max_seq_length_dec,
                 short_seq_prob, add_mask_tokens, pack_samples, seed,
                 *,
                 data_cache_path=None):

        # Params to store.
        self.name = name
        self.seed = seed
        self.masked_lm_prob = masked_lm_prob
        self.max_seq_length = max_seq_length
        self.max_seq_length_dec = max_seq_length_dec

        # Dataset.
        self.indexed_dataset = indexed_dataset
        self.pack_samples = pack_samples

        # Minimum number of tokens added: BOS and EOS.
        min_added_tokens = 2
        if self.pack_samples:
            (
                self.doc_idx, self.sample_idx, self.shuffle_idx,
                self.desc, self.desc_hash,
            ) = build_index_mappings_full_docs(
                self.name, data_prefix, self.indexed_dataset.get_doc_idx()[:-1],
                self.indexed_dataset.sizes, splits_string, max_num_samples,
                self.max_seq_length - min_added_tokens, self.seed,
                data_cache_path=data_cache_path)
        else:
            # Build the samples mapping.
            self.samples_mapping = get_samples_mapping(
                self.indexed_dataset,
                data_prefix,
                splits_string,
                num_epochs,
                max_num_samples,
                self.max_seq_length - min_added_tokens, # account for added tokens
                short_seq_prob,
                self.seed,
                self.name,
                False,
            )

        # Vocab stuff.
        tokenizer = get_tokenizer()
        self.vocab_id_list = list(tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_dict = tokenizer.inv_vocab
        self.cls_id = tokenizer.cls
        self.sep_id = tokenizer.sep
        self.mask_id = tokenizer.mask
        self.pad_id = tokenizer.pad
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        if add_mask_tokens:
            self.sentinel_tokens = tokenizer.additional_special_tokens_ids
            assert len(self.sentinel_tokens) > 0, \
                "Provide the argument --vocab-extra-ids 100 to the script"
        else:
            self.sentinel_tokens = None

    def __len__(self):
        if self.pack_samples:
            return self.sample_idx.shape[0]
        else:
            return self.samples_mapping.shape[0]

    def __getitem__(self, idx):
        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        np_rng = np.random.RandomState(seed=(self.seed + idx))
        if self.pack_samples:
            samples_dict = self._pack_samples(np_rng, idx)
        else:
            start_index, end_index, seq_length = self.samples_mapping[idx]
            sample = []
            for index in range(start_index, end_index):
                sample.append(self.indexed_dataset[index])
            samples_dict = build_training_sample(
                sample, seq_length,
                self.max_seq_length,  # needed for padding
                self.max_seq_length_dec, self.vocab_id_list,
                self.vocab_id_to_token_dict, self.cls_id, self.sep_id,
                self.mask_id, self.pad_id, self.masked_lm_prob, np_rng,
                self.bos_id, self.eos_id, self.sentinel_tokens)
        return samples_dict

    def _pack_samples(self, np_rng, idx):
        samples = get_samples(self.indexed_dataset, self.doc_idx,
                              self.sample_idx, self.shuffle_idx, idx)
        samples_dict = create_samples_dict(
            self.max_seq_length, self.max_seq_length_dec)
        prev_len = 0
        prev_len_dec = 0

        for sample in samples:
            remaining_seq_len = self.max_seq_length - prev_len
            seq_length = min(remaining_seq_len, len(sample))

            result_sample = build_training_sample(
                [sample], seq_length,
                self.max_seq_length,  # needed for padding
                self.max_seq_length_dec, self.vocab_id_list,
                self.vocab_id_to_token_dict, self.cls_id, self.sep_id,
                self.mask_id, self.pad_id, self.masked_lm_prob, np_rng,
                self.bos_id, self.eos_id, self.sentinel_tokens)
            maybe_lens = update_samples_dict(
                samples_dict,
                result_sample,
                self.max_seq_length,
                self.max_seq_length_dec,
                prev_len,
                prev_len_dec,
                self.pad_id,
            )
            if maybe_lens is None:
                # We are exceeding our sequence length already.
                break

            len_enc, len_dec = maybe_lens
            prev_len += len_enc
            prev_len_dec += len_dec

        add_final_padding(samples_dict, prev_len, prev_len_dec, self.pad_id)
        return samples_dict


def build_training_sample(sample, target_seq_length,
                          max_seq_length, max_seq_length_dec,
                          vocab_id_list, vocab_id_to_token_dict,
                          cls_id, sep_id, mask_id, pad_id,
                          masked_lm_prob, np_rng, bos_id=None,
                          eos_id=None, sentinel_tokens=None):
    """Build training sample.

    Arguments:
        sample: A list of sentences in which each sentence is a list token ids.
        target_seq_length: Desired sequence length.
        max_seq_length: Maximum length of the sequence. All values are padded to
            this length.
        max_seq_length_dec: Maximum length of the decoder input sequence. All
            values are padded to this length.
        vocab_id_list: List of vocabulary ids. Used to pick a random id.
        vocab_id_to_token_dict: A dictionary from vocab ids to text tokens.
        cls_id: Start of example id.
        sep_id: Separator id.
        mask_id: Mask token id.
        pad_id: Padding token id.
        masked_lm_prob: Probability to mask tokens.
        np_rng: Random number genenrator. Note that this rng state should be
              numpy and not python since python randint is inclusive for
              the opper bound whereas the numpy one is exclusive.
        bos_id: start of decoder example id
        eos_id: end of generation id
        sentinel_tokens: unique value to be substituted for every replaced span
    """

    assert target_seq_length <= max_seq_length

    # flatten sentences into one list
    tokens = [token for sentence in sample for token in sentence]

    # Truncate to `target_sequence_length`.
    max_num_tokens = target_seq_length
    truncated = len(tokens) > max_num_tokens
    tokens = tokens[:max_num_tokens]

    # Masking.
    max_predictions_per_seq = masked_lm_prob * max_num_tokens
    (tokens, masked_positions, masked_labels, _, masked_spans) = create_masked_lm_predictions(
        tokens, vocab_id_list, vocab_id_to_token_dict, masked_lm_prob,
        cls_id, sep_id, mask_id, max_predictions_per_seq, np_rng,
        max_ngrams=10, geometric_dist=True, masking_style="t5")

    # Padding.
    tokens_enc, tokens_dec_in, labels, enc_mask, \
    dec_mask, enc_dec_mask, loss_mask \
        = pad_and_convert_to_numpy(tokens, masked_positions,
                                   masked_labels, pad_id, max_seq_length,
                                   max_seq_length_dec, masked_spans,
                                   bos_id, eos_id, sentinel_tokens)

    train_sample = {
        'text_enc': tokens_enc,
        'text_dec': tokens_dec_in,
        'labels': labels,
        'loss_mask': loss_mask,
        'truncated': int(truncated),
        'enc_mask': enc_mask,
        'dec_mask': dec_mask,
        'enc_dec_mask': enc_dec_mask,
    }
    return train_sample


def merge_subsequent_masks(tokens, masked_spans=None, bos_id=None,
                           eos_id=None, sentinel_tokens=None,
                           prefix_lm=False):
    if prefix_lm:
        assert len(masked_spans) <= 1, \
            'Received more than one masked span for PrefixLM masking'
    elif sentinel_tokens is not None:
        sentinel_tokens = collections.deque(sentinel_tokens)

    insert_mask_tokens = not prefix_lm and sentinel_tokens is not None

    t5_input = []
    (t5_decoder_in, t5_decoder_out) = ([bos_id], [])
    (start_index, end_index) = (0, None)
    for span in masked_spans:
        end_index = span.index[0]
        # The part of the sequence that is visible before the masked
        # span starts. Starting from beginning or end of last masked
        # span.
        before_mask = tokens[start_index:end_index]

        if insert_mask_tokens:
            flag = sentinel_tokens.popleft()

            # Append the same tokens in decoder input and output
            t5_decoder_in.append(flag)
            t5_decoder_out.append(flag)
        elif not prefix_lm:
            # Append visible part of input sequence.
            t5_decoder_in.extend(before_mask)
            t5_decoder_out.extend(before_mask)
        t5_decoder_in.extend(span.label)
        t5_decoder_out.extend(span.label)

        t5_input.extend(before_mask)
        if insert_mask_tokens:
            t5_input.append(flag)

        # the next start index is the token after the last span token
        start_index = span.index[-1] + 1

    # Add <eos> token to the t5_decoder_out
    t5_decoder_out.append(eos_id)

    # Add the remaining tokens to the t5 input
    t5_input.extend(tokens[start_index:])
    return t5_input, t5_decoder_in, t5_decoder_out


def pad_and_convert_to_numpy(tokens, masked_positions,
                             masked_labels, pad_id,
                             max_seq_length, max_seq_length_dec,
                             masked_spans=None, bos_id=None,
                             eos_id=None, sentinel_tokens=None,
                             prefix_lm=False):
    """Pad sequences and convert them to numpy."""

    t5_input, t5_decoder_in, t5_decoder_out = merge_subsequent_masks(
        tokens, masked_spans, bos_id, eos_id, sentinel_tokens, prefix_lm)

    # assert (len(t5_input) - len(masked_spans)) + \
    #        (len(t5_decoder_in) - (len(masked_spans) + 1)) == len(tokens)

    # Some checks.

    # Encoder-side padding mask.
    num_tokens = len(t5_input)
    padding_length = max_seq_length - num_tokens
    if padding_length < 0:
        raise LengthExceededError()
    assert len(masked_positions) == len(masked_labels)

    # Tokens..
    filler = [pad_id] * padding_length
    tokens_enc = np.array(t5_input + filler, dtype=np.int64)

    # Decoder-side padding mask.
    num_tokens_dec = len(t5_decoder_in)
    padding_length_dec = max_seq_length_dec - num_tokens_dec
    if padding_length_dec < 0:
        raise DecoderLengthExceededError()
    filler_dec = [pad_id] * padding_length_dec
    tokens_dec_in = np.array(t5_decoder_in + filler_dec, dtype=np.int64)

    # Create attention masks
    enc_mask = make_attention_mask(tokens_enc, tokens_enc)
    enc_dec_mask = make_attention_mask(tokens_dec_in, tokens_enc)
    dec_mask = make_attention_mask(tokens_dec_in, tokens_dec_in)
    dec_mask = dec_mask * make_history_mask(tokens_dec_in)

    # Labels mask.
    labels = t5_decoder_out + ([-1] * padding_length_dec)
    labels = np.array(labels, dtype=np.int64)

    # Loss mask
    loss_mask = ([1] * num_tokens_dec) + ([0] * padding_length_dec)
    loss_mask = np.array(loss_mask, dtype=np.int64)

    return tokens_enc, tokens_dec_in, labels, enc_mask, \
           dec_mask, enc_dec_mask, loss_mask


def make_attention_mask(source_block, target_block):
    """
    Returns a 2-dimensional (2-D) attention mask
    :param source_block: 1-D array
    :param target_block: 1-D array
    """
    mask = (target_block[None, :] >= 1) * (source_block[:, None] >= 1)
    mask = mask.astype(np.int64)
    # (source_length, target_length)
    return mask


def make_attention_mask_3d(source_block, target_block):
    """
    Returns a 3-dimensional (3-D) attention mask
    :param source_block: 1-D array
    :param target_block: 1-D array
    """
    mask = (target_block[:, None, :] >= 1) * (source_block[:, :, None] >= 1)
    # (batch, source_length, target_length)
    # mask = mask.astype(np.int64)
    return mask


def make_history_mask(block):
    length = block.shape[0]
    arange = np.arange(length)
    history_mask = (arange[None, ] <= arange[:, None])
    history_mask = history_mask.astype(np.int64)
    return history_mask


def make_history_mask_3d(block):
    batch, length = block.shape
    arange = torch.arange(length, device=block.device)
    history_mask = (arange[None, ] <= arange[:, None])[None, ]
    history_mask = history_mask.expand(batch, length, length)
    return history_mask


def get_samples(indexed_dataset, doc_idx, sample_idx, shuffle_idx, idx):
    # Get the shuffled index.
    idx = shuffle_idx[idx]
    # Start and end documents.
    if idx == 0:
        doc_index_f = 0
    else:
        doc_index_f = sample_idx[idx - 1] + 1
    doc_index_l = sample_idx[idx]
    # If we are within the same document, just extract the chunk.
    if doc_index_f == doc_index_l:
        sample = indexed_dataset.get(doc_idx[doc_index_f])
        sample_list = [sample]
    else:
        # Otherwise, get the rest of the initial document.
        sample_list = [indexed_dataset.get(doc_idx[doc_index_f])]
        # Loop over all in between documents and add the entire document.
        for i in range(doc_index_f + 1, doc_index_l):
            sample_list.append(indexed_dataset.get(doc_idx[i]))
        # And finally add the relevant portion of last document.
        sample_list.append(indexed_dataset.get(
            doc_idx[doc_index_l]))
    return sample_list


def create_samples_dict(max_seq_length, max_seq_length_dec):
    samples_dict = {
        'text_enc': np.empty((max_seq_length,), dtype=np.int64),
        'text_dec': np.empty(
            (max_seq_length_dec,), dtype=np.int64),
        'labels': np.empty(
            (max_seq_length_dec,), dtype=np.int64),
        'loss_mask': np.zeros(
            (max_seq_length_dec,), dtype=np.int64),
        'truncated': 0,
        'enc_mask': np.zeros(
            (max_seq_length, max_seq_length),
            dtype=np.int64,
        ),
        'dec_mask': np.zeros(
            (max_seq_length_dec, max_seq_length_dec),
            dtype=np.int64,
        ),
        'enc_dec_mask': np.zeros(
            (max_seq_length_dec, max_seq_length),
            dtype=np.int64,
        ),
    }
    return samples_dict


def _remove_padding(result_sample, pad_id):
    # Remove padding
    padding_start = np.argmax(result_sample['text_enc'] == pad_id)
    padding_start_dec = np.argmax(result_sample['text_dec'] == pad_id)
    if padding_start == 0 and padding_start_dec == 0:
        return
    elif padding_start == 0:
        padding_start = None
    elif padding_start_dec == 0:
        padding_start_dec = None

    result_sample['text_enc'] = result_sample['text_enc'][:padding_start]
    for key in ['text_dec', 'labels', 'loss_mask']:
        result_sample[key] = result_sample[key][:padding_start_dec]
    result_sample['enc_mask'] = \
        result_sample['enc_mask'][:padding_start, :padding_start]
    result_sample['enc_dec_mask'] = \
        result_sample['enc_dec_mask'][:padding_start_dec, :padding_start]
    result_sample['dec_mask'] = \
        result_sample['dec_mask'][:padding_start_dec, :padding_start_dec]


def get_lens(key, prev_len, prev_len_dec, len_enc, len_dec):
    assert key != 'enc_dec_mask'
    if key in ['text_enc', 'enc_mask']:
        offset = prev_len
        length = len_enc
    else:
        offset = prev_len_dec
        length = len_dec
    return offset, length


def update_samples_dict(
        samples_dict,
        result_sample,
        max_seq_len,
        max_seq_len_dec,
        prev_len,
        prev_len_dec,
        pad_id,
):
    _remove_padding(result_sample, pad_id)

    len_enc = len(result_sample['text_enc'])
    len_dec = len(result_sample['text_dec'])

    if (
            prev_len + len_enc > max_seq_len
            or prev_len_dec + len_dec > max_seq_len_dec
    ):
        return None

    for key in ['text_enc', 'text_dec', 'labels']:
        curr_sample = result_sample[key]
        offset, length = get_lens(
            key, prev_len, prev_len_dec, len_enc, len_dec)
        samples_dict[key][offset:offset + length] = curr_sample

    samples_dict['loss_mask'][
        prev_len_dec:prev_len_dec + len_dec,
    ] += result_sample['loss_mask']
    samples_dict['enc_mask'][
        prev_len:prev_len + len_enc,
        prev_len:prev_len + len_enc,
    ] += result_sample['enc_mask']
    samples_dict['dec_mask'][
        prev_len_dec:prev_len_dec + len_dec,
        prev_len_dec:prev_len_dec + len_dec,
    ] += result_sample['dec_mask']
    samples_dict['enc_dec_mask'][
        prev_len_dec:prev_len_dec + len_dec,
        prev_len:prev_len + len_enc,
    ] += result_sample['enc_dec_mask']

    samples_dict['truncated'] += result_sample['truncated']
    return len_enc, len_dec


def add_final_padding(samples_dict, prev_len, prev_len_dec, pad_id):
    samples_dict['text_enc'][prev_len:] = pad_id
    samples_dict['text_dec'][prev_len_dec:] = pad_id
    samples_dict['labels'][prev_len_dec:] = -1
