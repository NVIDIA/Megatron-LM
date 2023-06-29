# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""UL2-style dataset."""

from collections import ChainMap
import math

import numpy as np
import torch

from megatron import get_tokenizer
from megatron.data.dataset_utils import (
    create_masked_lm_predictions,
    get_samples_mapping,
    SamplingStyle
)
from megatron.data.gpt_dataset import build_index_mappings_full_docs
from megatron.data.t5_dataset import (
    add_final_padding,
    create_samples_dict as t5_create_samples_dict,
    get_samples,
    LengthExceededError,
    make_history_mask,
    merge_subsequent_masks,
    pad_and_convert_to_numpy,
    update_samples_dict,
)
from megatron.model.enums import UL2ModelType


def is_decoder_only(ul2_model_type):
    """Return whether we use a decoder-only model."""
    assert isinstance(ul2_model_type, UL2ModelType)
    return ul2_model_type is not UL2ModelType.encoder_decoder


def is_prefix_lm(ul2_model_type):
    """Return whether we use a non-causal decoder-only model."""
    assert isinstance(ul2_model_type, UL2ModelType)
    return ul2_model_type is UL2ModelType.non_causal_decoder


class UL2Dataset(torch.utils.data.Dataset):
    def __init__(self, name, indexed_dataset, data_prefix,
                 splits_string, num_epochs, max_num_samples, model_type,
                 denoiser_ratios, denoisers, mean_span_lengths,
                 mask_ratios, add_mask_tokens, pack_samples,
                 denoiser_tokens, scale_normal_std, like_ul2r,
                 pack_any, pack_repeat_prompt,
                 max_seq_length, max_seq_length_dec, short_seq_prob, seed,
                 *,
                 data_cache_path=None):
        super().__init__()

        if denoiser_ratios is None:
            # Uniform distribution by default.
            denoiser_ratios = [1 / len(denoisers)] * len(denoisers)

        assert (
            len(denoiser_ratios) == len(denoisers)
            == len(mean_span_lengths) == len(mask_ratios)
        ), (
            'some UL2 configurations do not correspond to the amount of '
            'denoising objectives'
        )

        # Params to store.
        self.name = name
        self.seed = seed
        self.max_seq_length = max_seq_length
        self.max_seq_length_dec = max_seq_length_dec

        self.model_type = model_type
        self.denoiser_ratios = [
            denoiser_ratio / sum(denoiser_ratios)
            for denoiser_ratio in denoiser_ratios
        ]
        self.denoisers = [denoiser.upper() for denoiser in denoisers]
        self.mean_span_lengths = mean_span_lengths
        self.mask_ratios = mask_ratios
        self.scale_normal_std = scale_normal_std
        self.like_ul2r = like_ul2r

        # Dataset.
        self.indexed_dataset = indexed_dataset
        self.pack_samples = pack_samples
        self.pack_any = pack_any
        self.repeat_prompt = pack_repeat_prompt

        # Minimum number of tokens added: BOS and EOS.
        min_added_tokens = 2
        if is_decoder_only(model_type):
            # Here we also add a SEP token.
            min_added_tokens += 1

        # Build the samples mapping.
        if self.pack_samples:
            (
                self.doc_idx, self.sample_idx, self.shuffle_idx,
                self.desc, self.desc_hash,
            ) = build_index_mappings_full_docs(
                self.name, data_prefix,
                self.indexed_dataset.get_doc_idx()[:-1],
                self.indexed_dataset.sizes, splits_string, max_num_samples,
                self.max_seq_length - min_added_tokens, self.seed,
                data_cache_path=data_cache_path)
        else:
            self.samples_mapping = get_samples_mapping(
                self.indexed_dataset,
                data_prefix,
                splits_string,
                num_epochs,
                max_num_samples,
                # account for added tokens
                self.max_seq_length - min_added_tokens,
                short_seq_prob,
                self.seed,
                self.name,
                False,
            )

        # Vocab stuff.
        tokenizer = get_tokenizer()
        # Some tokenizers split their vocabularies. Here we handle both
        # cases.
        if (
                hasattr(tokenizer, 'tokenizer')
                and hasattr(tokenizer.tokenizer, 'special_tokens_decoder')
        ):
            inv_vocab = ChainMap(
                tokenizer.inv_vocab,
                tokenizer.tokenizer.special_tokens_decoder,
            )
            vocab = ChainMap(
                tokenizer.vocab, tokenizer.tokenizer.special_tokens)
        else:
            inv_vocab = tokenizer.inv_vocab
            vocab = tokenizer.vocab
        self.vocab_id_list = list(inv_vocab.keys())
        self.vocab_id_to_token_dict = inv_vocab
        # Replace empty string tokens with `None` – we want to ignore
        # those.
        self.cls_ids = {
            denoiser: vocab[token] if token else None
            for (denoiser, token) in denoiser_tokens.items()
        }
        # cls_token = self.vocab_id_to_token_dict[tokenizer.cls]
        # if cls_token not in self.cls_ids:
        #     self.cls_ids[cls_token] = tokenizer.cls
        self.sep_id = tokenizer.sep
        self.mask_id = tokenizer.mask
        self.pad_id = tokenizer.pad
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id

        if add_mask_tokens:
            # Filter out denoiser tokens.
            self.sentinel_tokens = [
                token
                for token in tokenizer.additional_special_tokens_ids
                if token not in self.cls_ids.values()
            ]
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
        # Denoiser selection
        denoiser_index = np_rng.choice(
            np.arange(len(self.denoisers)),
            p=self.denoiser_ratios,
        )

        if self.pack_samples:
            samples_dict = self._pack_samples(np_rng, idx, denoiser_index)
        else:
            start_index, end_index, seq_length = self.samples_mapping[idx]
            sample = []
            for index in range(start_index, end_index):
                sample.append(self.indexed_dataset[index])
            samples_dict = build_training_sample(
                sample, seq_length,
                self.max_seq_length,  # needed for padding
                self.max_seq_length_dec, self.vocab_id_list,
                self.vocab_id_to_token_dict, self.cls_ids, self.sep_id,
                self.mask_id, self.pad_id, self.model_type, denoiser_index,
                self.denoisers, self.mean_span_lengths,
                self.mask_ratios, self.scale_normal_std, self.like_ul2r,
                np_rng, self.bos_id, self.eos_id, self.sentinel_tokens)
        return samples_dict

    def _pack_samples(self, np_rng, idx, denoiser_index):
        samples = get_samples(self.indexed_dataset, self.doc_idx,
                              self.sample_idx, self.shuffle_idx, idx)
        samples_dict = create_samples_dict(
            self.max_seq_length, self.max_seq_length, self.model_type)
        prev_len = 0
        prev_len_dec = 0
        cls_ids = self.cls_ids

        for sample in samples:
            remaining_seq_len = self.max_seq_length - prev_len
            seq_length = min(remaining_seq_len, len(sample))

            result_sample = build_training_sample(
                [sample], seq_length,
                self.max_seq_length,  # needed for padding
                self.max_seq_length_dec, self.vocab_id_list,
                self.vocab_id_to_token_dict, cls_ids, self.sep_id,
                self.mask_id, self.pad_id, self.model_type, denoiser_index,
                self.denoisers, self.mean_span_lengths,
                self.mask_ratios, self.scale_normal_std, self.like_ul2r,
                np_rng, self.bos_id, self.eos_id, self.sentinel_tokens)
            if is_decoder_only(self.model_type):
                maybe_lens = update_samples_dict_decoder_only(
                    samples_dict,
                    result_sample,
                    self.max_seq_length,
                    prev_len,
                    self.pad_id,
                )
            else:
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

            if is_decoder_only(self.model_type):
                len_enc = maybe_lens
            else:
                len_enc, len_dec = maybe_lens
                prev_len_dec += len_dec
            prev_len += len_enc

            if not self.repeat_prompt and not self.pack_any:
                cls_ids = {self.denoisers[denoiser_index]: None}

            if self.pack_any:
                denoiser_index = np_rng.choice(
                    np.arange(len(self.denoisers)),
                    p=self.denoiser_ratios,
                )

        if is_decoder_only(self.model_type):
            samples_dict['text'][prev_len:] = self.pad_id
            samples_dict['labels'][prev_len:] = -1
        else:
            add_final_padding(
                samples_dict, prev_len, prev_len_dec, self.pad_id)
        return samples_dict


def build_training_sample(sample, target_seq_length,
                          max_seq_length, max_seq_length_dec,
                          vocab_id_list, vocab_id_to_token_dict,
                          cls_ids, sep_id, mask_id, pad_id,
                          model_type, denoiser_index, denoisers,
                          mean_span_lengths, mask_ratios,
                          scale_normal_std, like_ul2r,
                          np_rng, bos_id=None,
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
        cls_ids: Start of example ids.
        sep_id: Separator id.
        mask_id: Mask token id.
        pad_id: Padding token id.
        model_type: What type of model is used.
        denoiser_index: Index of selected denoising objective.
        denoisers: What type of UL2 denoising objective the other UL2
              configurations refer to.
        mean_span_lengths: Mean length for sampling span lengths. Numbers < 1
              indicate a mean length of the sequence length times that number.
        mask_ratios: Ratio of masked token in the full sequence.
        scale_normal_std: Whether to scale the standard deviation when using a
            normal distribution for span length sampling.
        like_ul2r: Whether to use the updated implementation as specified in
            the UL2R paper.
        np_rng: Random number genenrator. Note that this rng state should be
              numpy and not python since python randint is inclusive for
              the opper bound whereas the numpy one is exclusive.
        bos_id: start of decoder example id
        eos_id: end of generation id
        sentinel_tokens: unique value to be substituted for every replaced span
    """
    add_mask_tokens = sentinel_tokens is not None

    # Denoiser selection
    denoiser = denoisers[denoiser_index]
    masked_lm_prob = mask_ratios[denoiser_index]

    assert target_seq_length <= max_seq_length

    # flatten sentences into one list
    tokens = [token for sentence in sample for token in sentence]

    # Prepend objective token.
    cls_id = cls_ids.get(denoiser, False)
    if cls_id is False:
        raise ValueError('unknown denoiser')

    # If objective token is `None`, ignore it.
    if cls_id is not None:
        tokens = [cls_id] + tokens

    max_num_tokens = target_seq_length
    if (
            is_decoder_only(model_type)
            and denoiser != 'S'
            and add_mask_tokens
    ):
        # Keep space for repeated `extra_id` tokens; not the most data
        # efficient since we calculate this based on the maximum number
        # of possible `extra_id` tokens.
        safe_max_seq_len = math.floor(max_num_tokens / (1 + masked_lm_prob))
        truncated = len(tokens) > safe_max_seq_len
        tokens = tokens[:safe_max_seq_len]
    else:
        # If we are S-denoising, we know three tokens are going to be
        # added: `bos`, `sep`, and `eos`. Same when not adding mask
        # tokens.
        if (
                is_decoder_only(model_type) and denoiser == 'S'
                or not add_mask_tokens
        ):
            max_num_tokens -= 3

        # If we have a decoder-only model and do not add mask tokens, we
        # basically duplicate the sequence. So cut the maximum length in
        # half.
        if (
                is_decoder_only(model_type)
                and denoiser != 'S'
                and not add_mask_tokens
        ):
            max_num_tokens = max_num_tokens // 2

        # Truncate to `target_sequence_length`.
        truncated = len(tokens) > max_num_tokens
        tokens = tokens[:max_num_tokens]

    # Masking.
    mean_ngrams = mean_span_lengths[denoiser_index]
    if mean_ngrams < 1:
        # Ensure we always obtain at least one `max_ngrams`.
        mean_ngrams = max(1, round(len(tokens) * mean_ngrams))
    max_ngrams = mean_ngrams * 2 - 1

    if denoiser == 'R' or denoiser == 'X':
        if like_ul2r:
            sampling_style = SamplingStyle.UNIFORM
        elif scale_normal_std:
            sampling_style = SamplingStyle.NORMAL
        else:
            sampling_style = SamplingStyle.UNSCALED_NORMAL
        prefix_lm = False
        max_predictions_per_seq = len(tokens) - 1
    elif denoiser == 'S':
        sampling_style = SamplingStyle.UNIFORM
        prefix_lm = True
        max_predictions_per_seq = min(
            round(masked_lm_prob * len(tokens)) * 2 - 1,
            len(tokens) - 1,
        )
    else:
        raise ValueError('unknown denoiser')

    # Ensure we always have at least one prediction.
    max_predictions_per_seq = max(1, max_predictions_per_seq)
    (
        tokens, masked_positions, masked_labels, _, masked_spans,
    ) = create_masked_lm_predictions(
        tokens, vocab_id_list, vocab_id_to_token_dict, masked_lm_prob,
        cls_id, sep_id, mask_id, max_predictions_per_seq, np_rng,
        max_ngrams=max_ngrams, masking_style="t5",
        sampling_style=sampling_style, prefix_lm=prefix_lm,
    )

    if is_decoder_only(model_type):
        # Concatenate to one sequence.
        tokens_enc, tokens_dec_in, labels = merge_subsequent_masks(
            tokens, masked_spans, bos_id, eos_id, sentinel_tokens, prefix_lm)

        # Move EOS tokens to end of sequence.
        while tokens_enc and tokens_enc[-1] == eos_id:
            del tokens_enc[-1]
            tokens_dec_in.append(eos_id)
            labels.append(eos_id)

        # Move BOS token to start of sequence.
        tokens_dec_in = tokens_dec_in[1:]
        if not add_mask_tokens:
            # Do not reproduce objective token when not using masking
            # tokens.
            tokens_dec_in = tokens_dec_in[1:]
            labels = labels[1:]

        num_labels = len(labels)

        # Do not add separator token if S-denoising.
        separator = [sep_id] if denoiser != 'S' else []
        tokens = (
            [bos_id]
            + tokens_enc
            + separator
            + tokens_dec_in
        )

        # Pad and convert to NumPy.
        padding_length = max_seq_length - len(tokens)
        if padding_length < 0:
            raise LengthExceededError()
        filler = [pad_id] * padding_length

        tokens = np.array(tokens + filler, dtype=np.int64)
        labels = np.array((
            tokens_enc
            + separator
            + labels
            + filler
        ), dtype=np.int64)

        loss_mask = np.zeros(len(tokens), dtype=np.int64)
        labels_start_neg_index = -(num_labels + padding_length)
        labels_end_neg_index = -padding_length if padding_length > 0 else None
        loss_mask[labels_start_neg_index:labels_end_neg_index] = 1

        dec_mask = make_history_mask(tokens)
        if is_prefix_lm(model_type):
            dec_mask[:labels_start_neg_index, :labels_start_neg_index] = 1

        train_sample = {
            'text': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'truncated': int(truncated),
            'dec_mask': dec_mask,
        }
    else:
        # Padding.
        (
            tokens_enc, tokens_dec_in, labels, enc_mask,
            dec_mask, enc_dec_mask, loss_mask,
        ) = pad_and_convert_to_numpy(tokens, masked_positions,
                                     masked_labels, pad_id, max_seq_length,
                                     max_seq_length_dec, masked_spans,
                                     bos_id, eos_id, sentinel_tokens,
                                     prefix_lm)

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


def create_samples_dict(max_seq_length, max_seq_length_dec, model_type):
    if is_decoder_only(model_type):
        samples_dict = {
            'text': np.empty((max_seq_length,), dtype=np.int64),
            'labels': np.empty((max_seq_length,), dtype=np.int64),
            'loss_mask': np.zeros((max_seq_length,), dtype=np.int64),
            'truncated': 0,
            'dec_mask': np.zeros(
                (max_seq_length, max_seq_length),
                dtype=np.int64,
            ),
        }
    else:
        samples_dict = t5_create_samples_dict(
            max_seq_length, max_seq_length_dec)
    return samples_dict


def _remove_padding(result_sample, pad_id):
    # Remove padding
    padding_start = np.argmax(result_sample['text'] == pad_id)
    if padding_start == 0:
        return
    result_sample['text'] = result_sample['text'][:padding_start]
    for key in ['labels', 'loss_mask']:
        result_sample[key] = result_sample[key][:padding_start]
    result_sample['dec_mask'] = \
        result_sample['dec_mask'][:padding_start, :padding_start]


def update_samples_dict_decoder_only(
        samples_dict,
        result_sample,
        max_seq_len,
        prev_len,
        pad_id,
):
    _remove_padding(result_sample, pad_id)
    len_enc = len(result_sample['text'])

    if prev_len + len_enc > max_seq_len:
        return None

    for key in ['text', 'labels']:
        curr_sample = result_sample[key]
        samples_dict[key][prev_len:prev_len + len_enc] = curr_sample

    samples_dict['loss_mask'][
        prev_len:prev_len + len_enc,
    ] += result_sample['loss_mask']
    samples_dict['dec_mask'][
        prev_len:prev_len + len_enc,
        prev_len:prev_len + len_enc,
    ] += result_sample['dec_mask']

    samples_dict['truncated'] += result_sample['truncated']
    return len_enc
