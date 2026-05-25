# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import numpy as np
import torch

from megatron.training.datasets.fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig
from megatron.training.datasets.sft_dataset import IGNORE_INDEX, SFTDataset


class _TinyConversationTokenizer:
    eod = 99
    pad = 0

    def tokenize_conversation(self, conversation, return_target=True, add_generation_prompt=False):
        tokens = []
        targets = []
        for message in conversation:
            value = len(message["content"])
            tokens.append(value)
            targets.append(IGNORE_INDEX if message["role"] != "assistant" else value)
        tokens.append(self.eod)
        targets.append(self.eod)
        return torch.tensor(tokens, dtype=torch.int64), torch.tensor(targets, dtype=torch.int64)


def _build_sft_dataset(sequence_length=8, context_parallel_size=1):
    dataset = SFTDataset.__new__(SFTDataset)
    dataset.dataset = [
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "ok"},
            {"role": "system", "content": "s2"},
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "done"},
        ]
    ]
    dataset.indices = np.array([0], dtype=np.int64)
    dataset.num_samples = 1
    dataset.config = SimpleNamespace(
        tokenizer=_TinyConversationTokenizer(),
        sequence_length=sequence_length,
        context_parallel_size=context_parallel_size,
        reset_position_ids=False,
        create_attention_mask=False,
        reset_attention_mask=False,
    )
    return dataset


def test_sft_dataset_splits_conversations_and_packs_sample():
    dataset = _build_sft_dataset()

    conversations = dataset._split_conversations(dataset.dataset[0])
    sample = dataset[0]

    assert len(dataset) == 1
    assert len(conversations) == 2
    assert sample["tokens"].shape == (8,)
    assert sample["labels"].shape == (8,)
    assert sample["position_ids"].tolist()[:3] == [0, 1, 2]
    assert sample["loss_mask"][sample["labels"] == IGNORE_INDEX].sum().item() == 0
    assert sample["cu_seqlens"].dtype == torch.int32
    assert sample["max_seqlen"].item() > 0


def test_sft_dataset_context_parallel_padding_keeps_pack_aligned():
    sample = _build_sft_dataset(sequence_length=10, context_parallel_size=2)[0]

    assert sample["tokens"].shape == (10,)
    assert sample["cu_seqlens"][-1].item() == 10
    assert sample["loss_mask"][-1].item() == 0.0


class _TinyFimTokenizer:
    def __init__(self, text="abcdef"):
        self._tokenizer = self
        self.text = text
        self.vocab_size = 2048

    def ids_to_text(self, ids, remove_special_tokens=True):
        return self.text

    def text_to_ids(self, text):
        return [ord(char) % 50 + 1 for char in text]

    def tokens_to_ids(self, tokens):
        mapping = {
            "<fim-prefix>": 101,
            "<fim-middle>": 102,
            "<fim-suffix>": 103,
            "<fim-pad>": 0,
            "<eod>": 104,
            "<split>": 105,
        }
        if isinstance(tokens, list):
            return [mapping[token] for token in tokens]
        return mapping[tokens]


class _DeterministicFimRng:
    def __init__(self, binomial_results, randint_result=(1, 4)):
        self.binomial_results = list(binomial_results)
        self.randint_result = list(randint_result)

    def binomial(self, n, p):
        return self.binomial_results.pop(0)

    def randint(self, low, high=None, size=None):
        return np.array(self.randint_result)


def _build_fim_dataset(binomial_results=(1, 0), text="abcdef", split_token=None):
    dataset = GPTFIMDataset.__new__(GPTFIMDataset)
    dataset.np_rng = _DeterministicFimRng(binomial_results)
    dataset.fim_rate = 1.0
    dataset.fim_spm_rate = 0.0
    dataset.fragment_fim_rate = 1.0
    dataset.fim_split_sample = split_token
    dataset.no_fim_prefix = None
    dataset.suffix_tok_id = 103
    dataset.prefix_tok_id = 101
    dataset.middle_tok_id = 102
    dataset.pad_tok_id = 0
    dataset.eod_tok_id = 104
    dataset.config = SimpleNamespace(tokenizer=_TinyFimTokenizer(text=text))
    return dataset


def test_fim_config_and_permute_variants():
    config = GPTFIMDatasetConfig(
        random_seed=123,
        sequence_length=12,
        tokenizer=_TinyFimTokenizer(),
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        fim_rate=1.0,
        fim_spm_rate=0.0,
        fim_extra_tokens={
            "prefix": "<fim-prefix>",
            "middle": "<fim-middle>",
            "suffix": "<fim-suffix>",
            "pad": "<fim-pad>",
            "eod": "<eod>",
        },
    )
    dataset = _build_fim_dataset(binomial_results=(1, 0))
    sample = np.arange(12, dtype=np.int64)

    permuted = dataset._permute(
        sample,
        fim_rate=1.0,
        fim_spm_rate=0.0,
        tokenizer=dataset.config.tokenizer,
        suffix_tok_id=dataset.suffix_tok_id,
        prefix_tok_id=dataset.prefix_tok_id,
        middle_tok_id=dataset.middle_tok_id,
        pad_tok_id=dataset.pad_tok_id,
    )

    assert config.fim_rate == 1.0
    assert permuted.shape == sample.shape
    assert permuted[0] == dataset.prefix_tok_id
    assert dataset.middle_tok_id in permuted


def test_fim_permute_can_skip_by_rate_and_prefix():
    sample = np.arange(6, dtype=np.int64)

    skipped_by_rate = _build_fim_dataset(binomial_results=(0,))._permute(
        sample,
        fim_rate=0.0,
        fim_spm_rate=0.0,
        tokenizer=_TinyFimTokenizer(),
    )
    prefix_dataset = _build_fim_dataset(binomial_results=(1,), text="skip-this")
    prefix_dataset.no_fim_prefix = "skip"
    skipped_by_prefix = prefix_dataset._permute(
        sample,
        fim_rate=1.0,
        fim_spm_rate=0.0,
        tokenizer=prefix_dataset.config.tokenizer,
        no_fim_prefix="skip",
    )

    assert np.array_equal(skipped_by_rate, sample)
    assert np.array_equal(skipped_by_prefix, sample)


def test_fim_split_and_permute_sequence_handles_fragments():
    dataset = _build_fim_dataset(binomial_results=(1, 0, 0), split_token=9)
    sequence = np.array([1, 2, 9, 3, 4], dtype=np.int64)

    result = dataset._fim_split_and_permute_sequence(sequence)

    assert result.tolist() == sequence.tolist()
