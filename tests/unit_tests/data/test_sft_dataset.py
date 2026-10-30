# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the loss mask of :class:`SFTDataset` and :class:`MockSFTDataset`.

The datasets are built without ``MegatronDataset.__init__`` and with a stub tokenizer, so the
tests run on CPU without downloads or ``torch.distributed``.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.core.tokenizers import MegatronTokenizer
from megatron.training.datasets.sft_dataset import IGNORE_INDEX, MockSFTDataset, SFTDataset

EOD = 1
DISTINCT_PAD = 0
ROLE_IDS = {"system": 2, "user": 3, "assistant": 4}

# Two conversations packed into one sample. The stub tokenizer turns them into
# [2, 5, EOD, 3, 6, EOD, 4, 7, 8, EOD] (10 tokens) and
# [2, 5, EOD, 3, 9, EOD, 4, 10, EOD] (9 tokens).
CONVERSATIONS = [
    {"role": "system", "content": [5]},
    {"role": "user", "content": [6]},
    {"role": "assistant", "content": [7, 8]},
    {"role": "system", "content": [5]},
    {"role": "user", "content": [9]},
    {"role": "assistant", "content": [10]},
]


class _StubTokenizer:
    """The tokenizer interface that SFTDataset reads.

    Every message becomes ``[role id, *content, EOD]``, like a chat template that ends each turn
    with EOS. With ``mask_prompt``, system and user turns are IGNORE_INDEX in the targets, as in
    the nemotron prompt formats. Without it, targets equal tokens, as in the ``default`` format.
    """

    def __init__(self, pad, mask_prompt=False):
        self.eod = EOD
        self.pad = pad
        self.mask_prompt = mask_prompt

    def tokenize_conversation(self, conversation, return_target, add_generation_prompt):
        tokens, targets = [], []
        for message in conversation:
            ids = [ROLE_IDS[message["role"]], *message["content"], EOD]
            tokens.extend(ids)
            keep = message["role"] == "assistant" or not self.mask_prompt
            targets.extend(ids if keep else [IGNORE_INDEX] * len(ids))
        return np.array(tokens), np.array(targets)


def _make_dataset(cls, item, tokenizer, seq_length, context_parallel_size=1):
    """Build ``cls`` around a single item without running ``MegatronDataset.__init__``."""
    dataset = cls.__new__(cls)
    dataset.config = SimpleNamespace(
        tokenizer=tokenizer,
        sequence_length=seq_length,
        reset_position_ids=False,
        create_attention_mask=False,
        reset_attention_mask=False,
        hybrid_context_parallel=False,
        context_parallel_size=context_parallel_size,
        data_parallel_size=1,
        sequence_parallel_size=0,
    )
    dataset.dataset = [item]
    dataset.indices = np.arange(1)
    dataset.padding_divisor = dataset._calculate_padding_divisor()
    return dataset


@pytest.mark.parametrize("pad", [EOD, DISTINCT_PAD], ids=["pad_is_eod", "distinct_pad"])
def test_sft_dataset_masks_padding_by_position(pad):
    tokenizer = _StubTokenizer(pad)
    sample = _make_dataset(
        SFTDataset, CONVERSATIONS, tokenizer, seq_length=32, context_parallel_size=2
    )[0]
    labels, loss_mask = sample["labels"], sample["loss_mask"]

    # With cp=2 each conversation is padded to a multiple of 4 (10 -> 12, 9 -> 12) and the pack
    # is padded to seq_length + 1. Label i is target i + 1, so labels 9, 10 and 20 to 31 are
    # padding.
    is_pad = torch.zeros(32, dtype=torch.bool)
    is_pad[[9, 10, *range(20, 32)]] = True
    assert torch.equal(loss_mask, (~is_pad).float())
    # All six end-of-turn EOD labels are trained, also when pad == EOD.
    eod_labels = (labels == EOD) & ~is_pad
    assert eod_labels.sum() == 6
    assert torch.all(loss_mask[eod_labels] == 1.0)
    # Padded labels keep the pad id, which is a valid token id for the loss computation.
    assert torch.all(labels[is_pad] == pad)


def test_sft_dataset_keeps_assistant_eod_with_prompt_masking():
    tokenizer = _StubTokenizer(pad=EOD, mask_prompt=True)
    sample = _make_dataset(SFTDataset, CONVERSATIONS[:3], tokenizer, seq_length=16)[0]

    # Labels are the targets shifted by one: 5 IGNORE_INDEX from the system and user turns, the
    # assistant turn [4, 7, 8, EOD], then 7 padding labels.
    assert sample["labels"].tolist() == [IGNORE_INDEX] * 5 + [4, 7, 8, EOD] + [EOD] * 7
    assert sample["loss_mask"].tolist() == [0.0] * 5 + [1.0] * 4 + [0.0] * 7


def test_sft_dataset_truncation_masks_only_padding():
    tokenizer = _StubTokenizer(pad=EOD)
    sample = _make_dataset(
        SFTDataset, CONVERSATIONS, tokenizer, seq_length=16, context_parallel_size=2
    )[0]

    # With cp=2 the first conversation (10 tokens) is padded to 12 and kept whole. The second one
    # does not fit: the pack is cut to 16 targets and one padding target is appended.
    labels = sample["labels"].tolist()
    assert labels == [5, EOD, 3, 6, EOD, 4, 7, 8, EOD] + [EOD] * 2 + [2, 5, EOD, 3] + [EOD]
    assert sample["loss_mask"].tolist() == [1.0] * 9 + [0.0] * 2 + [1.0] * 4 + [0.0]


@pytest.mark.parametrize("pad", [EOD, DISTINCT_PAD], ids=["pad_is_eod", "distinct_pad"])
def test_mock_sft_dataset_masks_padding_by_position(pad):
    tokenizer = _StubTokenizer(pad)
    sample = _make_dataset(
        MockSFTDataset, np.array([5, 6]), tokenizer, seq_length=8, context_parallel_size=2
    )[0]

    # [5, 6, EOD] is padded to a multiple of 4 and then to seq_length + 1.
    assert sample["labels"].tolist() == [6, EOD] + [pad] * 6
    assert sample["loss_mask"].tolist() == [1.0, 1.0] + [0.0] * 6


def test_mock_sft_dataset_truncation_keeps_appended_eod():
    tokenizer = _StubTokenizer(pad=EOD)
    sample = _make_dataset(MockSFTDataset, np.arange(5, 20), tokenizer, seq_length=8)[0]

    # 15 ids plus EOD do not fit: 7 ids are kept, then EOD and one padding target are appended.
    assert sample["labels"].tolist() == [6, 7, 8, 9, 10, 11, EOD, EOD]
    assert sample["loss_mask"].tolist() == [1.0] * 7 + [0.0]


def test_default_prompt_format_without_pad_token_keeps_eos_labels(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")

    vocab = {"<unk>": 0, "<|eot|>": 1, "<|system|>": 2, "<|user|>": 3, "<|assistant|>": 4}
    vocab.update({"be": 5, "brief": 6, "hello": 7, "hi": 8})
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    # Like the Llama 3 and Mistral tokenizers: EOS ends each turn and no pad token is set.
    hf_tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<|eot|>"
    )
    hf_tokenizer.chat_template = (
        "{% for m in messages %}"
        "{{ '<|' + m['role'] + '|> ' + m['content'] + ' <|eot|> ' }}"
        "{% endfor %}"
    )
    hf_tokenizer.save_pretrained(tmp_path)
    tokenizer = MegatronTokenizer.from_pretrained(
        tokenizer_path=str(tmp_path), metadata_path={"library": "sft"}, prompt_format="default"
    )
    # The default prompt format falls back to EOS as the pad token.
    assert tokenizer.pad == tokenizer.eod == 1

    conversation = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    sample = _make_dataset(SFTDataset, conversation, tokenizer, seq_length=16)[0]

    # 10 tokens with every turn closed by <|eot|>, then 7 padding targets.
    assert sample["labels"].tolist() == [5, 6, 1, 3, 7, 1, 4, 8, 1] + [1] * 7
    assert sample["loss_mask"].tolist() == [1.0] * 9 + [0.0] * 7
