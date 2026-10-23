# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Plain-text encode/decode of SFTTokenizer with a tokenizer that prepends BOS."""

import json

import pytest
from tokenizers import Regex, Tokenizer, models, pre_tokenizers, processors
from transformers import AutoTokenizer

from megatron.core.tokenizers import MegatronTokenizer

_PIECES = [
    "<unk>",
    "<s>",
    "</s>",
    "<SPECIAL_10>",
    "<SPECIAL_11>",
    "<SPECIAL_12>",
    "System",
    "User",
    "Assistant",
    "\n",
    "abc",
    "hi",
    "yo",
]
_BOS_ID = 1
_ABC_ID = 10


@pytest.fixture
def sft_tokenizer(tmp_path):
    """Nemotron-style assets: tokenizer.json prepends <s> through a TemplateProcessing
    post-processor while tokenizer_config.json says add_bos_token=false."""
    backend = Tokenizer(
        models.WordLevel({piece: i for i, piece in enumerate(_PIECES)}, unk_token="<unk>")
    )
    backend.pre_tokenizer = pre_tokenizers.Split(
        Regex(r"<SPECIAL_\d+>|</?s>|\n| ?\w+|[^\w\s]"), "isolated"
    )
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A", pair="<s> $A $B", special_tokens=[("<s>", _BOS_ID)]
    )
    backend.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "add_bos_token": False,
                "add_eos_token": False,
                "clean_up_tokenization_spaces": False,
            }
        )
    )
    # The raw HF tokenizer does insert BOS on plain strings; that is the trigger.
    assert AutoTokenizer.from_pretrained(str(tmp_path)).encode("abc") == [_BOS_ID, _ABC_ID]
    return MegatronTokenizer.from_pretrained(
        tokenizer_path=str(tmp_path),
        metadata_path={"library": "sft"},
        prompt_format="nemotron-nano-v2",
    )


def test_plain_text_round_trip_does_not_insert_bos(sft_tokenizer):
    assert sft_tokenizer.tokenize("abc") == [_ABC_ID]
    assert sft_tokenizer.detokenize(sft_tokenizer.tokenize("abc")) == "abc"


def test_special_tokens_are_inserted_on_request(sft_tokenizer):
    assert sft_tokenizer._tokenizer.text_to_ids("abc", add_special_tokens=True) == [
        _BOS_ID,
        _ABC_ID,
    ]


@pytest.mark.parametrize("skip_special_tokens, expected", [(True, "abc"), (False, "<s> abc")])
def test_detokenize_honours_skip_special_tokens(sft_tokenizer, skip_special_tokens, expected):
    assert sft_tokenizer.detokenize(
        [_BOS_ID, _ABC_ID], skip_special_tokens=skip_special_tokens
    ) == (expected)


def test_conversation_tokenization_does_not_insert_bos(sft_tokenizer):
    conversation = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    ids = sft_tokenizer.tokenize_conversation(
        conversation, return_target=False, add_generation_prompt=False
    ).tolist()
    assert _BOS_ID not in ids
