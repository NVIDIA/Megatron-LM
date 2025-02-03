import base64
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import requests

from megatron.training import tokenizer
from megatron.training.tokenizer.gpt2_tokenization import PRETRAINED_VOCAB_ARCHIVE_MAP
from megatron.training.tokenizer.multimodal_tokenizer import MultimodalTokenizer

TOKENIZER_DIR = Path("~/data/tokenizers").expanduser()

# Copied over from test_preprocess_data.py
from tests.unit_tests.data.test_preprocess_data import __LOCAL_GPT2_VOCAB

GPT2_VOCAB_SIZE = 32768


def offsets_to_substrs(offsets, string):
    return [string[start:end] for start, end in zip([0] + offsets, offsets + [len(string)])]


def local_test_specs():
    return [
        Namespace(
            rank=0,
            tensor_model_parallel_size=8,
            make_vocab_size_divisible_by=128,
            tokenizer_type="GPTSentencePieceTokenizer",
            tokenizer_model=f"{TOKENIZER_DIR}/nemotron_2_256k.model",
        ),
        Namespace(
            rank=0,
            vocab_size=131072,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=8,
            tokenizer_type="TikTokenizer",
            tokenizer_model=f"{TOKENIZER_DIR}/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json",
            tiktoken_pattern="v2",
            tiktoken_num_special_tokens=1000,
            tiktoken_special_tokens=["<unk>", "<s>", "</s>"],
        ),
        Namespace(
            rank=0,
            vocab_size=131072,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=8,
            tokenizer_type="TikTokenizer",
            tokenizer_model=f"{TOKENIZER_DIR}/multiMixV5_fix_default_500000_128k.vocab.json",
            tiktoken_pattern="v1",
            tiktoken_num_special_tokens=1000,
            tiktoken_special_tokens=["<unk>", "<s>", "</s>"],
        ),
        Namespace(
            rank=0,
            vocab_size=128000,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=8,
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="meta-llama/Llama-2-7b-hf",
        ),
        Namespace(
            rank=0,
            vocab_size=128000,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=8,
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="meta-llama/Meta-Llama-3.1-8B",
        ),
    ]


@pytest.fixture(scope="session")
def gpt2_tiktok_vocab(tmp_path_factory):

    if Path(__LOCAL_GPT2_VOCAB).exists():
        with open(__LOCAL_GPT2_VOCAB, "r", encoding="utf-8") as reader:
            gpt2_vocab = json.load(reader)
    else:
        gpt2_vocab = json.loads(requests.get(PRETRAINED_VOCAB_ARCHIVE_MAP["gpt2"]).content)

    N = 256
    tiktok_vocab = [
        {"token_bytes": base64.b64encode(bytes([i])).decode("utf-8"), "token_str": str(i)}
        for i in range(N)
    ]
    tiktok_vocab_bytes = {x["token_bytes"] for x in tiktok_vocab}

    tiktok_vocab += [
        {"token_bytes": base64.b64encode(token.encode('utf-8')).decode("utf-8"), "token_str": token}
        for token in gpt2_vocab
        if base64.b64encode(token.encode('utf-8')).decode("utf-8") not in tiktok_vocab_bytes
    ]

    for i, entry in enumerate(tiktok_vocab):
        entry["rank"] = i

    for i, x in enumerate(tiktok_vocab):
        assert x.keys() == {"rank", "token_bytes", "token_str"}
        assert x["rank"] == i
        merge = base64.b64decode(x["token_bytes"])
        assert i >= 256 or merge == bytes([i]), f"{i} {merge} {bytes([i])}"

    file_name = tmp_path_factory.mktemp("data") / "gpt2_vocab.json"
    with open(file_name, "w") as f:
        json.dump(tiktok_vocab, f)

    return Namespace(
        rank=0,
        vocab_size=32768,
        make_vocab_size_divisible_by=128,
        tensor_model_parallel_size=8,
        tokenizer_type="TikTokenizer",
        tokenizer_model=str(file_name),
        tiktoken_pattern="v1",
        tiktoken_num_special_tokens=1000,
        tiktoken_special_tokens=["<unk>", "<s>", "</s>"],
    )


@pytest.mark.parametrize("args", local_test_specs())
def test_tokenizer(args):
    if not TOKENIZER_DIR.exists():
        pytest.skip("Skipping tokenizer tests because the tokenizer directory does not exist")

    tok = tokenizer.build_tokenizer(args)
    run_tokenizer_tests(tok)


def test_gpt2_tiktok_tokenizer(gpt2_tiktok_vocab):
    tok = tokenizer.build_tokenizer(gpt2_tiktok_vocab)
    run_tokenizer_tests(tok)


def run_tokenizer_tests(tok):
    string1 = (
        "The following are multiple choice questions (with answers) about college biology.\n"
        "Monoclonal antisera are distinguished from polyclonal antisera in which of the "
        "following ways?\n"
        "A. Each type of antibody in a monoclonal antiserum reacts against a single region of "
        "a single antigen; each type of antibody in a polyclonal antiserum reacts against "
        "multiple regions of different antigens.\n"
        "B. A monoclonal antibody reacts against multiple regions of a single antigen; a "
        "polyclonal antibody reacts against a single region of related antigens.\n"
        "C. A monoclonal antiserum contains antibodies secreted from the descendants of a "
        "single B lymphocyte; a polyclonal antiserum contains antibodies secreted from the "
        "descendants of different B lymphocytes.\n"
        "D. A monoclonal antiserum contains antibodies secreted from the descendants of a "
        "single B lymphocyte; a polyclonal antiserum contains antibodies secreted from the "
        "descendants of both B and T lymphocytes.\n"
        "Answer: C"
    )
    string2 = "Жизнь прекрасна и удивительна"
    string3 = "お誕生日おめでとう"
    strings = [string1, string2, string3]

    for test_string in strings:
        toks = tok.tokenize(test_string)
        offsets = tok.offsets(toks, test_string)
        dec = offsets_to_substrs(offsets, test_string)
        detok_str = ''.join(dec)
        # the following is not necessarily true by construction above,
        # since the many tokenizers may operate at the byte level and not
        # only at the character level.
        assert (
            detok_str == test_string
        ), f"Detokenized string {detok_str} does not match original {test_string}"
        assert len(toks) == len(
            offsets
        ), f"Tokenized string {toks} does not match original {offsets}"


def test_null_tokenizer():
    args = Namespace(
        tokenizer_type="NullTokenizer",
        rank=0,
        vocab_size=128000,
        make_vocab_size_divisible_by=128,
        tensor_model_parallel_size=8,
    )
    tok = tokenizer.build_tokenizer(args)
    test_string = "1 23 456 789"
    toks = tok.tokenize(test_string)
    offsets = tok.offsets(toks, test_string)
    dec = offsets_to_substrs(offsets, test_string)
    detok_str = ''.join(dec)

    assert (
        detok_str == test_string
    ), f"Detokenized string {detok_str} does not match original {test_string}"
    assert len(toks) == len(offsets), f"Tokenized string {toks} does not match original {offsets}"


class MockUnderlyingTokenizer:
    """Mock tokenizer for testing purposes."""

    def __init__(self):
        self.pad_token_id = 256

    def __len__(self):
        return 256

    def encode(self, text: str) -> list[int]:
        """Convert text to a list of token IDs."""
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        """Convert list of token IDs to plaintext."""
        return "".join([chr(t) for t in tokens])

    def apply_chat_template(self, conversation: list[dict], *args, **kwargs) -> list[int]:
        """Convert a conversation to token IDs."""
        out = []
        for turn in conversation:
            turn_tokens = self.encode(f"{turn['role']}:{turn['content']}")
            out.extend(turn_tokens)

        if kwargs.get("return_tensors", None) == "np":
            return [np.array(out)]

        return out

    def convert_tokens_to_ids(self, text: str) -> list[int]:
        """Convert plaintext to token IDs."""
        return self.encode(text)

    def add_tokens(self, extra_tokens: list[str], *args, **kwargs) -> int:
        """Add tokens to the tokenizer. No-op for this mock tokenizer."""
        return len(extra_tokens)


def test_multimodal_tokenizer():
    """Test MultimodalTokenizer."""
    underlying = MockUnderlyingTokenizer()
    prompt_format = "chatml"
    special_tokens = ["<image>"]
    image_tag_type = ""
    tokenizer = MultimodalTokenizer(underlying, prompt_format, special_tokens, image_tag_type)

    # Simple encode - decode roundtrip.
    assert (
        tokenizer.detokenize(tokenizer.tokenize("abc")) == "abc"
    ), "encode-decode roundtrip failed"

    # Apply chat template.
    conversation = [
        {"role": "system", "content": "abc"},
        {"role": "user", "content": "123<image>"},
        {"role": "assistant", "content": "xyz"},
    ]
    conv_tokens = tokenizer.tokenize_conversation(
        conversation, return_target=False, add_generation_prompt=False
    )
    assert len(conv_tokens) > 0, "failed to tokenize conversation"

    conv_tokens, target_tokens = tokenizer.tokenize_conversation(
        conversation, return_target=True, add_generation_prompt=True
    )
    assert len(conv_tokens) > 0 and len(conv_tokens) == len(
        target_tokens
    ), "failed to tokenize conversation and return target tokens"

    # Try converting tokens to ids.
    assert tokenizer.convert_tokens_to_ids("a"), "failed to convert tokens to ids."

    # Try image tags.
    image_tag_type = "nvlm"
    tokenizer = MultimodalTokenizer(underlying, prompt_format, special_tokens, image_tag_type)

    assert tokenizer._apply_image_tag("<image>hello") == "<Image><image></Image>hello"
    assert tokenizer._apply_image_tag([{"role": "user", "content": "<image>hello"}]) == [
        {"role": "user", "content": "<Image><image></Image>hello"}
    ]
