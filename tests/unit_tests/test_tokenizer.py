import base64
import json
from argparse import Namespace
from pathlib import Path

import pytest
import requests

from megatron.training import tokenizer
from megatron.training.tokenizer.gpt2_tokenization import PRETRAINED_VOCAB_ARCHIVE_MAP

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
