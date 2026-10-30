# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import json
import os
import runpy
import sys
import tempfile

import nltk
import pytest
import requests

from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.tokenizers.text.libraries.megatron_hf_tokenizer import MEGATRON_CONFIG_MAP
from tools.merge_datasets import main as merge_main
from tools.preprocess_data import Encoder, PunktTokenizer
from tools.preprocess_data import get_args as build_args
from tools.preprocess_data import main as build_main

__HUGGINGFACE_BERT_BASE_UNCASED_VOCAB = (
    "https://huggingface.co/bert-base-uncased/raw/main/vocab.txt"
)

__LOCAL_BERT_VOCAB = "/home/gitlab-runner/data/bert_data/vocab.txt"

__LOCAL_GPT2_MERGE = "/home/gitlab-runner/data/gpt3_data/gpt2-merges.txt"

__LOCAL_GPT2_VOCAB = "/home/gitlab-runner/data/gpt3_data/gpt2-vocab.json"


def dummy_jsonl(odir):
    # numbers
    list_numbers = [json.dumps({"text": str(i + 1)}) + "\n" for i in range(100)]
    with open(os.path.join(odir, "numbers.jsonl"), "w") as writer:
        writer.writelines(list_numbers)
    # numbers ascending
    list_numbers_ascending = [
        json.dumps({"text": " ".join([str(j + 1) for j in range(i + 1)])}) + "\n"
        for i in range(100)
    ]
    with open(os.path.join(odir, "numbers_ascending.jsonl"), "w") as writer:
        writer.writelines(list_numbers_ascending)
    # test
    list_test = []
    with open(__file__) as reader:
        for line in reader:
            list_test.append(json.dumps({"text": line}) + "\n")
    with open(os.path.join(odir, "test.jsonl"), "w") as writer:
        writer.writelines(list_test)


def build_datasets(idir, odir, extra_args=[]):
    for name in os.listdir(idir):
        sys.argv = [
            sys.argv[0],
            "--input",
            os.path.join(idir, name),
            "--output-prefix",
            os.path.join(odir, os.path.splitext(name)[0]),
        ] + extra_args
        build_main()


def merge_datasets(idir):
    sys.argv = [sys.argv[0], "--input", idir, "--output-prefix", os.path.join(idir, "merge")]
    merge_main()


def do_test_preprocess_data(temp_dir, extra_args=[]):
    # set the default nltk data path
    os.environ["NLTK_DATA"] = os.path.join(temp_dir, "nltk_data")
    nltk.data.path.append(os.environ["NLTK_DATA"])

    path_to_raws = os.path.join(temp_dir, "sample_raws")
    path_to_data = os.path.join(temp_dir, "sample_data")
    os.mkdir(path_to_raws)
    os.mkdir(path_to_data)

    # create the dummy resources
    dummy_jsonl(path_to_raws)

    # build the datasets
    build_datasets(path_to_raws, path_to_data, extra_args=extra_args)

    # merge the datasets
    merge_datasets(path_to_data)

    sys.argv = [sys.argv[0], "--input", None, "--output-prefix", None] + extra_args
    encoder = Encoder(build_args())
    encoder.initializer()

    def tokens_to_string(toks):
        for option in ["decode", "detokenize"]:
            try:
                return getattr(encoder.tokenizer, option)(toks)
            except:
                continue
        raise RuntimeError(f"{type(encoder.tokenizer)} tokenizer cannot decode or detokenize")

    merged_index = 0
    merged_dataset = IndexedDataset(os.path.join(path_to_data, "merge"))

    # sorted to ensure ordering matches merged dataset
    basenames = sorted(
        [
            name
            for name in os.listdir(path_to_data)
            if name.endswith(".idx") and not name.startswith("merge")
        ]
    )

    # index into the merged document index
    merged_doc_index_index = 0

    for basename in basenames:
        realpath_raw = f"{os.path.join(path_to_raws, '_'.join(basename.split('_')[:-2]))}.jsonl"
        realpath_doc = os.path.join(path_to_data, basename.split(".")[-2])

        dataset_index = 0
        dataset = IndexedDataset(realpath_doc)

        merged_doc_idx = merged_dataset.document_indices[
            merged_doc_index_index : merged_doc_index_index + len(dataset.document_indices)
        ]
        merged_doc_idx = merged_doc_idx - merged_doc_idx[0]

        assert (
            dataset.document_indices == merged_doc_idx
        ).all(), f"ERROR: {basename.split('_')[:-2]}: merged dataset document indices mismatch"

        merged_doc_index_index += len(dataset.document_indices) - 1

        with open(realpath_raw, "rt") as reader:
            for json_line in reader:
                toks = encoder.encode(json_line)[0]["text"]

                raw = tokens_to_string(toks)

                processed_toks = []
                while len(processed_toks) < len(toks):
                    processed_toks.extend(dataset[dataset_index])
                    dataset_index += 1
                processed = tokens_to_string(processed_toks)

                assert (
                    raw == processed
                ), f"ERROR: {basename.split('_')[:-2]}: raw and processed documents do not match"

                merged_toks = []
                while len(merged_toks) < len(toks):
                    merged_toks.extend(merged_dataset[merged_index])
                    merged_index += 1
                merged = tokens_to_string(merged_toks)

                assert (
                    raw == merged
                ), f"ERROR: {basename.split('_')[:-2]}: raw and merged documents do not match"

        print(
            f"INFO: {''.join(basename.split('_')[:-2])}: raw, processed, and merged documents match!"
        )

    print("INFO: Success!")


def gpt2_vocab(odir):
    if os.path.exists(__LOCAL_GPT2_VOCAB):
        return __LOCAL_GPT2_VOCAB
    path = os.path.join(odir, "vocab.json")
    with open(path, "wb") as writer:
        writer.write(requests.get(MEGATRON_CONFIG_MAP['GPT2BPETokenizer']['vocab']).content)
    return path


def gpt2_merge(odir):
    if os.path.exists(__LOCAL_GPT2_MERGE):
        return __LOCAL_GPT2_MERGE
    path = os.path.join(odir, "merge.txt")
    with open(path, "wb") as writer:
        writer.write(requests.get(MEGATRON_CONFIG_MAP['GPT2BPETokenizer']['merges_file']).content)
    return path


def test_preprocess_data_gpt():
    with tempfile.TemporaryDirectory() as temp_dir:

        # gpt specific args
        gpt_args = [
            "--tokenizer-type",
            "GPT2BPETokenizer",
            "--vocab-file",
            "/opt/data/tokenizers/megatron/gpt2-vocab.json",
            "--merge-file",
            "/opt/data/tokenizers/megatron/gpt2-merges.txt",
            "--append-eod",
            "--workers",
            "10",
            "--log-interval",
            "1",
        ]

        do_test_preprocess_data(temp_dir, extra_args=gpt_args)


def test_preprocess_data_gpt_optimal_workers():
    with tempfile.TemporaryDirectory() as temp_dir:

        # gpt specific args
        gpt_args = [
            "--input",
            "/opt/data/datasets/dclm/dclm.jsonl",
            "--output-prefix",
            f"{temp_dir}/optimal_workers",
            "--tokenizer-type",
            "GPT2BPETokenizer",
            "--vocab-file",
            "/opt/data/tokenizers/megatron/gpt2-vocab.json",
            "--merge-file",
            "/opt/data/tokenizers/megatron/gpt2-merges.txt",
            "--append-eod",
            "--workers",
            "2",
            "--log-interval",
            "1",
            "--find-optimal-num-workers",
            "--workers-to-check",
            "2",
            "4",
            "8",
            "--max-documents",
            "1002",
        ]
        sys.argv = ["/opt/megatron-lm/tools/preprocess_data.py"] + gpt_args
        runpy.run_path("/opt/megatron-lm/tools/preprocess_data.py", run_name="__main__")


def bert_vocab(odir):
    if os.path.exists(__LOCAL_BERT_VOCAB):
        return __LOCAL_BERT_VOCAB
    path = os.path.join(odir, "vocab.txt")
    with open(path, "wb") as writer:
        writer.write(requests.get(__HUGGINGFACE_BERT_BASE_UNCASED_VOCAB).content)
    return path


@pytest.mark.flaky
@pytest.mark.flaky_in_dev
def test_preprocess_data_bert():
    with tempfile.TemporaryDirectory() as temp_dir:

        # bert specific args
        bert_args = [
            "--tokenizer-type",
            "BertWordPieceLowerCase",
            "--vocab-file",
            "/opt/data/tokenizers/megatron/gpt2-vocab.json",
            "--split-sentences",
            "--workers",
            "10",
            "--log-interval",
            "1",
            "--partitions",
            "2",
            "--keep-sequential-samples",
        ]

        do_test_preprocess_data(temp_dir, extra_args=bert_args)


@pytest.mark.skipif(PunktTokenizer is None, reason="requires NLTK >= 3.9 (punkt_tab)")
@pytest.mark.parametrize("keep_newlines", [False, True])
def test_preprocess_data_split_sentences_single_invocation(tmp_path, monkeypatch, keep_newlines):
    """One --split-sentences run with a single partition writes the sentence-level dataset."""
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    # An empty Punkt model is enough to split these sentences and keeps the test offline.
    nltk_data = tmp_path / "nltk_data"
    punkt_dir = nltk_data / "tokenizers" / "punkt_tab" / "english"
    punkt_dir.mkdir(parents=True)
    for name in ["collocations.tab", "sent_starters.txt", "abbrev_types.txt", "ortho_context.tab"]:
        (punkt_dir / name).touch()
    monkeypatch.setenv("NLTK_DATA", str(nltk_data))
    monkeypatch.setattr(nltk.data, "path", [str(nltk_data)] + nltk.data.path)
    monkeypatch.setattr(nltk, "download", lambda *args, **kwargs: True)

    words = ["[UNK]", "<eod>", "Hello", "world", "This", "is", "a", "test", "Bye", ".", "!"]
    backend = Tokenizer(
        models.WordLevel(vocab={word: i for i, word in enumerate(words)}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_dir = tmp_path / "tokenizer"
    PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="<eod>"
    ).save_pretrained(tokenizer_dir)

    input_path = tmp_path / "docs.jsonl"
    with open(input_path, "w") as writer:
        writer.write(json.dumps({"text": "Hello world. This is a test."}) + "\n")
        writer.write(json.dumps({"text": "Bye!"}) + "\n")

    output_prefix = tmp_path / "docs"
    argv = [
        "preprocess_data.py",
        "--input",
        str(input_path),
        "--output-prefix",
        str(output_prefix),
        "--tokenizer-type",
        "HuggingFaceTokenizer",
        "--tokenizer-model",
        str(tokenizer_dir),
        "--split-sentences",
        "--workers",
        "1",
    ]
    if keep_newlines:
        argv.append("--keep-newlines")
    monkeypatch.setattr(sys, "argv", argv)
    build_main()

    # Two sentences in the first document, one in the second.
    dataset = IndexedDataset(f"{output_prefix}_text_sentence")
    sentences = [dataset[i].tolist() for i in range(len(dataset))]
    assert dataset.document_indices.tolist() == [0, 2, 3]
    assert sentences == [[2, 3, 9], [4, 5, 6, 7, 9], [8, 10]]


if __name__ == "__main__":
    test_preprocess_data_gpt()
    test_preprocess_data_bert()
    test_preprocess_data_gpt_optimal_workers()
