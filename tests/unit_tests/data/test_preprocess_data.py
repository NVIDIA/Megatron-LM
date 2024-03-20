# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import glob
import io
import json
import os
import sys
import tempfile

import boto3
import nltk
import requests

from botocore.exceptions import ClientError

from megatron.core.datasets.indexed_dataset import IndexedDataset, S3Config
from megatron.core.datasets.s3_utils import S3_PREFIX
from megatron.tokenizer.gpt2_tokenization import (
    PRETRAINED_MERGES_ARCHIVE_MAP,
    PRETRAINED_VOCAB_ARCHIVE_MAP,
)
from tools.merge_datasets import main as merge_main
from tools.preprocess_data import Encoder
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


def do_test_preprocess_data(temp_dir, extra_args=[], s3=False):
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
    build_datasets(
        path_to_raws, path_to_data, extra_args=extra_args,
    )

    # merge the datasets
    merge_datasets(path_to_data)

    sys.argv = [sys.argv[0], "--input", None, "--output-prefix", None,] + extra_args
    encoder = Encoder(build_args())
    encoder.initializer()

    def tokens_to_string(toks):
        for option in ["decode", "detokenize"]:
            try:
                return getattr(encoder.tokenizer, option)(toks)
            except:
                continue
        raise RuntimeError(f"{type(encoder.tokenizer)} tokenizer cannot decode or detokenize")

    prefix_for_path_prefix = ""
    indexed_dataset_kwargs = {}
    if s3:
        # Copy all the files in `temp_dir` to S3.
        s3_client = boto3.client("s3")
        bucket_name = "test-bucket"
        for path in glob.glob(os.path.join(temp_dir, "**/**")):
            assert path.startswith("/")
            s3_client.upload_file(path, bucket_name, path[1:])
        assert path_to_data.startswith("/")
        prefix_for_path_prefix = S3_PREFIX + bucket_name
        indexed_dataset_kwargs = {
            "mmap": False,
            "s3_config": S3Config(path_to_idx_cache=os.path.join(temp_dir, "idx_cache"))
        }

    merged_index = 0
    merged_dataset = IndexedDataset(
        prefix_for_path_prefix + os.path.join(path_to_data, "merge"), **indexed_dataset_kwargs
    )

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
        dataset = IndexedDataset(
            prefix_for_path_prefix + realpath_doc, **indexed_dataset_kwargs
        )

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
        writer.write(requests.get(PRETRAINED_VOCAB_ARCHIVE_MAP['gpt2']).content)
    return path


def gpt2_merge(odir):
    if os.path.exists(__LOCAL_GPT2_MERGE):
        return __LOCAL_GPT2_MERGE
    path = os.path.join(odir, "merge.txt")
    with open(path, "wb") as writer:
        writer.write(requests.get(PRETRAINED_MERGES_ARCHIVE_MAP['gpt2']).content)
    return path


def gpt_args(temp_dir):
    # gpt specific args
    return [
        "--tokenizer-type",
        "GPT2BPETokenizer",
        "--vocab-file",
        gpt2_vocab(temp_dir),
        "--merge-file",
        gpt2_merge(temp_dir),
        "--append-eod",
        "--workers",
        "10",
        "--log-interval",
        "1",
    ]


def test_preprocess_data_gpt():
    with tempfile.TemporaryDirectory() as temp_dir:
        do_test_preprocess_data(temp_dir, extra_args=gpt_args(temp_dir))


class MockS3Client:
    def __init__(self, *args, **kwargs):
        self._data = {}

    def download_file(self, bucket, key, local_path):
        with open(local_path, "wb") as fout:
            fout.write(self._data[(bucket, key)])

    def upload_file(self, local_path, bucket, key):
        with open(local_path, "rb") as fin:
            self._data[(bucket, key)] = fin.read()

    def head_object(self, Bucket, Key):
        return (Bucket, Key) in self._data

    def get_object(self, Bucket, Key, Range):
        assert Range.startswith("bytes=")
        parts = Range.split("=")
        assert len(parts) == 2
        subparts = parts[1].split("-")
        assert len(subparts) == 2
        start = int(subparts[0])
        # add 1 to convert inclusive index into exclusive index.
        end = int(subparts[1]) + 1
        return {"Body": io.BytesIO(self._data[(Bucket, Key)][start:end])}

    def close(self):
        pass


def test_preprocess_data_gpt_s3(monkeypatch):
    MOCK_S3_CLIENT = MockS3Client()

    def mock_s3_client(*args, **kwargs):
        return MOCK_S3_CLIENT

    monkeypatch.setattr("boto3.client", mock_s3_client)

    with tempfile.TemporaryDirectory() as temp_dir:
        do_test_preprocess_data(
            temp_dir, extra_args=gpt_args(temp_dir), s3=True
        )


def bert_vocab(odir):
    if os.path.exists(__LOCAL_BERT_VOCAB):
        return __LOCAL_BERT_VOCAB
    path = os.path.join(odir, "vocab.txt")
    with open(path, "wb") as writer:
        writer.write(requests.get(__HUGGINGFACE_BERT_BASE_UNCASED_VOCAB).content)
    return path


def test_preprocess_data_bert():
    with tempfile.TemporaryDirectory() as temp_dir:

        # bert specific args
        bert_args = [
            "--tokenizer-type",
            "BertWordPieceLowerCase",
            "--vocab-file",
            bert_vocab(temp_dir),
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


if __name__ == "__main__":
    test_preprocess_data_gpt()
    test_preprocess_data_bert()
