import os
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Any, Dict

import nltk

try:
    import boto3
    import botocore.exceptions as exceptions
except ModuleNotFoundError:
    boto3 = ModuleType("boto3")
    sys.modules[boto3.__name__] = boto3
    exceptions = ModuleType("botocore.exceptions")
    sys.modules[exceptions.__name__] = exceptions

from megatron.core.datasets.indexed_dataset import (
    IndexedDataset,
    S3Config,
    _FileBinReader,
    _MMapBinReader,
    _S3BinReader,
)
from megatron.core.datasets.utils_s3 import S3_PREFIX, S3Client
from tests.unit_tests.data.test_preprocess_data import (
    build_datasets,
    dummy_jsonl,
    gpt2_merge,
    gpt2_vocab,
)

##
# Overload client from boto3
##


class _LocalClient(S3Client):
    """Local test client"""

    def __init__(self, *args: Any) -> None:
        pass

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:
        os.system(f"cp {os.path.join('/', Bucket, Key)} {Filename}")
        assert os.path.exists(Filename)

    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
        raise NotImplementedError

    def head_object(self, Bucket: str, Key: str) -> Dict[str, Any]:
        assert os.path.exists(os.path.join("/", Bucket, Key))
        return {}

    def get_object(self, Bucket: str, Key: str, Range: str) -> Dict[str, Any]:
        _, _range = Range.split("=")
        _range_beg, _range_end = tuple(map(int, _range.split("-")))

        filename = os.path.join("/", Bucket, Key)

        with open(filename, mode='rb', buffering=0) as bin_buffer_file:
            bin_buffer_file.seek(_range_beg)
            _bytes = bin_buffer_file.read(_range_end - _range_beg)

        response = {"Body": SimpleNamespace(read=lambda: _bytes)}

        return response

    def close(self) -> None:
        pass


setattr(boto3, "client", _LocalClient)


##
# Overload ClientError from botocore.exceptions
##


class _LocalClientError(Exception):
    """ "Local test client error"""

    pass


setattr(exceptions, "ClientError", _LocalClientError)


def test_bin_reader():
    with tempfile.TemporaryDirectory() as temp_dir:
        # set the default nltk data path
        os.environ["NLTK_DATA"] = os.path.join(temp_dir, "nltk_data")
        nltk.data.path.append(os.environ["NLTK_DATA"])

        path_to_raws = os.path.join(temp_dir, "sample_raws")
        path_to_data = os.path.join(temp_dir, "sample_data")
        path_to_s3_cache = os.path.join(temp_dir, "s3_cache")
        os.mkdir(path_to_raws)
        os.mkdir(path_to_data)
        os.mkdir(path_to_s3_cache)

        # create the dummy resources
        dummy_jsonl(path_to_raws)

        # build the datasets
        build_datasets(
            path_to_raws,
            path_to_data,
            extra_args=[
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
            ],
        )

        prefixes = set(
            [
                os.path.join(temp_dir, "sample_data", path.split(".")[0])
                for path in os.listdir(path_to_data)
                if path.endswith(".bin") or path.endswith(".idx")
            ]
        )

        for prefix in prefixes:
            indexed_dataset_file = IndexedDataset(prefix, multimodal=False, mmap=False)
            assert isinstance(indexed_dataset_file.bin_reader, _FileBinReader)

            indexed_dataset_mmap = IndexedDataset(prefix, multimodal=False, mmap=True)
            assert isinstance(indexed_dataset_mmap.bin_reader, _MMapBinReader)

            indexed_dataset_s3 = IndexedDataset(
                S3_PREFIX + prefix,
                multimodal=False,
                mmap=False,
                s3_config=S3Config(path_to_idx_cache=path_to_s3_cache),
            )
            assert isinstance(indexed_dataset_s3.bin_reader, _S3BinReader)

            assert len(indexed_dataset_s3) == len(indexed_dataset_file)
            assert len(indexed_dataset_s3) == len(indexed_dataset_mmap)

            indices = random.sample(
                list(range(len(indexed_dataset_s3))), min(100, len(indexed_dataset_s3))
            )

            for idx in indices:
                assert (indexed_dataset_s3[idx] == indexed_dataset_file[idx]).all()
                assert (indexed_dataset_s3[idx] == indexed_dataset_mmap[idx]).all()


if __name__ == "__main__":
    test_bin_reader()
