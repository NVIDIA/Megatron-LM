import os
import random
import sys
import tempfile
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any, Dict

import nltk
import pytest

try:
    import boto3
    import botocore.exceptions as exceptions
except ModuleNotFoundError:
    boto3 = ModuleType("boto3")
    sys.modules[boto3.__name__] = boto3
    exceptions = ModuleType("botocore.exceptions")
    sys.modules[exceptions.__name__] = exceptions

try:
    import multistorageclient as msc
except ModuleNotFoundError:
    # Create mock msc module
    msc = ModuleType("multistorageclient")

    # Create mock types submodule
    types_module = ModuleType("multistorageclient.types")

    # Create Range class in types module
    class Range:
        def __init__(self, offset: int, size: int):
            self.offset = offset
            self.size = size

    # Add Range class to types module
    types_module.Range = Range  # type: ignore[attr-defined]

    # Add types submodule to msc
    msc.types = types_module

    # Register the mock module in sys.modules
    sys.modules[msc.__name__] = msc
    sys.modules[types_module.__name__] = types_module

from megatron.core.datasets.indexed_dataset import (
    IndexedDataset,
    ObjectStorageConfig,
    _FileBinReader,
    _MMapBinReader,
    _MultiStorageClientBinReader,
    _S3BinReader,
)
from megatron.core.datasets.object_storage_utils import MSC_PREFIX, S3_PREFIX, S3Client
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
        os.makedirs(os.path.dirname(Filename), exist_ok=True)
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

##
# Mock multistorageclient module
##


def _msc_download_file(remote_path, local_path):
    remote_path = remote_path.removeprefix(MSC_PREFIX + "default")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    os.system(f"cp {remote_path} {local_path}")


def _msc_resolve_storage_client(path):
    class StorageClient:
        def read(self, path, byte_range):
            with open(path, "rb") as f:
                f.seek(byte_range.offset)
                return f.read(byte_range.size)

    return StorageClient(), path.removeprefix(MSC_PREFIX + "default")


setattr(msc, "open", open)
setattr(msc, "download_file", _msc_download_file)
setattr(msc, "resolve_storage_client", _msc_resolve_storage_client)


@pytest.mark.flaky
@pytest.mark.flaky_in_dev
def test_bin_reader():
    with tempfile.TemporaryDirectory() as temp_dir:
        # set the default nltk data path
        os.environ["NLTK_DATA"] = os.path.join(temp_dir, "nltk_data")
        nltk.data.path.append(os.environ["NLTK_DATA"])

        path_to_raws = os.path.join(temp_dir, "sample_raws")
        path_to_data = os.path.join(temp_dir, "sample_data")
        path_to_object_storage_cache = os.path.join(temp_dir, "object_storage_cache")
        os.mkdir(path_to_raws)
        os.mkdir(path_to_data)
        os.mkdir(path_to_object_storage_cache)

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

            indexed_dataset_msc = IndexedDataset(
                MSC_PREFIX + "default" + prefix,  # use the default profile to access the filesystem
                multimodal=False,
                mmap=False,
                object_storage_config=ObjectStorageConfig(
                    path_to_idx_cache=path_to_object_storage_cache
                ),
            )
            assert isinstance(indexed_dataset_msc.bin_reader, _MultiStorageClientBinReader)
            assert len(indexed_dataset_msc) == len(indexed_dataset_file)
            assert len(indexed_dataset_msc) == len(indexed_dataset_mmap)

            indexed_dataset_s3 = IndexedDataset(
                S3_PREFIX + prefix,
                multimodal=False,
                mmap=False,
                object_storage_config=ObjectStorageConfig(
                    path_to_idx_cache=path_to_object_storage_cache
                ),
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
