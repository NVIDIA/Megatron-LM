# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import os
from dataclasses import dataclass
from typing import Any, Dict, Protocol, Tuple

import torch

try:
    import boto3
    import botocore.exceptions as exceptions
except ModuleNotFoundError:
    pass

from megatron.core.msc_utils import MultiStorageClientFeature

S3_PREFIX = "s3://"
MSC_PREFIX = "msc://"


@dataclass
class ObjectStorageConfig:
    """Config when the data (.bin) file and the index (.idx) file are in object storage

    Attributes:

        path_to_idx_cache (str): The local directory where we will store the index (.idx) file

        bin_chunk_nbytes (int): If the number of bytes is too small, then we send a request to S3
        at each call of the `read` method in _S3BinReader, which is slow, because each request
        has a fixed cost independent of the size of the byte range requested. If the number of
        bytes is too large, then we only rarely have to send requests to S3, but it takes a lot
        of time to complete the request when we do, which can block training. We've found that
        256 * 1024 * 1024 (i.e., 256 MiB) has worked well (though we have not put that much
        effort into tuning it), so we default to it.
    """

    path_to_idx_cache: str

    bin_chunk_nbytes: int = 256 * 1024 * 1024


class S3Client(Protocol):
    """The protocol which all s3 clients should abide by"""

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:
        """Download the file from S3 to the local file system"""
        ...

    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
        """Upload the file to S3"""
        ...

    def head_object(self, Bucket: str, Key: str) -> Dict[str, Any]:
        """Get the metadata of the file in S3"""
        ...

    def get_object(self, Bucket: str, Key: str, Range: str) -> Dict[str, Any]:
        """Get the file from S3"""
        ...

    def close(self) -> None:
        """Close the S3 client"""
        ...


def _remove_s3_prefix(path: str) -> str:
    """Remove the S3 prefix from a path

    Args:
        path (str): The path

    Returns:
        str: The path without the S3 prefix
    """
    return path.removeprefix(S3_PREFIX)


def _is_s3_path(path: str) -> bool:
    """Ascertain whether a path is in S3

    Args:
        path (str): The path

    Returns:
        bool: True if the path is in S3, False otherwise
    """
    return path.startswith(S3_PREFIX)


def _remove_msc_prefix(path: str) -> str:
    """
    Remove the MSC prefix from a path

    Args:
        path (str): The path

    Returns:
        str: The path without the MSC prefix
    """
    return path.removeprefix(MSC_PREFIX)


def _is_msc_path(path: str) -> bool:
    """Checks whether a path is in MSC path (msc://profile/path/to/file)

    Args:
        path (str): The path

    Returns:
        bool: True if the path is in MSC path, False otherwise
    """
    return path.startswith(MSC_PREFIX)


def _s3_download_file(client: S3Client, s3_path: str, local_path: str) -> None:
    """Download the object at the given S3 path to the given local file system path

    Args:
        client (S3Client): The S3 client

        s3_path (str): The S3 source path

        local_path (str): The local destination path
    """
    dirname = os.path.dirname(local_path)
    os.makedirs(dirname, exist_ok=True)
    parsed_s3_path = parse_s3_path(s3_path)
    client.download_file(parsed_s3_path[0], parsed_s3_path[1], local_path)


def _s3_object_exists(client: S3Client, path: str) -> bool:
    """Ascertain whether the object at the given S3 path exists in S3

    Args:
        client (S3Client): The S3 client

        path (str): The S3 path

    Raises:
        botocore.exceptions.ClientError: The error code is 404

    Returns:
        bool: True if the object exists in S3, False otherwise
    """
    parsed_s3_path = parse_s3_path(path)
    try:
        _ = client.head_object(bucket=parsed_s3_path[0], key=parsed_s3_path[1])
    except exceptions.ClientError as e:
        if e.response["Error"]["Code"] != "404":
            raise e
    return True


def is_object_storage_path(path: str) -> bool:
    """Ascertain whether a path is in object storage

    Args:
        path (str): The path

    Returns:
        bool: True if the path is in object storage (s3:// or msc://), False otherwise
    """
    return _is_s3_path(path) or _is_msc_path(path)


def get_index_cache_path(idx_path: str, object_storage_config: ObjectStorageConfig) -> str:
    """Get the index cache path for the given path

    Args:
        idx_path (str): The path to the index file

        object_storage_config (ObjectStorageConfig): The object storage config

    Returns:
        str: The index cache path
    """
    if _is_s3_path(idx_path):
        cache_idx_path = os.path.join(
            object_storage_config.path_to_idx_cache, _remove_s3_prefix(idx_path)
        )
    elif _is_msc_path(idx_path):
        cache_idx_path = os.path.join(
            object_storage_config.path_to_idx_cache, _remove_msc_prefix(idx_path)
        )
    else:
        raise ValueError(f"Invalid path: {idx_path}")

    return cache_idx_path


def parse_s3_path(path: str) -> Tuple[str, str]:
    """Parses the given S3 path returning correspsonding bucket and key.

    Args:
        path (str): The S3 path

    Returns:
        Tuple[str, str]: A (bucket, key) tuple
    """
    assert _is_s3_path(path)
    parts = path.replace(S3_PREFIX, "").split("/")
    bucket = parts[0]
    if len(parts) > 1:
        key = "/".join(parts[1:])
        assert S3_PREFIX + bucket + "/" + key == path
    else:
        key = ""
    return bucket, key


def get_object_storage_access(path: str) -> str:
    """Get the object storage access"""
    return "s3" if _is_s3_path(path) else "msc"


def dataset_exists(path_prefix: str, idx_path: str, bin_path: str) -> bool:
    """Check if the dataset exists on object storage

    Args:
        path_prefix (str): The prefix to the index (.idx) and data (.bin) files

        idx_path (str): The path to the index file

        bin_path (str): The path to the data file

    Returns:
        bool: True if the dataset exists on object storage, False otherwise
    """
    if _is_s3_path(path_prefix):
        s3_client = boto3.client("s3")
        return _s3_object_exists(s3_client, idx_path) and _s3_object_exists(s3_client, bin_path)
    elif _is_msc_path(path_prefix):
        msc = MultiStorageClientFeature.import_package()
        return msc.exists(idx_path) and msc.exists(bin_path)
    else:
        raise ValueError(f"Invalid path: {path_prefix}")


def cache_index_file(remote_path: str, local_path: str) -> None:
    """Download a file from object storage to a local path with distributed training support.
    The download only happens on Rank 0, and other ranks will wait for the file to be available.

    Note that this function does not include any barrier synchronization. The caller (typically
    in blended_megatron_dataset_builder.py) is responsible for ensuring proper synchronization
    between ranks using torch.distributed.barrier() after this function returns.

    Args:
        remote_path (str): The URL of the file to download (e.g., s3://bucket/path/file.idx
            or msc://profile/path/file.idx)
        local_path (str): The local destination path where the file should be saved

    Raises:
        ValueError: If the remote_path is not a valid S3 or MSC path
    """
    torch_dist_enabled = torch.distributed.is_initialized()

    if torch_dist_enabled:
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    if _is_s3_path(remote_path):
        s3_client = boto3.client("s3")

        if not torch_dist_enabled or rank == 0:
            _s3_download_file(s3_client, remote_path, local_path)

        assert os.path.exists(local_path)
    elif _is_msc_path(remote_path):
        msc = MultiStorageClientFeature.import_package()

        if not torch_dist_enabled or rank == 0:
            msc.download_file(remote_path, local_path)

        assert os.path.exists(local_path)
    else:
        raise ValueError(f"Invalid path: {remote_path}")
