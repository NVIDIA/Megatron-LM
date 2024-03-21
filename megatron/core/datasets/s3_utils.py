import boto3
import os
import torch
from typing import Any, Dict, NamedTuple, Protocol
from botocore.exceptions import ClientError


S3_PREFIX = "s3://"


class S3Client(Protocol):

    def download_file(self, bucket: str, key: str, local_path: str):
        ...

    def upload_file(self, local_path: str, bucket: str, key: str):
        ...

    def head_object(self, Bucket: str, Key: str):
        ...

    def get_object(self, Bucket: str, Key: str, Range: str) -> Dict[str, Any]:
        ...

    def close(self):
        ...


class ParsedS3Path(NamedTuple):
    """A parsed S3 path

    E.g., "s3://mybucket/path/to/file.ext" would have
    a value for `bucket` of "mybucket" and a value for
    `key` of "path/to/file.ext"
    """

    # The name of the S3 bucket.
    bucket: str

    # The S3 key, i.e., the part of the S3 path after the
    # first / that appears after the bucket name. If no such
    # / exists, then the key is empty.
    key: str


def is_s3_path(path: str) -> bool:
    """Returns True if the given path is an S3 path
    """
    return path.startswith(S3_PREFIX)


def parse_s3_path(path: str) -> ParsedS3Path:
    """Parses the given S3 path into a ParsedS3Path
    """
    assert is_s3_path(path)
    parts = path.replace(S3_PREFIX, "").split("/")
    bucket = parts[0]
    if len(parts) > 1:
        key = "/".join(parts[1:])
        assert S3_PREFIX + bucket + "/" + key == path
    else:
        key = ""
    return ParsedS3Path(bucket=bucket, key=key)


def object_exists(s3_client: S3Client, path: str) -> bool:
    """Return True if the object at the given S3 path exists
    """
    parsed_s3_path = parse_s3_path(path)
    try:
        response = s3_client.head_object(Bucket=parsed_s3_path.bucket, Key=parsed_s3_path.key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] != "404":
            raise e


def _download_file(s3_client: S3Client, s3_path: str, local_path: str):
    """Download the object at the given S3 path to the given local file system path
    """
    dirname = os.path.dirname(local_path)
    os.makedirs(dirname, exist_ok=True)
    parsed_s3_path = parse_s3_path(s3_path)
    s3_client.download_file(parsed_s3_path.bucket, parsed_s3_path.key, local_path)


def maybe_download_file(s3_path: str, local_path: str):
    """Download the object at the given S3 path to the given local file system path

    In a distributed setting, downloading the S3 object proceeds in stages in order
    to try to have the minimum number of processes download the object in order for
    all the ranks to have access to the downloaded object.
    """ 

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        # https://github.com/NVIDIA/Megatron-LM/blob/\
        # 89574689447d694bb19dd86fc8a6153b4467ba9d/megatron/initialize.py#L232
        local_rank = rank % torch.cuda.device_count()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    s3_client = boto3.client("s3")

    if (not os.path.exists(local_path)) and (rank == 0):
        _download_file(s3_client, s3_path, local_path)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # If the `local_path` is in a file system that is not
    # shared across all the ranks, then we assume it's in the
    # host file system and each host needs to download the file.
    if (not os.path.exists(local_path)) and (local_rank == 0):
        _download_file(s3_client, s3_path, local_path)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # If the `local_path` still does not exist, then we assume
    # each rank is saving to a separate location.
    if not os.path.exists(local_path):
        _download_file(s3_client, s3_path, local_path)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    assert os.path.exists(local_path)
