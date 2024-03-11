import os
import time
from collections import namedtuple
from typing import Optional, Tuple, Union

import boto3
import numpy
import torch
from botocore.exceptions import ClientError

from megatron.core.datasets.indexed_dataset import DType, _IndexReader, get_bin_path, get_idx_path


_S3_PREFIX = "s3://"


_ParsedS3Path = namedtuple("_ParsedS3Path", ["bucket", "key"])


def is_s3_path(path: str) -> str:
    return path.startswith(_S3_PREFIX)


def _get_local_idx_path(path_to_idx_cache, path_prefix):
    idx_path = get_idx_path(path_prefix)
    return os.path.join(path_to_idx_cache, os.path.basename(idx_path))


def _parse_s3_path(path: str) -> _ParsedS3Path:
    assert is_s3_path(path)
    parts = path.replace(_S3_PREFIX, "").split("/")
    bucket = parts[0]
    key = "/".join(parts[1:])
    assert _S3_PREFIX + bucket + "/" + key == path
    return _ParsedS3Path(bucket=bucket, key=key)


def _object_exists(s3_client, path: str) -> bool:
    parsed_s3_path = _parse_s3_path(path)
    try:
        response = s3_client.head_object(Bucket=parsed_s3_path.bucket, Key=parsed_s3_path.key)
        return True
    except ClientError:
        return False


class _S3Agent:
    """Retrieve byte ranges from S3 for the S3IndexedDataset."""

    def __init__(self, path, cache_nbytes):
        self._client = boto3.client('s3')
        assert path.startswith('s3://')
        path = path[len('s3://') :]
        self._bucket, self._key = path.split('/', 1)
        self._cache = None
        self._cache_bytes_start = None
        self._cache_bytes_end = None
        self._cache_nbytes = cache_nbytes

    def _extract_from_cache(self, offset, size):
        start = offset - self._cache_bytes_start
        assert start >= 0
        end = start + size
        assert end <= len(self._cache)
        return self._cache[start:end]

    def get_bytes(self, offset, size):
        """Get `size` bytes starting at `offset`.
        If the requested span of bytes [`offset`, `offset` + `size`) is covered
        by the in-memory cache maintained by this class, then this function
        extracts the requested span from that cache and returns it.
        Otherwise, this function first refreshes the cache and then extracts the
        requested span from the refreshed cache and returns it.
        The cache is refreshed based on `offset` and `size`. In particular, we
        divide all the bytes in an object into blocks, where each block contains
        `cache_size` bytes. We assign each block an index starting from 0.
        We take the block with index (`offset` // `cache_size`) to refresh the
        cache. If this new block still does not cover the requested span, we extend
        it just enough to include `offset` + `size`.
        """
        if (
            self._cache is not None
            and offset >= self._cache_bytes_start
            and offset + size <= self._cache_bytes_end
        ):
            return self._extract_from_cache(offset, size)

        bytes_start = (offset // self._cache_nbytes) * self._cache_nbytes
        assert bytes_start >= 0
        assert offset >= bytes_start
        bytes_end = max(bytes_start + self._cache_nbytes, offset + size)
        assert bytes_end >= 1
        self._cache = self._client.get_object(
            Bucket=self._bucket,
            Key=self._key,
            # Subtract 1, because the end of Range is inclusive.
            Range=f'bytes={bytes_start}-{bytes_end-1}',
        )['Body'].read()
        self._cache_bytes_start = bytes_start
        self._cache_bytes_end = bytes_end
        return self._extract_from_cache(offset, size)

    def close(self):
        self._client.close()


def _download_file(s3_client, s3_path, local_path):
    dirname = os.path.dirname(local_path)
    os.makedirs(dirname, exist_ok=True)
    parsed_s3_path = _parse_s3_path(s3_path)
    s3_client.download_file(parsed_s3_path.bucket, parsed_s3_path.key, local_path)


def _maybe_download_file(s3_path, local_path):
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


class S3IndexedDataset(torch.utils.data.Dataset):
    """The low-level interface dataset class for S3.

    Support a low-level dataset stored in the same format as the MMapIndexedDataset in S3.

    Args:
        path_prefix (str): The index (.idx) and data (.bin) prefix

        multimodal (bool): Whether the dataset is multimodal.

        path_to_idx_cache (str): Download the index (.idx) to this local cache.

        data_cache_nbytes (int): Stream the .bin file into memory in blocks of this number of bytes.
          If the cache size is too small, then we send a request to S3 at each call of `get_bytes`,
          which is slow, because each request has a fixed cost independent of the size of the byte range
          requested. If the cache size is too large, then we only rarely have to send requests to S3,
          but it takes a lot of time to complete the request when we do, which can block training.
          We have found that a cache size of 256 * 1024 * 1024 (i.e., 256 MiB) has worked well
          (though we have not put that much effort into tuning it), so we default to it.
    """

    def __init__(
        self,
        path_prefix: str,
        multimodal: bool,
        path_to_idx_cache: str,
        data_cache_nbytes: int = 256 * 1024 * 1024,
    ) -> None:
        super().__init__()
        self.path_prefix = None
        self.multimodal = None
        self.path_to_idx_cache = None
        self.data_cache_nbytes = None

        self.index = None
        self.s3_agent = None

        idx_path = get_idx_path(path_prefix)
        local_idx_path = _get_local_idx_path(path_to_idx_cache, path_prefix)
        _maybe_download_file(idx_path, local_idx_path)

        self.initialize(path_prefix, multimodal, path_to_idx_cache, data_cache_nbytes)

    def initialize(
        self, path_prefix: str, multimodal: bool, path_to_idx_cache: str, data_cache_nbytes: int
    ) -> None:
        """Initialize the dataset

        This method is called by S3IndexedDataset.__init__ during object creation and by
        S3IndexedDataset.__setstate__ during un-pickling

        Args:
            path_prefix (str): The index (.idx) and data (.bin) prefix

            multimodal (bool): Whether the dataset is multimodal

            path_to_idx_cache (str): Download the index (.idx) to this local cache.

            data_cache_nbytes (int): Stream the .bin file into memory in blocks of this number of bytes.
        """
        self.path_prefix = path_prefix
        self.multimodal = multimodal
        self.path_to_idx_cache = path_to_idx_cache
        self.data_cache_nbytes = data_cache_nbytes
        self.index = _IndexReader(
            _get_local_idx_path(self.path_to_idx_cache, self.path_prefix), self.multimodal
        )
        self.s3_agent = _S3Agent(get_bin_path(self.path_prefix), data_cache_nbytes)

    def __getstate__(self) -> Tuple[str, bool, str, int]:
        """Get the state during pickling

        Returns:
            Tuple[str, bool, str, int]: The state tuple
        """
        return self.path_prefix, self.multimodal, self.path_to_idx_cache, self.data_cache_nbytes

    def __setstate__(self, state: Tuple[str, bool, str, int]) -> None:
        """Set the state during un-pickling

        Args:
            state (Tuple[str, bool, str, int]): The state tuple
        """
        path_prefix, multimodal, path_to_idx_cache, data_cache_nbytes = state
        self.initialize(path_prefix, multimodal, path_to_idx_cache, data_cache_nbytes)

    def __del__(self) -> None:
        """Clean up the object
        """
        if self.s3_agent is not None:
            self.s3_agent.close()
            del self.s3_agent
        del self.index

    def __len__(self) -> int:
        """Return the length of the dataset i.e. the number of sequences in the index

        Returns:
            int: The length of the dataset
        """
        return len(self.index)

    def __getitem__(
        self, idx: Union[int, numpy.integer, slice]
    ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
        """Return from the dataset

        Args:
            idx (Union[int, numpy.integer]): The index into the dataset

        Raises:
            TypeError: When the index is of an unexpected type

        Returns:
            Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]: The sequence tokens and
            modes at the index
        """
        if isinstance(idx, (int, numpy.integer)):
            sequence_pointer, sequence_length, sequence_mode = self.index[idx]
            sequence = numpy.frombuffer(
                self.s3_agent.get_bytes(
                    sequence_pointer, sequence_length * DType.size(self.index.dtype)
                ),
                dtype=self.index.dtype,
            )
            return (sequence, sequence_mode) if sequence_mode is not None else sequence
        else:
            raise TypeError("Unexpected type received for idx: {}".format(type(idx)))

    def get(self, idx: int, offset: int = 0, length: Optional[int] = None) -> numpy.ndarray:
        """Retrieve a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        sequence_pointer, sequence_length, sequence_mode = self.index[idx]
        if length is None:
            length = sequence_length - offset
        sequence_pointer += offset * DType.size(self.index.dtype)
        sequence = numpy.frombuffer(
            self.s3_agent.get_bytes(sequence_pointer, length * DType.size(self.index.dtype)),
            dtype=self.index.dtype,
        )
        return (sequence, sequence_mode) if sequence_mode is not None else sequence

    @property
    def sequence_lengths(self) -> numpy.ndarray:
        """Get the sequence lengths

        Returns:
            numpy.ndarray: The sequence lengths
        """
        return self.index.sequence_lengths

    @property
    def document_indices(self) -> numpy.ndarray:
        """Get the document indices

        Returns:
            numpy.ndarray: The document indices
        """
        return self.index.document_indices

    def get_document_indices(self) -> numpy.ndarray:
        """Get the document indices

        This method is slated for deprecation.

        Returns:
            numpy.ndarray: The document indices
        """
        return self.index.document_indices

    def set_document_indices(self, document_indices: numpy.ndarray) -> None:
        """Set the document indices

        This method is slated for deprecation.

        Args:
            document_indices (numpy.ndarray): The document indices
        """
        self.index.document_indices = document_indices

    @property
    def sequence_modes(self) -> numpy.ndarray:
        """Get the sequence modes

        Returns:
            numpy.ndarray: The sequence modes
        """
        return self.index.sequence_modes

    @staticmethod
    def exists(path_prefix: str) -> bool:
        """Return whether the S3IndexedDataset exists in S3 at the prefix

        Args:
            path_prefix (str): The prefix to the index (.idx) and data (.bin) files

        Returns:
            bool: Whether the S3IndexedDataset exists in S3 at the prefix
        """
        s3_client = boto3.client("s3")
        return _object_exists(s3_client, get_idx_path(path_prefix)) and _object_exists(
            s3_client, get_bin_path(path_prefix)
        )
