# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Essentially re-written in entirety

import os
import shutil
import struct
from enum import Enum
from functools import lru_cache
from itertools import accumulate
from types import TracebackType
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import torch

from megatron import print_rank_0

_INDEX_HEADER = b"MMIDIDX\x00\x00"


class DType(Enum):
    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float64 = 6
    float32 = 7
    uint16 = 8

    @classmethod
    def code_from_dtype(cls, value: Type[np.number]) -> int:
        return cls[value.__name__].value

    @classmethod
    def dtype_from_code(cls, value: int) -> Type[np.number]:
        return getattr(np, cls(value).name)

    @staticmethod
    def size(key: Union[int, Type[np.number]]) -> int:
        if isinstance(key, int):
            return DType.dtype_from_code(key)().itemsize
        elif np.number in key.__mro__:
            return key().itemsize
        else:
            raise ValueError

    @staticmethod
    def optimal_dtype(cardinality: int) -> Type[np.number]:
        if cardinality is not None and cardinality < 65500:
            return np.uint16
        else:
            return np.int32


class _IndexWriter(object):
    """
    Object class to write the index file i.e. <data-path>.idx
    """

    def __init__(self, path: str, dtype: Type[np.number]) -> None:
        self.path = path
        self.dtype = dtype

    def __enter__(self) -> "_IndexWriter":
        self.idx_path = open(self.path, "wb")
        # fixed, vestigial practice
        self.idx_path.write(_INDEX_HEADER)
        # fixed, vestigial practice
        self.idx_path.write(struct.pack("<Q", 1))
        # the numeric code for the dtype
        self.idx_path.write(struct.pack("<B", DType.code_from_dtype(self.dtype)))
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        self.idx_path.close()

    def write(
        self,
        sequence_lengths: List[int],
        sequence_modes: Optional[List[int]],
        document_indices: List[int],
    ) -> None:
        sequence_pointers = self._sequence_pointers(sequence_lengths)

        # the number of sequences in the dataset
        sequence_count = len(sequence_lengths)
        self.idx_path.write(struct.pack("<Q", sequence_count))

        # the number of documents in the dataset
        document_count = len(document_indices)
        self.idx_path.write(struct.pack("<Q", document_count))

        # the number of tokens per sequence
        sequence_lengths = np.array(sequence_lengths, dtype=np.int32)
        self.idx_path.write(sequence_lengths.tobytes(order="C"))
        del sequence_lengths

        # the byte offsets for all sequences
        sequence_pointers = np.array(sequence_pointers, dtype=np.int64)
        self.idx_path.write(sequence_pointers.tobytes(order="C"))
        del sequence_pointers

        # the sequence indices marking the end of each document
        document_indices = np.array(document_indices, dtype=np.int64)
        self.idx_path.write(document_indices.tobytes(order="C"))

        # the mode per sequence
        if sequence_modes is not None:
            sequence_modes = np.array(sequence_modes, dtype=np.int32)
            self._file.write(sequence_modes.tobytes(order='C'))
            del sequence_modes

    def _sequence_pointers(self, sequence_lengths: List[int]) -> List[int]:
        itemsize = DType.size(self.dtype)
        curr_ptr = 0
        list_ptr = []
        for length in sequence_lengths:
            list_ptr.append(curr_ptr)
            curr_ptr += length * itemsize
        return list_ptr


class _IndexReader(object):
    """
    Object class to read the index file i.e. <data-path>.idx
    """

    def __init__(self, path: str, multimodal: bool) -> None:
        with open(path, "rb") as stream:
            header = stream.read(9)
            assert header == _INDEX_HEADER, f"bad header, cannot read: {path}"

            version = struct.unpack("<Q", stream.read(8))[0]
            assert version == 1, f"bad version, cannot read: {path}"

            code = struct.unpack("<B", stream.read(1))[0]
            self._dtype = DType.dtype_from_code(code)
            self._dtype_size = DType.size(self._dtype)

            self._sequence_count = struct.unpack("<Q", stream.read(8))[0]
            self._document_count = struct.unpack("<Q", stream.read(8))[0]

            offset = stream.tell()

        self._multimodal = multimodal

        self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

        print_rank_0("    reading sequence lengths...")
        self._sequence_lengths = np.frombuffer(
            self._bin_buffer, dtype=np.int32, count=self._sequence_count, offset=offset
        )

        print_rank_0("    reading sequence pointers...")
        self._sequence_pointers = np.frombuffer(
            self._bin_buffer,
            dtype=np.int64,
            count=self._sequence_count,
            offset=offset + self._sequence_lengths.nbytes,
        )

        print_rank_0("    reading document indices...")
        self._document_indices = np.frombuffer(
            self._bin_buffer,
            dtype=np.int64,
            count=self._document_count,
            offset=offset + self._sequence_lengths.nbytes + self._sequence_pointers.nbytes,
        )

        self._sequence_modes = None
        if self._multimodal:
            print_rank_0("    reading sequence modes...")
            self._sequence_modes = np.frombuffer(
                self._bin_buffer,
                dtype=np.int8,
                count=self._len,
                offset=offset
                + self._sequence_lengths.nbytes
                + self._sequence_pointers.nbytes
                + self._document_indices.nbytes,
            )

    def __del__(self) -> None:
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap

    def __len__(self) -> int:
        return self._sequence_count

    @lru_cache(maxsize=8)
    def __getitem__(self, i: int) -> Tuple[np.int32, np.int64, Optional[np.int8]]:
        return (
            self._sequence_pointers[i],
            self._sequence_lengths[i],
            self._sequence_modes[i] if self._multimodal else None,
        )

    @property
    def dtype(self) -> Type[np.number]:
        return self._dtype

    @property
    def sizes(self) -> np.ndarray:
        return self._sequence_lengths

    @property
    def doc_idx(self) -> np.ndarray:
        return self._document_indices

    @property
    def modes(self) -> np.ndarray:
        return self._sequence_modes


class MMapIndexedDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, skip_warmup: bool = False, multimodal: bool = False) -> None:
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None
        self._multimodal = multimodal

        self._do_init(path, skip_warmup, multimodal)

    def __getstate__(self) -> str:
        return self._path

    def __setstate__(self, path: str) -> None:
        self._do_init(path, skip_warmup=True, multimodal=False)

    def __del__(self) -> None:
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: Union[int, np.integer, slice]) -> np.ndarray:
        if isinstance(idx, (int, np.integer)):
            sequence_pointer, sequence_length, sequence_mode = self._index[idx]
            sequence = np.frombuffer(
                self._bin_buffer,
                dtype=self._index.dtype,
                count=sequence_length,
                offset=sequence_pointer,
            )
            return (sequence, sequence_mode) if sequence_mode is not None else sequence
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            sequence_lengths = self._index._sequence_lengths[idx]
            sequence_modes = self._index._sequence_modes[idx] if self._multimodal else None
            sequence_offsets = list(accumulate(sequence_lengths))
            sequences = np.split(
                np.frombuffer(
                    self._bin_buffer,
                    dtype=self._index.dtype,
                    count=sum(sequence_lengths),
                    offset=self._index._sequence_pointers[start],
                ),
                sequence_offsets[:-1],
            )
            return (sequences, sequence_modes) if sequence_modes is not None else sequences
        else:
            raise TypeError("Unexpected type received for idx: {}".format(type(idx)))

    def _do_init(self, path: str, skip_warmup: bool, multimodal: bool) -> None:
        self._path = path

        if not skip_warmup:
            print_rank_0("    warming up index mmap file...")
            self.warmup_mmap_file(get_idx_path(self._path))

        self._index = _IndexReader(get_idx_path(self._path), multimodal)

        if not skip_warmup:
            print_rank_0("    warming up data mmap file...")
            self.warmup_mmap_file(get_bin_path(self._path))

        print_rank_0("    creating np buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(get_bin_path(self._path), mode="r", order="C")

        print_rank_0("    creating memory view of np buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def get(self, idx: int, offset: int = 0, length: Optional[int] = None) -> np.ndarray:
        """Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        sequence_pointer, sequence_length, sequence_mode = self._index[idx]
        if length is None:
            length = sequence_length - offset
        sequence_pointer += offset * DType.size(self._index.dtype)
        sequence = np.frombuffer(
            self._bin_buffer, dtype=self._index.dtype, count=length, offset=sequence_pointer
        )
        return (sequence, sequence_mode) if sequence_mode is not None else sequence

    @property
    def sizes(self) -> np.ndarray:
        return self._index.sizes

    @property
    def doc_idx(self) -> np.ndarray:
        return self._index._document_indices

    def get_doc_idx(self) -> np.ndarray:
        return self._index._document_indices

    def set_doc_idx(self, doc_idx: np.ndarray) -> None:
        self._index._document_indices = doc_idx

    def modes(self) -> np.ndarray:
        return self._index.modes

    @property
    def supports_prefetch(self) -> bool:
        return False

    @staticmethod
    def exists(path_prefix: str) -> bool:
        return os.path.exists(get_idx_path(path_prefix)) and os.path.exists(
            get_bin_path(path_prefix)
        )

    @staticmethod
    def warmup_mmap_file(path: str) -> None:
        with open(path, "rb") as stream:
            while stream.read(100 * 1024 * 1024):
                pass


class MMapIndexedDatasetBuilder(object):
    def __init__(
        self, bin_path: str, dtype: Type[np.number] = np.int32, multimodal: bool = False
    ) -> None:
        self._data_file = open(bin_path, "wb")
        self._dtype = dtype
        self._multimodal = multimodal

        self._sequence_lengths = []
        self._document_indices = [0]
        self._sequence_modes = [] if self._multimodal else None

    def add_item(self, tensor: torch.Tensor, mode: int = 0) -> None:
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order="C"))
        self._sequence_lengths.append(np_array.size)
        if self._multimodal:
            self._sequence_modes.append(mode)

    def add_doc(
        self, tensor: torch.Tensor, lengths: List[int], modes: Optional[List[int]] = None
    ) -> None:
        np_array = np.array(tensor, dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order="C"))
        self._sequence_lengths.extend(lengths)
        self._document_indices.append(len(self._sequence_lengths))
        if self._multimodal:
            self._sequence_modes.extend(modes if modes is not None else [0] * lengths)

    def end_document(self) -> None:
        self._document_indices.append(len(self._sequence_lengths))

    def merge_file_(self, path_prefix: str) -> None:
        # Concatenate index
        index = _IndexReader(get_idx_path(path_prefix), multimodal=self._multimodal)
        assert index.dtype == self._dtype

        offset = len(self._sequence_lengths)
        self._sequence_lengths.extend(index.sizes)
        self._document_indices.extend((offset + index.doc_idx)[1:])

        if self._multimodal:
            self._sequence_modes.extend(index._sequence_modes)

        # Concatenate data
        with open(get_bin_path(path_prefix), "rb") as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, idx_path: str) -> None:
        self._data_file.close()
        with _IndexWriter(idx_path, self._dtype) as writer:
            writer.write(self._sequence_lengths, self._sequence_modes, self._document_indices)


def get_idx_path(path_prefix: str) -> str:
    return path_prefix + ".idx"


def get_bin_path(path_prefix: str) -> str:
    return path_prefix + ".bin"
