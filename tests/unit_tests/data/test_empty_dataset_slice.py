# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import numpy
import pytest

from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize(("count", "mmap"), [(0, False), (2, False), (2, True)])
def test_empty_indexed_dataset_slices(tmp_path, multimodal, count, mmap):
    """Empty contiguous slices work at and beyond the dataset's end."""
    prefix = str(tmp_path / "data")
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=numpy.int32, multimodal=multimodal)
    if count:
        builder.add_document([1, 2, 3], [1, 2], modes=[0, 1] if multimodal else None)
    builder.finalize(prefix + ".idx")
    dataset = IndexedDataset(prefix, multimodal=multimodal, mmap=mmap)
    for index in [slice(count, None), slice(count + 5, None), slice(1, 0), slice(0, 0)]:
        result = dataset[index]
        sequences, modes = result if multimodal else (result, None)
        assert sequences == []
        if multimodal:
            numpy.testing.assert_array_equal(modes, numpy.array([], dtype=numpy.int8))
    with pytest.raises(ValueError, match="contiguous"):
        _ = dataset[::2]
    if count:
        result = dataset[:]
        sequences, modes = result if multimodal else (result, None)
        assert [item.tolist() for item in sequences] == [[1], [2, 3]]
        if multimodal:
            numpy.testing.assert_array_equal(modes, [0, 1])
