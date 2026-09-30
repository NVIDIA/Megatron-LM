# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import numpy
import pytest

from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder


@pytest.mark.parametrize("modes", [[], [0], [1], [0, 1], [0, 0]])
def test_multimodal_sequence_modes(tmp_path, modes):
    """Modes are arrays whose presence must not depend on their truth value."""
    prefix = str(tmp_path / "multimodal")
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=numpy.int32, multimodal=True)
    builder.add_document(list(range(len(modes))), [1] * len(modes), modes=modes)
    builder.finalize(prefix + ".idx")
    dataset = IndexedDataset(prefix, multimodal=True, mmap=bool(modes))
    numpy.testing.assert_array_equal(dataset.sequence_modes, numpy.asarray(modes, dtype=numpy.int8))


def test_unimodal_sequence_modes_are_unavailable(tmp_path):
    """The modes property still rejects datasets that do not store modes."""
    prefix = str(tmp_path / "unimodal")
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=numpy.int32)
    builder.add_document([1], [1])
    builder.finalize(prefix + ".idx")
    dataset = IndexedDataset(prefix)
    with pytest.raises(AssertionError):
        _ = dataset.sequence_modes
