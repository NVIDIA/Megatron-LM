# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import numpy

from megatron.core.datasets.indexed_dataset import _IndexWriter


def test_sequence_pointers():
    writer = _IndexWriter("unused", numpy.int32)
    cases = (
        ([], []),
        ([1, 2, 3], [0, 4, 12]),
        (numpy.array([(1 << 31) - 100, 123], dtype=numpy.int32), [0, 8_589_934_192]),
    )

    for sequence_lengths, expected in cases:
        numpy.testing.assert_array_equal(
            writer._sequence_pointers(sequence_lengths), numpy.array(expected, dtype=numpy.int64)
        )
