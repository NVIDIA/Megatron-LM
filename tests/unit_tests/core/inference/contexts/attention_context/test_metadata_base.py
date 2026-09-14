# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.inference.contexts.attention_context.metadata_base import MetadataBase


# (is_cumulative, real, padded, pad_value, expected_pad_value_for_assert)
# In cumulative mode, padding repeats unpadded[real - 1] (the last real value).
# In non-cumulative mode, padding uses pad_value.
# In cumulative mode with real == 0, padding falls back to pad_value.
@pytest.mark.parametrize(
    "is_cumulative,real,padded,pad_value,expected_pad",
    [
        # Non-cumulative: pad with pad_value.
        (False, 3, 6, 99, 99),
        # Cumulative with real > 0: pad with unpadded[real - 1] (= 11 for [0, 5, 11, 17]).
        (True, 3, 5, 0, 11),
        # Cumulative with real == 0: falls back to pad_value.
        (True, 0, 3, 42, 42),
    ],
)
def test_tensor_copy_and_pad(is_cumulative, real, padded, pad_value, expected_pad):
    """tensor_copy_and_pad copies the first `real` entries verbatim, then pads to
    `padded` using either the last real value (cumulative) or pad_value."""
    m = MetadataBase()
    buf = torch.zeros(max(padded, 8), dtype=torch.int32)
    unpadded = torch.tensor([0, 5, 11, 17], dtype=torch.int32)
    out = m.tensor_copy_and_pad(
        buf,
        unpadded,
        real_batch_size=real,
        padded_batch_size=padded,
        is_cumulative_tensor=is_cumulative,
        pad_value=pad_value,
    )
    # Real prefix is verbatim.
    assert torch.equal(out[:real], unpadded[:real])
    # Padded suffix is the expected pad value.
    assert torch.all(out[real:padded] == expected_pad)


def test_tensor_copy_and_pad_rejects_oversized_real_batch():
    """A real_batch_size larger than padded_batch_size triggers an assertion (caller bug)."""
    m = MetadataBase()
    with pytest.raises(AssertionError):
        m.tensor_copy_and_pad(
            torch.zeros(8, dtype=torch.int32),
            torch.tensor([1, 2, 3, 4], dtype=torch.int32),
            real_batch_size=5,
            padded_batch_size=3,
        )
