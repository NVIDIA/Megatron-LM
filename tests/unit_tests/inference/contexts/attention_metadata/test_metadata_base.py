# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.inference.contexts.attention_context.metadata_base import MetadataBase


class TestMetadataBase:

    def test_init_starts_with_empty_state_data(self):
        """A fresh MetadataBase exposes an empty state_data dict."""
        m = MetadataBase()
        assert m.state_data == {}

    def test_update_is_noop(self):
        """The base update() method runs without error and does not mutate state_data."""
        m = MetadataBase()
        m.update(1, 2, kw=3)
        assert m.state_data == {}

    def test_reset_is_noop(self):
        """The base reset() method runs without error and does not mutate state_data."""
        m = MetadataBase()
        m.state_data = {"key": "value"}
        m.reset()
        # base reset is a no-op; subclasses override.
        assert m.state_data == {"key": "value"}

    def test_str_serialises_state_data(self):
        """__str__ serialises state_data as 'key: value' lines."""
        m = MetadataBase()
        m.state_data = {"a": 1, "b": 2}
        out = str(m)
        assert "a: 1" in out
        assert "b: 2" in out
        # Newline-delimited
        assert out.count("\n") == 1

    def test_tensor_copy_and_pad_non_cumulative(self):
        """Non-cumulative tensors are padded with the explicit pad_value."""
        m = MetadataBase()
        buf = torch.zeros(8, dtype=torch.int32)
        unpadded = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        out = m.tensor_copy_and_pad(
            buf, unpadded, real_batch_size=3, padded_batch_size=6, pad_value=99
        )
        assert torch.equal(out[:3], torch.tensor([1, 2, 3], dtype=torch.int32))
        assert torch.equal(out[3:6], torch.tensor([99, 99, 99], dtype=torch.int32))

    def test_tensor_copy_and_pad_cumulative_uses_last_value(self):
        """Cumulative tensors are padded with the last real value (not pad_value)."""
        m = MetadataBase()
        buf = torch.zeros(6, dtype=torch.int32)
        unpadded = torch.tensor([0, 5, 11, 17], dtype=torch.int32)
        out = m.tensor_copy_and_pad(
            buf, unpadded, real_batch_size=3, padded_batch_size=5, is_cumulative_tensor=True
        )
        # Real values copied:
        assert out[0] == 0 and out[1] == 5 and out[2] == 11
        # Cumulative padding uses unpadded[real_batch_size - 1] = 11
        assert out[3] == 11 and out[4] == 11

    def test_tensor_copy_and_pad_cumulative_zero_batch(self):
        """Cumulative tensor with real_batch_size=0 falls back to pad_value."""
        m = MetadataBase()
        buf = torch.zeros(4, dtype=torch.int32)
        unpadded = torch.tensor([7, 8, 9], dtype=torch.int32)
        out = m.tensor_copy_and_pad(
            buf,
            unpadded,
            real_batch_size=0,
            padded_batch_size=3,
            is_cumulative_tensor=True,
            pad_value=42,
        )
        assert torch.equal(out[:3], torch.tensor([42, 42, 42], dtype=torch.int32))

    def test_tensor_copy_and_pad_asserts_on_invalid_sizes(self):
        """real_batch_size > padded_batch_size triggers an assertion."""
        m = MetadataBase()
        buf = torch.zeros(8, dtype=torch.int32)
        unpadded = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        with pytest.raises(AssertionError):
            m.tensor_copy_and_pad(buf, unpadded, real_batch_size=5, padded_batch_size=3)
