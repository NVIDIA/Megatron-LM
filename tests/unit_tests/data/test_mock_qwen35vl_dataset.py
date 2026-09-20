# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MockQwen35VLDataset samples must be a pure function of (seed, split, index).

The dataset previously drew from ambient CPU RNG and ignored ``idx``, so all ranks of a
data-parallel group emitted the same stream and A/B comparisons across parallel layouts
ran on different data.
"""

import pytest
import torch

from examples.multimodal_dev.data.mock import MockQwen35VLDataset


def _payloads(seed=1234, split="train", idx=1):
    """Both random draws of one sample, from a deliberately tiny dataset."""
    ds = MockQwen35VLDataset(
        num_samples=8, seq_length=64, vocab_size=512, image_size=32, seed=seed, split=split
    )
    sample = ds[idx]
    return sample["input_ids"], sample["pixel_values"]


def test_sample_is_a_function_of_its_coordinates():
    torch.manual_seed(0)
    base = _payloads()
    torch.manual_seed(999)  # ambient RNG must not matter
    assert all(torch.equal(a, b) for a, b in zip(base, _payloads()))

    # Each coordinate changes both payloads. (seed=1235, idx=0) catches additive
    # mixing, where idx=1 under seed S collides with idx=0 under seed S+1.
    for other in [dict(idx=2), dict(split="valid"), dict(seed=1235), dict(seed=1235, idx=0)]:
        assert not any(torch.equal(a, b) for a, b in zip(base, _payloads(**other)))


def test_out_of_range_index_is_rejected():
    # The helper's dataset has 8 samples; negative indices are rejected too, not aliased.
    for bad in (8, -1):
        with pytest.raises(IndexError):
            _payloads(idx=bad)


def test_provider_wires_seed_and_separates_splits(monkeypatch):
    """The provider is where the regression lived: it dropped seed and shared one stream."""
    from types import SimpleNamespace

    import megatron.training

    args = SimpleNamespace(
        seed=777,
        total_seq_length=64,
        image_seq_length=256,
        padded_vocab_size=512,
        image_token_id=248056,
        image_size=32,
    )
    monkeypatch.setattr(megatron.training, "get_args", lambda: args, raising=False)

    from examples.multimodal_dev.data.mock import train_valid_test_datasets_provider

    train, valid, test = (
        ds[1]["input_ids"] for ds in train_valid_test_datasets_provider([4, 4, 4])
    )
    assert not torch.equal(train, valid)
    assert not torch.equal(valid, test)
    assert not torch.equal(train, test)

    args.seed = 778
    reseeded, _, _ = train_valid_test_datasets_provider([4, 4, 4])
    assert not torch.equal(reseeded[1]["input_ids"], train)


def test_image_seq_length_below_grid_is_rejected():
    """Clamping instead would only surface later, inside get_rope_index."""
    with pytest.raises(ValueError, match="merged vision tokens"):
        MockQwen35VLDataset(num_samples=1, seq_length=64, image_seq_length=1, image_size=224)
