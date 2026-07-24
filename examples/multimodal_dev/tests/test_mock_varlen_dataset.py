# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-only tests for the packed-window Qwen3.5-VL mock dataset."""

from types import SimpleNamespace

import pytest
import torch

import megatron.training
from examples.multimodal_dev.data.mock_varlen import (
    PackedWindowQwen35VLDataset,
    train_valid_test_varlen_datasets_provider,
)

_IMAGE_TOKEN_ID = 97
_VIDEO_TOKEN_ID = 98
_VISION_START_TOKEN_ID = 96
_VOCAB_SIZE = 100
_PATCH_SIZE = 2
_TEMPORAL_PATCH_SIZE = 2
_SPATIAL_MERGE_SIZE = 2
_PIXEL_DIM = 24

# Mixed stochastic profile: docs shorter than the 64-token window, so
# multi-segment windows and boundary handling are exercised.
_WINDOW_CONFIG = {
    "doc_length": {
        "components": [
            {"name": "short", "weight": 3, "min": 8, "max": 63, "mean": 24, "sigma": 0.8},
            {"name": "long", "weight": 1, "min": 64, "max": 256, "mean": 128, "sigma": 0.5},
        ]
    },
    "text_only_document_probability": 0.4,
    "image_poisson_rate_per_1k_text_tokens": 50,
    "image_density_gamma_shape": 1.0,
    # Miniature 64-token test windows naturally exceed the production fill
    # ceiling; distortion accounting is covered by the kernel tests.
    "max_boundary_fill_fraction": None,
}

# Deterministic profile (sigma-0 constant components): every document is
# exactly 24 text tokens, always interleaved.
_CONSTANT_CONFIG = {
    "doc_length": {
        "components": [{"name": "fixed", "weight": 1, "min": 24, "max": 24, "mean": 24, "sigma": 0}]
    },
    "text_only_document_probability": 0.0,
    "image_poisson_rate_per_1k_text_tokens": 50,
    "max_boundary_fill_fraction": None,
}


def _bucket_config(*resolutions, weights=None):
    config = {"mode": "buckets", "resolutions": [list(size) for size in resolutions]}
    if weights is not None:
        config["weights"] = list(weights)
    return config


def _make_dataset(**overrides):
    kwargs = {
        "num_samples": 32,
        "seq_length": 64,
        "seed": 1234,
        "vocab_size": _VOCAB_SIZE,
        "image_token_id": _IMAGE_TOKEN_ID,
        "video_token_id": _VIDEO_TOKEN_ID,
        "vision_start_token_id": _VISION_START_TOKEN_ID,
        "window_config": _WINDOW_CONFIG,
        "image_size_config": _bucket_config((8, 8), (8, 16)),
        "patch_size": _PATCH_SIZE,
        "temporal_patch_size": _TEMPORAL_PATCH_SIZE,
        "spatial_merge_size": _SPATIAL_MERGE_SIZE,
    }
    kwargs.update(overrides)
    return PackedWindowQwen35VLDataset(**kwargs)


def _assert_samples_equal(lhs, rhs):
    assert lhs.keys() == rhs.keys()
    for key in lhs:
        assert torch.equal(lhs[key], rhs[key]), key


def _assert_vision_contract(sample):
    """Vision starts, image blocks, grids, and pixel rows stay consistent."""
    input_ids = sample["input_ids"]
    grids = sample["image_grid_thw"]
    pixel_values = sample["pixel_values"]
    vision_starts = torch.where(input_ids == _VISION_START_TOKEN_ID)[0].tolist()

    assert tuple(grids.shape) == (len(vision_starts), 3)

    patch_offset = 0
    expected_image_tokens = 0
    block_ends = []
    for vision_start, (t, h, w) in zip(vision_starts, grids.tolist()):
        num_patches = t * h * w
        num_image_tokens = t * (h // _SPATIAL_MERGE_SIZE) * (w // _SPATIAL_MERGE_SIZE)
        image_start = vision_start + 1
        image_end = image_start + num_image_tokens

        assert torch.all(input_ids[image_start:image_end] == _IMAGE_TOKEN_ID)
        assert pixel_values[patch_offset : patch_offset + num_patches].shape == (
            num_patches,
            _PIXEL_DIM,
        )
        patch_offset += num_patches
        expected_image_tokens += num_image_tokens
        block_ends.append(image_end)

    assert patch_offset == pixel_values.shape[0]
    assert expected_image_tokens == int((input_ids == _IMAGE_TOKEN_ID).sum().item())
    assert all(end <= next_start for end, next_start in zip(block_ends, vision_starts[1:]))


class TestPackedWindowDataset:
    def test_windows_are_exactly_seq_length_with_matching_seq_lens(self):
        dataset = _make_dataset()
        saw_atoms = saw_multi_segment = False
        for idx in range(len(dataset)):
            sample = dataset[idx]
            assert sample.keys() == {
                "input_ids",
                "labels",
                "loss_mask",
                "pixel_values",
                "image_grid_thw",
                "seq_lens",
            }
            assert sample["input_ids"].shape == (64,)
            assert sample["labels"].shape == sample["loss_mask"].shape == (64,)
            assert int(sample["seq_lens"].sum().item()) == 64
            assert (sample["seq_lens"] > 0).all()
            _assert_vision_contract(sample)
            saw_atoms |= bool(sample["image_grid_thw"].shape[0])
            saw_multi_segment |= sample["seq_lens"].numel() > 1
        assert saw_atoms and saw_multi_segment

    def test_segment_final_positions_have_no_targets(self):
        dataset = _make_dataset()
        for idx in range(len(dataset)):
            sample = dataset[idx]
            boundary = 0
            for segment_length in sample["seq_lens"].tolist():
                boundary += segment_length
                assert sample["labels"][boundary - 1].item() == -100
                assert sample["loss_mask"][boundary - 1].item() == 0.0

    def test_image_targets_are_masked(self):
        dataset = _make_dataset()
        for idx in range(len(dataset)):
            sample = dataset[idx]
            for special_id in (_IMAGE_TOKEN_ID, _VIDEO_TOKEN_ID, _VISION_START_TOKEN_ID):
                assert not (sample["labels"] == special_id).any()

    def test_constant_components_give_deterministic_segment_grid(self):
        # sigma-0 docs of exactly 24 text tokens: segment structure is fully
        # determined by the walk (text + inserted atoms), not by length draws.
        dataset = _make_dataset(window_config=_CONSTANT_CONFIG, num_samples=8)
        for idx in range(8):
            sample = dataset[idx]
            assert int(sample["seq_lens"].sum().item()) == 64
            _assert_vision_contract(sample)

    def test_windows_are_deterministic_and_access_order_independent(self):
        lhs = _make_dataset()
        rhs = _make_dataset()
        for idx in reversed(range(8)):  # access in a different order
            _assert_samples_equal(lhs[idx], rhs[idx])

    def test_requires_bucket_image_size_config(self):
        with pytest.raises(ValueError, match="buckets"):
            _make_dataset(image_size_config=None)

    def test_streaming_mode_omits_pixel_values(self):
        eager = _make_dataset()
        streaming = _make_dataset(streaming_pixels=True)
        for idx in range(len(streaming)):
            sample = streaming[idx]
            assert "pixel_values" not in sample
            assert sample.keys() == {
                "input_ids",
                "labels",
                "loss_mask",
                "image_grid_thw",
                "seq_lens",
            }
            # Geometry and tokens are identical to the eager profile.
            reference = eager[idx]
            for key in sample:
                assert torch.equal(sample[key], reference[key]), key

    @pytest.mark.parametrize(
        ("image_size_config", "message"),
        [
            ({"mode": "buckets", "resolutions": [[8]]}, "two positive integers"),
            ({"mode": "buckets", "resolutions": [[8, 8, 8]]}, "two positive integers"),
            ({"mode": "buckets", "resolutions": [[0, 8]]}, "two positive integers"),
            ({"mode": "buckets", "resolutions": [[-8, 8]]}, "two positive integers"),
            ({"mode": "buckets", "resolutions": [[8.5, 8]]}, "two positive integers"),
            (
                {"mode": "buckets", "resolutions": [[8, 8], [8, 16]], "weights": [1]},
                "match 'resolutions' in length",
            ),
            ({"mode": "buckets", "resolutions": [[8, 8]], "weights": [float("nan")]}, "finite"),
        ],
    )
    def test_rejects_malformed_buckets(self, image_size_config, message):
        with pytest.raises(ValueError, match=message):
            _make_dataset(image_size_config=image_size_config)

    def test_rejects_zero_patch_budget(self):
        with pytest.raises(ValueError, match="positive integer or None"):
            _make_dataset(max_raw_patches_per_window=0)

    def test_rejects_unusable_vocabulary(self):
        with pytest.raises(ValueError, match="token IDs must be in"):
            _make_dataset(vocab_size=64)

    def test_over_budget_window_fails_before_pixels_materialize(self, monkeypatch):
        dataset = _make_dataset(max_raw_patches_per_window=1)

        def _no_pixels(*args, **kwargs):
            raise AssertionError("pixel tensor was materialized before the budget check")

        monkeypatch.setattr(torch.Tensor, "normal_", _no_pixels)
        with pytest.raises(ValueError, match="max_raw_patches_per_window"):
            for idx in range(len(dataset)):
                dataset[idx]  # first image-bearing window must fail pre-randn

    def test_virtual_length_decouples_from_plan_pool(self):
        # A training-schedule-sized virtual length must not build a
        # training-schedule-sized plan corpus.
        dataset = _make_dataset(
            num_samples=1_000_000, window_config={**_WINDOW_CONFIG, "plan_pool_windows": 8}
        )
        assert len(dataset) == 1_000_000
        assert dataset.plan_pool_windows == 8
        assert len(dataset.plan) == 8
        base, wrapped = dataset[3], dataset[3 + 8]
        # Same pool layout...
        assert torch.equal(base["seq_lens"], wrapped["seq_lens"])
        assert torch.equal(base["image_grid_thw"], wrapped["image_grid_thw"])
        # ...but content stays keyed by the virtual index.
        assert not torch.equal(base["input_ids"], wrapped["input_ids"])


def _provider_args(**overrides):
    args = SimpleNamespace(
        use_varlen_dataset=False,
        sequence_packing_scheduler=None,
        use_packed_sequence=True,
        use_vanilla_collate_fn=True,
        micro_batch_size=1,
        total_seq_length=64,
        seq_length=64,
        varlen_mock_dataset_config_json=(
            '{"mode":"packed_window",'
            '"doc_length":{"components":['
            '{"name":"short","weight":3,"min":8,"max":63,"mean":24,"sigma":0.8},'
            '{"name":"long","weight":1,"min":64,"max":256,"mean":128,"sigma":0.5}]},'
            '"text_only_document_probability":0.4,'
            '"image_poisson_rate_per_1k_text_tokens":50,'
            '"image_density_gamma_shape":1.0,'
            '"max_boundary_fill_fraction":null}'
        ),
        mock_image_size_config_json='{"mode":"buckets","resolutions":[[32,32],[64,32]]}',
        padded_vocab_size=248320,
        image_token_id=248056,
        seed=2026,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestPackedWindowProvider:
    def test_provides_three_splits_with_distinct_seeds(self, monkeypatch):
        monkeypatch.setattr(megatron.training, "get_args", lambda: _provider_args())
        train_ds, val_ds, test_ds = train_valid_test_varlen_datasets_provider((2, 1, 1))
        for dataset in (train_ds, val_ds, test_ds):
            assert isinstance(dataset, PackedWindowQwen35VLDataset)
        sample = train_ds[0]
        assert sample["input_ids"].shape == (64,)
        assert int(sample["seq_lens"].sum().item()) == 64
        assert train_ds.seed != val_ds.seed != test_ds.seed

    def test_rejects_micro_batch_size_above_one(self, monkeypatch):
        monkeypatch.setattr(
            megatron.training, "get_args", lambda: _provider_args(micro_batch_size=2)
        )
        with pytest.raises(ValueError, match="micro_batch_size == 1"):
            train_valid_test_varlen_datasets_provider((1, 1, 1))

    @pytest.mark.parametrize(
        "config",
        [
            None,
            '{"mode":"distribution","type":"lognormal","min_seq_len":32,"max_seq_len":64,'
            '"mean_seq_len":48,"lognormal_sigma":1.1}',
            '{"mode":"file","path":"lengths.csv"}',
        ],
    )
    def test_rejects_removed_legacy_modes(self, monkeypatch, config):
        monkeypatch.setattr(
            megatron.training,
            "get_args",
            lambda: _provider_args(varlen_mock_dataset_config_json=config),
        )
        with pytest.raises(ValueError, match="packed_window"):
            train_valid_test_varlen_datasets_provider((1, 1, 1))

    def test_rejects_collate_and_scheduler_conflicts(self, monkeypatch):
        monkeypatch.setattr(
            megatron.training, "get_args", lambda: _provider_args(use_vanilla_collate_fn=False)
        )
        with pytest.raises(ValueError, match="--use-vanilla-collate-fn"):
            train_valid_test_varlen_datasets_provider((1, 1, 1))
        monkeypatch.setattr(
            megatron.training,
            "get_args",
            lambda: _provider_args(sequence_packing_scheduler="dp_balanced"),
        )
        with pytest.raises(ValueError, match="--sequence-packing-scheduler"):
            train_valid_test_varlen_datasets_provider((1, 1, 1))

    def test_streaming_requires_chunked_encoder(self, monkeypatch):
        monkeypatch.setattr(
            megatron.training,
            "get_args",
            lambda: _provider_args(
                mock_synthetic_streaming_pixels=True, vision_encoder_chunk_patches=0
            ),
        )
        with pytest.raises(ValueError, match="--vision-encoder-chunk-patches"):
            train_valid_test_varlen_datasets_provider((1, 1, 1))

    def test_streaming_flag_reaches_the_dataset(self, monkeypatch):
        monkeypatch.setattr(
            megatron.training,
            "get_args",
            lambda: _provider_args(
                mock_synthetic_streaming_pixels=True, vision_encoder_chunk_patches=1024
            ),
        )
        train_ds, _, _ = train_valid_test_varlen_datasets_provider((2, 1, 1))
        assert train_ds.streaming_pixels
        assert "pixel_values" not in train_ds[0]

    def test_requires_packed_sequence(self, monkeypatch):
        monkeypatch.setattr(
            megatron.training, "get_args", lambda: _provider_args(use_packed_sequence=False)
        )
        with pytest.raises(ValueError, match="--use-packed-sequence"):
            train_valid_test_varlen_datasets_provider((1, 1, 1))

    def test_provider_wires_the_patch_budget_into_the_dataset(self, monkeypatch):
        monkeypatch.setattr(
            megatron.training,
            "get_args",
            lambda: _provider_args(max_vision_patches_per_microbatch=4096),
        )
        train_ds, _, _ = train_valid_test_varlen_datasets_provider((2, 1, 1))
        assert train_ds.max_raw_patches_per_window == 4096

    def test_requires_total_seq_length_to_match(self, monkeypatch):
        monkeypatch.setattr(
            megatron.training, "get_args", lambda: _provider_args(total_seq_length=128)
        )
        with pytest.raises(ValueError, match="--total-seq-length to equal"):
            train_valid_test_varlen_datasets_provider((1, 1, 1))
