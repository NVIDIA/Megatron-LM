# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Test for Qwen3-VL multimodal dataset.

##
# Compile megatron.core.datasets.helpers_cpp dependencies before BlendedDataset import
##

import json
import os
from typing import Dict

import pytest
import torch

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.multimodal_dataset import MockMultimodalDataset, MultimodalDatasetConfig
from megatron.core.datasets.qwen3vl_dataset import (
    Qwen3VLDataset,
    Qwen3VLDatasetBuilder,
    Qwen3VLDatasetConfig,
    qwen3vl_collate_fn,
)
from megatron.core.datasets.utils import compile_helpers
from megatron.core.tokenizers import MegatronTokenizer
from tests.unit_tests.test_utilities import Utils

_MOCK_VOCAB_SIZE = 8192


class MockQwen3VLDataset(MockMultimodalDataset):
    """Mock Qwen3-VL multimodal dataset.

    Extends MockMultimodalDataset to produce Qwen3-VL style samples with
    pixel_values (preprocessed patches) and image_grid_thw instead of raw
    image tensors.
    """

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get base multimodal sample (has "image" key with [C, H, W] tensor).
        sample = super().__getitem__(idx)

        image = sample.pop("image")
        _, h, w = image.shape

        # Simulate Qwen3-VL preprocessed pixel_values.
        # Real processor produces [num_patches, patch_dim] but for mock we use
        # a small fixed shape.
        patch_size = 14
        num_h_patches = h // patch_size
        num_w_patches = w // patch_size
        num_patches = num_h_patches * num_w_patches
        patch_dim = 3 * patch_size * patch_size  # 588

        sample["pixel_values"] = torch.zeros(num_patches, patch_dim, dtype=torch.float32)
        sample["image_grid_thw"] = torch.tensor([[1, num_h_patches, num_w_patches]])

        return sample


class TestQwen3VLCollate:
    """Tests for the qwen3vl_collate_fn function."""

    def test_empty_batch(self):
        result = qwen3vl_collate_fn([])
        assert result == {}

    def test_single_sample_text_only(self):
        sample = {
            "tokens": torch.tensor([1, 2, 3, 4, 5]),
            "labels": torch.tensor([2, 3, 4, 5, -100]),
            "loss_mask": torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0]),
            "position_ids": torch.tensor([0, 1, 2, 3, 4]),
        }
        result = qwen3vl_collate_fn([sample])

        assert result["tokens"].shape == torch.Size([1, 5])
        assert result["labels"].shape == torch.Size([1, 5])
        assert result["loss_mask"].shape == torch.Size([1, 5])
        assert result["position_ids"].shape == torch.Size([1, 5])
        assert "pixel_values" not in result

    def test_variable_length_padding(self):
        """Shorter sequences are zero-padded to the max length in the batch."""
        sample_short = {
            "tokens": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([2, 3, -100]),
            "loss_mask": torch.tensor([1.0, 1.0, 0.0]),
            "position_ids": torch.tensor([0, 1, 2]),
        }
        sample_long = {
            "tokens": torch.tensor([10, 20, 30, 40, 50]),
            "labels": torch.tensor([20, 30, 40, 50, -100]),
            "loss_mask": torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0]),
            "position_ids": torch.tensor([0, 1, 2, 3, 4]),
        }
        result = qwen3vl_collate_fn([sample_short, sample_long])

        # Padded to max_len=5.
        assert result["tokens"].shape == torch.Size([2, 5])
        assert torch.equal(result["tokens"][0], torch.tensor([1, 2, 3, 0, 0]))
        assert torch.equal(result["tokens"][1], torch.tensor([10, 20, 30, 40, 50]))
        # Labels padded with -100 (default fill).
        assert torch.equal(result["labels"][0], torch.tensor([2, 3, -100, -100, -100]))

    def test_with_pixel_values(self):
        """Pixel values from different samples are concatenated along dim 0."""
        sample1 = {
            "tokens": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([2, 3, -100]),
            "loss_mask": torch.tensor([1.0, 1.0, 0.0]),
            "position_ids": torch.tensor([0, 1, 2]),
            "pixel_values": torch.randn(10, 1176),
            "image_grid_thw": torch.tensor([[1, 5, 2]]),
        }
        sample2 = {
            "tokens": torch.tensor([4, 5, 6]),
            "labels": torch.tensor([5, 6, -100]),
            "loss_mask": torch.tensor([1.0, 1.0, 0.0]),
            "position_ids": torch.tensor([0, 1, 2]),
            "pixel_values": torch.randn(20, 1176),
            "image_grid_thw": torch.tensor([[1, 10, 2]]),
        }
        result = qwen3vl_collate_fn([sample1, sample2])

        # 10 + 20 patches concatenated.
        assert result["pixel_values"].shape == torch.Size([30, 1176])
        assert result["image_grid_thw"].shape == torch.Size([2, 3])

    def test_mixed_image_and_text_samples(self):
        """Batch with some image samples and some text-only samples."""
        sample_img = {
            "tokens": torch.tensor([1, 2]),
            "labels": torch.tensor([2, -100]),
            "loss_mask": torch.tensor([1.0, 0.0]),
            "position_ids": torch.tensor([0, 1]),
            "pixel_values": torch.randn(5, 1176),
            "image_grid_thw": torch.tensor([[1, 5, 1]]),
        }
        sample_txt = {
            "tokens": torch.tensor([3, 4]),
            "labels": torch.tensor([4, -100]),
            "loss_mask": torch.tensor([1.0, 0.0]),
            "position_ids": torch.tensor([0, 1]),
        }
        result = qwen3vl_collate_fn([sample_img, sample_txt])

        assert "pixel_values" in result
        assert result["pixel_values"].shape == torch.Size([5, 1176])

    def test_none_pixel_values_ignored(self):
        """Samples with pixel_values=None are treated as text-only."""
        sample = {
            "tokens": torch.tensor([1, 2]),
            "labels": torch.tensor([2, -100]),
            "loss_mask": torch.tensor([1.0, 0.0]),
            "position_ids": torch.tensor([0, 1]),
            "pixel_values": None,
        }
        result = qwen3vl_collate_fn([sample])
        assert "pixel_values" not in result


class TestQwen3VLDatasetConfig:
    """Tests for the Qwen3VLDatasetConfig dataclass."""

    def test_defaults(self):
        config = Qwen3VLDatasetConfig()
        assert config.jsonl_paths is None
        assert config.image_base_dir is None
        assert config.img_h == 384
        assert config.img_w == 384
        assert config.processor_name == "Qwen/Qwen3-VL-8B-Instruct"
        assert config.sequence_length == 4096
        assert config.random_seed == 42

    def test_custom_values(self):
        config = Qwen3VLDatasetConfig(
            jsonl_paths=["/data/train.jsonl"],
            image_base_dir="/data/images",
            img_h=224,
            img_w=224,
            sequence_length=2048,
            random_seed=123,
        )
        assert config.jsonl_paths == ["/data/train.jsonl"]
        assert config.image_base_dir == "/data/images"
        assert config.img_h == 224
        assert config.img_w == 224
        assert config.sequence_length == 2048


class TestParseConversation:
    """Tests for Qwen3VLDataset._parse_conversation.

    Uses a stub instance since the method only operates on its text argument.
    """

    def _make_stub(self):
        return object.__new__(Qwen3VLDataset)

    def test_single_turn(self):
        ds = self._make_stub()
        text = "User: What is this?\n\nAssistant: This is a test."
        messages = ds._parse_conversation(text)

        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "What is this?"}
        assert messages[1] == {"role": "assistant", "content": "This is a test."}

    def test_multi_turn(self):
        ds = self._make_stub()
        text = (
            "User: Hello\n\n"
            "Assistant: Hi there!\n\n"
            "User: How are you?\n\n"
            "Assistant: I'm fine."
        )
        messages = ds._parse_conversation(text)

        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"
        assert messages[0]["content"] == "Hello"
        assert messages[3]["content"] == "I'm fine."

    def test_with_image_token(self):
        ds = self._make_stub()
        text = "User: <image> Describe this image.\n\nAssistant: A cat."
        messages = ds._parse_conversation(text)

        assert len(messages) == 2
        assert "<image>" in messages[0]["content"]
        assert messages[1]["content"] == "A cat."

    def test_empty_text(self):
        ds = self._make_stub()
        assert ds._parse_conversation("") == []

    def test_no_role_markers(self):
        ds = self._make_stub()
        assert ds._parse_conversation("Just plain text without roles.") == []

    def test_multiline_content(self):
        ds = self._make_stub()
        text = "User: Line one\nLine two\nLine three\n\nAssistant: Response."
        messages = ds._parse_conversation(text)

        assert len(messages) == 2
        assert "Line one\nLine two\nLine three" == messages[0]["content"]


class TestLoadJsonl:
    """Tests for Qwen3VLDataset._load_jsonl."""

    def _make_stub(self):
        stub = object.__new__(Qwen3VLDataset)
        stub.samples = []
        return stub

    def test_load_valid_jsonl(self, tmp_path):
        jsonl_file = tmp_path / "data.jsonl"
        samples = [
            {"text": "User: Hello\nAssistant: Hi", "images": []},
            {"text": "User: <image> What?\nAssistant: A cat.", "images": ["img.jpg"]},
        ]
        jsonl_file.write_text("\n".join(json.dumps(s) for s in samples))

        ds = self._make_stub()
        ds._load_jsonl(str(jsonl_file))

        assert len(ds.samples) == 2
        assert ds.samples[0]["text"] == samples[0]["text"]
        assert ds.samples[1]["images"] == ["img.jpg"]
        assert ds.samples[0]["_jsonl_dir"] == str(tmp_path)

    def test_skip_empty_lines(self, tmp_path):
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text('{"text": "line1"}\n\n\n{"text": "line2"}\n')

        ds = self._make_stub()
        ds._load_jsonl(str(jsonl_file))
        assert len(ds.samples) == 2

    def test_skip_invalid_json(self, tmp_path):
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text('{"text": "valid"}\nnot json\n{"text": "also valid"}\n')

        ds = self._make_stub()
        ds._load_jsonl(str(jsonl_file))
        assert len(ds.samples) == 2

    def test_empty_file(self, tmp_path):
        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.write_text("")

        ds = self._make_stub()
        ds._load_jsonl(str(jsonl_file))
        assert len(ds.samples) == 0

    def test_len(self, tmp_path):
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text('{"text":"a"}\n{"text":"b"}\n{"text":"c"}\n')

        ds = self._make_stub()
        ds._load_jsonl(str(jsonl_file))
        assert len(ds) == 3


class TestResolveImagePath:
    """Tests for Qwen3VLDataset._resolve_image_path_uncached."""

    def _make_stub(self, image_base_dir=None):
        stub = object.__new__(Qwen3VLDataset)
        stub.config = Qwen3VLDatasetConfig(image_base_dir=image_base_dir)
        stub._dir_listing_cache = {}
        stub._resolved_paths = {}
        return stub

    def test_absolute_path(self):
        ds = self._make_stub()
        result = ds._resolve_image_path_uncached("/absolute/path/img.jpg", "/some/dir")
        assert result == "/absolute/path/img.jpg"

    def test_relative_to_image_base_dir(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        (img_dir / "photo.jpg").touch()

        ds = self._make_stub(image_base_dir=str(img_dir))
        result = ds._resolve_image_path_uncached("photo.jpg", str(tmp_path))
        assert result == str(img_dir / "photo.jpg")

    def test_relative_to_jsonl_dir(self, tmp_path):
        (tmp_path / "photo.jpg").touch()

        ds = self._make_stub()
        result = ds._resolve_image_path_uncached("photo.jpg", str(tmp_path))
        assert result == str(tmp_path / "photo.jpg")

    def test_source_images_subdir(self, tmp_path):
        src_dir = tmp_path / "source_images"
        src_dir.mkdir()
        (src_dir / "photo.jpg").touch()

        ds = self._make_stub()
        result = ds._resolve_image_path_uncached("photo.jpg", str(tmp_path))
        assert result == str(src_dir / "photo.jpg")

    def test_extracted_subdir(self, tmp_path):
        extracted = tmp_path / "source_images" / "extracted"
        extracted.mkdir(parents=True)
        (extracted / "photo.jpg").touch()

        ds = self._make_stub()
        result = ds._resolve_image_path_uncached("photo.jpg", str(tmp_path))
        assert result == str(extracted / "photo.jpg")

    def test_extracted_archive_subdir(self, tmp_path):
        archive_dir = tmp_path / "source_images" / "extracted" / "archive1"
        archive_dir.mkdir(parents=True)
        (archive_dir / "photo.jpg").touch()

        ds = self._make_stub()
        result = ds._resolve_image_path_uncached("photo.jpg", str(tmp_path))
        assert result == str(archive_dir / "photo.jpg")

    def test_fallback_returns_original(self):
        ds = self._make_stub()
        result = ds._resolve_image_path_uncached("nonexistent.jpg", "/no/such/dir")
        assert result == "nonexistent.jpg"

    def test_cached_resolve_uses_cache(self, tmp_path):
        (tmp_path / "img.jpg").touch()

        ds = self._make_stub()
        ds._resolved_paths[("img.jpg", str(tmp_path))] = "/cached/path.jpg"

        result = ds._resolve_image_path("img.jpg", str(tmp_path))
        assert result == "/cached/path.jpg"

    def test_cached_resolve_fallback(self, tmp_path):
        """Cache miss falls back to uncached resolution."""
        (tmp_path / "img.jpg").touch()

        ds = self._make_stub()
        result = ds._resolve_image_path("img.jpg", str(tmp_path))
        assert result == str(tmp_path / "img.jpg")
        # Now cached.
        assert ("img.jpg", str(tmp_path)) in ds._resolved_paths


class TestCachedListdir:
    """Tests for Qwen3VLDataset._cached_listdir."""

    def _make_stub(self):
        stub = object.__new__(Qwen3VLDataset)
        stub._dir_listing_cache = {}
        return stub

    def test_lists_directory(self, tmp_path):
        (tmp_path / "a.txt").touch()
        (tmp_path / "b.txt").touch()

        ds = self._make_stub()
        result = ds._cached_listdir(str(tmp_path))
        assert sorted(result) == ["a.txt", "b.txt"]

    def test_returns_cached_result(self, tmp_path):
        (tmp_path / "file.txt").touch()

        ds = self._make_stub()
        result1 = ds._cached_listdir(str(tmp_path))
        result2 = ds._cached_listdir(str(tmp_path))
        assert result1 is result2  # Same object from cache.

    def test_nonexistent_dir(self):
        ds = self._make_stub()
        result = ds._cached_listdir("/nonexistent/dir/path")
        assert result == []


class TestQwen3VLDatasetBuilder:
    """Tests for Qwen3VLDatasetBuilder._get_jsonl_paths."""

    def test_get_jsonl_paths(self, tmp_path):
        # Create the JSONL file that the path conversion should find.
        jsonl_file = tmp_path / "chartqa_train.jsonl"
        jsonl_file.write_text('{"text": "sample"}\n')

        blend_config = {
            "train": [0.5, str(tmp_path / "preprocessed" / "chartqa_train_text_document")]
        }
        blend_path = tmp_path / "blend.json"
        blend_path.write_text(json.dumps(blend_config))

        config = Qwen3VLDatasetConfig()
        builder = Qwen3VLDatasetBuilder(
            config=config, blend_path=str(blend_path), train_val_test_num_samples=[100, 10, 10]
        )

        paths = builder._get_jsonl_paths("train")
        assert str(jsonl_file) in paths

    def test_missing_jsonl_files(self, tmp_path):
        blend_config = {"train": [1.0, "/nonexistent/path_text_document"]}
        blend_path = tmp_path / "blend.json"
        blend_path.write_text(json.dumps(blend_config))

        config = Qwen3VLDatasetConfig()
        builder = Qwen3VLDatasetBuilder(
            config=config, blend_path=str(blend_path), train_val_test_num_samples=[100, 10, 10]
        )

        assert builder._get_jsonl_paths("train") == []

    def test_empty_split(self, tmp_path):
        blend_config = {"train": []}
        blend_path = tmp_path / "blend.json"
        blend_path.write_text(json.dumps(blend_config))

        config = Qwen3VLDatasetConfig()
        builder = Qwen3VLDatasetBuilder(
            config=config, blend_path=str(blend_path), train_val_test_num_samples=[100, 10, 10]
        )

        assert builder._get_jsonl_paths("valid") == []

    def test_multiple_paths(self, tmp_path):
        # Two JSONL files.
        (tmp_path / "ds1_train.jsonl").write_text('{"text":"a"}\n')
        (tmp_path / "ds2_train.jsonl").write_text('{"text":"b"}\n')

        blend_config = {
            "train": [
                0.5,
                str(tmp_path / "preprocessed" / "ds1_train_text_document"),
                0.5,
                str(tmp_path / "preprocessed" / "ds2_train_text_document"),
            ]
        }
        blend_path = tmp_path / "blend.json"
        blend_path.write_text(json.dumps(blend_config))

        config = Qwen3VLDatasetConfig()
        builder = Qwen3VLDatasetBuilder(
            config=config, blend_path=str(blend_path), train_val_test_num_samples=[100, 10, 10]
        )

        paths = builder._get_jsonl_paths("train")
        assert len(paths) == 2


def test_mock_qwen3vl_dataset():
    """Test MockQwen3VLDataset via BlendedMegatronDatasetBuilder.

    Follows the same pattern as test_multimodal_dataset.py but validates
    Qwen3-VL specific output fields (pixel_values, image_grid_thw).
    """
    if torch.distributed.is_available():
        Utils.initialize_distributed()
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path={"library": "null-text"}, vocab_size=_MOCK_VOCAB_SIZE
    )
    config = MultimodalDatasetConfig(
        random_seed=1234,
        sequence_length=1024,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=True,
        image_h=336,
        image_w=336,
        split="990,9,1",
        tokenizer=tokenizer,
        mid_level_dataset_surplus=0.005,
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockQwen3VLDataset, [100, 100, 100], lambda: True, config
    ).build()

    patch_size = 14
    num_h = 336 // patch_size  # 24
    num_w = 336 // patch_size  # 24
    expected_patches = num_h * num_w  # 576
    expected_patch_dim = 3 * patch_size * patch_size  # 588

    for ds in datasets:
        sample = ds[0]

        # Should have Qwen3-VL keys, not raw "image".
        assert "image" not in sample
        assert "tokens" in sample
        assert "pixel_values" in sample
        assert "image_grid_thw" in sample

        # Validate shapes.
        assert sample["pixel_values"].shape == torch.Size([expected_patches, expected_patch_dim])
        assert sample["image_grid_thw"].shape == torch.Size([1, 3])
        assert sample["image_grid_thw"][0, 0].item() == 1  # temporal
        assert sample["image_grid_thw"][0, 1].item() == num_h
        assert sample["image_grid_thw"][0, 2].item() == num_w

    # Test collation of mock samples.
    batch = [datasets[0][i] for i in range(3)]
    collated = qwen3vl_collate_fn(batch)

    assert collated["tokens"].shape[0] == 3  # batch size
    assert collated["pixel_values"].shape == torch.Size([expected_patches * 3, expected_patch_dim])
    assert collated["image_grid_thw"].shape == torch.Size([3, 3])
