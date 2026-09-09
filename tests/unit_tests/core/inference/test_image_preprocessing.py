# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import io
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.config import ImageProcessingConfig, VideoProcessingConfig
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.image_preprocessing import (
    _decode_sampled_video_frames,
    _load_frame_sequence_manifest,
    _video_sample_indices,
    preprocess_image_bytes_list,
    preprocess_video_bytes_list,
)


class _FakeFrame:
    def __init__(self, index):
        self.index = index
        self.conversions = 0

    def to_image(self):
        self.conversions += 1
        return self

    def convert(self, mode):
        assert mode == "RGB"
        return self.index


class _FakeContainer:
    def __init__(self, frames, declared_frames):
        self._frames = frames
        self.streams = SimpleNamespace(video=[SimpleNamespace(frames=declared_frames)])

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def decode(self, _stream):
        return iter(self._frames)


def _install_fake_av(monkeypatch, *, total_frames, declared_frames):
    frames = [_FakeFrame(index) for index in range(total_frames)]
    fake_av = SimpleNamespace(open=lambda _: _FakeContainer(frames, declared_frames))
    monkeypatch.setitem(sys.modules, "av", fake_av)
    return frames


def test_video_sample_indices_preserve_temporal_rounding():
    config = SimpleNamespace(num_frames=3, temporal_patch_size=2)

    assert _video_sample_indices(3, config) == [0, 2]


def test_declared_frame_count_converts_only_sampled_frames(monkeypatch):
    frames = _install_fake_av(monkeypatch, total_frames=10, declared_frames=10)
    config = SimpleNamespace(num_frames=3, temporal_patch_size=1)

    sampled = _decode_sampled_video_frames(b"video", config)

    assert sampled == [0, 4, 9]
    assert sum(frame.conversions for frame in frames) == 3


def test_unindexed_stream_counts_then_converts_only_sampled_frames(monkeypatch):
    frames = _install_fake_av(monkeypatch, total_frames=10, declared_frames=0)
    config = SimpleNamespace(num_frames=3, temporal_patch_size=1)

    sampled = _decode_sampled_video_frames(b"video", config)

    assert sampled == [0, 4, 9]
    assert sum(frame.conversions for frame in frames) == 3


def test_frame_sequence_manifest_loads_rgb_copies(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    frame_paths = []
    for index, mode in enumerate(("RGBA", "L")):
        frame_path = tmp_path / f"frame-{index}.png"
        image_module.new(mode, (2, 2)).save(frame_path)
        frame_paths.append(str(frame_path))

    magic = b"frames:"
    payload = magic + json.dumps({"frame_paths": frame_paths}).encode()
    frames = _load_frame_sequence_manifest(payload, magic)

    assert [frame.mode for frame in frames] == ["RGB", "RGB"]
    assert [frame.size for frame in frames] == [(2, 2), (2, 2)]


@pytest.mark.parametrize(
    ("manifest", "error"),
    [
        (b"{", "Invalid frame-sequence manifest JSON"),
        (json.dumps([]).encode(), "must be a JSON object"),
        (json.dumps({"frame_paths": []}).encode(), "requires non-empty string frame_paths"),
    ],
)
def test_frame_sequence_manifest_rejects_invalid_payloads(manifest, error):
    with pytest.raises(ValueError, match=error):
        _load_frame_sequence_manifest(b"frames:" + manifest, b"frames:")


def test_image_bytes_list_preserves_per_image_aspect_ratios():
    image_module = pytest.importorskip("PIL.Image")
    pytest.importorskip("torchvision")
    encoded_images = []
    for size in ((4, 2), (2, 4)):
        buffer = io.BytesIO()
        image_module.new("RGB", size, color="red").save(buffer, format="PNG")
        encoded_images.append(buffer.getvalue())
    config = ImageProcessingConfig(
        patch_dim=2,
        dynamic_resolution=True,
        dynamic_resolution_max_patches=16,
        pixel_mean=[0.0, 0.0, 0.0],
        pixel_std=[1.0, 1.0, 1.0],
    )

    result = preprocess_image_bytes_list(encoded_images, config)

    assert result["imgs"].shape == (1, 4, 12)
    assert result["imgs_sizes"].tolist() == [[2, 4], [4, 2]]


def test_video_manifest_packs_frames_with_one_shared_resolution(monkeypatch):
    image_module = pytest.importorskip("PIL.Image")
    frames = [
        image_module.new("RGB", (8, 4), color="red"),
        image_module.new("RGB", (4, 8), color="blue"),
    ]
    target_shapes = []

    from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
        image_preprocessing,
    )

    monkeypatch.setattr(
        image_preprocessing, "_load_frame_sequence_manifest", lambda _payload, _magic: frames
    )
    monkeypatch.setattr(
        image_preprocessing,
        "dynamic_res_preprocess",
        lambda *_args, **_kwargs: SimpleNamespace(height=4, width=6),
    )

    def fake_preprocess_image(_frame, _config, target_hw=None, device=None):
        assert device is None
        target_shapes.append(target_hw)
        return torch.ones(1, 1, 2), torch.tensor([target_hw], dtype=torch.int32)

    monkeypatch.setattr(image_preprocessing, "preprocess_image", fake_preprocess_image)
    config = VideoProcessingConfig(
        image_config=ImageProcessingConfig(patch_dim=2, dynamic_resolution=True),
        num_frames=2,
        frame_manifest_magic=b"frames:",
    )

    result = preprocess_video_bytes_list([b"frames:{}"], config)

    assert target_shapes == [(4, 6), (4, 6)]
    assert result["imgs"].shape == (1, 2, 2)
    assert result["imgs_sizes"].tolist() == [[4, 6], [4, 6]]
    assert result["num_frames"].tolist() == [2]


def test_video_manifest_rejects_wrong_frame_count(monkeypatch):
    image_module = pytest.importorskip("PIL.Image")
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
        image_preprocessing,
    )

    monkeypatch.setattr(
        image_preprocessing,
        "_load_frame_sequence_manifest",
        lambda _payload, _magic: [image_module.new("RGB", (2, 2))],
    )
    config = VideoProcessingConfig(
        image_config=ImageProcessingConfig(patch_dim=2, dynamic_resolution=True),
        num_frames=2,
        frame_manifest_magic=b"frames:",
    )

    with pytest.raises(ValueError, match="Frame-sequence count must match"):
        preprocess_video_bytes_list([b"frames:{}"], config)
