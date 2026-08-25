# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
from types import SimpleNamespace

from megatron.core.inference.text_generation_server.dynamic_text_gen_server.image_preprocessing import (
    _decode_sampled_video_frames,
    _video_sample_indices,
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
