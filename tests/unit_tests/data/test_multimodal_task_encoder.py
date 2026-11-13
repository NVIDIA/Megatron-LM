# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import importlib
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.multimodal.data_loading import task_encoder
from examples.multimodal.data_loading.task_encoder import (
    MultiModalTaskEncoder,
    _normalize_thinking_trace,
)


@pytest.mark.parametrize("consumer", ["tiling", "task_encoder", "legacy_task_encoder"])
def test_preprocessing_rejects_non_vision_model(monkeypatch, run_config, consumer):
    # The legacy example imports sibling modules by name. Set its script path
    # explicitly instead of relying on another test module to modify sys.path.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples/multimodal"))
    from examples.multimodal import dataset_helpers
    from examples.multimodal.data_loading.image_processing import create_image_tiling_strategy

    args = SimpleNamespace()
    monkeypatch.setattr(task_encoder, "get_args", lambda: args)
    monkeypatch.setattr(dataset_helpers, "get_args", lambda: args)
    calls = {
        "tiling": lambda: create_image_tiling_strategy(args, run_config),
        "task_encoder": MultiModalTaskEncoder,
        "legacy_task_encoder": dataset_helpers.TaskEncoder,
    }
    with pytest.raises(TypeError, match="Expected cfg.model.*got GPTModelConfig"):
        calls[consumer]()


def _encoder(thread_count=8):
    encoder = object.__new__(MultiModalTaskEncoder)
    encoder.args = SimpleNamespace(video_decode_thread_count=thread_count)
    return encoder


@pytest.mark.parametrize("sequence_bound,has_trainable_tokens", [(3, False), (4, True)])
def test_legacy_encoder_limits_targets_to_model_sequence_bound(
    monkeypatch, run_config, sequence_bound, has_trainable_tokens
):
    from megatron.training.config.training_config import MultiModalTokenizerConfig
    from megatron.training.models import LLaVAModelConfig

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples/multimodal"))
    from examples.multimodal import dataset_helpers

    transformer = run_config.model.transformer
    transformer.max_seq_length_per_sample = sequence_bound
    run_config.model = LLaVAModelConfig(transformer=transformer, seq_length=sequence_bound)
    run_config.tokenizer = MultiModalTokenizerConfig()
    args = SimpleNamespace(
        prompt_path=None,
        packing_buffer_size=None,
        vision_model_type="clip",
        pixel_shuffle=False,
        use_tile_tags=False,
        max_num_tiles=1,
        use_area_weighted_aspect_ratio=False,
        dynamic_resolution=False,
        dynamic_resolution_min_patches=1,
        dynamic_resolution_min_side=1,
        match_tiling_dynamic_resolution=False,
        thumbnail_area_threshold=1,
    )
    monkeypatch.setattr(dataset_helpers, "get_args", lambda: args)
    monkeypatch.setattr(
        dataset_helpers, "get_tokenizer", lambda: SimpleNamespace(image_token_index=-200, pad=0)
    )
    monkeypatch.setattr(dataset_helpers, "get_num_image_embeddings", lambda **kwargs: 1)
    monkeypatch.setattr(dataset_helpers, "ImageTransform", lambda *args, **kwargs: None)
    encoder = dataset_helpers.TaskEncoder(image_token_budget=16)
    assert not hasattr(encoder, "cfg")
    # A trainable next-token target is just outside the shorter execution window.
    result = encoder.target_has_trainable_tokens(
        np.array([1, 2, 3, 4, 5]), [], np.array([-100, -100, -100, -100, 9]), []
    )
    assert bool(result) is has_trainable_tokens


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("<think>reasoning</think>answer", "<think>\nreasoning</think>answer"),
        (
            "<think>\n  multi-line\nreasoning  \n</think>\n\nanswer",
            "<think>\nmulti-line\nreasoning</think>answer",
        ),
        ("<think>  </think>\nanswer", "<think></think>answer"),
    ],
)
def test_normalize_thinking_trace_ultra(content, expected):
    assert _normalize_thinking_trace(content, thinking_trace_format="ultra") == expected


def test_normalize_thinking_trace_uses_nemotron6_separator():
    assert (
        _normalize_thinking_trace(
            "<think>reasoning</think>answer", thinking_trace_format="normalized"
        )
        == "<think>\nreasoning\n</think>\nanswer"
    )


class _FakeVideoStream:
    def __init__(self):
        self.codec_context = SimpleNamespace(thread_count=None, thread_type=None)
        self.type = "video"


class _FakeContainer:
    def __init__(self):
        self.closed = False
        self.video_stream = _FakeVideoStream()
        self.streams = [self.video_stream]

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _decoder_module():
    return importlib.import_module(task_encoder.AVDecoder.__module__)


def _fake_decoder(decoder_module, frames):
    class FakeEnergonDecoder:
        def get_clips(self, **kwargs):
            with decoder_module.av_open(io.BytesIO(b"video")):
                pass
            return SimpleNamespace(video_clips=[[frame] for frame in frames])

    return FakeEnergonDecoder()


def test_video_decode_configures_threads_and_restores_hook(monkeypatch):
    container = _FakeContainer()
    decoder_module = _decoder_module()

    def original_av_open(*args, **kwargs):
        return container

    monkeypatch.setattr(decoder_module, "av_open", original_av_open, raising=False)
    monkeypatch.setattr(task_encoder, "tensor_to_pil", lambda image: image)

    images = _encoder()._decode_video_frames_with_energon(
        _fake_decoder(decoder_module, ["frame-0", "frame-1"]), [1.5, 0.25], thread_count=8
    )

    assert images == ["frame-0", "frame-1"]
    assert container.video_stream.codec_context.thread_count == 8
    assert container.video_stream.codec_context.thread_type == "FRAME"
    assert decoder_module.av_open is original_av_open
    assert container.closed


def test_video_decode_disabled_preserves_original_open(monkeypatch):
    container = _FakeContainer()
    decoder_module = _decoder_module()

    def original_av_open(*args, **kwargs):
        return container

    monkeypatch.setattr(decoder_module, "av_open", original_av_open, raising=False)
    monkeypatch.setattr(task_encoder, "tensor_to_pil", lambda image: image)

    images = _encoder(thread_count=0)._decode_video_frames(
        _fake_decoder(decoder_module, ["frame"]), [0.0]
    )

    assert images == ["frame"]
    assert container.video_stream.codec_context.thread_count is None
    assert container.video_stream.codec_context.thread_type is None


def test_video_decode_restores_hook_when_decode_raises(monkeypatch):
    decoder_module = _decoder_module()

    def original_av_open(*args, **kwargs):
        return _FakeContainer()

    class FailingDecoder:
        def get_clips(self, **kwargs):
            with decoder_module.av_open(io.BytesIO(b"video")):
                raise RuntimeError("decode failed")

    monkeypatch.setattr(decoder_module, "av_open", original_av_open, raising=False)

    with pytest.raises(RuntimeError, match="decode failed"):
        _encoder()._decode_video_frames_with_energon(FailingDecoder(), [0.0], thread_count=8)

    assert decoder_module.av_open is original_av_open


def test_video_decode_rejects_negative_thread_count():
    decoder = SimpleNamespace(
        get_clips=lambda **kwargs: pytest.fail("invalid config must fail before decode")
    )
    with pytest.raises(ValueError, match="must be non-negative"):
        _encoder(thread_count=-1)._decode_video_frames(decoder, [0.0])


@pytest.mark.parametrize("container_type", [tuple, list])
def test_load_media_unwraps_grouped_video_results(monkeypatch, container_type):
    class FakeAVDecoder:
        suppress_warnings = False

    decoder = FakeAVDecoder()

    class FakeLazy:
        def get(self, sample):
            return container_type([decoder, "ignored metadata"])

    lazy_media = FakeLazy()
    frames = [
        SimpleNamespace(media=SimpleNamespace(value=lazy_media, timestamp=0.0)),
        SimpleNamespace(media=SimpleNamespace(value=lazy_media, timestamp=1.0)),
    ]
    encoder = _encoder()
    monkeypatch.setattr(task_encoder, "AVDecoder", FakeAVDecoder)
    monkeypatch.setattr(
        encoder,
        "_decode_video_frames",
        lambda resolved_decoder, timestamps: [f"frame-{timestamp}" for timestamp in timestamps],
    )

    encoder._load_media(SimpleNamespace(images=frames))

    assert [frame.media.value for frame in frames] == ["frame-0.0", "frame-1.0"]
    assert decoder.suppress_warnings


@pytest.mark.parametrize("container_type", [tuple, list])
def test_load_media_unwraps_single_image_results(container_type):
    resolved_image = object()
    lazy_media = SimpleNamespace(
        get=lambda sample: container_type([resolved_image, "ignored metadata"])
    )
    image = SimpleNamespace(media=SimpleNamespace(value=lazy_media))

    _encoder()._load_media(SimpleNamespace(images=[image]))

    assert image.media.value is resolved_image
