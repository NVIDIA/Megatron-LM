# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from examples.mimo.data import avlm_sample_loader as sample_loader_mod
from examples.mimo.data.avlm_sample_loader import reject_clip_list, sample_loader
from examples.mimo.data.utils.calculate_audio_tokens import (
    calculate_num_audio_tokens,
    capped_whisper_lengths,
    expand_audio_placeholders,
    stack_or_cat,
)
from examples.mimo.model_providers.hf_whisper_encoder import (
    HFWhisperEncoderWrapper,
    scale_whisper_tp_grad,
)


class _RecordEmbed(nn.Embedding):
    def forward(self, input):
        self.last_index_device = input.device
        return super().forward(input)


class _IdentityLayer(nn.Module):
    def forward(self, hidden, attention_mask=None):
        return hidden


class _TinyEncoder(nn.Module):
    def __init__(self, n_mels=4, width=4):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, stride=2, padding=1)
        self.embed_positions = _RecordEmbed(3000, width)
        self.embed_positions.weight.requires_grad_(False)
        self.layers = nn.ModuleList([_IdentityLayer()])
        self.layer_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(0.0)
        self.config = type(
            "Cfg",
            (),
            {
                "dropout": 0.0,
                "attention_dropout": 0.0,
                "activation_dropout": 0.0,
                "encoder_layerdrop": 0.0,
            },
        )()

    def forward(self, input_features, **_kwargs):
        if input_features.shape[-1] != 3000:
            raise ValueError("mel length must be 3000")
        hidden = torch.nn.functional.gelu(self.conv1(input_features))
        hidden = torch.nn.functional.gelu(self.conv2(hidden))
        hidden = hidden.permute(0, 2, 1)
        hidden = hidden + self.embed_positions.weight[: hidden.shape[1]].to(hidden.dtype)
        hidden = self.layer_norm(hidden)
        return type("Out", (), {"last_hidden_state": hidden})()


def _unpadded_body(encoder: _TinyEncoder, features: Tensor) -> Tensor:
    hidden = features.unsqueeze(0)
    hidden = torch.nn.functional.gelu(encoder.conv1(hidden))
    hidden = torch.nn.functional.gelu(encoder.conv2(hidden))
    hidden = hidden.permute(0, 2, 1)
    positions = encoder.embed_positions(torch.arange(hidden.shape[1], device=hidden.device))
    hidden = hidden + positions.to(dtype=hidden.dtype)
    hidden = torch.nn.functional.dropout(hidden, p=0.0, training=encoder.training)
    for layer in encoder.layers:
        hidden = layer(hidden, None)
    hidden = encoder.layer_norm(hidden)
    return hidden.squeeze(0)


def _batch(prefix_frames: int, tail: float) -> tuple[Tensor, Tensor]:
    torch.manual_seed(0)
    features = torch.randn(1, 4, 3000)
    features[:, :, prefix_frames:] = tail
    prefix = features[:, :, :prefix_frames].clone()
    features[:, :, :prefix_frames] = prefix
    return features, prefix


def test_crop_ignores_the_padded_tail():
    encoder = _TinyEncoder()
    wrapper = HFWhisperEncoderWrapper(encoder=encoder, finetune=False)
    features, prefix = _batch(13, tail=9.0)
    zeros = features.clone()
    zeros[:, :, 13:] = 0
    feature_lengths = torch.tensor([13])
    seq_lengths = torch.tensor([7])
    cropped = wrapper(features, seq_lengths=seq_lengths, feature_lengths=feature_lengths)
    zero_tail = wrapper(zeros, seq_lengths=seq_lengths, feature_lengths=feature_lengths)
    reference = _unpadded_body(encoder, prefix.squeeze(0))
    assert torch.allclose(cropped, reference)
    assert torch.allclose(zero_tail, reference)
    assert encoder.embed_positions.last_index_device == cropped.device

    full = encoder(features).last_hidden_state
    masked = full[0, :7]
    assert not torch.allclose(masked, reference)

    with pytest.raises(ValueError, match="3000"):
        encoder(prefix)


def test_seq_lengths_without_feature_lengths_raises():
    wrapper = HFWhisperEncoderWrapper(encoder=_TinyEncoder(), finetune=False)
    features = torch.randn(1, 4, 3000)
    with pytest.raises(ValueError, match="not invertible"):
        wrapper(features, seq_lengths=torch.tensor([7]))


def test_wrong_seq_lengths_raises():
    wrapper = HFWhisperEncoderWrapper(encoder=_TinyEncoder(), finetune=False)
    features = torch.randn(1, 4, 3000)
    with pytest.raises(ValueError, match="does not match"):
        wrapper(features, seq_lengths=torch.tensor([3]), feature_lengths=torch.tensor([13]))


def test_missing_lengths_still_calls_encoder_forward():
    encoder = _TinyEncoder()
    wrapper = HFWhisperEncoderWrapper(encoder=encoder, finetune=False)
    features = torch.randn(1, 4, 3000)
    out = wrapper(features)
    assert torch.allclose(out, encoder(features).last_hidden_state)


def test_capped_lengths_match_short_clips_and_cap_thirty_seconds():
    for samples in (1, 160, 159, 161, 16000, 479999):
        _mel, tokens = capped_whisper_lengths(samples)
        today = calculate_num_audio_tokens(torch.zeros(1, samples), "openai/whisper-base")
        assert tokens == today
    for samples in (480000, 480001):
        mel, tokens = capped_whisper_lengths(samples)
        assert mel == 3000
        assert tokens == 1500


def test_finetune_grad_and_dropout():
    frozen = _TinyEncoder()
    wrapper = HFWhisperEncoderWrapper(encoder=frozen, finetune=False)
    frozen.conv1.weight.requires_grad_(True)
    features = torch.randn(1, 4, 3000)
    out = wrapper(features, seq_lengths=torch.tensor([7]), feature_lengths=torch.tensor([13]))
    assert not out.requires_grad
    assert frozen.conv1.weight.grad is None

    trained = _TinyEncoder()
    trained.conv1.weight.requires_grad_(True)
    finetuned = HFWhisperEncoderWrapper(encoder=trained, finetune=True)
    out = finetuned(features, seq_lengths=torch.tensor([7]), feature_lengths=torch.tensor([13]))
    out.sum().backward()
    assert trained.conv1.weight.grad is not None
    assert trained.embed_positions.weight.grad is None

    bad = _TinyEncoder()
    bad.config.dropout = 0.1
    try:
        HFWhisperEncoderWrapper(encoder=bad, finetune=True)
    except ValueError as exc:
        assert "dropout" in str(exc)
    else:
        raise AssertionError("nonzero dropout must raise")


def test_tp_hook_averages_the_sum(monkeypatch):
    group = object()
    seen = {}

    def all_reduce(grad, op=None, group=None):
        seen["op"] = op
        seen["group"] = group
        grad.mul_(4)
        return grad

    def get_world_size(group=None):
        if group is None:
            raise AssertionError("get_world_size requires the TP group")
        return 4

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(torch.distributed, "get_world_size", get_world_size)
    grad = torch.ones(2)
    scaled = scale_whisper_tp_grad(grad, group)
    assert seen["op"] == torch.distributed.ReduceOp.SUM
    assert seen["group"] is group
    assert torch.equal(scaled, torch.ones(2))


def test_expand_audio_placeholders_splits_once():
    prompt = expand_audio_placeholders("<audio> hello <audio>", [3, 2])
    assert prompt == "<audio><audio><audio> hello <audio><audio>"


def test_stack_or_cat_keeps_variable_clip_counts():
    one = [torch.zeros(80, 4), torch.ones(80, 4)]
    assert stack_or_cat(one).shape == (2, 80, 4)
    many = [torch.zeros(2, 80, 4), torch.ones(1, 80, 4)]
    assert stack_or_cat(many).shape == (3, 80, 4)


def _raw(conversations, audios, blobs):
    raw = {
        "__key__": "clip-key",
        "json": {"conversations": conversations, "audios": audios},
        "img": b"img",
    }
    raw.update(blobs)
    return raw


def test_sample_loader_flag_false_keeps_the_first_clip(monkeypatch):
    monkeypatch.setattr(sample_loader_mod, "_all_audio_clips", lambda: False)
    loaded = sample_loader(
        _raw(
            [
                {"value": "<audio> a <audio>"},
                {"value": "answer"},
                {"value": "<audio>"},
                {"value": "later"},
            ],
            ["k.first.wav", "k.second.wav"],
            {"first.wav": b"A", "second.wav": b"B"},
        )
    )
    assert loaded["audio"] == b"A"
    assert loaded["context"] == "<audio> a "
    assert loaded["answers"] == "answer"


def test_sample_loader_flag_true_returns_every_clip(monkeypatch):
    monkeypatch.setattr(sample_loader_mod, "_all_audio_clips", lambda: True)
    loaded = sample_loader(
        _raw(
            [{"value": "<audio> a"}, {"value": "one"}, {"value": "b <audio>"}, {"value": "two"}],
            ["k.first.wav", "k.second.wav"],
            {"first.wav": b"A", "second.wav": b"B"},
        )
    )
    assert loaded["audio"] == [b"A", b"B"]
    assert loaded["context"] == ["<audio> a", "b <audio>"]


def test_sample_loader_flag_true_rejects_bad_samples(monkeypatch):
    monkeypatch.setattr(sample_loader_mod, "_all_audio_clips", lambda: True)
    try:
        sample_loader(
            _raw(
                [{"value": "<audio>"}, {"value": "one"}],
                ["k.first.wav", "k.second.wav"],
                {"first.wav": b"A", "second.wav": b"B"},
            )
        )
    except ValueError as exc:
        assert "1" in str(exc) and "2" in str(exc) and "clip-key" in str(exc)
    else:
        raise AssertionError("tag/file mismatch must raise")

    try:
        sample_loader(
            _raw(
                [{"value": "<audio>"}, {"value": "has <audio>"}],
                ["k.first.wav"],
                {"first.wav": b"A"},
            )
        )
    except ValueError as exc:
        assert "assistant" in str(exc)
    else:
        raise AssertionError("assistant audio must raise")

    try:
        sample_loader(_raw([{"value": "<audio>"}, {"value": "one"}], ["k.first.wav"], {}))
    except ValueError as exc:
        assert "missing audio blob" in str(exc)
    else:
        raise AssertionError("missing blob must raise")


def test_encode_sample_rejects_a_clip_list_when_the_flag_is_off():
    try:
        reject_clip_list([b"a", b"b"], False)
    except ValueError as exc:
        assert "list" in str(exc)
    else:
        raise AssertionError("a clip list must be rejected")
    reject_clip_list([b"a"], True)
    reject_clip_list(b"a", False)
