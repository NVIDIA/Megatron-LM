# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.optimizer.qk_clip import clip_qk
from megatron.core.transformer.attention import Attention


class _FakeCoreAttention(torch.nn.Module):
    def __init__(self, max_logits):
        super().__init__()
        self.current_max_attn_logits = torch.tensor(max_logits, dtype=torch.float32)


class _FakeAttention(Attention):
    def __init__(self, max_logits):
        torch.nn.Module.__init__(self)
        self.core_attention = _FakeCoreAttention(max_logits)
        self.clip_calls = 0

    def clip_qk(self):
        self.clip_calls += 1
        self.core_attention.current_max_attn_logits = None

    def get_query_key_value_tensors(self, *args, **kwargs):
        raise NotImplementedError


class _FakeQKLikeModule(torch.nn.Module):
    """A non-Attention module that happens to expose the same attribute names."""

    def __init__(self):
        super().__init__()
        self.core_attention = _FakeCoreAttention([1000.0])
        self.clip_calls = 0

    def clip_qk(self):
        self.clip_calls += 1


class _FakeHybridModel(HybridModel):
    def __init__(self, decoder_attention, mtp_attention):
        torch.nn.Module.__init__(self)
        self.decoder = torch.nn.ModuleList([decoder_attention])
        self.mtp = torch.nn.Module()
        self.mtp.mtp_model_layer = torch.nn.ModuleList([mtp_attention])
        self.unrelated = _FakeQKLikeModule()


class _ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = torch.nn.Module()
        self.module.module = model


def test_clip_qk_reaches_nested_mtp(monkeypatch):
    decoder_attention = _FakeAttention([80.0, 120.0])
    mtp_attention = _FakeAttention([150.0, 90.0])
    model = _FakeHybridModel(decoder_attention, mtp_attention)
    model_chunk = _ModelWrapper(model)
    all_reduce_calls = []

    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, **kwargs: all_reduce_calls.append((tensor, kwargs)),
    )
    monkeypatch.setattr(
        "megatron.core.optimizer.qk_clip.parallel_state.get_data_parallel_group",
        lambda **kwargs: None,
    )

    assert clip_qk([model_chunk]) == 150.0
    assert decoder_attention.clip_calls == 1
    assert mtp_attention.clip_calls == 1
    assert model.unrelated.clip_calls == 0
    assert len(all_reduce_calls) == 2


def test_clip_qk_log_only_resets_nested_mtp(monkeypatch):
    decoder_attention = _FakeAttention([80.0])
    mtp_attention = _FakeAttention([150.0])
    model = _FakeHybridModel(decoder_attention, mtp_attention)
    model_chunk = _ModelWrapper(model)

    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "megatron.core.optimizer.qk_clip.parallel_state.get_data_parallel_group",
        lambda **kwargs: None,
    )

    assert clip_qk([model_chunk], log_max_only=True) == 150.0
    assert decoder_attention.clip_calls == 0
    assert mtp_attention.clip_calls == 0
    assert decoder_attention.core_attention.current_max_attn_logits is None
    assert mtp_attention.core_attention.current_max_attn_logits is None
