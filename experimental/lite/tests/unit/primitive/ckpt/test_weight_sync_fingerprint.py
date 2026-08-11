# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import torch

from megatron.lite.primitive.ckpt.weight_sync_fingerprint import (
    _sample_indices,
    stream_fingerprint,
    tensor_fingerprint_record,
)


def test_stream_fingerprint_is_order_independent():
    a = tensor_fingerprint_record("model.embed_tokens.weight", torch.arange(32))
    b = tensor_fingerprint_record("model.layers.0.weight", torch.arange(8))
    assert stream_fingerprint([a, b]) == stream_fingerprint([b, a])


def test_stream_fingerprint_detects_sampled_payload_change():
    before = tensor_fingerprint_record("model.embed_tokens.weight", torch.arange(32))
    value = torch.arange(32)
    value[-1] = 100
    after = tensor_fingerprint_record("model.embed_tokens.weight", value)
    assert before["sha256"] != after["sha256"]


def test_sample_indices_stay_in_bounds_for_multi_billion_byte_tensor():
    numel = 9_000_000_001
    indices = _sample_indices(numel, 256)
    assert indices[0].item() == 0
    assert indices[-1].item() == numel - 1
    assert indices.min().item() >= 0
    assert indices.max().item() < numel
