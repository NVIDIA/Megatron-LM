# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-only tests for MimoModel.zero_grad_buffer fan-out."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from megatron.core.models.mimo.model.base import MimoModel


def _stub(language_model, modality_submodules):
    """Minimal stand-in carrying the real zero_grad_buffer / _active_submodules."""
    stub = SimpleNamespace(language_model=language_model, modality_submodules=modality_submodules)
    stub.zero_grad_buffer = MimoModel.zero_grad_buffer.__get__(stub)
    stub._active_submodules = MimoModel._active_submodules.__get__(stub)
    return stub


def test_zero_grad_buffer_fans_out_to_present_submodules():
    language_model = MagicMock()
    vision = MagicMock()
    _stub(language_model, {"vision": vision}).zero_grad_buffer()

    language_model.zero_grad_buffer.assert_called_once_with()
    vision.zero_grad_buffer.assert_called_once_with()


def test_zero_grad_buffer_skips_none_submodules():
    vision = MagicMock()
    # Encoder-only rank: language_model is None, plus a None modality entry.
    _stub(None, {"vision": vision, "audio": None}).zero_grad_buffer()

    vision.zero_grad_buffer.assert_called_once_with()
