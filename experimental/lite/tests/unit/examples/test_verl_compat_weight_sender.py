# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

VERL_EXAMPLE_ROOT = Path(__file__).resolve().parents[3] / "examples" / "verl"
if str(VERL_EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_EXAMPLE_ROOT))

from verl_mlite import compat

pytestmark = pytest.mark.optional


class _Socket:
    def __init__(self) -> None:
        self.messages = []

    def send_pyobj(self, value) -> None:
        self.messages.append(value)

    def recv(self):
        return b"ack"


class _Sender:
    use_shm = False
    bucket_size = 64
    _mlite_prefetch_allow_cpu = True

    def __init__(self) -> None:
        self.socket = _Socket()
        self.buffer = None
        self.cleaned = False

    async def async_send_weights(self, _weights):
        raise AssertionError("the MLite bucket packer should handle a sync generator")

    def _init_socket(self) -> None:
        pass

    def _init_buffer(self) -> None:
        self.buffer = torch.empty(self.bucket_size, dtype=torch.uint8)

    def _direct_send_large_weight(self, _name, _weight) -> None:
        raise AssertionError("test weight unexpectedly exceeded the bucket")

    def _cleanup(self) -> None:
        self.cleaned = True


def test_lazy_weight_generator_is_not_submitted_to_an_executor(monkeypatch) -> None:
    consumed = []

    def forbidden_submit(*_args, **_kwargs):
        raise AssertionError("lazy model-weight production must not enter an executor")

    monkeypatch.setattr(ThreadPoolExecutor, "submit", forbidden_submit)

    def weights():
        consumed.append(True)
        yield "model.layers.0.weight", torch.arange(8, dtype=torch.bfloat16)

    assert compat._install_bucketed_sender_prefetch(_Sender)
    sender = _Sender()
    asyncio.run(sender.async_send_weights(weights()))

    assert consumed == [True]
    assert sender.cleaned
    assert sender.socket.messages[0]["is_last"] is True
    assert "model.layers.0.weight" in sender.socket.messages[0]["bucket_meta"]
