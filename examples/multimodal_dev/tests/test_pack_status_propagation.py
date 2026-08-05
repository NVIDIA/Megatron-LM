# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-only tests for the TP pack-status handshake in forward_step.

``_propagate_pack_status`` converts source-only pack/validation failures in
``pack_or_pad_batch`` from a collective hang into a synchronized failure on
every TP rank: the source broadcasts a ``[failed, msg_len]`` header (plus the
UTF-8 message bytes on failure) before the payload broadcast, then all ranks
raise together. The same handshake covers source-side data-fetch failures in
``_fetch_batch_with_status`` (the fetch-and-propagate portion of
``get_batch``), tested here alongside a CPU run of the padded (BSHD) branch
of ``pack_or_pad_batch``. These tests drive the helpers directly with a
monkeypatched ``forward_step.mpu`` and a recording
``torch.distributed.broadcast`` stub, so no process group (and no GPU) is
needed.
"""

from types import SimpleNamespace

import pytest
import torch

from examples.multimodal_dev import forward_step
from examples.multimodal_dev.forward_step import _PACK_ERROR_MSG_MAX_BYTES, _propagate_pack_status

_PEER_PREFIX = "TP source rank failed while fetching/packing/validating the microbatch: "


def _fake_mpu(world_size):
    return SimpleNamespace(
        get_tensor_model_parallel_world_size=lambda: world_size,
        get_tensor_model_parallel_rank=lambda: 0,
        get_tensor_model_parallel_src_rank=lambda: 0,
        get_tensor_model_parallel_group=lambda: "fake-tp-group",
    )


class TestWorldSizeOne:
    def test_error_raises_original_without_collectives(self, monkeypatch):
        monkeypatch.setattr(forward_step, "mpu", _fake_mpu(world_size=1))

        def _fail_broadcast(*args, **kwargs):
            raise AssertionError("no collective may run at TP world size 1")

        monkeypatch.setattr(torch.distributed, "broadcast", _fail_broadcast)

        original = ValueError("seq_lens sum mismatch")
        with pytest.raises(ValueError) as excinfo:
            _propagate_pack_status(True, original, device="cpu")
        assert excinfo.value is original

    def test_no_error_returns_without_collectives(self, monkeypatch):
        monkeypatch.setattr(forward_step, "mpu", _fake_mpu(world_size=1))

        def _fail_broadcast(*args, **kwargs):
            raise AssertionError("no collective may run at TP world size 1")

        monkeypatch.setattr(torch.distributed, "broadcast", _fail_broadcast)

        assert _propagate_pack_status(True, None, device="cpu") is None


class TestSourceSide:
    def test_success_broadcasts_zero_header_only(self, monkeypatch):
        monkeypatch.setattr(forward_step, "mpu", _fake_mpu(world_size=2))
        calls = []
        monkeypatch.setattr(
            torch.distributed,
            "broadcast",
            lambda tensor, src, group=None: calls.append((tensor.clone(), src, group)),
        )

        assert _propagate_pack_status(True, None, device="cpu") is None

        # Exactly one collective: the [failed=0, msg_len=0] header. No payload.
        assert len(calls) == 1
        header, src, group = calls[0]
        assert header.dtype == torch.int64
        assert header.tolist() == [0, 0]
        assert src == 0
        assert group == "fake-tp-group"

    def test_error_broadcasts_header_and_payload_then_reraises(self, monkeypatch):
        monkeypatch.setattr(forward_step, "mpu", _fake_mpu(world_size=2))
        calls = []
        monkeypatch.setattr(
            torch.distributed,
            "broadcast",
            lambda tensor, src, group=None: calls.append(tensor.clone()),
        )

        original = ValueError("BSHD over-length reject: sample of length 9000")
        with pytest.raises(ValueError) as excinfo:
            _propagate_pack_status(True, original, device="cpu")
        # The source re-raises the ORIGINAL exception object.
        assert excinfo.value is original

        msg_bytes = str(original).encode("utf-8")
        assert len(calls) == 2
        assert calls[0].tolist() == [1, len(msg_bytes)]
        assert calls[1].dtype == torch.uint8
        assert bytes(calls[1].tolist()) == msg_bytes

    def test_error_message_truncated_to_max_bytes(self, monkeypatch):
        monkeypatch.setattr(forward_step, "mpu", _fake_mpu(world_size=2))
        calls = []
        monkeypatch.setattr(
            torch.distributed,
            "broadcast",
            lambda tensor, src, group=None: calls.append(tensor.clone()),
        )

        original = RuntimeError("x" * (_PACK_ERROR_MSG_MAX_BYTES + 500))
        with pytest.raises(RuntimeError):
            _propagate_pack_status(True, original, device="cpu")

        assert calls[0].tolist() == [1, _PACK_ERROR_MSG_MAX_BYTES]
        assert calls[1].numel() == _PACK_ERROR_MSG_MAX_BYTES


class TestPeerSide:
    def test_peer_raises_runtime_error_with_source_message(self, monkeypatch):
        monkeypatch.setattr(forward_step, "mpu", _fake_mpu(world_size=2))
        src_msg = "seq_lens sum 4097 != sample length 4096"
        src_msg_bytes = src_msg.encode("utf-8")
        calls = []

        def _peer_broadcast(tensor, src, group=None):
            # Simulate the source's values landing in the peer's buffer.
            calls.append(tensor)
            if len(calls) == 1:
                tensor.copy_(torch.tensor([1, len(src_msg_bytes)], dtype=torch.int64))
            else:
                tensor.copy_(torch.tensor(list(src_msg_bytes), dtype=torch.uint8))

        monkeypatch.setattr(torch.distributed, "broadcast", _peer_broadcast)

        with pytest.raises(RuntimeError) as excinfo:
            _propagate_pack_status(False, None, device="cpu")
        assert str(excinfo.value) == _PEER_PREFIX + src_msg
        assert len(calls) == 2

    def test_peer_success_returns_without_payload_collective(self, monkeypatch):
        monkeypatch.setattr(forward_step, "mpu", _fake_mpu(world_size=2))
        calls = []

        def _peer_broadcast(tensor, src, group=None):
            calls.append(tensor)
            tensor.copy_(torch.tensor([0, 0], dtype=torch.int64))

        monkeypatch.setattr(torch.distributed, "broadcast", _peer_broadcast)

        assert _propagate_pack_status(False, None, device="cpu") is None
        assert len(calls) == 1


def _patch_tp_world(monkeypatch, world_size, rank):
    """Patch both ``forward_step.mpu`` and the module-level parallel-state imports."""
    monkeypatch.setattr(forward_step, "mpu", _fake_mpu(world_size=world_size))
    monkeypatch.setattr(forward_step, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(forward_step, "get_tensor_model_parallel_src_rank", lambda: 0)
    monkeypatch.setattr(forward_step, "get_tensor_model_parallel_group", lambda: "fake-tp-group")


class _BoomIterator:
    """Data iterator whose ``next()`` raises like a crashed DataLoader worker."""

    def __init__(self, error):
        self.error = error

    def __iter__(self):
        return self

    def __next__(self):
        raise self.error


class TestFetchBatchWithStatus:
    """``_fetch_batch_with_status`` is the fetch-and-propagate portion of
    ``get_batch``: a source-side iterator error must reach the handshake
    BEFORE the ``has_data`` broadcast, while ``StopIteration`` keeps its
    normal end-of-data (``has_data = 0``) semantics."""

    def test_src_fetch_error_reraised_with_failure_header_before_has_data(self, monkeypatch):
        _patch_tp_world(monkeypatch, world_size=2, rank=0)
        calls = []
        monkeypatch.setattr(
            torch.distributed,
            "broadcast",
            lambda tensor, src, group=None: calls.append(tensor.clone()),
        )

        original = RuntimeError("boom")
        with pytest.raises(RuntimeError) as excinfo:
            forward_step._fetch_batch_with_status(_BoomIterator(original), device="cpu")
        # The source re-raises the ORIGINAL exception object.
        assert excinfo.value is original

        # Exactly the failure handshake ran: int64 [failed=1, msg_len] header
        # then the uint8 message payload. The has_data broadcast (a 1-element
        # uint8 tensor) never happened — peers learn of the failure first.
        assert len(calls) == 2
        assert calls[0].dtype == torch.int64
        assert calls[0].tolist() == [1, len(b"boom")]
        assert calls[1].dtype == torch.uint8
        assert bytes(calls[1].tolist()) == b"boom"

    def test_peer_raises_runtime_error_on_src_fetch_failure(self, monkeypatch):
        _patch_tp_world(monkeypatch, world_size=2, rank=1)
        src_msg_bytes = b"boom"
        calls = []

        def _peer_broadcast(tensor, src, group=None):
            # Simulate the source's failure header/payload landing in the
            # peer's buffers.
            calls.append(tensor)
            if len(calls) == 1:
                tensor.copy_(torch.tensor([1, len(src_msg_bytes)], dtype=torch.int64))
            else:
                tensor.copy_(torch.tensor(list(src_msg_bytes), dtype=torch.uint8))

        monkeypatch.setattr(torch.distributed, "broadcast", _peer_broadcast)

        # The peer never touches the iterator (it is None on non-source ranks).
        with pytest.raises(RuntimeError, match="boom"):
            forward_step._fetch_batch_with_status(None, device="cpu")
        # Header + payload only; the peer raised before the has_data collective.
        assert len(calls) == 2

    def test_stop_iteration_keeps_has_data_zero_semantics(self, monkeypatch):
        _patch_tp_world(monkeypatch, world_size=2, rank=0)
        calls = []
        monkeypatch.setattr(
            torch.distributed,
            "broadcast",
            lambda tensor, src, group=None: calls.append(tensor.clone()),
        )

        data, has_data = forward_step._fetch_batch_with_status(iter([]), device="cpu")

        assert data is None
        assert has_data is False
        # Success handshake header (no failure payload), then has_data=0.
        assert len(calls) == 2
        assert calls[0].dtype == torch.int64
        assert calls[0].tolist() == [0, 0]
        assert calls[1].dtype == torch.uint8
        assert calls[1].tolist() == [0]

    def test_src_success_returns_data_and_has_data_one(self, monkeypatch):
        _patch_tp_world(monkeypatch, world_size=2, rank=0)
        calls = []
        monkeypatch.setattr(
            torch.distributed,
            "broadcast",
            lambda tensor, src, group=None: calls.append(tensor.clone()),
        )

        batch = [{"input_ids": torch.arange(4)}]
        data, has_data = forward_step._fetch_batch_with_status(iter([batch]), device="cpu")

        assert data is batch
        assert has_data is True
        assert len(calls) == 2
        assert calls[0].tolist() == [0, 0]
        assert calls[1].tolist() == [1]
