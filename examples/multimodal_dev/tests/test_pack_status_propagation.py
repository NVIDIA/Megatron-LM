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


class TestBshdOverLengthGuardCpu:
    """Drive the real padded (BSHD) branch of ``pack_or_pad_batch`` on CPU."""

    @staticmethod
    def _setup(monkeypatch):
        # TP world 1 (handshake short-circuits, no collectives needed for the
        # failure path) with a CP-aware mpu stub for pack_or_pad_batch.
        mpu_stub = SimpleNamespace(
            get_tensor_model_parallel_world_size=lambda: 1,
            get_context_parallel_world_size=lambda: 1,
            get_tensor_model_parallel_rank=lambda: 0,
            get_tensor_model_parallel_src_rank=lambda: 0,
            get_tensor_model_parallel_group=lambda: "fake-tp-group",
        )
        monkeypatch.setattr(forward_step, "mpu", mpu_stub)
        monkeypatch.setattr(forward_step, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(forward_step, "get_tensor_model_parallel_src_rank", lambda: 0)
        monkeypatch.setattr(
            forward_step, "get_tensor_model_parallel_group", lambda: "fake-tp-group"
        )
        args = SimpleNamespace(
            sequence_parallel=False,
            pad_packed_seq_alignment=None,
            cuda_graph_impl="none",
            max_vision_patches_per_microbatch=None,
            max_vision_patches_per_image=None,
            seq_length=8,
        )
        monkeypatch.setattr(forward_step, "get_args", lambda: args)
        # broadcast is a rank-0 no-op at TP world 1 (only the happy path
        # reaches broadcast_data_batch; the source keeps its own tensors).
        monkeypatch.setattr(torch.distributed, "broadcast", lambda *a, **k: None)

    @staticmethod
    def _sample(length):
        return {
            "input_ids": torch.arange(length, dtype=torch.long),
            "labels": torch.arange(length, dtype=torch.long),
            "loss_mask": torch.ones(length, dtype=torch.float32),
            "pixel_values": torch.zeros(0, 1176, dtype=torch.float32),
            "image_grid_thw": torch.zeros(0, 3, dtype=torch.long),
        }

    def test_over_length_sample_raises_value_error(self, monkeypatch):
        self._setup(monkeypatch)
        with pytest.raises(
            ValueError, match=r"A sample of length 12 exceeds the --seq-length cap 8"
        ):
            forward_step.pack_or_pad_batch(
                [self._sample(12)], use_packed_sequence=False, seq_length=8, device="cpu"
            )

    def test_in_range_sample_pads_on_cpu(self, monkeypatch):
        # Companion check: the padded branch genuinely runs end-to-end on CPU
        # with these stubs, so the guard test above fails in the guard, not in
        # the setup.
        self._setup(monkeypatch)
        batch = forward_step.pack_or_pad_batch(
            [self._sample(6)], use_packed_sequence=False, seq_length=8, device="cpu"
        )
        assert batch["input_ids"].shape == (1, 6)
        assert batch["labels"].shape == (1, 6)
        assert batch["loss_mask"].shape == (1, 6)


class TestPackedThdMaxAlignment:
    """``--pad-packed-seq-alignment max`` is the ONLY mode mock_varlen permits
    (data/mock_varlen/qwen35_vl.py rejects every other value), yet the integer
    branch is what the alignment fixtures elsewhere exercise. Drive the real
    packed path through it end to end on CPU.
    """

    @staticmethod
    def _setup(monkeypatch, *, max_seqlen_per_dp_cp_rank):
        mpu_stub = SimpleNamespace(
            get_tensor_model_parallel_world_size=lambda: 1,
            get_context_parallel_world_size=lambda: 1,
            get_tensor_model_parallel_rank=lambda: 0,
            get_tensor_model_parallel_src_rank=lambda: 0,
            get_tensor_model_parallel_group=lambda: "fake-tp-group",
        )
        monkeypatch.setattr(forward_step, "mpu", mpu_stub)
        monkeypatch.setattr(forward_step, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(forward_step, "get_tensor_model_parallel_src_rank", lambda: 0)
        monkeypatch.setattr(
            forward_step, "get_tensor_model_parallel_group", lambda: "fake-tp-group"
        )
        args = SimpleNamespace(
            sequence_parallel=False,
            pad_packed_seq_alignment="max",
            max_seqlen_per_dp_cp_rank=max_seqlen_per_dp_cp_rank,
            pad_packed_seq_by_appending_dummy_seq=True,
            cuda_graph_impl="none",
            max_vision_patches_per_microbatch=None,
            max_vision_patches_per_image=None,
            seq_length=64,
        )
        monkeypatch.setattr(forward_step, "get_args", lambda: args)
        monkeypatch.setattr(torch.distributed, "broadcast", lambda *a, **k: None)
        return args

    @staticmethod
    def _sample(seq_lens):
        total = int(sum(seq_lens))
        return {
            "input_ids": torch.arange(1, total + 1, dtype=torch.long),
            "labels": torch.arange(1, total + 1, dtype=torch.long),
            "loss_mask": torch.ones(total, dtype=torch.float32),
            "seq_lens": torch.tensor(seq_lens, dtype=torch.int32),
            "pixel_values": torch.zeros(0, 1176, dtype=torch.float32),
            "image_grid_thw": torch.zeros(0, 3, dtype=torch.long),
        }

    def test_max_alignment_pads_to_the_fixed_target(self, monkeypatch):
        self._setup(monkeypatch, max_seqlen_per_dp_cp_rank=64)
        batch = forward_step.pack_or_pad_batch(
            [self._sample([10, 7])], use_packed_sequence=True, seq_length=64, device="cpu"
        )
        params = batch["packed_seq_params"]

        # The target comes from max_seqlen_per_dp_cp_rank, NOT from the data:
        # every microbatch lands on the same physical width, which is the whole
        # point of "max" (a fixed shape for the graph/memory envelope).
        assert batch["input_ids"].shape[-1] == 64
        assert int(params.cu_seqlens_q_padded[-1].item()) == 64
        assert params.total_tokens == 64

        # The 17 real tokens become two logical sequences plus one dummy tail
        # sequence covering the rest; the tail is padding everywhere.
        assert params.cu_seqlens_q_padded.tolist() == [0, 10, 17, 64]
        assert batch["padding_mask"][..., 17:].all()
        assert batch["labels"][..., 17:].eq(-100).all()
        assert batch["loss_mask"][..., 17:].eq(0).all()

    def test_batch_longer_than_the_target_is_rejected_loudly(self, monkeypatch):
        self._setup(monkeypatch, max_seqlen_per_dp_cp_rank=16)
        with pytest.raises(ValueError, match="increase --max-seqlen-per-dp-cp-rank"):
            forward_step.pack_or_pad_batch(
                [self._sample([10, 7])], use_packed_sequence=True, seq_length=64, device="cpu"
            )


class TestSourceSideFailuresPrecedeAnyPayloadCollective:
    """Everything that can fail on the source must fail BEFORE the handshake.

    ``broadcast_data_batch`` stages tensors and encodes dtypes inside its
    per-key loop, so a failure there lands between payload collectives with the
    peers already blocked — the handshake cannot help once it has reported
    success. These pin the two known offenders to the protected region.
    """

    def test_unencodable_dtype_is_rejected_before_any_collective(self):
        with pytest.raises(TypeError, match="cannot encode"):
            forward_step._stage_batch_for_broadcast(
                {"weird": torch.zeros(4, dtype=torch.float64)}, "cpu"
            )

    def test_patch_budget_verdict_precedes_the_handshake(self, monkeypatch):
        # An over-budget microbatch must be refused while the payload is still
        # on the host: staging it to the device is the OOM the guard pre-empts.
        TestBshdOverLengthGuardCpu._setup(monkeypatch)
        forward_step.get_args().max_vision_patches_per_microbatch = 1

        propagated = []
        monkeypatch.setattr(
            forward_step,
            "_propagate_pack_status",
            lambda is_src, error, device="cuda": propagated.append(error),
        )
        monkeypatch.setattr(
            forward_step,
            "_stage_batch_for_broadcast",
            lambda data, device: pytest.fail("staging ran despite an over-budget payload"),
        )

        sample = TestBshdOverLengthGuardCpu._sample(6)
        sample["pixel_values"] = torch.zeros(64, 1176, dtype=torch.float32)
        sample["image_grid_thw"] = torch.tensor([[1, 8, 8]], dtype=torch.long)
        forward_step.pack_or_pad_batch(
            [sample], use_packed_sequence=False, seq_length=8, device="cpu"
        )
        # The failure reached the handshake as a value, not as a raise.
        assert len(propagated) == 1 and isinstance(propagated[0], Exception)
