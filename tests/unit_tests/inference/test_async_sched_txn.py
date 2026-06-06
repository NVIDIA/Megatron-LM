# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Targeted battery for the transactional async-scheduling decode pipeline.

The spine of this battery is :func:`assert_async_equals_serial`, which runs the
same prompts/seed twice -- once with ``enable_async_scheduling=False`` (the serial
ground truth) and once ``True`` -- and asserts identical generated token ids. The
serial path is the oracle; the async path must reproduce it token-for-token.

In the earliest commits the flag is a no-op, so the equivalence check passes
trivially and exists to lock in the harness. As the async pipeline is built up
(launch-before-commit, closed survivor set, hybrid single-bank, EP, MTP), the same
check becomes the load-bearing correctness gate.

Run (8 GPUs, in its own ``torch.distributed.run`` invocation)::

    /opt/venv/bin/python -m torch.distributed.run --nproc-per-node 8 -m pytest -v \
        tests/unit_tests/inference/test_async_sched_txn.py
"""

import argparse

import pytest
import torch

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.metadata_slot import (
    DecodeMetadataSlot,
    DepthTwoRing,
    assert_slot_pointer_identity,
    make_decode_metadata_slots,
)
from megatron.core.inference.engines.step_transaction import (
    LaunchDecision,
    LaunchEligibility,
    LaunchGateReason,
    LaunchSignals,
    RetireQueue,
    StepTransactionManager,
    StepTxn,
    StepTxnDiagnostics,
    TxnPhase,
    classify_launch_eligibility,
)
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicEngineTestConfig,
    DynamicInferenceEngineTestBase,
    set_rounder,
)
from tests.unit_tests.test_utilities import Utils


def _to_token_list(tokens):
    """Normalize a generated-token container (list or tensor) to a list of ints."""
    if tokens is None:
        return []
    if isinstance(tokens, torch.Tensor):
        return tokens.detach().cpu().tolist()
    return list(tokens)


def _collect_outputs(env):
    """Map request_id -> (status, generated token ids) after a completed run."""
    outputs = {}
    for request in env.requests:
        outputs[request.request_id] = (
            request.status,
            _to_token_list(getattr(request, "output", None)),
        )
    return outputs


class AsyncSchedTxnTestBase(DynamicInferenceEngineTestBase):
    """Shared helpers for the async-scheduling battery.

    Reuses the model/context/engine fixtures from the dynamic-engine test base so
    the async path is exercised against the exact same tiny GPT / hybrid configs.
    """

    @classmethod
    def assert_async_equals_serial(cls, **test_config_kwargs):
        """Token-exact ``async == serial`` equivalence.

        Builds and runs the engine twice from identical seeds/config -- once serial
        (``enable_async_scheduling=False``) and once async (``True``) -- and asserts
        every request produced the identical status and generated token ids.

        Returns the (serial_env, async_env) pair for any further assertions.
        """
        # Disallow the caller overriding the toggle; this helper owns it.
        test_config_kwargs.pop("enable_async_scheduling", None)

        serial_env = cls._run_test(enable_async_scheduling=False, **test_config_kwargs)
        serial_outputs = _collect_outputs(serial_env)

        async_env = cls._run_test(enable_async_scheduling=True, **test_config_kwargs)
        async_outputs = _collect_outputs(async_env)

        assert set(serial_outputs.keys()) == set(async_outputs.keys()), (
            f"request id sets differ: serial={sorted(serial_outputs)} "
            f"async={sorted(async_outputs)}"
        )

        mismatches = []
        for request_id, (serial_status, serial_tokens) in serial_outputs.items():
            async_status, async_tokens = async_outputs[request_id]
            if serial_status != async_status or serial_tokens != async_tokens:
                mismatches.append(
                    f"  request {request_id}: "
                    f"serial=({serial_status}, {serial_tokens}) "
                    f"async=({async_status}, {async_tokens})"
                )
        assert not mismatches, "async != serial for:\n" + "\n".join(mismatches)

        return serial_env, async_env


@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestAsyncSchedTxnFlag(AsyncSchedTxnTestBase):
    """C0: opt-in flag plumbing + the serial-equivalence spine (flag is a no-op here)."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )

    @classmethod
    def teardown_class(cls):
        delete_cuda_graphs()
        set_rounder(64)
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_arg_maps_to_config_dest(self) -> None:
        """The Megatron arg parses to the documented dest, defaulting to False."""
        from megatron.training.arguments import _add_inference_args

        parser = argparse.ArgumentParser()
        _add_inference_args(parser)

        # Default is off.
        default_args = parser.parse_args([])
        assert default_args.inference_dynamic_batching_async_scheduling is False

        # Enable flag.
        on_args = parser.parse_args(["--inference-dynamic-batching-async-scheduling"])
        assert on_args.inference_dynamic_batching_async_scheduling is True

        # BooleanOptionalAction provides an explicit --no- variant.
        off_args = parser.parse_args(["--no-inference-dynamic-batching-async-scheduling"])
        assert off_args.inference_dynamic_batching_async_scheduling is False

    @pytest.mark.internal
    def test_config_field_defaults_false(self) -> None:
        """The config field exists and defaults to False (disabled == legacy)."""
        assert InferenceConfig().enable_async_scheduling is False
        assert InferenceConfig(enable_async_scheduling=True).enable_async_scheduling is True

    @pytest.mark.internal
    def test_flag_plumbs_to_context(self) -> None:
        """The flag reaches the live DynamicInferenceContext both off and on."""
        set_rounder(4)
        for enabled in (False, True):
            env = self._build_test_env(
                DynamicEngineTestConfig(model_provider="gpt", enable_async_scheduling=enabled)
            )
            assert env.engine.context.enable_async_scheduling is enabled

    @pytest.mark.internal
    def test_async_equals_serial_gpt_decode(self) -> None:
        """Spine: tiny GPT greedy decode is identical with the flag off vs on.

        In C0 the flag is a no-op, so this is a determinism + harness check; the
        same assertion becomes the real correctness gate in later commits.
        """
        self.assert_async_equals_serial(
            model_provider="gpt",
            num_tokens_to_generate=8,
            num_requests=4,
            num_gap_steps=1,
        )


class _StubEvent:
    """A CUDA-event stand-in with a controllable ``query()`` for deterministic tests."""

    def __init__(self, ready: bool):
        self.ready = ready
        self.synchronized = False

    def query(self) -> bool:
        return self.ready

    def synchronize(self) -> None:
        self.synchronized = True
        self.ready = True


@pytest.mark.internal
class TestStepTransactionPrimitives:
    """C1 unit tests for the transaction data structures (no model parallel needed)."""

    def test_lease_ownership_single_owner(self) -> None:
        """Resources are owned by the transaction via lease fields -- no global arrays."""
        txn = StepTxn(step_id=7, active_request_count=3)
        assert txn.phase is TxnPhase.SERIAL
        assert not txn.has_leases()

        txn.lease_kv_block(11)
        txn.lease_kv_block(12)
        txn.reserve_boundary_block(request_id=5, block_column=2, block_id=99)
        txn.lease_mamba_slot(4)

        assert txn.kv_block_leases == [11, 12]
        assert txn.reserved_boundary_blocks == {(5, 2): 99}
        assert txn.mamba_slot_leases == [4]
        # Reserved boundary blocks are included in the full KV-block ownership view.
        assert sorted(txn.leased_kv_block_ids()) == [11, 12, 99]
        assert txn.has_leases()

    def test_retire_queue_two_step_delay_no_event(self) -> None:
        """Without a fence, a resource is released only after the two-step delay."""
        released = []
        q = RetireQueue()
        assert q.RETIRE_DELAY_STEPS == 2
        q.enqueue(enqueue_step=0, release=lambda: released.append("blk"), tag="kv")

        assert q.drain(0) == 0  # same step
        assert q.drain(1) == 0  # one step later
        assert released == []
        assert q.pending() == 1
        assert q.drain(2) == 1  # two steps later -> released
        assert released == ["blk"]
        assert q.pending() == 0

    def test_retire_queue_fence_gate(self) -> None:
        """Even past the delay, a resource is held until its fence completes."""
        released = []
        ev = _StubEvent(ready=False)
        q = RetireQueue()
        q.enqueue(enqueue_step=0, release=lambda: released.append("blk"), event=ev, tag="kv")

        # Delay satisfied, but fence not yet complete -> still held.
        assert q.drain(5) == 0
        assert released == []
        assert q.pending() == 1

        # Fence completes -> released on next drain.
        ev.ready = True
        assert q.drain(5) == 1
        assert released == ["blk"]

    def test_retire_queue_real_cuda_event(self) -> None:
        """Integration: a recorded+synchronized CUDA event gates the release."""
        released = []
        stream = torch.cuda.current_stream()
        x = torch.ones(256, 256, device="cuda")
        _ = x @ x  # enqueue real work
        ev = torch.cuda.Event()
        ev.record(stream)

        q = RetireQueue()
        q.enqueue(enqueue_step=0, release=lambda: released.append("blk"), event=ev)
        ev.synchronize()  # guarantee completion
        assert q.drain(2) == 1
        assert released == ["blk"]

    def test_retire_queue_drain_all_blocking(self) -> None:
        """Shutdown drain synchronizes fences and releases everything."""
        released = []
        ev = _StubEvent(ready=False)
        q = RetireQueue()
        q.enqueue(enqueue_step=10, release=lambda: released.append("a"), event=ev)
        q.enqueue(enqueue_step=10, release=lambda: released.append("b"))
        assert q.drain_all_blocking() == 2
        assert sorted(released) == ["a", "b"]
        assert ev.synchronized is True
        assert q.pending() == 0

    def test_retire_queue_out_of_order_ready(self) -> None:
        """A ready entry is not blocked behind an earlier not-ready one."""
        released = []
        not_ready = _StubEvent(ready=False)
        ready = _StubEvent(ready=True)
        q = RetireQueue()
        q.enqueue(enqueue_step=0, release=lambda: released.append("first"), event=not_ready)
        q.enqueue(enqueue_step=0, release=lambda: released.append("second"), event=ready)

        assert q.drain(2) == 1  # only the ready one
        assert released == ["second"]
        assert q.pending() == 1

    def test_manager_serial_txn_lifecycle_and_diagnostics(self) -> None:
        """begin_serial_txn -> commit updates phase and the always-on counters."""
        mgr = StepTransactionManager(context=None)
        d = mgr.diagnostics
        assert d.as_dict()["serial_steps"] == 0

        txn = mgr.begin_serial_txn(step_id=0, active_request_count=2)
        assert txn.phase is TxnPhase.ADOPTED
        assert txn.speculative is False
        assert (d.prepared, d.adopted, d.serial_steps, d.launched) == (1, 1, 1, 0)
        assert mgr.current_txn is txn

        mgr.commit(txn)
        assert txn.phase is TxnPhase.COMMITTED

        # Diagnostic hooks are O(1) and route to the right counters.
        mgr.note_sync_step("admission")
        mgr.barrier("graph_recapture")
        mgr.note_skip("paused_request")
        mgr.note_guard_failure()
        snap = d.as_dict()
        assert snap["sync_steps"] == 1
        assert snap["barrier_skips"] == 1
        assert snap["guard_failures"] == 1
        assert snap["skip_reasons"] == {
            "admission": 1,
            "graph_recapture": 1,
            "paused_request": 1,
        }

    def test_diagnostics_dict_keys(self) -> None:
        """The diagnostics snapshot exposes the full documented counter set."""
        keys = set(StepTxnDiagnostics().as_dict().keys())
        assert keys == {
            "prepared",
            "launched",
            "adopted",
            "serial_steps",
            "sync_steps",
            "barrier_skips",
            "retired",
            "guard_failures",
            "skip_reasons",
        }


@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestAsyncSchedTxnScaffold(AsyncSchedTxnTestBase):
    """C1: the serial step wrapped as an always-adopted transaction (still serial)."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )

    @classmethod
    def teardown_class(cls):
        delete_cuda_graphs()
        set_rounder(64)
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_manager_only_when_enabled(self) -> None:
        """The manager (and its overhead) exists only when the flag is on."""
        set_rounder(4)
        off_env = self._build_test_env(
            DynamicEngineTestConfig(model_provider="gpt", enable_async_scheduling=False)
        )
        assert off_env.engine.step_txn_manager is None

        on_env = self._build_test_env(
            DynamicEngineTestConfig(model_provider="gpt", enable_async_scheduling=True)
        )
        assert on_env.engine.step_txn_manager is not None
        assert on_env.engine.step_txn_manager.diagnostics.as_dict() == StepTxnDiagnostics().as_dict()

    @pytest.mark.internal
    def test_no_global_async_resource_arrays(self) -> None:
        """The context must not carry global _async_reserved_* / _async_deferred_* arrays."""
        set_rounder(4)
        env = self._build_test_env(
            DynamicEngineTestConfig(model_provider="gpt", enable_async_scheduling=True)
        )
        ctx = env.engine.context
        offenders = [
            name
            for name in vars(ctx)
            if name.startswith("_async_reserved") or name.startswith("_async_deferred")
        ]
        assert offenders == [], f"global async resource arrays present: {offenders}"

    @pytest.mark.internal
    def test_serial_wrapped_txn_equals_serial_bs1(self) -> None:
        """bs=1 tiny GPT decode is token-identical when wrapped as a serial txn."""
        serial_env, async_env = self.assert_async_equals_serial(
            model_provider="gpt",
            num_requests=1,
            num_tokens_to_generate=8,
            num_gap_steps=1,
        )
        # The wrapped path ran as always-adopted serial transactions: every step was
        # prepared+adopted, none launched speculatively, and no guard ever tripped.
        d = async_env.engine.step_txn_manager.diagnostics
        assert d.serial_steps > 0
        assert d.prepared == d.serial_steps
        assert d.adopted == d.serial_steps
        assert d.launched == 0
        assert d.guard_failures == 0
        assert d.retired == 0


class _StubGpuView:
    """A ContextGPUView stand-in exposing a ``_buf`` with a real ``data_ptr()``."""

    def __init__(self, nbytes: int = 64):
        self._buf = torch.zeros(nbytes, dtype=torch.uint8)


@pytest.mark.internal
class TestMetadataSlotPrimitives:
    """C2 unit tests for the decode-metadata slot + depth-2 ring abstractions."""

    def test_slot_reuse_gated_by_fences(self) -> None:
        """A slot is free only when not in use and both fences have retired."""
        slot = DecodeMetadataSlot(slot_id=1, gpu_view=_StubGpuView())
        assert slot.is_free()  # fresh slot, no fences

        slot.acquire()
        assert not slot.is_free()  # in use
        with pytest.raises(AssertionError):
            slot.acquire()  # cannot double-acquire
        slot.release()
        assert slot.is_free()

        # Pending fences keep the slot busy even after release.
        slot.h2d_done_event = _StubEvent(ready=True)
        slot.forward_done_event = _StubEvent(ready=False)
        assert not slot.is_free()
        with pytest.raises(AssertionError):
            slot.acquire()

        # Once the forward fence completes, the slot frees up.
        slot.forward_done_event.ready = True
        assert slot.is_free()
        slot.acquire()  # now allowed
        slot.release()
        slot.reset_fences()
        assert slot.h2d_done_event is None and slot.forward_done_event is None

    def test_slot_identity_and_pointer_assertion(self) -> None:
        """Distinct slots have distinct graph keys; the pointer guard catches moves."""
        a = DecodeMetadataSlot(slot_id=0, gpu_view=_StubGpuView())
        b = DecodeMetadataSlot(slot_id=1, gpu_view=_StubGpuView())
        assert a.graph_slot_key() == a.base_ptr
        assert a.graph_slot_key() != b.graph_slot_key()

        # Captured-against-current-address: guard passes.
        assert_slot_pointer_identity(a, a.base_ptr)
        # Captured-against-the-other-slot's-address: guard fires.
        with pytest.raises(AssertionError):
            assert_slot_pointer_identity(a, b.base_ptr)

    def test_depth_two_ring(self) -> None:
        """The ring has two distinct entries selected by step parity."""
        ring = DepthTwoRing(factory=lambda i: f"buf{i}")
        assert len(ring) == 2
        assert ring[0] == "buf0" and ring[1] == "buf1"
        assert ring[2] == "buf0" and ring[3] == "buf1"  # parity
        assert ring.current(0) == "buf0" and ring.other(0) == "buf1"
        assert ring.current(1) == "buf1" and ring.other(1) == "buf0"

    def test_make_decode_metadata_slots(self) -> None:
        """slot_current wraps the live view; slot_child wraps a distinct fresh view."""
        live = _StubGpuView()
        current, child = make_decode_metadata_slots(
            live, child_gpu_view_factory=lambda: _StubGpuView()
        )
        assert current.gpu_view is live
        assert current.slot_id == 0 and child.slot_id == 1
        assert current.base_ptr != child.base_ptr
        assert current.is_free() and child.is_free()


@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestMetadataSlotsInContext(AsyncSchedTxnTestBase):
    """C2: the context wires slot_current/slot_child only when async is enabled."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )

    @classmethod
    def teardown_class(cls):
        delete_cuda_graphs()
        set_rounder(64)
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_slots_present_only_when_enabled(self) -> None:
        """Serial path: no slots. Async path: slot_current wraps the live gpu_view."""
        set_rounder(4)
        off = self._build_test_env(
            DynamicEngineTestConfig(model_provider="gpt", enable_async_scheduling=False)
        ).engine.context
        assert off.slot_current is None and off.slot_child is None

        ctx = self._build_test_env(
            DynamicEngineTestConfig(model_provider="gpt", enable_async_scheduling=True)
        ).engine.context
        assert ctx.slot_current is not None and ctx.slot_child is not None
        # slot_current is the live gpu_view; slot_child is a distinct free buffer.
        assert ctx.slot_current.gpu_view is ctx.gpu_view
        assert ctx.slot_child.gpu_view is not ctx.gpu_view
        assert ctx.slot_current.base_ptr != ctx.slot_child.base_ptr
        assert ctx.slot_child.is_free()

    @pytest.mark.internal
    @torch.inference_mode()
    def test_serial_metadata_copied_to_child_slot_matches(self) -> None:
        """slot_child is a faithful, identically-laid-out, independent metadata buffer.

        The prestage (a later commit) builds the next step's bookkeeping into the
        child slot and copies it via a single coalesced H2D. This verifies the child
        slot has the exact same byte layout as the live gpu_view (so the copy is
        sound) and uses independent storage (so a running forward on slot_current is
        never disturbed by writes to slot_child).
        """
        set_rounder(4)
        ctx = self._build_test_env(
            DynamicEngineTestConfig(
                model_provider="gpt",
                num_requests=2,
                num_tokens_to_generate=4,
                enable_async_scheduling=True,
            )
        ).engine.context
        live = ctx.slot_current.gpu_view
        child = ctx.slot_child.gpu_view

        # Same backing-buffer size => identical field layout, distinct storage.
        assert live._buf.shape == child._buf.shape
        assert live._buf.data_ptr() != child._buf.data_ptr()

        # Stamp the live buffer with a deterministic pattern, then copy into the
        # child slot exactly as the prestage's coalesced H2D will.
        live._buf.copy_(
            (torch.arange(live._buf.numel(), device=live._buf.device) % 251).to(torch.uint8)
        )
        child._buf.copy_(live._buf)

        for field in (
            "token_to_input_ids",
            "token_to_pos_ids",
            "token_to_block_idx",
            "request_query_lengths",
            "request_kv_length_offsets",
            "mha_cu_query_seq_lengths",
            "mha_cu_kv_seq_lengths",
        ):
            assert torch.equal(getattr(live, field), getattr(child, field)), (
                f"field {field} differs after copy"
            )

        # Independence: mutating the live buffer must not perturb the child slot.
        before = child.token_to_input_ids.clone()
        live.token_to_input_ids[:] += 7
        assert torch.equal(child.token_to_input_ids, before)

    @pytest.mark.internal
    def test_async_equals_serial_with_slots(self) -> None:
        """C2 regression: allocating slots (flag on) keeps decode token-exact vs serial."""
        self.assert_async_equals_serial(
            model_provider="gpt",
            num_requests=4,
            num_tokens_to_generate=8,
            num_gap_steps=1,
        )


@pytest.mark.internal
class TestLaunchEligibilityGate:
    """C5 (diagnostics-only) / section 1.9: the static launch-eligibility gate.

    Pure classification logic for whether a step may launch a speculative child
    forward before commit. No hot path / CUDA-graph surgery -- the prestage that
    builds the child metadata and the launch that consumes it land in later commits.
    """

    @staticmethod
    def _eligible_signals(**overrides) -> LaunchSignals:
        """A baseline of signals for which a speculative launch IS eligible."""
        base = dict(decode_only=True, active_request_count=4, using_cuda_graph=True)
        base.update(overrides)
        return LaunchSignals(**base)

    def test_pure_decode_is_eligible(self) -> None:
        decision = classify_launch_eligibility(self._eligible_signals())
        assert decision.eligibility is LaunchEligibility.ELIGIBLE
        assert decision.reason is LaunchGateReason.PURE_DECODE
        assert decision.eligible is True

    def test_each_ineligible_condition_maps_to_its_reason(self) -> None:
        """Exhaustive: each non-absorbable condition forces a sync step with its reason."""
        cases = [
            (dict(active_request_count=0), LaunchGateReason.NO_ACTIVE_REQUESTS),
            (dict(decode_only=False), LaunchGateReason.NOT_DECODE_ONLY),
            (dict(using_cuda_graph=False), LaunchGateReason.GRAPH_INELIGIBLE),
            (dict(graph_recapture=True), LaunchGateReason.GRAPH_RECAPTURE),
            (dict(chunked_prefill_active=True), LaunchGateReason.CHUNKED_PREFILL),
            (dict(pending_admission=True), LaunchGateReason.PENDING_ADMISSION),
            (dict(resume_pending=True), LaunchGateReason.RESUME),
            (dict(evict_pending=True), LaunchGateReason.EVICT),
            (dict(forced_pause_overflow=True), LaunchGateReason.FORCED_PAUSE_OVERFLOW),
            (dict(mtp_layout_change=True), LaunchGateReason.MTP_DEPENDENT_LAYOUT),
            (dict(kv_reservation_fits=False), LaunchGateReason.KV_RESERVATION_UNAVAILABLE),
        ]
        # Every non-eligible reason is reachable, and no two cases collapse.
        reasons_seen = set()
        for overrides, expected_reason in cases:
            decision = classify_launch_eligibility(self._eligible_signals(**overrides))
            assert decision.eligibility is LaunchEligibility.SYNC, overrides
            assert decision.reason is expected_reason, overrides
            assert not decision.eligible
            reasons_seen.add(expected_reason)
        # The cases cover every LaunchGateReason except the eligible one.
        assert reasons_seen == set(LaunchGateReason) - {LaunchGateReason.PURE_DECODE}

    def test_reason_priority_is_deterministic(self) -> None:
        """When several gates fail, the most fundamental one is reported."""
        # not-decode-only outranks a pending admission.
        d = classify_launch_eligibility(
            self._eligible_signals(decode_only=False, pending_admission=True)
        )
        assert d.reason is LaunchGateReason.NOT_DECODE_ONLY
        # no-active-requests outranks everything.
        d = classify_launch_eligibility(
            self._eligible_signals(active_request_count=0, decode_only=False)
        )
        assert d.reason is LaunchGateReason.NO_ACTIVE_REQUESTS
        # graph recapture outranks downstream layout reasons.
        d = classify_launch_eligibility(
            self._eligible_signals(graph_recapture=True, evict_pending=True)
        )
        assert d.reason is LaunchGateReason.GRAPH_RECAPTURE

    def test_absorbable_events_are_not_gate_inputs(self) -> None:
        """Finish / stop-word / cancel must not appear as launch-gate signals.

        They are absorbable (a launched forward discards the finished row), so they
        never force a sync step -- encoded by their absence from LaunchSignals.
        """
        field_names = set(LaunchSignals.__dataclass_fields__.keys())
        for forbidden in ("finish", "finished", "stop_word", "stopword", "cancel"):
            assert not any(forbidden in name for name in field_names), (
                f"absorbable signal '{forbidden}' must not gate launches"
            )

    def test_launch_decision_dataclass(self) -> None:
        d = LaunchDecision(LaunchEligibility.ELIGIBLE, LaunchGateReason.PURE_DECODE)
        assert d.eligible is True
        d2 = LaunchDecision(LaunchEligibility.SYNC, LaunchGateReason.RESUME)
        assert d2.eligible is False
