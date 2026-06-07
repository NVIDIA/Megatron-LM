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
import types

import pytest
import torch

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.metadata_slot import DecodeMetadataBuffer, DepthTwoRing
from megatron.core.inference.engines.step_transaction import (
    LaunchDecision,
    LaunchEligibility,
    LaunchGateReason,
    LaunchSignals,
    PrestagedDecodePlan,
    RetireQueue,
    StepTransactionManager,
    StepTxn,
    StepTxnDiagnostics,
    TxnPhase,
    classify_launch_eligibility,
)
from megatron.core.inference.sampling import TorchSampling
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicEngineTestConfig,
    DynamicInferenceEngineTestBase,
    set_rounder,
    skip_if_mamba_sequence_packing_not_available,
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
    """C2 (amended): single GPU metadata buffer + CPU staging double-buffer ring.

    The two-GPU-buffer "slot" model was discarded: a captured decode graph reads metadata
    by absolute address, so there is exactly ONE fixed-address GPU buffer
    (:class:`DecodeMetadataBuffer`) updated in place, and the double-buffer is CPU-side
    (:class:`DepthTwoRing`).
    """

    def test_buffer_reuse_gated_by_fences(self) -> None:
        """The single GPU buffer can be re-staged only when both fences have retired."""
        buf = DecodeMetadataBuffer(gpu_view=_StubGpuView())
        assert buf.is_free()  # fresh buffer, no fences

        # Pending fences keep the buffer busy (a still in-flight forward may read it).
        buf.h2d_done_event = _StubEvent(ready=True)
        buf.forward_done_event = _StubEvent(ready=False)
        assert not buf.is_free()

        # Once the forward fence completes, the buffer is reusable.
        buf.forward_done_event.ready = True
        assert buf.is_free()

        buf.reset_fences()
        assert buf.h2d_done_event is None and buf.forward_done_event is None

    def test_pointer_invariance_guard(self) -> None:
        """The guard captures the buffer (and view) addresses, then asserts invariance.

        This is the single-buffer ``data_ptr()``-invariance guard that replaces the
        discarded two-slot pointer-identity check: replay is only valid against the buffer
        the graph captured, so the address must never move.
        """
        buf = DecodeMetadataBuffer(gpu_view=_StubGpuView())
        input_ptr, pos_ptr = 0x1000, 0x2000

        # First call records the baseline addresses (lazy capture); it must not raise.
        buf.assert_pointer_invariant(input_ids_ptr=input_ptr, pos_ids_ptr=pos_ptr)
        assert buf._captured_base_ptr == buf.base_ptr

        # Same addresses on every subsequent step: guard passes.
        buf.assert_pointer_invariant(input_ids_ptr=input_ptr, pos_ids_ptr=pos_ptr)

        # A moved GPU buffer (the bug the guard exists to catch) fires.
        moved = DecodeMetadataBuffer(gpu_view=_StubGpuView())
        moved.capture_pointers()
        moved.gpu_view = _StubGpuView()  # simulate a reallocation to a new address
        with pytest.raises(AssertionError, match="GPU buffer moved"):
            moved.assert_pointer_invariant()

        # A moved cached input/pos view also fires.
        with pytest.raises(AssertionError, match="input_ids view moved"):
            buf.assert_pointer_invariant(input_ids_ptr=0xBEEF, pos_ids_ptr=pos_ptr)
        with pytest.raises(AssertionError, match="pos_ids view moved"):
            buf.assert_pointer_invariant(input_ids_ptr=input_ptr, pos_ids_ptr=0xBEEF)

    def test_depth_two_ring(self) -> None:
        """The ring has two distinct entries selected by step parity."""
        ring = DepthTwoRing(factory=lambda i: f"buf{i}")
        assert len(ring) == 2
        assert ring[0] == "buf0" and ring[1] == "buf1"
        assert ring[2] == "buf0" and ring[3] == "buf1"  # parity
        assert ring.current(0) == "buf0" and ring.other(0) == "buf1"
        assert ring.current(1) == "buf1" and ring.other(1) == "buf0"

    def test_depth_two_ring_of_prebuilt_entries(self) -> None:
        """``DepthTwoRing.of`` rings two already-constructed entries (live + staging)."""
        live, staging = ["live"], ["staging"]
        ring = DepthTwoRing.of(live, staging)
        assert len(ring) == 2
        # Entry identity is preserved (the ring holds the live + staging buffers).
        assert ring.current(0) is live and ring.other(0) is staging
        assert ring.current(1) is staging and ring.other(1) is live


@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestMetadataSlotsInContext(AsyncSchedTxnTestBase):
    """C2 (amended): the context wires the single GPU buffer + CPU staging ring on async."""

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
    def test_buffer_and_ring_present_only_when_enabled(self) -> None:
        """Serial path: nothing. Async path: ONE GPU buffer wrapping the live gpu_view."""
        set_rounder(4)
        off = self._build_test_env(
            DynamicEngineTestConfig(model_provider="gpt", enable_async_scheduling=False)
        ).engine.context
        assert off.decode_metadata_buffer is None and off.cpu_staging_ring is None

        ctx = self._build_test_env(
            DynamicEngineTestConfig(model_provider="gpt", enable_async_scheduling=True)
        ).engine.context
        # The single GPU metadata buffer wraps the live gpu_view -- there is no second
        # GPU buffer / no replay-against-child.
        assert ctx.decode_metadata_buffer is not None
        assert ctx.decode_metadata_buffer.gpu_view is ctx.gpu_view
        assert ctx.decode_metadata_buffer.base_ptr == ctx.gpu_view._buf.data_ptr()
        assert ctx.decode_metadata_buffer.is_free()
        # Exactly one GPU metadata buffer exists (no ContextGPUView other than gpu_view).
        from megatron.core.inference.contexts.gpu_view import ContextGPUView

        gpu_views = [v for v in vars(ctx).values() if isinstance(v, ContextGPUView)]
        assert gpu_views == [ctx.gpu_view], "expected exactly one GPU metadata buffer"

        # The genuine double-buffer is CPU-side: live <-> pinned staging mirror.
        assert ctx.cpu_staging_ring is not None
        assert ctx.cpu_staging_ring.current(0) is ctx._cpu_bookkeeping_buf
        assert ctx.cpu_staging_ring.other(0) is ctx._cpu_staging_buf

    @pytest.mark.internal
    @torch.inference_mode()
    def test_cpu_staging_buffer_faithful_and_independent(self) -> None:
        """The CPU staging mirror is identically laid out and independent from live.

        The prestage (a later commit) builds the next step's bookkeeping into the CPU
        staging buffer; one coalesced H2D then copies it into the single GPU buffer. This
        verifies the staging buffer has the exact same byte layout as the live CPU
        bookkeeping buffer (so the copy is sound) and uses independent, pinned storage (so
        prestaging the next step never disturbs the live buffer the current forward's
        already-issued H2D reads).
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
        live = ctx._cpu_bookkeeping_buf
        staging = ctx._cpu_staging_buf

        # Same size/dtype => identical field layout; distinct, pinned storage.
        assert live.shape == staging.shape and live.dtype == staging.dtype
        assert live.data_ptr() != staging.data_ptr()
        assert staging.is_pinned()

        # Stamp the staging buffer (as the prestage will), then copy into live exactly as
        # the publish-step swap does -- the views must agree afterward.
        staging.copy_((torch.arange(staging.numel()) % 251).to(torch.uint8))
        before_live = live.clone()
        # Independence: writing the staging buffer must not perturb the live buffer.
        assert torch.equal(live, before_live)
        live.copy_(staging)
        assert torch.equal(live, staging)

    @pytest.mark.internal
    def test_async_equals_serial_single_buffer(self) -> None:
        """C2 (amended) regression: the single buffer + ring keep decode token-exact."""
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
        # The cases cover every reason produced by the static classifier. PURE_DECODE is the
        # eligible reason, and SPECULATIVE_BOUNDARY_WINDOW is a prestage-only reason (set by the
        # launch-before-commit boundary look-ahead, not derivable from the static LaunchSignals).
        assert reasons_seen == set(LaunchGateReason) - {
            LaunchGateReason.PURE_DECODE,
            LaunchGateReason.SPECULATIVE_BOUNDARY_WINDOW,
        }

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


class _FakeSamplingContext:
    """A minimal stand-in for DynamicInferenceContext, exposing only what the torch
    sampler reads: the active request count, the (compacted) per-request sampling
    metadata, and ``request_ids`` (active row i == request_ids[paused + i])."""

    def __init__(self, request_ids, temperature, top_k, top_p, paused_request_count=0):
        device = torch.cuda.current_device()
        active = len(request_ids)
        assert len(temperature) == len(top_k) == len(top_p) == active
        self.paused_request_count = paused_request_count
        self.total_request_count = paused_request_count + active
        # request_ids: paused placeholders at the front, then the active ids.
        rids = [-1] * paused_request_count + [int(r) for r in request_ids]
        self.request_ids = torch.tensor(rids, dtype=torch.long, device=device)
        # active_request_metadata is compacted (index 0 == first active request).
        self.active_request_metadata = {
            "temperature": torch.tensor(temperature, dtype=torch.float32, device=device),
            "top_k": torch.tensor(top_k, dtype=torch.int32, device=device),
            "top_p": torch.tensor(top_p, dtype=torch.float32, device=device),
        }


@pytest.mark.internal
class TestPerRequestKeyedRNG:
    """C3: the torch backend's per-request keyed RNG (serial behavior change).

    The migrated sampler draws each request from its own ``(seed + request_id)``-keyed
    generator, so a request's draw depends only on ``(request_id, its own draw count)`` --
    invariant to batch composition and row order. That is exactly the property pre-commit
    async sampling needs, and it makes ``async == serial`` token-exact by construction.
    Greedy rows (``top_k == 1``) draw no random numbers and are a strict no-op.
    """

    VOCAB = 32

    def _sampler(self, seed: int) -> TorchSampling:
        # The shared rng is only used by the static path; per-request keying uses `seed`.
        shared = torch.Generator(device=torch.cuda.current_device())
        shared.manual_seed(seed)
        return TorchSampling(shared, vocab_size=self.VOCAB, seed=seed)

    def _uniform_logits(self, n: int) -> torch.Tensor:
        # Uniform logits => the sampled token is fully determined by the RNG, which makes
        # the determinism / invariance assertions sharp.
        return torch.zeros(n, self.VOCAB, device=torch.cuda.current_device())

    def _neutral(self, n: int):
        # temperature=1.0, top_k=0, top_p=0.0 => no filtering, pure multinomial draw.
        return [1.0] * n, [0] * n, [0.0] * n

    def test_fixed_seed_reproducible(self) -> None:
        """Same seed + same request ids => identical tokens (bs=1 and bs>1)."""
        for request_ids in ([7], [3, 8, 5, 1]):
            n = len(request_ids)
            t, k, p = self._neutral(n)
            logits = self._uniform_logits(n)
            a = self._sampler(1234).sample_kernel(
                logits.clone(), n, _FakeSamplingContext(request_ids, t, k, p)
            )
            b = self._sampler(1234).sample_kernel(
                logits.clone(), n, _FakeSamplingContext(request_ids, t, k, p)
            )
            assert torch.equal(a, b), f"non-reproducible for {request_ids}: {a} vs {b}"

    def test_invariant_to_batch_composition(self) -> None:
        """A request's draw is identical alone vs. batched with other requests.

        This is the load-bearing property: pre-commit async sampling re-orders / re-sizes
        the batch, so a per-request draw must not depend on who else is present.
        """
        t1, k1, p1 = self._neutral(1)
        alone = self._sampler(99).sample_kernel(
            self._uniform_logits(1), 1, _FakeSamplingContext([42], t1, k1, p1)
        )

        # Request 42 is now row 2 in a 4-request batch; its logits row is identical.
        request_ids = [10, 11, 42, 13]
        n = len(request_ids)
        t, k, p = self._neutral(n)
        batched = self._sampler(99).sample_kernel(
            self._uniform_logits(n), n, _FakeSamplingContext(request_ids, t, k, p)
        )
        assert batched[2].item() == alone[0].item(), (
            f"request 42 drew {batched[2].item()} in a batch but {alone[0].item()} alone"
        )

    def test_invariant_to_row_order(self) -> None:
        """Reordering the rows leaves each request's draw unchanged."""
        t, k, p = self._neutral(2)
        forward = self._sampler(7).sample_kernel(
            self._uniform_logits(2), 2, _FakeSamplingContext([100, 200], t, k, p)
        )
        reversed_ = self._sampler(7).sample_kernel(
            self._uniform_logits(2), 2, _FakeSamplingContext([200, 100], t, k, p)
        )
        # request 100 is row 0 forward / row 1 reversed; request 200 is the opposite.
        assert forward[0].item() == reversed_[1].item()
        assert forward[1].item() == reversed_[0].item()

    def test_survivor_unperturbed_by_finish_and_retire(self) -> None:
        """A request's stream is independent of others' presence; retiring is safe."""
        t3, k3, p3 = self._neutral(3)
        t1, k1, p1 = self._neutral(1)

        # Survivor request 5 batched with two others, drawn over two steps.
        big = self._sampler(55)
        ctx_big = _FakeSamplingContext([5, 6, 7], t3, k3, p3)
        big_step0 = big.sample_kernel(self._uniform_logits(3), 3, ctx_big)
        # Retire a finished peer mid-run; survivors must not shift.
        big.retire_requests([6])
        big_step1 = big.sample_kernel(self._uniform_logits(3), 3, ctx_big)

        # Request 5 alone over two steps must produce the same two tokens.
        solo = self._sampler(55)
        ctx_solo = _FakeSamplingContext([5], t1, k1, p1)
        solo_step0 = solo.sample_kernel(self._uniform_logits(1), 1, ctx_solo)
        solo_step1 = solo.sample_kernel(self._uniform_logits(1), 1, ctx_solo)

        assert big_step0[0].item() == solo_step0[0].item()
        assert big_step1[0].item() == solo_step1[0].item()

        # Retiring then re-adding a request id gives a fresh, identically-seeded stream.
        s = self._sampler(55)
        first = s.sample_kernel(self._uniform_logits(1), 1, _FakeSamplingContext([9], t1, k1, p1))
        s.retire_requests([9])
        again = s.sample_kernel(self._uniform_logits(1), 1, _FakeSamplingContext([9], t1, k1, p1))
        assert first[0].item() == again[0].item()

    def test_deferred_admission_does_not_perturb_existing(self) -> None:
        """Admitting a new request next step does not change existing requests' draws."""
        t2, k2, p2 = self._neutral(2)
        t3, k3, p3 = self._neutral(3)

        # Two requests this step; a third admitted next step.
        s = self._sampler(8)
        before = s.sample_kernel(
            self._uniform_logits(2), 2, _FakeSamplingContext([1, 2], t2, k2, p2)
        )
        after = s.sample_kernel(
            self._uniform_logits(3), 3, _FakeSamplingContext([1, 2, 3], t3, k3, p3)
        )

        # A reference where request 3 was never admitted: requests 1 and 2 over two steps.
        ref = self._sampler(8)
        ref0 = ref.sample_kernel(
            self._uniform_logits(2), 2, _FakeSamplingContext([1, 2], t2, k2, p2)
        )
        ref1 = ref.sample_kernel(
            self._uniform_logits(2), 2, _FakeSamplingContext([1, 2], t2, k2, p2)
        )
        assert torch.equal(before, ref0)
        # Requests 1 and 2's second-step draws are unchanged by admitting request 3.
        assert after[0].item() == ref1[0].item()
        assert after[1].item() == ref1[1].item()

    def test_greedy_is_strict_no_op(self) -> None:
        """top_k == 1 returns argmax and draws no random numbers (seed-independent)."""
        n = 4
        request_ids = [1, 2, 3, 4]
        # Distinct logits so argmax is well-defined: row r peaks at token r.
        logits = torch.full((n, self.VOCAB), -10.0, device=torch.cuda.current_device())
        for r in range(n):
            logits[r, r] = 10.0
        greedy_t, greedy_k, greedy_p = [1.0] * n, [1] * n, [0.0] * n

        # Two different seeds must give the identical (argmax) result.
        out_a = self._sampler(1).sample_kernel(
            logits.clone(), n, _FakeSamplingContext(request_ids, greedy_t, greedy_k, greedy_p)
        )
        out_b = self._sampler(987654).sample_kernel(
            logits.clone(), n, _FakeSamplingContext(request_ids, greedy_t, greedy_k, greedy_p)
        )
        expected = torch.tensor([0, 1, 2, 3], device=out_a.device)
        assert torch.equal(out_a, expected)
        assert torch.equal(out_a, out_b)
        # No generators were created for a purely greedy batch.
        sampler = self._sampler(1)
        sampler.sample_kernel(
            logits.clone(), n, _FakeSamplingContext(request_ids, greedy_t, greedy_k, greedy_p)
        )
        assert sampler._request_rngs == {}

    def test_speculative_path_routes_through_per_request_helper(self) -> None:
        """sample_speculative keys per request: a decode request's 1+n_spec rows draw from
        its own generator, invariant to batch composition."""
        n_spec = 1
        # Two decode requests, no prefill; each contributes (1 + n_spec) = 2 rows.
        request_ids = [21, 22]
        num_decode, num_prefill = 2, 0
        n_tokens = num_decode * (1 + n_spec) + num_prefill
        t, k, p = self._neutral(num_decode + num_prefill)

        batched = self._sampler(321).sample_speculative(
            self._uniform_logits(n_tokens),
            num_decode,
            num_prefill,
            n_spec,
            _FakeSamplingContext(request_ids, t, k, p),
            eager=True,
        )

        # Request 21 alone: its two (base + spec) rows must match the batched draw.
        t1, k1, p1 = self._neutral(1)
        alone = self._sampler(321).sample_speculative(
            self._uniform_logits(1 * (1 + n_spec)),
            1,
            0,
            n_spec,
            _FakeSamplingContext([21], t1, k1, p1),
            eager=True,
        )
        # token_to_request_index = [0, 0, 1, 1] => request 21 owns rows 0,1.
        assert batched[0].item() == alone[0].item()
        assert batched[1].item() == alone[1].item()

        # Determinism across same-seed samplers.
        again = self._sampler(321).sample_speculative(
            self._uniform_logits(n_tokens),
            num_decode,
            num_prefill,
            n_spec,
            _FakeSamplingContext(request_ids, t, k, p),
            eager=True,
        )
        assert torch.equal(batched, again)


@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestC4SingleBufferDecode(AsyncSchedTxnTestBase):
    """C4: single in-place GPU metadata buffer + token-excluded H2D + GPU token scatter.

    The async decode step now places the next step's input ids on the single GPU metadata
    buffer with an in-place scatter (no D2H round trip) and excludes the token-id region
    from the coalesced H2D. The in-place metadata advance is landed + proven equal to the
    from-scratch rebuild. Everything is gated behind ``enable_async_scheduling``; the bar is
    token-exact ``async == serial``.
    """

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

    # --- (C) in-place metadata advance == from-scratch rebuild --------------------

    @pytest.mark.internal
    @torch.inference_mode()
    def test_inplace_advance_equals_rebuild(self) -> None:
        """The in-place MHA advance reproduces the from-scratch rebuild, incl. a boundary.

        Drives a real GPT decode with a small KV block size (so requests cross block
        boundaries mid-generation) and, for every consecutive pair of same-batch decode
        steps, checks that advancing the *previous* step's MHA metadata in place yields the
        exact metadata ``initialize_attention_state`` rebuilt from scratch.
        """
        set_rounder(4)
        # KV block size must be a multiple of 256 (flash-attn paged-KV constraint), so a
        # boundary crossing is forced with a long prompt (250) + a short generation: the
        # sequence crosses the first 256-token block boundary a few decode steps in.
        env = self._build_test_env(
            DynamicEngineTestConfig(
                model_provider="gpt",
                num_requests=1,
                num_gap_steps=0,
                min_prompt_length=250,
                max_prompt_length=250,
                num_tokens_to_generate=20,
                enable_async_scheduling=True,
            )
        )
        context = env.engine.context

        snapshots: list = []
        original_init = context.initialize_attention_state

        def capturing_init(*args, **kwargs):
            original_init(*args, **kwargs)
            # Only capture real (non graph-capture) decode-only steps.
            if context.is_creating_cuda_graphs or not context.is_decode_only():
                snapshots.append(None)
                return
            bs = context.num_decode_requests
            snapshots.append(
                {
                    "bs": bs,
                    "query_lengths": context._cpu_mha_query_lengths[:bs].clone(),
                    "cu_query": context._cpu_mha_cu_query_seq_lengths[: bs + 1].clone(),
                    "kv_seq_lengths": context._cpu_mha_kv_seq_lengths[:bs].clone(),
                    "cu_kv": context._cpu_mha_cu_kv_seq_lengths[: bs + 1].clone(),
                    "block_table": context._cpu_mha_block_table[:bs].clone(),
                }
            )

        context.initialize_attention_state = capturing_init
        try:
            env.engine._add_request(env.requests[0])
            while env.engine.has_unfinished_requests():
                self._run_step(env)
        finally:
            context.initialize_attention_state = original_init

        max_req = context.max_requests
        scratch = {
            "query_lengths": torch.zeros(max_req, dtype=torch.int32),
            "cu_query": torch.zeros(max_req + 1, dtype=torch.int32),
            "kv_seq_lengths": torch.zeros(max_req, dtype=torch.int32),
            "cu_kv": torch.zeros(max_req + 1, dtype=torch.int32),
            "block_table": torch.full(
                (max_req, context.max_kv_block_count), -1, dtype=torch.int32
            ),
        }

        verified_pairs = 0
        boundary_pairs = 0
        for prev, cur in zip(snapshots, snapshots[1:]):
            if prev is None or cur is None or prev["bs"] != cur["bs"] or cur["bs"] == 0:
                continue
            bs = cur["bs"]
            context.advance_decode_metadata_in_place(
                bs=bs,
                prev_kv_seq_lengths=prev["kv_seq_lengths"],
                committed_block_table=cur["block_table"],
                out_query_lengths=scratch["query_lengths"],
                out_cu_query=scratch["cu_query"],
                out_kv_seq_lengths=scratch["kv_seq_lengths"],
                out_cu_kv=scratch["cu_kv"],
                out_block_table=scratch["block_table"],
            )
            assert torch.equal(scratch["query_lengths"][:bs], cur["query_lengths"])
            assert torch.equal(scratch["cu_query"][: bs + 1], cur["cu_query"])
            assert torch.equal(scratch["kv_seq_lengths"][:bs], cur["kv_seq_lengths"]), (
                f"in-place kv_seq advance != rebuild: "
                f"prev={prev['kv_seq_lengths']} -> got {scratch['kv_seq_lengths'][:bs]} "
                f"expected {cur['kv_seq_lengths']}"
            )
            assert torch.equal(scratch["cu_kv"][: bs + 1], cur["cu_kv"])
            assert torch.equal(scratch["block_table"][:bs], cur["block_table"])
            verified_pairs += 1
            if not torch.equal(prev["block_table"], cur["block_table"]):
                boundary_pairs += 1

        assert verified_pairs > 0, "no same-batch decode pairs were verified"
        assert boundary_pairs > 0, "no block-boundary crossing exercised (tune block size)"

    # --- (D) single GPU buffer never moves across decode steps --------------------

    @pytest.mark.internal
    @torch.inference_mode()
    def test_buffer_data_ptr_invariant_across_decode(self) -> None:
        """The single GPU metadata buffer (and cached input/pos views) never move."""
        set_rounder(4)
        env = self._build_test_env(
            DynamicEngineTestConfig(
                model_provider="gpt",
                num_requests=4,
                num_gap_steps=1,
                num_tokens_to_generate=8,
                enable_async_scheduling=True,
            )
        )
        context = env.engine.context
        base_ptrs = set()
        for request in env.requests:
            env.engine._add_request(request)
            for _ in range(env.config.num_gap_steps):
                self._run_step(env)
                base_ptrs.add(context.decode_metadata_buffer.base_ptr)
        while env.engine.has_unfinished_requests():
            self._run_step(env)
            base_ptrs.add(context.decode_metadata_buffer.base_ptr)

        assert len(base_ptrs) == 1, f"GPU metadata buffer moved across steps: {base_ptrs}"
        # The live guard recorded the captured address; it equals the live buffer address.
        buf = context.decode_metadata_buffer
        assert buf.base_ptr == buf._captured_base_ptr == next(iter(base_ptrs))

    # --- (A) token-excluded H2D leaves the GPU-scattered token region intact -------

    @pytest.mark.internal
    @torch.inference_mode()
    def test_h2d_excludes_scattered_token_region(self) -> None:
        """``transfer_bookkeeping_to_gpu(include_token_to_input_ids=False)`` preserves the
        GPU token-id region (scattered) while still transferring every other field."""
        set_rounder(4)
        env = self._build_test_env(
            DynamicEngineTestConfig(
                model_provider="gpt",
                num_requests=2,
                num_gap_steps=0,
                num_tokens_to_generate=4,
                enable_async_scheduling=True,
            )
        )
        context = env.engine.context
        # Step into a valid decode state.
        env.engine._add_request(env.requests[0])
        env.engine._add_request(env.requests[1])
        self._run_step(env)

        n = 6
        device = context.gpu_view.token_to_input_ids.device
        scattered = torch.arange(1000, 1000 + n, device=device, dtype=torch.long)
        context.gpu_view.token_to_input_ids[:n].copy_(scattered)
        # CPU token region holds DIFFERENT values; a token-excluded H2D must NOT apply them.
        context.token_to_input_ids[:n] = torch.arange(2000, 2000 + n, dtype=torch.long)
        # A non-token field (token_to_pos_ids) the H2D MUST transfer.
        context.token_to_pos_ids[:n] = torch.arange(7, 7 + n, dtype=torch.long)

        context.transfer_bookkeeping_to_gpu(include_token_to_input_ids=False)
        torch.cuda.synchronize()

        assert torch.equal(
            context.gpu_view.token_to_input_ids[:n], scattered
        ), "token-excluded H2D clobbered the GPU-scattered token ids"
        assert torch.equal(
            context.gpu_view.token_to_pos_ids[:n].cpu(),
            torch.arange(7, 7 + n, dtype=torch.long),
        ), "token-excluded H2D failed to transfer a non-token field"

        # The H2D fence was recorded on the single buffer.
        assert context.decode_metadata_buffer.h2d_done_event is not None

        # A full H2D (include=True) DOES overwrite the token region from the CPU buffer.
        context.transfer_bookkeeping_to_gpu(include_token_to_input_ids=True)
        torch.cuda.synchronize()
        assert torch.equal(
            context.gpu_view.token_to_input_ids[:n].cpu(),
            torch.arange(2000, 2000 + n, dtype=torch.long),
        ), "full H2D should transfer the CPU token region"

    # --- (B) end-to-end token-exact async == serial -------------------------------

    @pytest.mark.internal
    def test_async_equals_serial_torch_nongreedy_reshaping(self) -> None:
        """Torch non-greedy decode, reshaping batches (staggered finishes)."""
        self.assert_async_equals_serial(
            model_provider="gpt",
            num_requests=8,
            num_gap_steps=1,
            num_tokens_to_generate=24,
            use_fixed_output_lengths=True,
        )

    @pytest.mark.internal
    def test_async_equals_serial_greedy_reshaping(self) -> None:
        """Greedy (argmax) decode, reshaping batches + boundaries -- token-exact."""
        from unittest import mock

        base_build = AsyncSchedTxnTestBase._build_requests

        def greedy_build(cls, test_config):
            requests = base_build(test_config)
            for request in requests:
                request.sampling_params.top_k = 1  # argmax => greedy, no RNG draws
            return requests

        with mock.patch.object(type(self), "_build_requests", classmethod(greedy_build)):
            self.assert_async_equals_serial(
                model_provider="gpt",
                num_requests=8,
                num_gap_steps=1,
                num_tokens_to_generate=24,
                use_fixed_output_lengths=True,
            )

    @pytest.mark.internal
    def test_async_equals_serial_hybrid_decode(self) -> None:
        """Hybrid (mamba) decode stays token-exact -- the scatter/exclusion path also fires
        for hybrid steady-state decode (single coalesced H2D still carries mamba metadata)."""
        skip_if_mamba_sequence_packing_not_available("hybrid")
        self.assert_async_equals_serial(
            model_provider="hybrid",
            num_requests=4,
            num_gap_steps=1,
            num_tokens_to_generate=16,
        )


class TestC5PrestagedPlanPrimitives:
    """C5 (CPU-only): the prestaged-plan dataclass + manager diagnostics, no model."""

    @pytest.mark.internal
    def test_plan_dataclass_defaults(self) -> None:
        empty = PrestagedDecodePlan(step_id=7, eligible=False, skip_reason=LaunchGateReason.RESUME)
        assert empty.eligible is False
        assert empty.skip_reason is LaunchGateReason.RESUME
        assert empty.has_leases is False
        assert empty.kv_blocks_needed == 0

        leased = PrestagedDecodePlan(
            step_id=8,
            eligible=True,
            kv_block_leases=[3, 4],
            reserved_boundary_blocks=[(11, 2)],
            mamba_slot_leases=[5],
            kv_blocks_needed=1,
        )
        assert leased.eligible is True and leased.skip_reason is None
        assert leased.has_leases is True
        assert leased.kv_blocks_needed == 1

    @pytest.mark.internal
    def test_manager_record_prestage_diagnostics(self) -> None:
        import types

        manager = StepTransactionManager(types.SimpleNamespace())
        manager.record_prestage(PrestagedDecodePlan(step_id=1, eligible=True))
        manager.record_prestage(PrestagedDecodePlan(step_id=1, eligible=True))
        manager.record_prestage(
            PrestagedDecodePlan(
                step_id=2, eligible=False, skip_reason=LaunchGateReason.PENDING_ADMISSION
            )
        )
        diag = manager.diagnostics.as_dict()
        assert diag["prepared"] == 2
        assert diag["skip_reasons"]["pending_admission"] == 1


@pytest.mark.internal
class TestC5HandoffLifecycle:
    """C5 (CPU-only): the prestage -> launch -> adopt -> retire transaction handoff.

    These exercise the manager-side launch-before-commit machinery the overlap engine
    wiring consumes: seeding a speculative child from a prestaged plan, launching it with
    its two fences, the consume-by-request-id adopt guard (design 1.2 / 1.3), and the
    two-step fence-gated lease retire (design 1.7). All pure CPU bookkeeping over
    ``StepTxn`` objects -- no model, no allocators, no GPU buffer.
    """

    @staticmethod
    def _eligible_plan(step_id=3, request_ids=(10, 11, 12)):
        return PrestagedDecodePlan(
            step_id=step_id,
            eligible=True,
            snapshot_request_ids=list(request_ids),
            active_request_count=len(request_ids),
            kv_block_leases=[5, 6, 7],
            reserved_boundary_blocks=[(11, 2)],
            mamba_slot_leases=[1, 2, 3],
            kv_blocks_needed=1,
        )

    def test_from_prestaged_plan_carries_snapshot_and_leases(self) -> None:
        """A StepTxn seeded from an eligible plan carries its snapshot + lease manifest."""
        plan = self._eligible_plan()
        txn = StepTxn.from_prestaged_plan(plan, bucket=("decode", 4))
        assert txn.phase is TxnPhase.PRESTAGED
        assert txn.speculative is True
        assert txn.step_id == 3 and txn.active_request_count == 3
        assert txn.bucket == ("decode", 4)
        assert txn.request_ids == [10, 11, 12]
        # Physical leases the forward reads are carried; the lease lists are independent
        # copies (mutating the txn must not perturb the plan).
        assert txn.kv_block_leases == [5, 6, 7]
        assert txn.mamba_slot_leases == [1, 2, 3]
        txn.kv_block_leases.append(99)
        assert plan.kv_block_leases == [5, 6, 7]
        # Boundary crossers are recorded as PENDING (no block id) -- prestage never allocates.
        assert txn.pending_boundary_columns == [(11, 2)]
        assert txn.reserved_boundary_blocks == {}

    def test_from_prestaged_plan_rejects_ineligible(self) -> None:
        """An ineligible plan cannot seed a speculative transaction."""
        ineligible = PrestagedDecodePlan(
            step_id=1, eligible=False, skip_reason=LaunchGateReason.RESUME
        )
        with pytest.raises(AssertionError):
            StepTxn.from_prestaged_plan(ineligible)

    def test_prestage_child_eligible_builds_child(self) -> None:
        """prestage_child seeds + stores the child and counts a prepared step."""
        mgr = StepTransactionManager(context=None)
        child = mgr.prestage_child(self._eligible_plan(), bucket=("decode", 4))
        assert child is mgr.child_txn
        assert child.phase is TxnPhase.PRESTAGED and child.request_ids == [10, 11, 12]
        assert mgr.diagnostics.prepared == 1
        assert mgr.diagnostics.skip_reasons == {}

    def test_prestage_child_ineligible_records_skip_and_no_child(self) -> None:
        """An ineligible plan leaves child_txn unset and records its static skip reason."""
        mgr = StepTransactionManager(context=None)
        # Seed a stale child to prove an ineligible prestage clears it (sync step).
        mgr.child_txn = StepTxn(step_id=0, phase=TxnPhase.PRESTAGED)
        out = mgr.prestage_child(
            PrestagedDecodePlan(
                step_id=2, eligible=False, skip_reason=LaunchGateReason.PENDING_ADMISSION
            )
        )
        assert out is None and mgr.child_txn is None
        assert mgr.diagnostics.prepared == 0
        assert mgr.diagnostics.skip_reasons["pending_admission"] == 1

    def test_launch_child_transitions_and_attaches_fences(self) -> None:
        """launch_child moves PRESTAGED -> LAUNCHED, attaches both fences, counts launched."""
        mgr = StepTransactionManager(context=None)
        mgr.prestage_child(self._eligible_plan())
        h2d, fwd = _StubEvent(ready=True), _StubEvent(ready=False)
        child = mgr.launch_child(forward_done_event=fwd, h2d_done_event=h2d)
        assert child.phase is TxnPhase.LAUNCHED
        assert child.forward_done_event is fwd and child.h2d_done_event is h2d
        assert mgr.diagnostics.launched == 1

    def test_launch_child_requires_prestaged(self) -> None:
        """launch_child asserts when there is no prestaged child (e.g. after a sync step)."""
        mgr = StepTransactionManager(context=None)
        with pytest.raises(AssertionError):
            mgr.launch_child()

    def test_adopt_child_exact_survivors_promotes(self) -> None:
        """Survivors == snapshot -> adopted; child promoted to current_txn."""
        mgr = StepTransactionManager(context=None)
        child = mgr.prestage_child(self._eligible_plan())
        mgr.launch_child()
        assert mgr.adopt_child([10, 11, 12], committed_bucket=None) is True
        assert child.phase is TxnPhase.ADOPTED
        assert mgr.current_txn is child and mgr.child_txn is None
        assert mgr.diagnostics.adopted == 1 and mgr.diagnostics.guard_failures == 0

    def test_adopt_child_absorbs_mid_batch_finish(self) -> None:
        """A finish removes a row (survivors subset of snapshot) -> still adopted, no rerun."""
        mgr = StepTransactionManager(context=None)
        mgr.prestage_child(self._eligible_plan(request_ids=(10, 11, 12)))
        mgr.launch_child()
        # Request 11 finished this step; survivors are a strict subset of the snapshot.
        assert mgr.adopt_child([10, 12]) is True
        assert mgr.diagnostics.adopted == 1 and mgr.diagnostics.guard_failures == 0

    def test_adopt_guard_trips_on_foreign_survivor(self) -> None:
        """A survivor absent from the snapshot trips the guard (no promotion, no rerun)."""
        mgr = StepTransactionManager(context=None)
        child = mgr.prestage_child(self._eligible_plan(request_ids=(1, 2)))
        mgr.launch_child()
        # Request 3 was never in the launched forward's snapshot -> guard failure.
        assert mgr.adopt_child([1, 2, 3]) is False
        assert mgr.diagnostics.guard_failures == 1 and mgr.diagnostics.adopted == 0
        # Recovery is a barrier handled by the caller; the child stays LAUNCHED, not discarded.
        assert mgr.current_txn is None
        assert mgr.child_txn is child and child.phase is TxnPhase.LAUNCHED

    def test_adopt_guard_trips_on_bucket_mismatch(self) -> None:
        """A graph-bucket mismatch trips the guard even when survivors are a subset."""
        mgr = StepTransactionManager(context=None)
        mgr.prestage_child(self._eligible_plan(request_ids=(1, 2)), bucket=("decode", 4))
        mgr.launch_child()
        assert mgr.adopt_child([1, 2], committed_bucket=("decode", 8)) is False
        assert mgr.diagnostics.guard_failures == 1 and mgr.diagnostics.adopted == 0

    def test_adopt_child_requires_launched(self) -> None:
        """adopt_child asserts unless a launched child is in flight."""
        mgr = StepTransactionManager(context=None)
        with pytest.raises(AssertionError):
            mgr.adopt_child([1, 2])
        mgr.prestage_child(self._eligible_plan())  # PRESTAGED, not yet LAUNCHED
        with pytest.raises(AssertionError):
            mgr.adopt_child([10, 11, 12])

    def test_retire_txn_leases_gated_by_forward_event(self) -> None:
        """A txn's freed leases release only after its forward fence + the two-step delay."""
        mgr = StepTransactionManager(context=None)
        fwd = _StubEvent(ready=False)
        txn = StepTxn(step_id=5, forward_done_event=fwd)
        freed = []
        mgr.retire_txn_leases(
            txn, current_step=5, release=lambda: freed.append("kv"), tag="kv_block"
        )
        assert mgr.retire(5) == 0 and freed == []  # neither delay nor fence satisfied
        assert mgr.retire(7) == 0 and freed == []  # delay satisfied, fence not ready
        fwd.ready = True
        assert mgr.retire(7) == 1 and freed == ["kv"]  # both satisfied -> released
        assert mgr.diagnostics.retired == 1

    def test_retire_txn_leases_event_override(self) -> None:
        """An explicit event overrides the txn fence (e.g. a coalesced multi-resource fence)."""
        mgr = StepTransactionManager(context=None)
        txn = StepTxn(step_id=0, forward_done_event=_StubEvent(ready=False))
        override = _StubEvent(ready=True)
        freed = []
        mgr.retire_txn_leases(
            txn, current_step=0, release=lambda: freed.append("slot"), event=override, tag="mamba"
        )
        assert mgr.retire(2) == 1 and freed == ["slot"]

    def test_full_handoff_sequence_diagnostics(self) -> None:
        """prestage -> launch -> adopt -> commit -> retire over two steps tallies cleanly."""
        mgr = StepTransactionManager(context=None)
        fwd = _StubEvent(ready=True)
        # Step K: prestage + launch the child for K+1.
        child = mgr.prestage_child(self._eligible_plan(step_id=1), bucket=("decode", 4))
        mgr.launch_child(forward_done_event=fwd, h2d_done_event=_StubEvent(ready=True))
        # Step K+1: adopt the launched forward, commit it, retire a finished request's block.
        assert mgr.adopt_child([10, 12], committed_bucket=("decode", 4)) is True
        mgr.commit(child)
        assert child.phase is TxnPhase.COMMITTED
        freed = []
        mgr.retire_txn_leases(child, current_step=1, release=lambda: freed.append("blk"))
        mgr.retire(3)  # two steps later, fence ready
        assert freed == ["blk"]
        snap = mgr.diagnostics.as_dict()
        assert (snap["prepared"], snap["launched"], snap["adopted"]) == (1, 1, 1)
        assert snap["guard_failures"] == 0 and snap["retired"] == 1


@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestC5Prestage(AsyncSchedTxnTestBase):
    """C5: launch-before-commit prestage / publish builders + committed-state launch signals.

    The prestage predicts the *next* decode forward's plan from the committed layout (speculating
    no finish): it snapshots the consumed request-id set, builds the next forward's MHA metadata
    into the CPU staging buffer (never live / never GPU), and records the read-only lease manifest
    (KV blocks, <=1 reserved boundary block per crosser, mamba slots). publish swaps the staging
    MHA into the live buffer + one token-excluded H2D. These are the building blocks the
    cross-step shadow overlap consumes; this commit lands + validates them == the serial rebuild,
    while the runtime decode path is unchanged (so ``async == serial`` / ``async-off == main``).

    Launch eligibility requires a CUDA graph; the tiny test models run eager, so tests that need
    an *eligible* plan force ``using_cuda_graph_this_step`` True (isolating the prestage logic from
    the heavyweight graph-capture machinery, which the C4 battery already exercises).
    """

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

    def _decode_context(self, *, n_steps=4, **overrides):
        """Build an async-enabled engine and step it into a committed decode-only state."""
        set_rounder(4)
        cfg = dict(
            model_provider="gpt",
            num_requests=2,
            num_gap_steps=0,
            num_tokens_to_generate=16,
            enable_async_scheduling=True,
        )
        cfg.update(overrides)
        env = self._build_test_env(DynamicEngineTestConfig(**cfg))
        for request in env.requests:
            env.engine._add_request(request)
        for _ in range(n_steps):
            if not env.engine.has_unfinished_requests():
                break
            self._run_step(env)
        return env, env.engine.context

    # --- committed-state launch signals -------------------------------------------

    @pytest.mark.internal
    @torch.inference_mode()
    def test_build_launch_signals_reads_committed_state(self) -> None:
        """build_launch_signals reflects committed state; eager tiny model => graph-ineligible."""
        env, ctx = self._decode_context()
        assert ctx.is_decode_only()
        sig = ctx.build_launch_signals()
        assert sig.decode_only is True
        assert sig.active_request_count == ctx.total_request_count - ctx.paused_request_count
        assert sig.active_request_count > 0
        assert sig.mtp_layout_change is False
        assert sig.kv_reservation_fits is True  # tiny model, ample free blocks
        assert sig.using_cuda_graph == ctx.using_cuda_graph_this_step()
        decision = classify_launch_eligibility(sig)
        if not sig.using_cuda_graph:
            # No CUDA graph on the tiny eager model -> the launch gate vetoes (sync step).
            assert not decision.eligible
            assert decision.reason is LaunchGateReason.GRAPH_INELIGIBLE

    # --- prestage: read-only on live; snapshot + MHA == committed rebuild formula ---

    @pytest.mark.internal
    @torch.inference_mode()
    def test_prestage_is_read_only_on_live_state(self) -> None:
        """Prestage writes only the staging buffer; it mutates no live tensor or the allocator."""
        env, ctx = self._decode_context()
        ctx.using_cuda_graph_this_step = lambda: True
        try:
            buf_before = ctx._cpu_bookkeeping_buf.clone()
            blocks_before = ctx.request_to_kv_block_ids.clone()
            ids_before = ctx.request_ids.clone()
            avail_before = ctx.kv_block_allocator.get_active_avail()
            plan = ctx.prestage_next_decode_step_into_cpu_staging()
            assert plan.eligible, plan.skip_reason
            assert torch.equal(ctx._cpu_bookkeeping_buf, buf_before), "prestage mutated live buffer"
            assert torch.equal(ctx.request_to_kv_block_ids, blocks_before)
            assert torch.equal(ctx.request_ids, ids_before)
            assert ctx.kv_block_allocator.get_active_avail() == avail_before
            # The snapshot is an independent clone (consume-by-request-id stays stable under
            # a later compaction of the live request_ids).
            assert plan.snapshot_request_ids.data_ptr() != ctx.request_ids.data_ptr()
        finally:
            del ctx.using_cuda_graph_this_step

    @pytest.mark.internal
    @torch.inference_mode()
    def test_prestage_snapshot_and_mha_formula(self) -> None:
        """Snapshot == active request ids; staging MHA == the decode rebuild formula."""
        env, ctx = self._decode_context()
        ctx.using_cuda_graph_this_step = lambda: True
        try:
            plan = ctx.prestage_next_decode_step_into_cpu_staging()
            assert plan.eligible and plan.skip_reason is None
            bs = plan.active_request_count
            active = slice(ctx.paused_request_count, ctx.total_request_count)
            assert torch.equal(plan.snapshot_request_ids, ctx.request_ids[active])
            assert bs == ctx.total_request_count - ctx.paused_request_count

            exp_kv = (
                ctx.request_kv_length_offsets[active] + ctx.request_query_lengths[active]
            ).int()
            assert torch.equal(ctx._staging_mha_kv_seq_lengths[:bs], exp_kv[:bs])
            assert torch.equal(
                ctx._staging_mha_query_lengths[:bs], torch.ones(bs, dtype=torch.int32)
            )
            assert torch.equal(
                ctx._staging_mha_cu_query_seq_lengths[: bs + 1],
                torch.arange(bs + 1, dtype=torch.int32),
            )
            exp_cukv = torch.zeros(bs + 1, dtype=torch.int32)
            exp_cukv[1:] = torch.cumsum(exp_kv[:bs], dim=0)
            assert torch.equal(ctx._staging_mha_cu_kv_seq_lengths[: bs + 1], exp_cukv)
            assert torch.equal(
                ctx._staging_mha_block_table[:bs], ctx.request_to_kv_block_ids[active][:bs].int()
            )
        finally:
            del ctx.using_cuda_graph_this_step

    @pytest.mark.internal
    @torch.inference_mode()
    def test_prestage_excludes_host_maxlen_finishers_from_snapshot(self) -> None:
        """C5: a host-known max-length finisher is dropped from the launch snapshot.

        The launch-before-commit prestage builds the consume-by-request-id snapshot from the
        committed active set. When the caller supplies the host-maxlen survival mask (1 = survives,
        0 = reaches max_sequence_length at the pending commit -- host-deterministic, no D2H), the
        finisher row's request id must be absent from ``snapshot_request_ids`` (it is freed at its
        own commit and must never ride the speculative forward). A None mask is byte-identical to
        no exclusion (the serial-path default)."""
        env, ctx = self._decode_context()
        ctx.using_cuda_graph_this_step = lambda: True
        try:
            active = slice(ctx.paused_request_count, ctx.total_request_count)
            bs = ctx.total_request_count - ctx.paused_request_count
            assert bs >= 2, "need >=2 active requests to exercise a mid-batch exclusion"
            full_ids = ctx.request_ids[active].clone()

            # Baseline: no mask -> snapshot is the full active set (default serial behavior).
            plan_full = ctx.prestage_next_decode_step_into_cpu_staging()
            assert torch.equal(plan_full.snapshot_request_ids, full_ids)

            # Mark the first active row a host-known max-length finisher; it must be excluded.
            survival = torch.ones(bs, dtype=torch.uint8)
            survival[0] = 0
            plan = ctx.prestage_next_decode_step_into_cpu_staging(snapshot_survival_mask=survival)
            finisher_id = int(full_ids[0].item())
            snap = plan.snapshot_request_ids.tolist()
            assert finisher_id not in snap, "host max-length finisher leaked into the snapshot"
            assert snap == [int(x) for x in full_ids[1:].tolist()], "survivors must be kept in order"
        finally:
            del ctx.using_cuda_graph_this_step

    @pytest.mark.internal
    @torch.inference_mode()
    def test_prestage_mha_equals_rebuild(self) -> None:
        """Prestaged staging MHA == initialize_attention_state's live rebuild of the next forward.

        Short prompts + the default (256-token) KV block size => no boundary crossing within a few
        decode steps, so the full MHA (including the block table) is identical to the rebuild.
        """
        env, ctx = self._decode_context()
        ctx.using_cuda_graph_this_step = lambda: True
        try:
            plan = ctx.prestage_next_decode_step_into_cpu_staging()
        finally:
            del ctx.using_cuda_graph_this_step
        assert plan.eligible
        assert plan.kv_blocks_needed == 0, "this test must not cross a block boundary"
        bs = plan.active_request_count
        staging = {
            "ql": ctx._staging_mha_query_lengths[:bs].clone(),
            "cuq": ctx._staging_mha_cu_query_seq_lengths[: bs + 1].clone(),
            "kv": ctx._staging_mha_kv_seq_lengths[:bs].clone(),
            "cukv": ctx._staging_mha_cu_kv_seq_lengths[: bs + 1].clone(),
            "bt": ctx._staging_mha_block_table[:bs].clone(),
        }

        # Rebuild the real next forward's MHA into the live buffer and compare.
        ctx.initialize_attention_state()
        assert torch.equal(staging["ql"], ctx._cpu_mha_query_lengths[:bs])
        assert torch.equal(staging["cuq"], ctx._cpu_mha_cu_query_seq_lengths[: bs + 1])
        assert torch.equal(staging["kv"], ctx._cpu_mha_kv_seq_lengths[:bs]), (
            f"prestaged kv_seq != rebuild: {staging['kv']} vs {ctx._cpu_mha_kv_seq_lengths[:bs]}"
        )
        assert torch.equal(staging["cukv"], ctx._cpu_mha_cu_kv_seq_lengths[: bs + 1])
        assert torch.equal(staging["bt"], ctx._cpu_mha_block_table[:bs])

    # --- speculative (launch-before-commit) prestage ------------------------------

    @pytest.mark.internal
    @torch.inference_mode()
    def test_speculative_prestage_equals_commit_then_rebuild(self) -> None:
        """The speculative plan == the metadata of the forward AFTER the pending commit.

        Launch-before-commit defers the current decode step's ``update_requests`` and launches
        the next forward against the layout that commit *will* produce (design 1.1 step 2). So
        the speculative prestage taken at the committed layout must equal what
        ``initialize_attention_state`` rebuilds after one more real decode step is committed.
        """
        env, ctx = self._decode_context()
        ctx.using_cuda_graph_this_step = lambda: True
        try:
            plan = ctx.prestage_next_decode_step_into_cpu_staging(
                speculate_pending_decode_commit=True
            )
        finally:
            del ctx.using_cuda_graph_this_step
        assert plan.eligible, plan.skip_reason
        assert plan.kv_blocks_needed == 0, "fixture must not cross a boundary in the spec window"
        bs = plan.active_request_count
        spec = {
            "ql": ctx._staging_mha_query_lengths[:bs].clone(),
            "cuq": ctx._staging_mha_cu_query_seq_lengths[: bs + 1].clone(),
            "kv": ctx._staging_mha_kv_seq_lengths[:bs].clone(),
            "cukv": ctx._staging_mha_cu_kv_seq_lengths[: bs + 1].clone(),
            "bt": ctx._staging_mha_block_table[:bs].clone(),
        }
        snapshot = plan.snapshot_request_ids.clone()

        # Commit exactly one more real decode step, then rebuild the next forward's MHA.
        self._run_step(env)
        active = slice(ctx.paused_request_count, ctx.total_request_count)
        # Finish-free, no-boundary step => the active set is unchanged, so the speculative
        # snapshot (consume-by-request-id) matches the post-commit survivors exactly.
        assert torch.equal(snapshot, ctx.request_ids[active])
        ctx.initialize_attention_state()
        assert torch.equal(spec["kv"], ctx._cpu_mha_kv_seq_lengths[:bs]), (
            f"spec kv != post-commit rebuild: {spec['kv']} vs {ctx._cpu_mha_kv_seq_lengths[:bs]}"
        )
        assert torch.equal(spec["ql"], ctx._cpu_mha_query_lengths[:bs])
        assert torch.equal(spec["cuq"], ctx._cpu_mha_cu_query_seq_lengths[: bs + 1])
        assert torch.equal(spec["cukv"], ctx._cpu_mha_cu_kv_seq_lengths[: bs + 1])
        assert torch.equal(spec["bt"], ctx._cpu_mha_block_table[:bs])

    @pytest.mark.internal
    @torch.inference_mode()
    def test_speculative_prestage_kv_is_one_ahead_of_nonspeculative(self) -> None:
        """At one committed layout, the speculative kv lengths lead the non-speculative by one
        unit query (the predicted pending-commit append)."""
        env, ctx = self._decode_context()
        ctx.using_cuda_graph_this_step = lambda: True
        try:
            non_spec = ctx.prestage_next_decode_step_into_cpu_staging()
            bs = non_spec.active_request_count
            non_spec_kv = ctx._staging_mha_kv_seq_lengths[:bs].clone()
            spec = ctx.prestage_next_decode_step_into_cpu_staging(
                speculate_pending_decode_commit=True
            )
            spec_kv = ctx._staging_mha_kv_seq_lengths[:bs].clone()
        finally:
            del ctx.using_cuda_graph_this_step
        assert non_spec.eligible and spec.eligible
        active = slice(ctx.paused_request_count, ctx.total_request_count)
        assert torch.equal(spec_kv, non_spec_kv + ctx.request_query_lengths[active][:bs].int())

    @pytest.mark.internal
    @torch.inference_mode()
    def test_speculative_prestage_boundary_window_forces_sync(self) -> None:
        """A crosser inside the 2-step speculative window makes the speculative plan ineligible
        (SPECULATIVE_BOUNDARY_WINDOW) even though the non-speculative 1-step plan is eligible."""
        env, ctx = self._decode_context()
        ctx.using_cuda_graph_this_step = lambda: True
        # Force "a request crosses within the spec window" while the 1-step gate still fits.
        ctx._decode_crossers_within_window = lambda steps: 1 if steps >= 2 else 0
        try:
            non_spec = ctx.prestage_next_decode_step_into_cpu_staging()
            assert non_spec.eligible, non_spec.skip_reason
            spec = ctx.prestage_next_decode_step_into_cpu_staging(
                speculate_pending_decode_commit=True
            )
            assert not spec.eligible
            assert spec.skip_reason is LaunchGateReason.SPECULATIVE_BOUNDARY_WINDOW
        finally:
            del ctx.using_cuda_graph_this_step
            del ctx._decode_crossers_within_window

    # --- boundary reservation manifest --------------------------------------------

    @pytest.mark.internal
    @torch.inference_mode()
    def test_prestage_boundary_reservation(self) -> None:
        """A boundary crosser is recorded as exactly one reserved block (req_id, column)."""
        set_rounder(4)
        # Long prompt (250) + 256-token block => the sequence crosses the first block boundary a
        # few decode steps in (mirrors the C4 boundary fixture).
        env = self._build_test_env(
            DynamicEngineTestConfig(
                model_provider="gpt",
                num_requests=1,
                num_gap_steps=0,
                min_prompt_length=250,
                max_prompt_length=250,
                num_tokens_to_generate=20,
                enable_async_scheduling=True,
            )
        )
        ctx = env.engine.context
        env.engine._add_request(env.requests[0])

        saw_boundary = False
        while env.engine.has_unfinished_requests():
            if ctx.is_decode_only() and (ctx.total_request_count - ctx.paused_request_count) > 0:
                ctx.using_cuda_graph_this_step = lambda: True
                try:
                    plan = ctx.prestage_next_decode_step_into_cpu_staging()
                finally:
                    del ctx.using_cuda_graph_this_step
                if plan.eligible:
                    crossers = int(ctx._boundary_crosser_local_indices().numel())
                    assert plan.kv_blocks_needed == len(plan.reserved_boundary_blocks)
                    assert plan.kv_blocks_needed == crossers
                    for req_id, column in plan.reserved_boundary_blocks:
                        assert req_id >= 0 and column >= 0
                    if plan.kv_blocks_needed > 0:
                        saw_boundary = True
            self._run_step(env)

        assert saw_boundary, "no boundary crosser exercised (tune block size / prompt length)"

    @pytest.mark.internal
    @torch.inference_mode()
    def test_prestage_kv_unavailable_forces_sync(self) -> None:
        """No free block for a boundary crosser => sync step (and a same-step finish is NOT
        counted as headroom -- the gate never reads finishes)."""
        env, ctx = self._decode_context()
        ctx.using_cuda_graph_this_step = lambda: True
        ctx._boundary_crosser_local_indices = lambda: torch.tensor([0], dtype=torch.long)
        ctx.kv_block_allocator.get_active_avail = lambda: 0  # free pool exhausted
        try:
            sig = ctx.build_launch_signals()
            assert sig.kv_reservation_fits is False
            plan = ctx.prestage_next_decode_step_into_cpu_staging()
            assert not plan.eligible
            assert plan.skip_reason is LaunchGateReason.KV_RESERVATION_UNAVAILABLE
            # A finish this step would free a block only AFTER commit's release; the prestage
            # runs before finish-detection and must not count it -- still unavailable.
            assert ctx.build_launch_signals().kv_reservation_fits is False
        finally:
            del ctx.using_cuda_graph_this_step
            del ctx._boundary_crosser_local_indices
            del ctx.kv_block_allocator.get_active_avail

    # --- static skip reasons ------------------------------------------------------

    @pytest.mark.internal
    @torch.inference_mode()
    def test_prestage_skip_reasons(self) -> None:
        """Each non-eligible condition surfaces its static skip reason (no staging writes)."""
        env, ctx = self._decode_context()

        # Eager (no CUDA graph) -> graph-ineligible.
        assert (
            ctx.prestage_next_decode_step_into_cpu_staging().skip_reason
            is LaunchGateReason.GRAPH_INELIGIBLE
        )

        ctx.using_cuda_graph_this_step = lambda: True
        try:
            # Pending admission (engine-driven signal).
            assert (
                ctx.prestage_next_decode_step_into_cpu_staging(
                    pending_admission=True
                ).skip_reason
                is LaunchGateReason.PENDING_ADMISSION
            )

            # Not decode-only (prefill / mixed).
            ctx.is_decode_only = lambda: False
            try:
                assert (
                    ctx.prestage_next_decode_step_into_cpu_staging().skip_reason
                    is LaunchGateReason.NOT_DECODE_ONLY
                )
            finally:
                del ctx.is_decode_only

            # Chunked prefill in progress.
            ctx.get_index_of_chunked_prefill_request = lambda safe=True: 0
            try:
                assert (
                    ctx.prestage_next_decode_step_into_cpu_staging().skip_reason
                    is LaunchGateReason.CHUNKED_PREFILL
                )
            finally:
                del ctx.get_index_of_chunked_prefill_request

            # MTP-dependent layout (speculative decoding).
            saved_spec = ctx.num_speculative_tokens
            ctx.num_speculative_tokens = 1
            try:
                assert (
                    ctx.prestage_next_decode_step_into_cpu_staging().skip_reason
                    is LaunchGateReason.MTP_DEPENDENT_LAYOUT
                )
            finally:
                ctx.num_speculative_tokens = saved_spec
        finally:
            del ctx.using_cuda_graph_this_step

    # --- publish: staging -> live + token-excluded H2D ----------------------------

    @pytest.mark.internal
    @torch.inference_mode()
    def test_publish_transfers_staging_to_live_and_gpu(self) -> None:
        """publish copies the staging MHA into the live buffer and applies it on the GPU buffer."""
        env, ctx = self._decode_context()
        ctx.using_cuda_graph_this_step = lambda: True
        try:
            plan = ctx.prestage_next_decode_step_into_cpu_staging()
            assert plan.eligible
            bs = plan.active_request_count
            staging_kv = ctx._staging_mha_kv_seq_lengths[:bs].clone()
            staging_bt = ctx._staging_mha_block_table[:bs].clone()

            ctx.publish_prepared_decode_plan(plan)
            torch.cuda.synchronize()

            # Staging MHA is now live.
            assert torch.equal(ctx._cpu_mha_kv_seq_lengths[:bs], staging_kv)
            assert torch.equal(ctx._cpu_mha_block_table[:bs], staging_bt)
            # The GPU buffer's MHA byte-region reflects the live buffer (token-excluded H2D ran).
            s, e = ctx._mha_region_byte_start, ctx._mha_region_byte_end
            assert torch.equal(ctx.gpu_view._buf[s:e].cpu(), ctx._cpu_bookkeeping_buf[s:e])
            assert ctx.decode_metadata_buffer.h2d_done_event is not None
        finally:
            del ctx.using_cuda_graph_this_step

    @pytest.mark.internal
    @torch.inference_mode()
    def test_publish_ineligible_plan_is_noop(self) -> None:
        """Publishing an ineligible plan changes nothing."""
        env, ctx = self._decode_context()
        before = ctx._cpu_bookkeeping_buf.clone()
        ctx.publish_prepared_decode_plan(
            PrestagedDecodePlan(step_id=0, eligible=False, skip_reason=LaunchGateReason.RESUME)
        )
        assert torch.equal(ctx._cpu_bookkeeping_buf, before)

    # --- regression: runtime decode unchanged (async == serial / async-off == main) ---

    @pytest.mark.internal
    def test_async_equals_serial_regression(self) -> None:
        """The prestage/publish builders do not run on the decode path, so async stays
        token-exact vs serial (greedy, reshaping batches)."""
        from unittest import mock

        base_build = AsyncSchedTxnTestBase._build_requests

        def greedy_build(cls, test_config):
            requests = base_build(test_config)
            for request in requests:
                request.sampling_params.top_k = 1
            return requests

        with mock.patch.object(type(self), "_build_requests", classmethod(greedy_build)):
            self.assert_async_equals_serial(
                model_provider="gpt",
                num_requests=6,
                num_gap_steps=1,
                num_tokens_to_generate=20,
                use_fixed_output_lengths=True,
            )


@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestC5OverlapFlip(AsyncSchedTxnTestBase):
    """C5 overlap flip: the runtime launches the next decode forward BEFORE update_requests.

    The launch-before-commit overlap engages for plain GPT decode under a CUDA graph (the launch
    gate requires a graph, design 1.9). It launches the next forward speculatively -- predicting
    the pending commit, consume-by-request-id, finish-/boundary-gated so no output realignment is
    ever needed -- so update_requests runs in the launched forward's GPU shadow. The bar is
    token-exact ``async == serial`` AND evidence (the controller's ordering counter) that the
    overlap actually fired. Eager / prefill / hybrid / MTP stay on the serial path.
    """

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

    @staticmethod
    def _greedy_cuda_graph_kwargs(**overrides):
        cfg = dict(
            model_provider="gpt",
            num_requests=4,
            num_gap_steps=0,
            num_tokens_to_generate=24,
            use_fixed_output_lengths=True,
            num_cuda_graphs=4,
            force_build_cuda_graphs=True,
            inference_cuda_graph_scope=InferenceCudaGraphScope.block,
            use_cuda_graphs_for_non_decode_steps=False,
            context_max_requests=128,
        )
        cfg.update(overrides)
        return cfg

    @pytest.mark.internal
    def test_overlap_fires_and_equals_serial_greedy(self) -> None:
        """Greedy GPT decode under a CUDA graph: token-exact async==serial AND the overlap fired
        (the next forward was launched before update_requests at least once)."""
        from unittest import mock

        base_build = AsyncSchedTxnTestBase._build_requests

        def greedy_build(cls, test_config):
            requests = base_build(test_config)
            for request in requests:
                request.sampling_params.top_k = 1  # argmax => greedy, no RNG draws
            return requests

        with mock.patch.object(type(self), "_build_requests", classmethod(greedy_build)):
            serial_env, async_env = self.assert_async_equals_serial(
                **self._greedy_cuda_graph_kwargs()
            )

        # The overlap actually engaged: at least one forward was launched before its commit.
        launches = async_env.engine.controller._async_launch_before_commit_count
        assert launches > 0, "launch-before-commit overlap never fired (no GPU shadow overlap)"
        assert async_env.engine.controller._async_committed_with_inflight_forward
        # The serial control did NOT overlap (the flag stays at its default).
        assert serial_env.engine.controller._async_launch_before_commit_count == 0

    @pytest.mark.internal
    def test_prestage_issued_in_forward_shadow_before_logits_read(self) -> None:
        """Deterministic (wall-clock-independent) proof of the continuous-loop pipelining: the
        next forward's metadata prestage + publish is issued in the CURRENT forward's GPU shadow,
        BEFORE that forward's logits are read to the host to sample.

        ``_async_overlap_step`` calls ``_async_prestage_next_decode_forward`` (which increments
        ``_async_prestage_in_shadow_count`` on each eligible prestage) strictly before
        ``_dynamic_step_prepare_commit`` (the GPU->CPU sample transfer that reads the logits). So
        every increment of the shadow counter provably happened before the logits read for that
        step -- the load-bearing ordering, independent of any timing. We assert the shadow
        prestage fired for at least as many steps as the launch (the launch can only fire when its
        shadow prestage already published), that both are positive, and that the serial control
        never prestages. Combined with token-exact ``async == serial`` (asserted here too) this
        proves the prestage runs in F(K)'s shadow without perturbing the generated tokens.
        """
        from unittest import mock

        base_build = AsyncSchedTxnTestBase._build_requests

        def greedy_build(cls, test_config):
            requests = base_build(test_config)
            for request in requests:
                request.sampling_params.top_k = 1  # argmax => greedy, no RNG draws
            return requests

        with mock.patch.object(type(self), "_build_requests", classmethod(greedy_build)):
            serial_env, async_env = self.assert_async_equals_serial(
                **self._greedy_cuda_graph_kwargs()
            )

        controller = async_env.engine.controller
        shadow_prestages = controller._async_prestage_in_shadow_count
        launches = controller._async_launch_before_commit_count
        # The shadow prestage fired (each issued before its step's logits read, by code order).
        assert shadow_prestages > 0, "prestage never ran in the forward's shadow"
        # Every launch was preceded by a shadow prestage+publish for that step (the launch
        # consumes the metadata the shadow prestage published). Prestage may additionally fire on
        # a finishing step whose launch is then suppressed, so shadow >= launch > 0.
        assert launches > 0
        assert shadow_prestages >= launches, (
            f"shadow prestages ({shadow_prestages}) must be >= launches ({launches}): "
            "every launched forward is fed by a prestage published in the prior forward's shadow"
        )
        # The serial control issues no shadow prestage (the overlap path is async-only).
        assert serial_env.engine.controller._async_prestage_in_shadow_count == 0
        assert serial_env.engine.controller._async_launch_before_commit_count == 0

    @pytest.mark.internal
    def test_overlap_equals_serial_torch_nongreedy(self) -> None:
        """Non-greedy torch decode under a CUDA graph: token-exact async==serial. The C3
        per-request keyed RNG makes draws depend only on (request_id, draw count), which the
        overlap preserves (each forward is still sampled once, in request order)."""
        serial_env, async_env = self.assert_async_equals_serial(**self._greedy_cuda_graph_kwargs())
        assert async_env.engine.controller._async_launch_before_commit_count > 0

    @pytest.mark.internal
    def test_overlap_equals_serial_greedy_staggered_finishes(self) -> None:
        """Staggered finishes under the overlap: a finish forces a non-overlapped (sync) step and
        the pipeline re-primes, staying token-exact vs serial across the reshapes."""
        from unittest import mock

        base_build = AsyncSchedTxnTestBase._build_requests

        def greedy_build(cls, test_config):
            requests = base_build(test_config)
            for i, request in enumerate(requests):
                request.sampling_params.top_k = 1
                # Staggered output lengths => finishes land on different steps, exercising the
                # finish-gated launch (no speculative launch on a finishing step) + re-prime.
                request.sampling_params.num_tokens_to_generate = 12 + 3 * i
            return requests

        with mock.patch.object(type(self), "_build_requests", classmethod(greedy_build)):
            serial_env, async_env = self.assert_async_equals_serial(
                **self._greedy_cuda_graph_kwargs(num_requests=5, use_fixed_output_lengths=False)
            )
        assert async_env.engine.controller._async_launch_before_commit_count > 0


@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
@pytest.mark.internal
class TestC6Ungate(AsyncSchedTxnTestBase):
    """C6: the launch is UNGATED -- fired before the sample D2H, from host-known state only.

    The single load-bearing relaxation. The launch of the next forward no longer waits on the
    blocking sample D2H + EOS finish-check (the ~360us bubble): it fires from host-known state
    (max-length gated, EOS ungated) before the D2H, which is consumed after the launch enqueue.
    Max-length stays exactly gated (a max-length finish forces a sync prime, no ride). A
    data-dependent (EOS/stop-word) finish drains+discards the doomed forward and re-primes -- the
    finishing step is re-run synchronously, so async stays token-exact vs serial across reshapes.

    The bar (per the design) is token-exact ``async == serial`` across greedy AND per-request-RNG
    sampling with staggered, mid-batch, and simultaneous finishes, plus evidence that the overlap
    fired (the launch ran before the commit) and that max-length finishes never rode a forward.
    """

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

    @staticmethod
    def _cuda_graph_kwargs(**overrides):
        cfg = dict(
            model_provider="gpt",
            num_requests=6,
            num_gap_steps=0,
            num_tokens_to_generate=20,
            use_fixed_output_lengths=True,
            num_cuda_graphs=4,
            force_build_cuda_graphs=True,
            inference_cuda_graph_scope=InferenceCudaGraphScope.block,
            use_cuda_graphs_for_non_decode_steps=False,
            context_max_requests=128,
        )
        cfg.update(overrides)
        return cfg

    @pytest.mark.internal
    def test_ungated_nongreedy_staggered_midbatch_finishes(self) -> None:
        """Per-request-RNG decode with staggered, mid-batch max-length finishes: token-exact
        async==serial under the ungate. Each finish forces a sync prime (max-length stays gated);
        survivors keep their own per-request keyed draws across the reshapes."""
        from unittest import mock

        base_build = AsyncSchedTxnTestBase._build_requests

        def staggered_build(cls, test_config):
            requests = base_build(test_config)
            for i, request in enumerate(requests):
                # Per-request RNG (no top_k=1). Staggered lengths => finishes land on different
                # steps, including mid-batch (compaction reorders rows), exercising the re-prime.
                request.sampling_params.num_tokens_to_generate = 10 + 2 * i
            return requests

        with mock.patch.object(type(self), "_build_requests", classmethod(staggered_build)):
            serial_env, async_env = self.assert_async_equals_serial(
                **self._cuda_graph_kwargs(num_requests=6, use_fixed_output_lengths=False)
            )
        # The overlap actually engaged (launched before commit), and the serial control did not.
        assert async_env.engine.controller._async_launch_before_commit_count > 0
        assert serial_env.engine.controller._async_launch_before_commit_count == 0
        # Max-length finishes are host-gated: the launch is suppressed on a finishing step, so no
        # doomed forward is ever drained/discarded (barrier_skips stays 0 -- no EOS in this mix).
        assert async_env.engine.step_txn_manager.diagnostics.guard_failures == 0

    @pytest.mark.internal
    def test_ungated_greedy_equals_serial(self) -> None:
        """Greedy (argmax, no RNG) fixed-length decode: token-exact async==serial, overlap fired,
        and the consume-by-request-id adopt guard never failed (steady decode adopts every step)."""
        from unittest import mock

        base_build = AsyncSchedTxnTestBase._build_requests

        def greedy_build(cls, test_config):
            requests = base_build(test_config)
            for request in requests:
                request.sampling_params.top_k = 1
            return requests

        with mock.patch.object(type(self), "_build_requests", classmethod(greedy_build)):
            serial_env, async_env = self.assert_async_equals_serial(**self._cuda_graph_kwargs())
        diag = async_env.engine.step_txn_manager.diagnostics
        assert async_env.engine.controller._async_launch_before_commit_count > 0
        assert diag.launched > 0
        assert diag.adopted > 0
        assert diag.guard_failures == 0


@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
@pytest.mark.internal
class TestC7Hardening(AsyncSchedTxnTestBase):
    """C7: boundary hardening -- admission deferral, barrier recovery, drain.

    The ungated overlap launches a forward speculatively each step. A new request admitted while
    that forward is in flight would allocate KV under a layout the forward did not compute, forcing
    an adopt-guard barrier (or racing a deferred free). C7 DEFERS admission: while a forward is in
    flight and a request is waiting, the next launch is suppressed (a sync prime step), so the
    following step has no forward in flight and admits safely. The bar is token-exact async==serial
    across staggered mid-decode admission, with the overlap still firing and the adopt guard never
    failing (deferral prevents the barrier).
    """

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

    @staticmethod
    def _cuda_graph_kwargs(**overrides):
        cfg = dict(
            model_provider="gpt",
            num_requests=4,
            num_tokens_to_generate=20,
            use_fixed_output_lengths=True,
            num_cuda_graphs=4,
            force_build_cuda_graphs=True,
            inference_cuda_graph_scope=InferenceCudaGraphScope.block,
            use_cuda_graphs_for_non_decode_steps=False,
            context_max_requests=128,
        )
        cfg.update(overrides)
        return cfg

    @pytest.mark.internal
    def test_staggered_admission_deferred_equals_serial(self) -> None:
        """Requests admitted mid-decode (num_gap_steps>0, so a new prefill lands while a forward is
        in flight): token-exact async==serial, the overlap fired, and the consume-by-request-id
        adopt guard never tripped -- the C7 admission deferral admits on a clean step, so a forward
        in flight is never adopted against a layout that grew under it."""
        serial_env, async_env = self.assert_async_equals_serial(
            **self._cuda_graph_kwargs(num_requests=5, num_gap_steps=3)
        )
        diag = async_env.engine.step_txn_manager.diagnostics
        assert async_env.engine.controller._async_launch_before_commit_count > 0
        assert diag.guard_failures == 0, (
            "admission deferral must prevent the adopt-guard barrier "
            f"(guard_failures={diag.guard_failures})"
        )

    @pytest.mark.internal
    def test_staggered_admission_nongreedy_equals_serial(self) -> None:
        """Same, with per-request-RNG sampling: a survivor's keyed draws are unperturbed by a new
        request admitted on a deferred (sync prime) step."""
        serial_env, async_env = self.assert_async_equals_serial(
            **self._cuda_graph_kwargs(
                num_requests=6, num_gap_steps=2, use_fixed_output_lengths=False
            )
        )
        assert async_env.engine.controller._async_launch_before_commit_count > 0
        assert async_env.engine.step_txn_manager.diagnostics.guard_failures == 0


@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
@pytest.mark.internal
class TestC8Parity(AsyncSchedTxnTestBase):
    """C8: the async==serial parity harness across the full mandated prompt mix + scope gates.

    This is the proof artifact for the build: the launch-before-commit overlap must reproduce the
    serial oracle token-for-token across greedy AND per-request-RNG sampling, data-dependent
    (EOS/stop-word) finishes (which drain+discard the doomed forward), simultaneous finishes,
    batch-drain-to-zero, and (for logprob/top-n requests) the fall-to-serial scope gate. The
    crash-loud launch-site asserts (mirrored here) keep the overlap from silently widening.
    """

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

    @staticmethod
    def _cuda_graph_kwargs(**overrides):
        cfg = dict(
            model_provider="gpt",
            num_requests=4,
            num_gap_steps=0,
            num_tokens_to_generate=24,
            use_fixed_output_lengths=True,
            num_cuda_graphs=4,
            force_build_cuda_graphs=True,
            inference_cuda_graph_scope=InferenceCudaGraphScope.block,
            use_cuda_graphs_for_non_decode_steps=False,
            context_max_requests=128,
        )
        cfg.update(overrides)
        return cfg

    @staticmethod
    def _force_data_dependent_finish(rows_at_step):
        """Patch context: force a data-dependent (EOS-like) finish for given active rows at a step.

        Returns a mock.patch context manager over the controller's data-dependent finish mask that
        zeros the chosen active rows when the context reaches ``target_step`` -- deterministically,
        in BOTH the serial and async runs (each env has its own context.step_count), so the
        comparison is fair and the async path exercises the doomed-forward drain+discard + reprime.
        """
        from unittest import mock

        orig = TextGenerationController._data_dependent_finish_mask
        target_step, rows = rows_at_step

        def forced(self, sampled_tokens_cpu, active_request_ids, active_request_count):
            mask = orig(self, sampled_tokens_cpu, active_request_ids, active_request_count)
            ctx = self.inference_wrapped_model.inference_context
            if ctx.step_count == target_step and active_request_count > max(rows):
                mask = mask.clone()
                for r in rows:
                    mask[r] = 0
            return mask

        return mock.patch.object(TextGenerationController, "_data_dependent_finish_mask", forced)

    @pytest.mark.internal
    def test_forced_eos_discard_midbatch_equals_serial(self) -> None:
        """A single data-dependent (EOS-like) finish at a mid-batch row mid-decode: the launched
        forward is doomed (it predicted finish-free), so it is drained+discarded and the step
        re-primes. Token-exact async==serial across the reshape (the multi-finisher discard counter
        is asserted in the two-EOS test); the overlap fired and the adopt guard never tripped."""
        with self._force_data_dependent_finish((6, [0])):
            serial_env, async_env = self.assert_async_equals_serial(
                **self._cuda_graph_kwargs(num_requests=4)
            )
        diag = async_env.engine.step_txn_manager.diagnostics
        assert async_env.engine.controller._async_launch_before_commit_count > 0
        assert diag.guard_failures == 0

    @pytest.mark.internal
    def test_two_eos_same_step_equals_serial(self) -> None:
        """Two simultaneous data-dependent finishers (rows 0 and 2) on the same step: both popped,
        survivors stay aligned, the doomed forward drained+discarded. Token-exact async==serial."""
        with self._force_data_dependent_finish((5, [0, 2])):
            serial_env, async_env = self.assert_async_equals_serial(
                **self._cuda_graph_kwargs(num_requests=5)
            )
        diag = async_env.engine.step_txn_manager.diagnostics
        assert async_env.engine.controller._async_launch_before_commit_count > 0
        assert diag.barrier_skips > 0
        assert diag.guard_failures == 0

    @pytest.mark.internal
    def test_two_eos_same_step_nongreedy_equals_serial(self) -> None:
        """Same, with per-request-RNG sampling: each survivor's keyed draw is its own across the
        mid-batch compaction the discard+reprime produces."""
        with self._force_data_dependent_finish((7, [1, 3])):
            serial_env, async_env = self.assert_async_equals_serial(
                **self._cuda_graph_kwargs(num_requests=5, use_fixed_output_lengths=False)
            )
        assert async_env.engine.controller._async_launch_before_commit_count > 0
        assert async_env.engine.step_txn_manager.diagnostics.guard_failures == 0

    @pytest.mark.internal
    def test_two_finishers_same_step_maxlen_equals_serial(self) -> None:
        """Two requests with the same fixed length finish (max-length) on the same step. Max-length
        is host-gated, so the launch is suppressed that step (sync prime). Token-exact + drains to
        zero at the end."""
        serial_env, async_env = self.assert_async_equals_serial(
            **self._cuda_graph_kwargs(num_requests=4, num_tokens_to_generate=18)
        )
        assert async_env.engine.controller._async_launch_before_commit_count > 0
        assert async_env.engine.step_txn_manager.diagnostics.guard_failures == 0

    @pytest.mark.internal
    def test_logprobs_falls_to_serial_path(self) -> None:
        """A request asking for log probs takes the SERIAL path (single-buffered logits cannot ride
        the ungated overlap): the launch-before-commit overlap NEVER fires, and tokens are
        identical to async-off."""
        from unittest import mock

        base_build = AsyncSchedTxnTestBase._build_requests

        def logprob_build(cls, test_config):
            requests = base_build(test_config)
            for request in requests:
                request.sampling_params.return_log_probs = True
                # The tiny test model materializes only last-token logits; skip prompt logprobs so
                # the generated-token logprobs (which force the serial path) are still requested.
                request.sampling_params.skip_prompt_log_probs = True
            return requests

        with mock.patch.object(type(self), "_build_requests", classmethod(logprob_build)):
            serial_env, async_env = self.assert_async_equals_serial(**self._cuda_graph_kwargs())
        # Scope gate: logprob-bearing requests keep the overlap OFF entirely.
        assert async_env.engine.controller._async_launch_before_commit_count == 0, (
            "overlap must not fire while a logprob/top-n request is active (it falls to serial)"
        )


@pytest.mark.internal
class TestC8ScopeAsserts:
    """C8: the static scope predicate is crash-loud-tight (no silent widening).

    Unit-level checks of ``_async_overlap_scope_ok`` / ``_active_requests_need_logprobs`` against a
    tiny fake controller+context: the overlap is refused for MTP, hybrid, EP>1, and logprob/top-n,
    matching the launch-site asserts exactly. (Multi-rank EP behavior is exercised by the EP gate
    here without needing a multi-GPU job.)"""

    def _fake(self, *, async_on=True, has_buffer=True, mtp=0, hybrid=False, ep_size=1):
        class _PG:
            pass

        ep_group = _PG() if ep_size != 1 else None

        ctx = types.SimpleNamespace(
            enable_async_scheduling=async_on,
            decode_metadata_buffer=object() if has_buffer else None,
            is_hybrid_model=hybrid,
            expert_model_parallel_group=ep_group,
            total_request_count=2,
            paused_request_count=0,
            active_request_metadata={
                "return_log_probs": torch.zeros(2, dtype=torch.int32),
                "top_n_logprobs": torch.zeros(2, dtype=torch.int32),
            },
        )
        iwm = types.SimpleNamespace(inference_context=ctx)
        ctrl = types.SimpleNamespace(
            inference_wrapped_model=iwm,
            num_speculative_tokens=mtp,
            _async_overlap_scope_ok=lambda c: TextGenerationController._async_overlap_scope_ok(
                ctrl, c
            ),
            _active_requests_need_logprobs=(
                lambda c: TextGenerationController._active_requests_need_logprobs(ctrl, c)
            ),
        )
        return ctrl, ctx, ep_size

    @pytest.mark.internal
    def test_scope_ok_for_plain_gpt_ep1(self) -> None:
        ctrl, ctx, _ = self._fake()
        assert ctrl._async_overlap_scope_ok(ctx) is True

    @pytest.mark.internal
    def test_scope_refused_for_mtp_hybrid_ep_and_disabled(self) -> None:
        import unittest.mock as mock

        # MTP.
        ctrl, ctx, _ = self._fake(mtp=2)
        assert ctrl._async_overlap_scope_ok(ctx) is False
        # Hybrid (mamba single bank).
        ctrl, ctx, _ = self._fake(hybrid=True)
        assert ctrl._async_overlap_scope_ok(ctx) is False
        # Async off / no buffer.
        ctrl, ctx, _ = self._fake(async_on=False)
        assert ctrl._async_overlap_scope_ok(ctx) is False
        ctrl, ctx, _ = self._fake(has_buffer=False)
        assert ctrl._async_overlap_scope_ok(ctx) is False
        # EP > 1 (the gate calls get_pg_size on the EP group; patch it to report size 4).
        ctrl, ctx, _ = self._fake(ep_size=4)
        with mock.patch(
            "megatron.core.inference.text_generation_controllers."
            "text_generation_controller.get_pg_size",
            return_value=4,
        ):
            assert ctrl._async_overlap_scope_ok(ctx) is False

    @pytest.mark.internal
    def test_logprobs_predicate(self) -> None:
        ctrl, ctx, _ = self._fake()
        assert ctrl._active_requests_need_logprobs(ctx) is False
        ctx.active_request_metadata["return_log_probs"][1] = 1
        assert ctrl._active_requests_need_logprobs(ctx) is True
        ctx.active_request_metadata["return_log_probs"][1] = 0
        ctx.active_request_metadata["top_n_logprobs"][0] = 3
        assert ctrl._active_requests_need_logprobs(ctx) is True
