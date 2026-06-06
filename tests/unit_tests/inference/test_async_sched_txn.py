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
from megatron.core.inference.engines.step_transaction import (
    RetireQueue,
    StepTransactionManager,
    StepTxn,
    StepTxnDiagnostics,
    TxnPhase,
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
