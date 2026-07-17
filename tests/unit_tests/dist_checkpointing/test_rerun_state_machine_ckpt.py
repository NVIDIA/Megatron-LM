# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import torch

from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.rerun_state_machine import RerunMode, RerunState, RerunStateMachine
from tests.unit_tests.test_utilities import Utils


class TestRerunStateMachineCheckpointContract:
    """Unit tests for ``RerunStateMachine.state_dict()``.

    These are fast, single-process tests that directly exercise the
    ``state_dict()`` contract. They do not require a full save/load round
    trip because the bug lives entirely in the structure returned by
    ``state_dict()`` on the *first* save, which is what gets cached by
    ``TorchDistSaveShardedStrategy`` when
    ``--ckpt-assume-constant-structure`` is set.
    """

    def setup_method(self, method):
        Utils.initialize_distributed()

    def teardown_method(self, method):
        pass

    def _assert_sharded_object_shape(self, sh_obj):
        """The ShardedObject must be uniquely keyed per rank with a shape
        equal to the world size; this is what lets torch DCP reconstruct
        the global object on load."""
        assert isinstance(sh_obj, ShardedObject)
        assert sh_obj.key == "rerun_state_machine_state"
        assert sh_obj.global_shape == (torch.distributed.get_world_size(),)
        assert sh_obj.global_offset == (torch.distributed.get_rank(),)

    def test_steady_state_emits_sharded_object(self):
        """Regression test for issue #4378.

        In the steady state (no pending rerun), ``state_dict()`` used to
        return ``None`` which left the cached SavePlan with no entry for
        ``rerun_state_machine_state``. After the fix, ``state_dict()`` must
        emit a ShardedObject sentinel so the plan always includes the
        rerun shard from the first save onwards.
        """
        machine = RerunStateMachine(mode=RerunMode.VALIDATE_RESULTS)
        assert machine.state == RerunState.NOT_RUNNING_YET

        sd = machine.state_dict(data_iterator=None, ckpt_format="torch_dist")

        assert sd is not None, (
            "state_dict() must not return None in the steady state when rerun is"
            " enabled; otherwise --ckpt-assume-constant-structure caches a plan"
            " that is missing the rerun_state_machine_state shard and the fault"
            " save silently drops it (issue #4378)."
        )
        assert set(sd.keys()) >= {"mode", "state", "current_iteration", "sharded"}
        assert sd["state"] == RerunState.NOT_RUNNING_YET
        self._assert_sharded_object_shape(sd["sharded"])

    def test_fault_state_emits_sharded_object(self):
        """When a fault is in flight, ``state_dict()`` continues to emit
        the same ShardedObject structure, now carrying the real fault
        payload instead of the sentinel values."""
        machine = RerunStateMachine(mode=RerunMode.VALIDATE_RESULTS)
        machine.state = RerunState.WILL_RERUN_FROM_CHECKPOINT
        machine.rerun_requested = True
        machine.checkpoint_requested = True

        sd = machine.state_dict(data_iterator=None, ckpt_format="torch_dist")

        assert sd is not None
        assert sd["state"] == RerunState.WILL_RERUN_FROM_CHECKPOINT
        self._assert_sharded_object_shape(sd["sharded"])

    def test_structure_constant_across_rerun_transition(self):
        """This is the core invariant the Option-2 fix establishes: the
        ShardedObject's key / global_shape / global_offset are identical
        across the steady-state save and a subsequent fault save, so the
        cached SavePlan built on the first save remains valid when a fault
        triggers the second save."""
        machine = RerunStateMachine(mode=RerunMode.VALIDATE_RESULTS)

        # Save #1: steady state (mirrors every normal checkpoint during a
        # healthy run).
        sd_steady = machine.state_dict(data_iterator=None, ckpt_format="torch_dist")

        # Simulate the transition performed by should_run_forward_backward
        # when a mismatching rerun result demands a fault checkpoint.
        machine.state = RerunState.WILL_RERUN_FROM_CHECKPOINT
        machine.rerun_requested = True
        machine.checkpoint_requested = True

        # Save #2: fault save, performed in the same process with the same
        # (already-cached) TorchDistSaveShardedStrategy.
        sd_fault = machine.state_dict(data_iterator=None, ckpt_format="torch_dist")

        steady_obj = sd_steady["sharded"]
        fault_obj = sd_fault["sharded"]
        assert steady_obj.key == fault_obj.key
        assert steady_obj.global_shape == fault_obj.global_shape
        assert steady_obj.global_offset == fault_obj.global_offset
        # Sanity: the payloads do differ between the two saves (sentinel vs
        # real fault context). torch DCP caches structure, not contents, so
        # this is fine.
        assert steady_obj.data["rerun_requested"] is False
        assert fault_obj.data["rerun_requested"] is True

    def test_steady_state_does_not_require_wrapped_data_iterator(self):
        """In the steady state we skip ``_sanitize_data_iterators`` so the
        caller isn't forced to wrap its training iterator in
        ``RerunDataIterator`` just to satisfy checkpointing. The
        requirement to wrap only kicks in once a rerun is pending."""
        machine = RerunStateMachine(mode=RerunMode.VALIDATE_RESULTS)

        # An unwrapped iterator would assert inside
        # _sanitize_data_iterators. In steady state it must be accepted.
        sd = machine.state_dict(data_iterator=iter([1, 2, 3]), ckpt_format="torch_dist")

        assert sd is not None
        assert sd["sharded"].data["data_iterator_checkpoints"] is None

    def test_disabled_mode_returns_none(self):
        """When the rerun state machine is disabled, ``state_dict()``
        returns ``None`` so disabled jobs don't pay any checkpoint
        overhead."""
        machine = RerunStateMachine(mode=RerunMode.DISABLED)

        sd = machine.state_dict(data_iterator=None, ckpt_format="torch_dist")

        assert sd is None

    def test_non_torch_dist_format_returns_none(self):
        """``ShardedObject`` is only supported by the ``torch_dist``
        format; for other formats ``state_dict()`` returns ``None``
        regardless of machine state."""
        machine = RerunStateMachine(mode=RerunMode.VALIDATE_RESULTS)
        assert machine.state_dict(data_iterator=None, ckpt_format="torch") is None

        machine.state = RerunState.WILL_RERUN_FROM_CHECKPOINT
        assert machine.state_dict(data_iterator=None, ckpt_format="torch") is None

    def test_force_overrides_short_circuits(self):
        """``force=True`` is used on the load path to build a template
        that mirrors whatever the checkpoint happens to contain. It
        bypasses both the DISABLED short-circuit and the ckpt_format
        short-circuit, matching the pre-existing load-side contract in
        ``megatron/training/checkpointing.py``."""
        machine = RerunStateMachine(mode=RerunMode.DISABLED)

        sd = machine.state_dict(data_iterator=None, ckpt_format="torch_dist", force=True)
        assert sd is not None
        self._assert_sharded_object_shape(sd["sharded"])

        # For legacy formats used on the load path (e.g. fsdp_dtensor),
        # force=True still produces the template.
        sd = machine.state_dict(data_iterator=None, ckpt_format="fsdp_dtensor", force=True)
        assert sd is not None
        self._assert_sharded_object_shape(sd["sharded"])
