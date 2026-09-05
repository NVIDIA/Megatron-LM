# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Failure-path tests for the persistent asynchronous checkpoint worker."""

import os
from functools import partial

from megatron.core.dist_checkpointing.strategies.async_utils import (
    AsyncCallsQueue,
    AsyncRequest,
    PersistentAsyncCaller,
    PersistentAsyncWorkerError,
)
from tests.unit_tests.test_utilities import Utils


def _fail_on_rank(rank: int, failing_rank: int) -> None:
    """Raise in the persistent worker belonging to ``failing_rank``."""
    if rank == failing_rank:
        raise RuntimeError(f"injected checkpoint failure on rank {rank}")


def _complete_successfully() -> None:
    """A picklable no-op used to verify that a replacement worker can start."""


def _exit_worker(exit_code: int) -> None:
    """Exit a persistent worker without publishing a completion result."""
    os._exit(exit_code)


def _preload_on_rank(rank: int, failing_rank: int, hard_exit: bool):
    """Fail during preloading on one rank and return staged data elsewhere."""
    if rank == failing_rank:
        if hard_exit:
            os._exit(17)
        raise RuntimeError(f"injected preload failure on rank {rank}")
    return object()


def _consume_preloaded(_header, _payload, _footer) -> None:
    """A picklable async function with the three-argument preload contract."""


class TestPersistentAsyncWorkerFailure:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_worker_failure_is_propagated_to_all_ranks(self):
        Utils.initialize_model_parallel(1, 1)
        async_calls = AsyncCallsQueue(persistent=True)
        request = AsyncRequest(
            async_fn=_fail_on_rank, async_fn_args=(Utils.rank, 0), finalize_fns=[]
        )

        try:
            async_calls.schedule_async_request(request)
            async_calls.schedule_async_request(
                AsyncRequest(async_fn=_complete_successfully, async_fn_args=(), finalize_fns=[])
            )
            try:
                async_calls.maybe_finalize_async_calls(blocking=True)
            except PersistentAsyncWorkerError as exc:
                reported_error = str(exc)
            else:
                raise AssertionError("persistent checkpoint worker failure was not propagated")
        finally:
            async_calls.close()

        assert "injected checkpoint failure on rank 0" in reported_error
        assert async_calls.get_num_unfinalized_calls() == 0
        assert AsyncCallsQueue._persistent_caller is None
        assert PersistentAsyncCaller._persistent_process is None

        replacement_calls = AsyncCallsQueue(persistent=True)
        replacement_request = AsyncRequest(
            async_fn=_complete_successfully, async_fn_args=(), finalize_fns=[]
        )
        try:
            replacement_calls.schedule_async_request(replacement_request)
            assert replacement_calls.maybe_finalize_async_calls(blocking=True) == [0]
        finally:
            replacement_calls.close()

    def test_hard_exit_during_async_call_is_propagated_to_all_ranks(self):
        Utils.initialize_model_parallel(1, 1)
        async_calls = AsyncCallsQueue(persistent=True)
        request = AsyncRequest(async_fn=_exit_worker, async_fn_args=(17,), finalize_fns=[])

        try:
            async_calls.schedule_async_request(request)
            try:
                async_calls.maybe_finalize_async_calls(blocking=True)
            except PersistentAsyncWorkerError as exc:
                reported_error = str(exc)
            else:
                raise AssertionError("persistent checkpoint worker exit was not propagated")
        finally:
            async_calls.close()

        assert "exit code 17" in reported_error
        assert async_calls.get_num_unfinalized_calls() == 0
        assert AsyncCallsQueue._persistent_caller is None
        assert PersistentAsyncCaller._persistent_process is None

    def test_preload_failure_is_propagated_to_all_ranks(self):
        Utils.initialize_model_parallel(1, 1)
        async_calls = AsyncCallsQueue(persistent=True)
        request = AsyncRequest(
            async_fn=_consume_preloaded,
            async_fn_args=(None, None, None),
            finalize_fns=[],
            preload_fn=partial(_preload_on_rank, Utils.rank, 0, False),
        )

        try:
            async_calls.schedule_async_request(request)
            try:
                async_calls.maybe_finalize_async_calls(blocking=True)
            except PersistentAsyncWorkerError as exc:
                reported_error = str(exc)
            else:
                raise AssertionError("persistent checkpoint preload failure was not propagated")
        finally:
            async_calls.close()

        assert "injected preload failure on rank 0" in reported_error
        assert AsyncCallsQueue._persistent_caller is None
        assert PersistentAsyncCaller._persistent_process is None

    def test_hard_exit_during_preload_is_propagated_to_all_ranks(self):
        Utils.initialize_model_parallel(1, 1)
        async_calls = AsyncCallsQueue(persistent=True)
        request = AsyncRequest(
            async_fn=_consume_preloaded,
            async_fn_args=(None, None, None),
            finalize_fns=[],
            preload_fn=partial(_preload_on_rank, Utils.rank, 0, True),
        )

        try:
            async_calls.schedule_async_request(request)
            try:
                async_calls.maybe_finalize_async_calls(blocking=True)
            except PersistentAsyncWorkerError as exc:
                reported_error = str(exc)
            else:
                raise AssertionError("persistent checkpoint worker exit was not propagated")
        finally:
            async_calls.close()

        assert "exit code 17" in reported_error
        assert AsyncCallsQueue._persistent_caller is None
        assert PersistentAsyncCaller._persistent_process is None
