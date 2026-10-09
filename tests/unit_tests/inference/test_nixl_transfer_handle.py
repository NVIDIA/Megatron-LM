# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""NIXL failure cleanup and buffer ownership regressions."""

from unittest import mock

import pytest

from megatron.core.inference.disaggregation.kv_reshard import KVShardLayout
from megatron.core.inference.disaggregation.transfer_backends.base import TransferStartError
from megatron.core.inference.disaggregation.transfer_backends.nixl import (
    NixlPullHandle,
    NixlTransferBackend,
)


@pytest.mark.parametrize("state", ["ERR", Exception("NIXL_ERR_INVALID_PARAM")])
def test_poll_failure_releases_all_transfers_once(state):
    agent = mock.Mock()
    agent.check_xfer_state.side_effect = [state]
    xfers = [object(), object()]
    # Even an expired deadline must not leave the exceptional transfer untracked.
    handle = NixlPullHandle(agent, xfers, ["failed", "active"], submitted_at=-60, timeout_s=30)

    for _ in range(3):
        with pytest.raises(RuntimeError, match="NIXL transfer failed"):
            handle.poll()
        assert handle.done
        assert handle.storage_safe

    agent.check_xfer_state.assert_called_once_with(xfers[0])
    assert agent.release_xfer_handle.call_args_list == [mock.call(xfer) for xfer in xfers]
    assert handle.xfers == []


@pytest.mark.parametrize("failed_release", [0, 1])
def test_release_failure_retains_storage_without_repolling(failed_release):
    agent = mock.Mock()
    agent.check_xfer_state.side_effect = Exception("NIXL_ERR_INVALID_PARAM")
    xfers = [object(), object()]
    agent.release_xfer_handle.side_effect = [
        Exception("cannot cancel") if i == failed_release else None for i in range(2)
    ]
    handle = NixlPullHandle(agent, xfers, ["first", "second"], submitted_at=0)

    for _ in range(3):
        with pytest.raises(RuntimeError, match="storage remains quarantined"):
            handle.poll()
        assert not handle.done
        assert not handle.storage_safe

    agent.check_xfer_state.assert_called_once_with(xfers[0])
    assert agent.release_xfer_handle.call_args_list == [mock.call(xfer) for xfer in xfers]
    assert handle.xfers == [xfers[failed_release]]
    assert handle.contexts == [["first", "second"][failed_release]]


def test_timeout_does_not_establish_storage_safety():
    agent = mock.Mock()
    agent.check_xfer_state.side_effect = ["PROC", "DONE"]
    handle = NixlPullHandle(agent, [object()], ["slow"], submitted_at=-60, timeout_s=30)

    with pytest.raises(TimeoutError):
        handle.poll()
    assert not handle.storage_safe
    assert handle.poll()
    assert handle.storage_safe
    agent.release_xfer_handle.assert_not_called()


@pytest.mark.parametrize("wrapped_start_error", [False, True])
def test_partial_submission_retains_unsafely_released_transfers(wrapped_start_error):
    backend = object.__new__(NixlTransferBackend)
    backend._agent = mock.Mock()
    backend._agent.check_xfer_state.side_effect = Exception("NIXL_ERR_INVALID_PARAM")
    backend._agent.release_xfer_handle.side_effect = Exception("cannot cancel")
    backend._ssm_layout = None
    backend._layout = KVShardLayout(
        num_layers=1, num_heads=2, tp_size=1, tp_rank=0, pp_size=1, pp_rank=0, global_rank=2
    )
    backend._num_outer = 2
    backend._blocks_axis = 2
    backend._bytes_per_slice = 32
    backend._validate_peer = mock.Mock()
    xfer = object()
    start_error = (
        TransferStartError("submission failed", storage_safe=False)
        if wrapped_start_error
        else ValueError("submission failed")
    )
    backend._begin_transfer = mock.Mock(side_effect=[(xfer, "first"), start_error])
    peers = [
        {
            "global_rank": rank,
            "tp_size": 2,
            "tp_rank": rank,
            "pp_size": 1,
            "pp_rank": 0,
            "num_layers_global": 1,
            "num_kv_heads_global": 2,
            "layer_start": 0,
            "layer_end": 1,
            "heads_per_partition": 1,
            "num_outer": 2,
            "bytes_per_slice": 16,
            "blocks_axis": 2,
        }
        for rank in range(2)
    ]

    with pytest.raises(TransferStartError, match="submission failed") as failure:
        backend.begin_pull_blocks(peers, [0], [0])

    assert not failure.value.storage_safe
    assert len(failure.value.cleanup_handles) == 1
    cleanup = failure.value.cleanup_handles[0]
    assert cleanup.xfers == [xfer]
    assert not cleanup.storage_safe
    if wrapped_start_error:
        assert failure.value is start_error
