# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Real-NCCL protocol tests for full-iteration CUDA graph consensus.

Run this module through ``torchrun`` with at least two local GPUs.  These tests
deliberately use the WORLD process group for the wrapper's safety collectives;
rank-local mocks cannot detect mismatched collective selection or asymmetric
failure behavior.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from megatron.core.datasets.data_schedule import _build_thd_full_iteration_dynamic_cp_signature
from megatron.core.full_cuda_graph import (
    FULL_CUDA_GRAPH_STATIC_METADATA_KEY,
    FullCudaGraphWrapper,
    StaticBufferLoader,
)
from megatron.core.packed_seq_params import PackedSeqParams
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(Utils.world_size < 2, reason="requires torchrun with at least two ranks"),
    pytest.mark.timeout(60),
]


@pytest.fixture(autouse=True)
def _distributed_wrapper_state():
    """Keep every test/rank at the same process-group and wrapper lifecycle."""
    Utils.initialize_distributed()
    torch.distributed.barrier()
    FullCudaGraphWrapper.reset_cuda_graph()
    yield
    torch.distributed.barrier()
    FullCudaGraphWrapper.reset_cuda_graph()
    torch.distributed.barrier()


@dataclass
class _CallCounts:
    data_read: int = 0
    forward: int = 0


class _CountingWrapper(FullCudaGraphWrapper):
    """Minimal wrapper probe whose eager body contains no distributed work."""

    def __init__(self, *, batch_supplier=None):
        self.counts = _CallCounts()
        self._batch_supplier = batch_supplier
        super().__init__(
            self._forward_backward,
            cuda_graph_warmup_steps=100,
            batch_preparation_fn=(None if batch_supplier is None else self._prepare_batch),
            require_global_static_metadata_consensus=True,
        )

    def _prepare_batch(self, data_iterator, vp_stage):
        return self._batch_supplier()

    def data_read(self, data_iterator, model, training, num_microbatches):
        self.counts.data_read += 1
        return super().data_read(data_iterator, model, training, num_microbatches)

    def _forward_backward(self, **kwargs):
        self.counts.forward += 1
        return torch.ones((), device=torch.cuda.current_device())


def _call_kwargs(*, num_microbatches=1, forward_only=False):
    return {
        'model': torch.nn.Identity().cuda(),
        'data_iterator': None,
        'num_microbatches': num_microbatches,
        'seq_length': 64,
        'micro_batch_size': 1,
        'forward_only': forward_only,
    }


def _capture_signature(*, num_microbatches=1):
    return {
        'num_microbatches': num_microbatches,
        'num_model_chunks': 1,
        'seq_length': 64,
        'micro_batch_size': 1,
        'decoder_seq_length': None,
    }


def _invoke(call):
    try:
        return call(), None
    except Exception as exc:  # The collective outcome is asserted on every rank below.
        return None, exc


def _gather_ints(*values):
    local = torch.tensor(values, dtype=torch.int64, device=torch.cuda.current_device())
    world_size = torch.distributed.get_world_size()
    gathered = torch.empty(world_size * local.numel(), dtype=local.dtype, device=local.device)
    torch.distributed.all_gather_into_tensor(gathered, local)
    return gathered.view(world_size, local.numel()).cpu()


def _assert_collective_rejection(error, wrapper, *, message, expected_data_reads, stage='training'):
    outcomes = _gather_ints(
        isinstance(error, RuntimeError),
        message in str(error),
        wrapper.counts.data_read,
        wrapper.counts.forward,
        len(StaticBufferLoader.static_buffers[stage]),
        FullCudaGraphWrapper.curr_iteration[stage],
        wrapper.static_loader.batch_stage is None,
        wrapper.static_loader.batch_start_length is None,
        not wrapper.static_loader.batch_dynamic_cp_metadata,
    )
    assert torch.all(outcomes[:, 0] == 1), str(error)
    assert torch.all(outcomes[:, 1] == 1), str(error)
    assert torch.all(outcomes[:, 2] == expected_data_reads), outcomes
    assert torch.all(outcomes[:, 3:] == torch.tensor([0, 0, 0, 1, 1, 1])), outcomes


def _dynamic_cp_batch(cp_group, local_cp_size):
    params = PackedSeqParams(
        qkv_format='thd', local_cp_size=local_cp_size, cp_group=cp_group, cp_partition_mode='zigzag'
    )
    return {
        'tokens': torch.tensor(
            [torch.distributed.get_rank()], dtype=torch.int64, device=torch.cuda.current_device()
        ),
        FULL_CUDA_GRAPH_STATIC_METADATA_KEY: {
            'thd_dynamic_cp': _build_thd_full_iteration_dynamic_cp_signature(
                params, config=SimpleNamespace(mtp_num_layers=0)
            )
        },
    }


def test_rank_zero_only_current_signature_mismatch_fails_before_data_read():
    """One rank changing N cannot leave peers entering batch preparation."""
    rank = torch.distributed.get_rank()
    wrapper = _CountingWrapper()

    _, error = _invoke(lambda: wrapper(**_call_kwargs(num_microbatches=2 if rank == 0 else 1)))

    _assert_collective_rejection(
        error, wrapper, message='before reading data, capture, or replay', expected_data_reads=0
    )


def test_stage_mismatch_fails_before_data_read():
    """Training/validation disagreement is part of the WORLD signature."""
    rank = torch.distributed.get_rank()
    wrapper = _CountingWrapper()

    _, error = _invoke(lambda: wrapper(**_call_kwargs(forward_only=(rank != 0))))

    # The local stage differs, so inspect both stage-local state vectors rather
    # than assuming every process indexed the training dictionaries.
    local_stage = 'training' if rank == 0 else 'validation'
    _assert_collective_rejection(
        error,
        wrapper,
        message='before reading data, capture, or replay',
        expected_data_reads=0,
        stage=local_stage,
    )


def test_rank_zero_only_captured_signature_mismatch_fails_before_data_read():
    """A stale captured signature on one rank must reject the whole WORLD."""
    rank = torch.distributed.get_rank()
    wrapper = _CountingWrapper()
    FullCudaGraphWrapper.capture_signature['training'] = _capture_signature(
        num_microbatches=2 if rank == 0 else 1
    )

    _, error = _invoke(lambda: wrapper(**_call_kwargs(num_microbatches=1)))

    _assert_collective_rejection(
        error, wrapper, message='before reading data, capture, or replay', expected_data_reads=0
    )


def test_first_batch_rejects_incoherent_dynamic_cp_topology():
    """Overlapping, non-reciprocal DCP memberships fail before eager forward."""
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    # All ranks must create subgroups in the same order.  Only rank zero uses
    # this singleton; peers report WORLD, creating an incoherent overlap.
    singleton = torch.distributed.new_group(ranks=[0], backend='nccl')
    cp_group = singleton if rank == 0 else torch.distributed.group.WORLD
    local_cp_size = 1 if rank == 0 else world_size
    batch = _dynamic_cp_batch(cp_group, local_cp_size)
    wrapper = _CountingWrapper(batch_supplier=lambda: batch)

    _, error = _invoke(lambda: wrapper(**_call_kwargs()))

    _assert_collective_rejection(
        error, wrapper, message='before capture or replay', expected_data_reads=1
    )


def test_invalid_first_batch_rolls_back_then_valid_retry_succeeds():
    """A rejected first slot must not poison the next valid static batch."""
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    current = {
        'batch': _dynamic_cp_batch(torch.distributed.group.WORLD, 1 if rank == 0 else world_size)
    }
    wrapper = _CountingWrapper(batch_supplier=lambda: current['batch'])

    _, first_error = _invoke(lambda: wrapper(**_call_kwargs()))
    _assert_collective_rejection(
        first_error, wrapper, message='before capture or replay', expected_data_reads=1
    )

    torch.distributed.barrier()
    current['batch'] = _dynamic_cp_batch(torch.distributed.group.WORLD, world_size)
    _, retry_error = _invoke(lambda: wrapper(**_call_kwargs()))

    outcomes = _gather_ints(
        retry_error is None,
        wrapper.counts.data_read,
        wrapper.counts.forward,
        len(StaticBufferLoader.static_buffers['training']),
        FullCudaGraphWrapper.curr_iteration['training'],
        wrapper.static_loader.batch_stage is None,
        wrapper.static_loader.batch_start_length is None,
        not wrapper.static_loader.batch_dynamic_cp_metadata,
    )
    assert torch.all(outcomes == torch.tensor([1, 2, 1, 1, 1, 1, 1, 1])), (retry_error, outcomes)
