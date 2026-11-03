# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Engram's token-prefetch wrapper must survive the full-iteration data read."""

import pytest
import torch

from megatron.core.full_cuda_graph import FullCudaGraphWrapper, StaticBufferLoader
from megatron.training.utils import get_pipeline_prefetched_tokens
from megatron.training.utils.common_utils import PipelineTokenPrefetchIterator

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

MBS, SEQ, NUM_MICROBATCHES = 1, 8, 2


@pytest.fixture(autouse=True)
def _reset_static_buffers():
    StaticBufferLoader.static_buffers = {'training': [], 'validation': []}
    yield
    StaticBufferLoader.static_buffers = {'training': [], 'validation': []}


def _batches(offset):
    return [
        {"tokens": torch.arange(SEQ, dtype=torch.int64).view(MBS, SEQ) + offset + 100 * b}
        for b in range(NUM_MICROBATCHES)
    ]


def _single_rank_wrapper(batches):
    """The TP=PP=1 wrapper: tokens are derived from the batch the scheduler consumed."""
    return PipelineTokenPrefetchIterator(iter(batches), [], [], token_spec=(MBS, SEQ, "cuda"))


def _distributed_wrapper(batches):
    """The TP/PP>1 wrapper: tokens were broadcast before the schedule (fresh tensors)."""
    tokens = [batch["tokens"].cuda() for batch in batches]
    return PipelineTokenPrefetchIterator(None, list(batches), tokens)


def _consume(iterator):
    """What forward_step does per microbatch: next() then read the Engram tokens."""
    seen = []
    for _ in range(NUM_MICROBATCHES):
        batch = next(iterator)
        seen.append((batch, get_pipeline_prefetched_tokens(iterator)))
    return seen


def _full_iteration_read(iterator):
    wrapper = FullCudaGraphWrapper(lambda **kwargs: None)
    (replay,) = wrapper.data_read(iterator, [object()], True, NUM_MICROBATCHES)
    return replay


@pytest.mark.parametrize("make_wrapper", [_single_rank_wrapper, _distributed_wrapper])
def test_full_iteration_read_keeps_tokens_in_static_buffers(make_wrapper):
    pointers = None
    for step in range(3):
        batches = _batches(offset=step)
        seen = _consume(_full_iteration_read(make_wrapper(batches)))
        step_pointers = []
        for (batch, tokens), source in zip(seen, batches):
            torch.testing.assert_close(tokens.cpu(), source["tokens"])
            step_pointers.append(tokens.data_ptr())
        # A replayed graph reads the addresses captured on the first step.
        if pointers is None:
            pointers = step_pointers
        assert step_pointers == pointers


def test_rank_without_data_gets_an_empty_replay_with_tokens():
    """Middle pipeline stages own no batches but still read every microbatch's tokens."""
    tokens = [torch.full((MBS, SEQ), b, dtype=torch.int64, device="cuda") for b in range(2)]
    middle_stage = PipelineTokenPrefetchIterator(None, [], tokens)
    replay = _full_iteration_read(middle_stage)
    for b in range(NUM_MICROBATCHES):
        assert torch.all(get_pipeline_prefetched_tokens(replay) == b)


def test_plain_iterators_are_unchanged():
    batches = _batches(offset=0)
    replay = _full_iteration_read(iter(batches))
    assert not isinstance(replay, PipelineTokenPrefetchIterator)
    for batch, source in zip(replay, batches):
        torch.testing.assert_close(batch["tokens"].cpu(), source["tokens"])
