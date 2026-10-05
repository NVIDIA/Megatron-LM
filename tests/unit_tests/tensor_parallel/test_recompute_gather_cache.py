# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""recompute_gather_cache: a gather inside a recompute phase is kept once and served to the next
gather of the same unchanged local input (the backward's) without communicating; an in-place
change, a leftover entry and a quantized gather are handled.  CPU only, no Transformer Engine."""

import torch

from megatron.core.tensor_parallel import recompute_gather_cache as rgc


def _fresh():
    rgc._cache.clear()
    for k in rgc._stats:
        rgc._stats[k] = 0


def _fake_gather_factory(calls):
    def gather(inp, process_group, async_op=False, quantizer=None):
        calls.append(async_op)
        return torch.cat([inp, inp + 1], 0), None

    return gather


def test_backward_gather_is_served_from_the_recompute_forward_gather():
    _fresh()
    calls = []
    gather = rgc._cached_gather(_fake_gather_factory(calls))
    x = torch.arange(6.0).view(3, 2)
    expected = torch.cat([x, x + 1], 0)

    # outside a recompute phase: pass-through, nothing is kept
    out, handle = gather(x, None)
    assert torch.equal(out, expected) and handle is None and not rgc._cache

    # the recompute forward's gather is kept; the caller's own reference may be emptied
    with rgc.recompute_phase():
        out1, _ = gather(x, None)
        assert torch.equal(out1, expected) and len(rgc._cache) == 1
        out1.data = torch.Tensor()  # what Transformer Engine's clear_tensor_data does

    # the backward's gather (a saved-tensor alias of the same storage, async) hits and consumes
    out2, handle2 = gather(x.detach(), None, async_op=True)
    assert torch.equal(out2, expected) and handle2 is None
    assert not rgc._cache and calls == [False, False]
    assert rgc.stats()["stored"] == 1 and rgc.stats()["hits"] == 1


def test_a_changed_input_a_leftover_entry_and_a_quantized_gather():
    _fresh()
    calls = []
    gather = rgc._cached_gather(_fake_gather_factory(calls))
    x = torch.arange(6.0).view(3, 2)

    # an in-place change between the two gathers invalidates the entry: the backward gathers
    with rgc.recompute_phase():
        gather(x, None)
    x.add_(1)
    out, _ = gather(x, None, async_op=True)
    assert torch.equal(out, torch.cat([x, x + 1], 0)) and calls[-1] is True
    assert rgc.stats()["invalid"] == 1 and not rgc._cache

    # an entry nobody consumed is dropped when the next outermost recompute phase begins
    with rgc.recompute_phase():
        gather(x, None)
    assert len(rgc._cache) == 1
    with rgc.recompute_phase():
        with rgc.recompute_phase():  # nested phases do not drop
            pass
    assert not rgc._cache and rgc.stats()["unconsumed"] == 1

    # a quantized gather is passed through untouched
    with rgc.recompute_phase():
        gather(x, None, False, object())
    assert not rgc._cache and rgc.stats()["bypassed"] == 1
