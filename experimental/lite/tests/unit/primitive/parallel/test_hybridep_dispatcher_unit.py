# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contract tests for the HybridEP token-dispatcher path (fake HybridEPBuffer)."""

import pytest
import torch

from megatron.lite.primitive.parallel import ParallelState


class _FakeHybridEPBuffer:
    """Single-rank stand-in: permute by expert on dispatch, index_add back on combine."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.calls = []

    def dispatch_with_permute(self, *, hidden, probs=None, routing_map=None, topk_idx=None,
                              num_of_experts=None, handle=None, **kwargs):
        self.calls.append(("dispatch", kwargs))
        if handle is not None:
            row_ids, expert_ids = handle
            return hidden[row_ids], None, None, None, handle
        if routing_map is None:
            routing_map = torch.zeros(hidden.size(0), num_of_experts, dtype=torch.bool)
            valid = topk_idx >= 0
            routing_map[torch.arange(hidden.size(0)).unsqueeze(1).expand_as(topk_idx)[valid],
                        topk_idx[valid].long()] = True
        row_ids, expert_ids = routing_map.T.nonzero(as_tuple=True)[::-1]
        order = torch.argsort(expert_ids * hidden.size(0) + row_ids)
        row_ids, expert_ids = row_ids[order], expert_ids[order]
        tpe = routing_map.sum(0).to(torch.int32)
        handle = (row_ids, expert_ids)
        return hidden[row_ids], probs[row_ids, expert_ids], None, tpe, handle

    def combine_with_unpermute(self, *, hidden, probs=None, handle, **kwargs):
        self.calls.append(("combine", kwargs))
        row_ids, expert_ids = handle
        n = int(row_ids.max()) + 1
        combined = torch.zeros(n, hidden.size(1), dtype=hidden.dtype).index_add_(0, row_ids, hidden)
        combined_probs = None
        if probs is not None:
            combined_probs = torch.zeros(n, self.init_kwargs["num_local_experts"], dtype=probs.dtype)
            combined_probs.index_put_((row_ids, expert_ids), probs, accumulate=True)
        return combined, combined_probs


@pytest.fixture
def hybridep_dispatcher(monkeypatch, transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import dispatcher as mod

    monkeypatch.setattr(mod, "HybridEPBuffer", _FakeHybridEPBuffer)
    monkeypatch.setattr(mod, "_hybridep_buffers", {})
    monkeypatch.setattr(mod.dist, "all_reduce", lambda tensor, op=None, group=None: None)

    def make(num_experts=4, **kwargs):
        ps = ParallelState(ep_size=1, ep_rank=0, ep_group=object(), tp_ep_group=object())
        d = mod.TokenDispatcher(num_experts=num_experts, hidden_size=2, ps=ps,
                                dispatch_backend="hybridep", **kwargs)
        d.ep_size = 1  # fake buffer holds every expert, dispatcher still takes the hybridep path
        d.use_hybridep = True
        d.num_local_experts = num_experts
        return d, mod

    return make


@pytest.mark.parametrize("dense", [True, False])
def test_hybridep_roundtrip_and_grads(hybridep_dispatcher, dense):
    dispatcher, mod = hybridep_dispatcher(hybridep_dense_routing=dense)
    hidden = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], requires_grad=True)
    topk_indices = torch.tensor([[0, 2], [2, 1], [3, 0]])
    topk_scores = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]], requires_grad=True)

    dispatched, tpe, probs = dispatcher.dispatch(hidden, topk_scores, topk_indices)
    kwargs = dispatcher._hybridep_buffer.calls[0][1]
    assert kwargs["num_of_tokens_per_rank"] == 3 and kwargs["num_of_experts_per_rank"] == 4
    assert dispatcher._hybridep_buffer.init_kwargs["max_num_of_tokens_per_rank"] == 3
    torch.testing.assert_close(tpe, torch.tensor([2, 1, 2, 1]))
    assert tpe.dtype == torch.int64 and probs.dtype == torch.float32
    # expert-major order: e0<-(t0,t2) e1<-(t1) e2<-(t0,t1) e3<-(t2)
    torch.testing.assert_close(dispatched, hidden.detach()[[0, 2, 1, 0, 1, 2]])
    torch.testing.assert_close(probs, torch.tensor([0.6, 0.5, 0.3, 0.4, 0.7, 0.5]))

    combined = dispatcher.combine(dispatched)
    torch.testing.assert_close(combined, 2 * hidden.detach())
    assert dispatcher._handle is None and dispatcher._num_permuted_tokens is None

    (combined.sum() + probs.sum()).backward()
    torch.testing.assert_close(hidden.grad, torch.full_like(hidden, 2.0))
    torch.testing.assert_close(topk_scores.grad, torch.ones_like(topk_scores))


def test_hybridep_dense_routing_passes_int16_topk(hybridep_dispatcher):
    dispatcher, _ = hybridep_dispatcher()
    dispatcher.dispatch(torch.ones(2, 2), torch.ones(2, 1), torch.tensor([[1], [3]]))
    kind, _ = dispatcher._hybridep_buffer.calls[0]
    assert kind == "dispatch"
    dispatcher_sparse, _ = hybridep_dispatcher(hybridep_dense_routing=False)
    dispatcher_sparse.dispatch(torch.ones(2, 2), torch.ones(2, 1), torch.tensor([[1], [3]]))
    assert dispatcher.hybridep_dense_routing and not dispatcher_sparse.hybridep_dense_routing


def test_hybridep_buffer_shared_per_group(hybridep_dispatcher):
    d1, mod = hybridep_dispatcher()
    d2, _ = hybridep_dispatcher()
    d2.ps = d1.ps
    d1.dispatch(torch.ones(2, 2), torch.ones(2, 1), torch.tensor([[0], [1]]))
    d2.dispatch(torch.ones(2, 2), torch.ones(2, 1), torch.tensor([[0], [1]]))
    assert d1._hybridep_buffer is d2._hybridep_buffer
    assert len(mod._hybridep_buffers) == 1


def test_dispatch_backend_validation(transformer_engine_import_stub, monkeypatch):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import dispatcher as mod

    ps = ParallelState(ep_size=1, ep_rank=0)
    assert mod.TokenDispatcher(3, 2, ps, use_deepep=False).dispatch_backend == "alltoall"
    assert mod.TokenDispatcher(3, 2, ps, use_deepep=True).dispatch_backend == "deepep"
    with pytest.raises(ValueError):
        mod.TokenDispatcher(3, 2, ps, dispatch_backend="nccl")
    monkeypatch.setattr(mod, "HybridEPBuffer", None)
    with pytest.raises(RuntimeError):
        mod.TokenDispatcher(3, 2, ps, dispatch_backend="hybridep")
