from types import SimpleNamespace

import pytest
import torch

from megatron.lite.model.deepseek_v4.vllm.primitive.moe import hybridep
from megatron.lite.model.deepseek_v4.vllm.primitive.moe.communication import (
    VLLMAlignedHybridEPDispatcher,
)


class _FakeHybridBuffer:
    runtime = object()

    def dispatch_with_permute(self, *, hidden, topk_idx=None, handle=None, **kwargs):
        if handle is not None:
            route_rows = handle[0]
            output = hidden.new_zeros(
                (int(kwargs["num_permuted_tokens"]), hidden.shape[1])
            )
            output.index_copy_(0, route_rows, hidden)
            return output, None, None, None, None
        assert kwargs["pad_multiple"] == 128
        if topk_idx.numel() == 0:
            empty_rows = torch.empty(0, dtype=torch.long)
            return (
                hidden.new_empty((0, hidden.shape[1])),
                None,
                None,
                torch.tensor([0, 0], dtype=torch.int64),
                (empty_rows,),
            )
        assert topk_idx.reshape(-1).tolist() == [0, 0, 1]
        route_rows = torch.tensor([0, 1, 128], dtype=torch.long)
        output = hidden.new_zeros((256, hidden.shape[1]))
        output.index_copy_(0, route_rows, hidden)
        return (
            output,
            None,
            None,
            torch.tensor([128, 128], dtype=torch.int64),
            (route_rows,),
        )

    def combine_with_unpermute(self, *, hidden, handle, **_kwargs):
        return hidden.index_select(0, handle[0]), None


def _inputs():
    hidden = torch.stack(
        (
            torch.ones(16, dtype=torch.bfloat16),
            torch.full((16,), 2, dtype=torch.bfloat16),
        )
    )
    indices = torch.tensor([[0, 0], [1, 9]], dtype=torch.int64)
    weights = torch.tensor([[0.25, 0.75], [0.4, 0.6]], dtype=torch.float32)
    return hidden, weights, indices


def _install_fake_ep_gather(monkeypatch) -> None:
    import vllm.model_executor.layers.fused_moe.deep_gemm_utils as deep_gemm_utils

    def fake_ep_gather(
        input_tensor, _ids, recv_weights, input_index, _expert_map, output
    ):
        output.zero_()
        for token in range(input_index.shape[0]):
            for slot in range(input_index.shape[1]):
                row = int(input_index[token, slot])
                if row >= 0:
                    output[token].add_(
                        input_tensor[row] * recv_weights[token, slot]
                    )

    monkeypatch.setattr(deep_gemm_utils, "ep_gather", fake_ep_gather)


def test_impl_config_accepts_only_clean_dispatchers() -> None:
    from megatron.lite.model.deepseek_v4.vllm import protocol

    assert protocol.ImplConfig().moe_token_dispatcher_type == "deepep"
    assert (
        protocol.ImplConfig(moe_token_dispatcher_type="hybridep")
        .moe_token_dispatcher_type
        == "hybridep"
    )
    with pytest.raises(ValueError, match="deepep.*hybridep"):
        protocol.ImplConfig(moe_token_dispatcher_type="alltoall")


def test_hybridep_fails_closed_without_runtime(monkeypatch) -> None:
    monkeypatch.setattr(hybridep, "deep_ep", SimpleNamespace(Buffer=object))
    with pytest.raises(RuntimeError, match="HybridEPBuffer"):
        VLLMAlignedHybridEPDispatcher(
            4,
            16,
            SimpleNamespace(ep_size=2, ep_group=object()),
        )


def test_hybridep_rejects_invalid_topology(monkeypatch) -> None:
    monkeypatch.setattr(
        hybridep, "deep_ep", SimpleNamespace(HybridEPBuffer=object)
    )
    monkeypatch.setenv("NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN", "3")
    monkeypatch.setattr(
        hybridep.dist, "get_world_size", lambda *, group: 4
    )
    with pytest.raises(RuntimeError, match="not divisible"):
        VLLMAlignedHybridEPDispatcher(
            4,
            16,
            SimpleNamespace(ep_size=2, ep_group=object()),
        )


def test_hybridep_preserves_duplicate_slots_and_compacts_invalid_routes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        hybridep, "_get_buffer", lambda *_args: _FakeHybridBuffer()
    )
    hidden, weights, indices = _inputs()
    result = hybridep.dispatch_routes(
        hidden,
        weights,
        indices,
        num_experts=4,
        num_local_experts=2,
        group=object(),
    )

    assert result.tokens_per_expert.tolist() == [128, 128]
    assert result.hidden.shape == (256, 16)
    assert result.state.source_output_index.tolist() == [[0, 1], [2, -1]]
    source = hybridep.combine_routes(result.hidden, result.state)
    torch.testing.assert_close(source[0], hidden[0])
    torch.testing.assert_close(source[1], hidden[0])
    torch.testing.assert_close(source[2], hidden[1])


def test_hybridep_keeps_zero_route_rank_in_collective(monkeypatch) -> None:
    hidden, weights, indices = _inputs()
    indices.fill_(-1)
    monkeypatch.setattr(
        hybridep, "_get_buffer", lambda *_args: _FakeHybridBuffer()
    )
    result = hybridep.dispatch_routes(
        hidden,
        weights,
        indices,
        num_experts=4,
        num_local_experts=2,
        group=object(),
    )
    assert result.hidden.shape == (0, hidden.shape[1])
    assert result.tokens_per_expert.tolist() == [0, 0]
    assert torch.equal(
        result.state.source_output_index,
        torch.full_like(indices, -1),
    )


def test_hybridep_combine_lifecycle_and_autograd(monkeypatch) -> None:
    _install_fake_ep_gather(monkeypatch)
    monkeypatch.setattr(
        hybridep,
        "deep_ep",
        SimpleNamespace(HybridEPBuffer=object),
    )
    monkeypatch.setenv("NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN", "2")
    monkeypatch.setattr(
        hybridep.dist, "get_world_size", lambda *, group: 2
    )
    monkeypatch.setattr(
        hybridep, "_get_buffer", lambda *_args: _FakeHybridBuffer()
    )
    dispatcher = VLLMAlignedHybridEPDispatcher(
        4,
        16,
        SimpleNamespace(ep_size=2, ep_group=object()),
    )
    hidden, weights, indices = _inputs()
    hidden.requires_grad_(True)

    dispatched, counts, _ = dispatcher.dispatch(hidden, weights, indices)
    with pytest.raises(RuntimeError, match="awaiting combine"):
        dispatcher.dispatch(hidden, weights, indices)
    output = dispatcher.combine(dispatched)
    assert counts.tolist() == [128, 128]
    output.float().sum().backward()
    assert hidden.grad is not None
    assert torch.equal(hidden.grad[0], torch.ones_like(hidden.grad[0]))
    assert torch.equal(hidden.grad[1], torch.full_like(hidden.grad[1], 0.4))
    with pytest.raises(RuntimeError, match="no matching dispatch"):
        dispatcher.combine(dispatched)
