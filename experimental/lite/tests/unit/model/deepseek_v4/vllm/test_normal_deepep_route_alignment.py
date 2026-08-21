import types

import torch

from megatron.lite.model.deepseek_v4.vllm.primitive.moe.dispatcher import (
    VLLMAlignedNormalDeepEPDispatcher,
)
from megatron.lite.primitive.parallel import ParallelState


def _capture_aligned_dispatch_contract(dispatcher, monkeypatch):
    captured = {}

    def fake_aligned(self, hidden, scores, indices):
        captured["indices"] = indices
        return hidden, torch.empty(0, dtype=torch.int64), scores

    monkeypatch.setattr(
        dispatcher,
        "_dispatch_aligned",
        types.MethodType(fake_aligned, dispatcher),
    )
    return captured


def test_aligned_dispatch_fixed_topk_contract_matches_slime(monkeypatch) -> None:
    dispatcher = VLLMAlignedNormalDeepEPDispatcher.__new__(
        VLLMAlignedNormalDeepEPDispatcher
    )
    captured = _capture_aligned_dispatch_contract(dispatcher, monkeypatch)
    hidden = torch.zeros(2, 16, dtype=torch.bfloat16)
    scores = torch.ones(2, 2, dtype=torch.float32)
    indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)

    dispatcher.dispatch(hidden, scores, indices)

    assert captured["indices"] is indices


def test_route_alignment_preserves_duplicate_slots_and_fp32_gather(monkeypatch) -> None:
    import vllm.model_executor.layers.fused_moe.deep_gemm_utils as deep_gemm_utils

    def fake_ep_gather(
        input_tensor, recv_ids, recv_weights, input_index, expert_map, output
    ):
        assert expert_map is None
        output.zero_()
        for token in range(recv_ids.shape[0]):
            accumulator = torch.zeros(input_tensor.shape[1], dtype=torch.float32)
            for slot in range(recv_ids.shape[1]):
                row = int(input_index[token, slot])
                if row >= 0:
                    accumulator += input_tensor[row].float() * recv_weights[token, slot]
            output[token].copy_(accumulator.to(output.dtype))

    monkeypatch.setattr(deep_gemm_utils, "ep_gather", fake_ep_gather)
    dispatcher = VLLMAlignedNormalDeepEPDispatcher(
        num_experts=2,
        hidden_size=16,
        ps=ParallelState(ep_size=1, ep_rank=0),
        use_deepep=False,
    )
    hidden = torch.stack(
        (
            torch.ones(16, dtype=torch.bfloat16),
            torch.full((16,), 2, dtype=torch.bfloat16),
        )
    )
    # Token 0 deliberately selects expert 0 twice. Ordinary boolean routing
    # maps collapse these slots; rollout-aligned routing must retain both.
    indices = torch.tensor([[0, 0], [1, 0]], dtype=torch.int64)
    weights = torch.tensor([[0.25, 0.75], [0.4, 0.6]], dtype=torch.float32)

    dispatched, tokens_per_expert, _ = dispatcher.dispatch(hidden, weights, indices)

    assert tokens_per_expert.tolist() == [3, 1]
    torch.testing.assert_close(dispatched[0], hidden[0])
    torch.testing.assert_close(dispatched[1], hidden[0])
    torch.testing.assert_close(dispatched[2], hidden[1])
    torch.testing.assert_close(dispatched[3], hidden[1])

    expert_output = dispatched.clone()
    expert_output[:3].mul_(2)  # expert 0
    expert_output[3:].mul_(3)  # expert 1
    actual = dispatcher.combine(expert_output)
    expected = torch.stack(
        (
            torch.full((16,), 2.0, dtype=torch.bfloat16),
            torch.full((16,), 4.8, dtype=torch.bfloat16),
        )
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
