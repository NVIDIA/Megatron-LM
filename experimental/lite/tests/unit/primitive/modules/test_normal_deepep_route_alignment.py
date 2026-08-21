import types

import pytest
import torch

import megatron.lite.primitive.modules.dispatcher as dispatcher_module
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.model.deepseek_v4.vllm.dispatcher import (
    VLLMAlignedNormalDeepEPDispatcher,
)
from megatron.lite.primitive.parallel import ParallelState


def test_deepep_disables_unsafe_deterministic_empty_fill(monkeypatch) -> None:
    monkeypatch.setattr(
        dispatcher_module.torch,
        "are_deterministic_algorithms_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        dispatcher_module.torch.utils.deterministic,
        "fill_uninitialized_memory",
        True,
    )

    dispatcher_module._configure_deepep_deterministic_allocator()

    assert not dispatcher_module.torch.utils.deterministic.fill_uninitialized_memory


def test_deepep_initialization_matches_mcore_num_sms(monkeypatch) -> None:
    calls = []

    class FakeBuffer:
        @staticmethod
        def set_num_sms(value):
            calls.append(value)

    group = object()
    monkeypatch.setattr(
        dispatcher_module,
        "deep_ep",
        type("FakeDeepEP", (), {"Buffer": FakeBuffer}),
    )
    ps = types.SimpleNamespace(ep_size=2, tp_ep_group=group)

    dispatcher = TokenDispatcher(4, 16, ps, use_deepep=True)

    assert calls == [20]
    assert dispatcher.buffer is None


def test_deepep_buffer_is_created_lazily_at_dispatch(monkeypatch) -> None:
    import megatron.lite.primitive.modules.dispatcher as dispatcher_module

    class FakeBuffer:
        @staticmethod
        def set_num_sms(_value):
            pass

    group = object()
    sentinel = object()
    monkeypatch.setattr(
        dispatcher_module,
        "deep_ep",
        type("FakeDeepEP", (), {"Buffer": FakeBuffer}),
    )
    monkeypatch.setattr(
        dispatcher_module,
        "_get_deepep_buffer",
        lambda actual_group, _hidden_bytes: (
            sentinel if actual_group is group else None
        ),
    )
    dispatcher = TokenDispatcher(
        4,
        16,
        types.SimpleNamespace(ep_size=2, tp_ep_group=group),
        use_deepep=True,
    )
    dispatcher._ensure_deepep_buffer(
        torch.zeros(2, 16, dtype=torch.bfloat16)
    )

    assert dispatcher.buffer is sentinel


def _capture_aligned_dispatch_contract(dispatcher, monkeypatch):
    captured = {}

    def fake_aligned(self, hidden, scores, indices, *, source_fixed_topk_valid):
        captured["indices"] = indices
        captured["source_fixed_topk_valid"] = source_fixed_topk_valid
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
    dispatcher.capacity_factor = None
    captured = _capture_aligned_dispatch_contract(dispatcher, monkeypatch)
    hidden = torch.zeros(2, 16, dtype=torch.bfloat16)
    scores = torch.ones(2, 2, dtype=torch.float32)
    indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)

    dispatcher.dispatch(hidden, scores, indices)

    assert captured["source_fixed_topk_valid"] is True
    assert captured["indices"] is indices


def test_aligned_dispatch_masks_routes_like_slime(monkeypatch) -> None:
    dispatcher = VLLMAlignedNormalDeepEPDispatcher.__new__(
        VLLMAlignedNormalDeepEPDispatcher
    )
    dispatcher.capacity_factor = 1.0
    captured = _capture_aligned_dispatch_contract(dispatcher, monkeypatch)
    hidden = torch.zeros(2, 16, dtype=torch.bfloat16)
    scores = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float32)
    indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    token_mask = torch.tensor([False, True])

    dispatcher.dispatch(
        hidden,
        scores,
        indices,
        router_token_masks=token_mask,
    )

    assert captured["source_fixed_topk_valid"] is False
    assert torch.equal(
        captured["indices"],
        torch.tensor([[0, -1], [-1, -1]], dtype=torch.int64),
    )


def test_aligned_deepep_buffer_matches_mcore_process_wide_reuse(monkeypatch) -> None:
    import megatron.lite.primitive.modules.dispatcher as dispatcher_module

    class FakeConfig:
        def get_nvl_buffer_size_hint(self, hidden_bytes, group_size):
            return hidden_bytes * group_size

        def get_rdma_buffer_size_hint(self, hidden_bytes, group_size):
            return hidden_bytes * group_size * 2

    class FakeBuffer:
        created = 0

        @staticmethod
        def get_dispatch_config(group_size):
            return FakeConfig()

        @staticmethod
        def get_combine_config(group_size):
            return FakeConfig()

        def __init__(self, *, group, num_nvl_bytes, num_rdma_bytes, explicitly_destroy):
            type(self).created += 1
            self.group = group
            self.num_nvl_bytes = num_nvl_bytes
            self.num_rdma_bytes = num_rdma_bytes
            self.explicitly_destroy = explicitly_destroy
            self.runtime = object()

    group = object()
    monkeypatch.setattr(dispatcher_module, "deep_ep", type("FakeDeepEP", (), {"Buffer": FakeBuffer}))
    monkeypatch.setattr(dispatcher_module.dist, "get_world_size", lambda *, group: 4)
    monkeypatch.setattr(dispatcher_module.torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(dispatcher_module, "_deepep_buffer", None)
    layer0_primary = dispatcher_module._build_deepep_buffer(group, 4096)
    layer0_metadata = dispatcher_module._build_deepep_buffer(group, 4096)
    layer1_primary = dispatcher_module._build_deepep_buffer(group, 4096)

    assert layer0_primary is layer0_metadata is layer1_primary
    assert FakeBuffer.created == 1
    assert layer0_primary.num_rdma_bytes == 0

    grown = dispatcher_module._build_deepep_buffer(group, 8192)
    assert grown is not layer0_primary
    assert FakeBuffer.created == 2


def test_deepep_receive_counts_keep_mcore_cpu_contract(monkeypatch) -> None:
    import megatron.lite.primitive.modules.dispatcher as dispatcher_module

    class FakeBuffer:
        def get_dispatch_layout(self, *_args, **_kwargs):
            return None, None, None, None, None

        def dispatch(self, hidden, **_kwargs):
            return hidden, None, None, [3, 5], (), None

    monkeypatch.setattr(dispatcher_module, "_get_deepep_buffer", lambda *_args: FakeBuffer())
    ctx = types.SimpleNamespace()
    hidden = torch.zeros(2, 16, dtype=torch.bfloat16)
    indices = torch.zeros(2, 1, dtype=torch.int64)
    scores = torch.ones(2, 1, dtype=torch.float32)

    result = dispatcher_module._DeepEPDispatch.forward(
        ctx, object(), hidden, indices, scores, 2, False, False
    )

    counts = result[3]
    assert counts.device.type == "cpu"
    assert counts.dtype == torch.int64
    assert counts.tolist() == [3, 5]


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

    dispatched, tokens_per_expert, _ = dispatcher.dispatch(
        hidden, weights, indices
    )

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


def test_normal_deepep_finish_deduplicates_hash_routes() -> None:
    dispatcher = TokenDispatcher.__new__(TokenDispatcher)
    dispatcher.num_local_experts = 2
    dispatcher.moe_permute_fusion = False
    dispatcher._local_tpe_list = None
    dispatcher._row_id_map = None
    dispatcher._restore_shape = None

    hidden = torch.stack(
        (
            torch.ones(16, dtype=torch.bfloat16),
            torch.full((16,), 2, dtype=torch.bfloat16),
        )
    )
    indices = torch.tensor([[0, 0], [1, 0]], dtype=torch.int64)
    weights = torch.tensor([[0.25, 0.75], [0.4, 0.6]], dtype=torch.float32)

    dispatched, tokens_per_expert, routed_weights = (
        dispatcher._finish_deepep_dispatch(
            hidden,
            indices,
            weights,
            # DeepEP counts top-k slots before duplicate expert IDs are folded.
            [3, 1],
        )
    )

    assert tokens_per_expert.tolist() == [2, 1]
    assert dispatched.shape == (3, 16)
    torch.testing.assert_close(
        routed_weights,
        torch.tensor([1.0, 0.6, 0.4], dtype=torch.float32),
        rtol=0,
        atol=0,
    )


def test_normal_deepep_finish_rejects_permute_count_mismatch(monkeypatch) -> None:
    dispatcher = TokenDispatcher.__new__(TokenDispatcher)
    dispatcher.num_local_experts = 2
    dispatcher.moe_permute_fusion = False
    dispatcher.ps = types.SimpleNamespace(ep_group=object())
    monkeypatch.setattr(dispatcher_module.dist, "get_rank", lambda *, group: 0)
    monkeypatch.setattr(
        dispatcher_module,
        "permute",
        lambda hidden, *_args, **_kwargs: (
            hidden[:1],
            torch.ones(1),
            torch.zeros(1, dtype=torch.long),
        ),
    )

    with pytest.raises(RuntimeError, match="dispatch metadata mismatch"):
        dispatcher._finish_deepep_dispatch(
            torch.ones(2, 16, dtype=torch.bfloat16),
            torch.tensor([[0], [1]], dtype=torch.int64),
            torch.ones(2, 1, dtype=torch.float32),
            [1, 1],
        )
