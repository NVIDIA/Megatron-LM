# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""HybridEP FakePG allocation contracts; synthetic outputs are not numerical references."""

import gc
import os
import weakref

import pytest
import torch

from megatron.core.ops.moe.fake_deepep import validate_fake_flex_config
from megatron.core.ops.moe.fake_hybridep import FakeHybridEPBuffer
from megatron.core.transformer.moe import fused_a2a


@pytest.fixture
def fake_group():
    from torch.testing._internal.distributed.fake_pg import FakeStore

    if torch.distributed.is_initialized():
        pytest.skip("requires an isolated process without a pre-existing process group")
    torch.distributed.init_process_group("fake", store=FakeStore(), rank=0, world_size=4)
    try:
        yield torch.distributed.group.WORLD
    finally:
        fused_a2a.reset_hybrid_ep_buffer()
        torch.distributed.destroy_process_group()


def hybrid_config(**overrides):
    return {
        "bf16": True,
        "moe_token_dispatcher_type": "flex",
        "moe_flex_dispatcher_backend": "hybridep",
        "expert_tensor_parallel_size": 1,
        "moe_router_force_load_balancing": True,
        **overrides,
    }


def test_hybrid_memory_scope():
    model = validate_fake_flex_config(hybrid_config(moe_use_grouped_tensor=True))
    assert model["policy"] == "hybridep-balanced-expert-activations-v1"
    assert model["capacity_guarantee"] is False
    assert model["peak_is_upper_bound"] is False
    assert "not total device capacity" in model["measurement_scope"]
    assert any("transport buffers" in item for item in model["unmodeled_memory"])


@pytest.mark.parametrize(
    "options",
    [
        {"bf16": False},
        {"fp8": "hybrid"},
        {"fp4": "nvfp4"},
        {"expert_tensor_parallel_size": 2},
        {"expert_tensor_parallel_size": None, "tensor_model_parallel_size": 2},
        {"moe_expert_capacity_factor": 1.0},
        {"moe_expert_rank_capacity_factor": 1.0},
        {"moe_pad_expert_input_to_capacity": True},
        {"moe_router_padding_for_quantization": True},
        {"moe_hybridep_pad_uneven_dispatch_inputs": True},
        {"moe_virtual_expert_load_balance": True},
        {"cuda_graph_impl": "local"},
        {"moe_router_force_load_balancing": False},
    ],
)
def test_hybrid_rejects_unsupported_models(options):
    with pytest.raises(ValueError):
        validate_fake_flex_config(hybrid_config(**options))


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("alignment,expected_counts", [(None, [5, 5, 4]), (16, [16, 16, 16])])
def test_hybrid_allocations_and_cached_dispatch(fake_group, dense, alignment, expected_counts):
    buffer = FakeHybridEPBuffer(fake_group, 16, 7, 3, device="cpu")
    hidden = torch.zeros((7, 16), dtype=torch.bfloat16)
    probs = torch.ones((7, 12), dtype=torch.float32)
    ids = torch.arange(2, dtype=torch.int16).expand(7, -1)
    routes = torch.zeros_like(probs, dtype=torch.bool).scatter_(1, ids.long(), True)
    metadata = {"topk_idx": ids, "num_of_experts": 12} if dense else {"routing_map": routes}
    received, weights, scaling, counts, handle = buffer.dispatch_with_permute(
        hidden=hidden, probs=probs, num_of_experts_per_rank=3, pad_multiple=alignment, **metadata
    )
    assert counts.tolist() == expected_counts
    assert counts.device.type == "cpu"
    assert received.shape == (sum(expected_counts), 16)
    assert weights.shape == (sum(expected_counts),)
    assert torch.count_nonzero(weights).item() == 14
    assert scaling is None
    combined, grad_probs = buffer.combine_with_unpermute(
        hidden=received, probs=weights, handle=handle, pad_multiple=alignment
    )
    assert combined.shape == hidden.shape
    assert grad_probs.shape == probs.shape
    replay = buffer.dispatch_with_permute(hidden=combined, handle=handle, pad_multiple=alignment)
    assert replay[0].shape == received.shape
    assert replay[-1] is handle
    other = FakeHybridEPBuffer(fake_group, 16, 7, 3, device="cpu")
    with pytest.raises(ValueError, match="another buffer"):
        other.combine_with_unpermute(hidden=received, handle=handle, pad_multiple=alignment)
    with pytest.raises(ValueError, match="alignment changed"):
        buffer.combine_with_unpermute(hidden=received, handle=handle, pad_multiple=32)


@pytest.mark.parametrize("bad_routes", ["masked", "duplicate", "out_of_range"])
def test_hybrid_rejects_invalid_routes(fake_group, bad_routes):
    buffer = FakeHybridEPBuffer(fake_group, 16, 7, 3, device="cpu")
    hidden = torch.zeros((7, 16), dtype=torch.bfloat16)
    probs = torch.ones((7, 12), dtype=torch.float32)
    if bad_routes == "masked":
        routes = torch.zeros_like(probs, dtype=torch.bool)
        routes[:, :2] = True
        routes[0, 0] = False
        metadata = {"routing_map": routes}
    else:
        ids = torch.zeros((7, 2), dtype=torch.int16)
        ids[:, 1] = 12 if bad_routes == "out_of_range" else 0
        metadata = {"topk_idx": ids, "num_of_experts": 12}
    with pytest.raises(ValueError):
        buffer.dispatch_with_permute(
            hidden=hidden, probs=probs, num_of_experts_per_rank=3, **metadata
        )


@pytest.mark.parametrize("alignment", [None, 16])
def test_hybrid_autograd_after_buffer_reset(fake_group, alignment, monkeypatch):
    # Only device selection is substituted. Run the actual wrappers with a real
    # FakePG and CPU tensors, keeping three microbatches alive through a reset.
    def init_cpu_buffer(group, hidden_dim, num_tokens, local_experts, *args):
        fused_a2a._hybrid_ep_buffer = FakeHybridEPBuffer(
            group, hidden_dim, num_tokens, local_experts, device="cpu"
        )

    monkeypatch.setattr(fused_a2a, "init_hybrid_ep_buffer", init_cpu_buffer)
    losses, inputs, handles = [], [], []
    for tokens in (5, 7, 11):
        x = torch.zeros((tokens, 16), dtype=torch.bfloat16, requires_grad=True)
        probs = torch.ones((tokens, 12), dtype=torch.float32, requires_grad=True)
        routes = torch.zeros_like(probs, dtype=torch.bool)
        routes[:, :2] = True
        received, weights, _, _, handle = fused_a2a.HybridEPDispatch.apply(
            x,
            routes,
            probs,
            fake_group,
            3,
            None,
            None,
            None,
            None,
            True,
            None,
            alignment,
            108,
            None,
            None,
        )
        weight = torch.ones((16, 16), dtype=torch.bfloat16, requires_grad=True)
        local = (received @ weight) * weights.to(torch.bfloat16).unsqueeze(1)
        result = fused_a2a.HybridEPCombine.apply(local, handle, None, alignment, True)
        losses.append(result.float().sum())
        inputs.append((x, probs, weight))
        handles.append(weakref.ref(handle))
    del handle
    fused_a2a.reset_hybrid_ep_buffer()
    assert all(ref() is not None for ref in handles)
    sum(losses).backward()
    for x, probs, weight in inputs:
        assert x.grad is not None and x.grad.shape == x.shape
        assert probs.grad is not None and probs.grad.shape == probs.shape
        assert weight.grad is not None
    del losses, result, local, received, weights
    gc.collect()
    assert all(ref() is None for ref in handles)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU memory probe")
@pytest.mark.parametrize("tp,rank,grouped,fused", [(1, 0, False, False), (2, 15, True, True)])
def test_native_fake_hybrid_dispatcher(tp, rank, grouped, fused, monkeypatch):
    from torch.testing._internal.distributed.fake_pg import FakeStore

    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.moe.moe_utils import get_align_size_for_quantization
    from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher
    from megatron.core.transformer.transformer_config import TransformerConfig

    if not fused_a2a.HAVE_HYBRIDEP:
        pytest.skip("HybridEP bindings required")
    if torch.distributed.is_initialized():
        pytest.skip("requires an isolated process without a pre-existing process group")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    torch.distributed.init_process_group("fake", store=FakeStore(), rank=rank, world_size=8 * tp)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=2,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
        create_gloo_process_groups=False,
    )
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        config = TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=2,
            expert_model_parallel_size=4,
            expert_tensor_parallel_size=1,
            sequence_parallel=tp > 1,
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            num_moe_experts=16,
            moe_router_topk=4,
            moe_router_dtype="fp32",
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="hybridep",
            moe_router_force_load_balancing=True,
            moe_grouped_gemm=True,
            moe_use_grouped_tensor=grouped,
            moe_permute_fusion_into_hybridep=fused,
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("real HybridEP Buffer initialized in fake mode")

        monkeypatch.setattr(fused_a2a.HybridEPBuffer, "__init__", forbidden)
        ep_rank = groups.ep.rank()
        dispatcher = MoEFlexTokenDispatcher(
            4, list(range(ep_rank * 4, ep_rank * 4 + 4)), config, groups
        )
        losses, inputs, handles = [], [], []
        for tokens in (65, 129, 193):
            x = torch.zeros(
                (tokens, 1, 128), dtype=torch.bfloat16, device="cuda", requires_grad=True
            )
            probs = torch.ones((tokens, 16), device="cuda", requires_grad=True)
            routes = torch.zeros_like(probs, dtype=torch.bool)
            routes[:, :4] = True
            local, weights = dispatcher.dispatch_preprocess(x, routes, probs)
            received, weights = dispatcher.token_dispatch(local, weights)
            local, counts, weights = dispatcher.dispatch_postprocess(received, weights)
            alignment = max(get_align_size_for_quantization(config), 1)
            expected = (tokens + alignment - 1) // alignment * alignment
            assert counts.tolist() == [expected] * 4
            assert received.shape == (4 * expected, 128)
            weight = torch.ones((128, 128), dtype=torch.bfloat16, device="cuda", requires_grad=True)
            expert = (local @ weight) * weights.to(torch.bfloat16).unsqueeze(1)
            handles.append(weakref.ref(dispatcher._comm_manager.handle))
            local = dispatcher.combine_preprocess(expert)
            result = dispatcher.combine_postprocess(dispatcher.token_combine(local))
            assert result.shape == x.shape
            losses.append(result.float().sum())
            inputs.append((x, probs, weight))
        sum(losses).backward()
        for x, probs, weight in inputs:
            assert x.grad is not None and x.grad.shape == x.shape
            assert probs.grad is not None and probs.grad.shape == probs.shape
            assert weight.grad is not None
        del losses, result, local, expert, received, weights
        gc.collect()
        assert all(ref() is None for ref in handles)
        torch.cuda.synchronize()
    finally:
        fused_a2a.reset_hybrid_ep_buffer()
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()
