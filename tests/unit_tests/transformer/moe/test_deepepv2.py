# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.common.utils import should_free_input
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.moe import fused_a2a
from megatron.core.transformer.moe.token_dispatcher import _DeepepV2Manager
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def fresh_nccl_group(monkeypatch):
    """Create a group whose communicator has not been initialized by a collective."""
    Utils.initialize_distributed()
    group = torch.distributed.new_group(backend="nccl")
    monkeypatch.setattr(fused_a2a, "_elastic_buffer", None)
    yield group
    torch.cuda.synchronize()
    fused_a2a._elastic_buffer = None
    torch.distributed.destroy_process_group(group)


def test_elastic_buffer_initializes_communicator_and_shares_capacity(fresh_nccl_group, monkeypatch):
    """Size a fresh buffer safely and keep every rank consistent on reuse and growth."""
    group = fresh_nccl_group
    capacities = []

    class BufferStub(SimpleNamespace):
        @staticmethod
        def get_buffer_size_hint(group, num_max_tokens_per_rank, hidden, num_topk):
            backend = group._get_backend(torch.device("cuda"))
            if hasattr(backend, "_comm_ptr"):
                assert backend._comm_ptr() != 0
            capacities.append(num_max_tokens_per_rank)
            return num_max_tokens_per_rank * hidden * 2

        def __init__(self, group, **kwargs):
            super().__init__(group=group, **kwargs)

    monkeypatch.setattr(fused_a2a, "ElasticBuffer", BufferStub, raising=False)
    first = fused_a2a.get_elastic_buffer(group, (group.rank() + 1) * 16, 1024, 2)
    assert first.num_max_tokens_per_rank == group.size() * 16

    # The largest local input moves to a different rank and shrinks. Reuse the
    # existing capacity so dispatch still gets the same upper bound on every rank.
    reused = fused_a2a.get_elastic_buffer(group, (group.size() - group.rank()) * 8, 1024, 2)
    assert reused is first
    assert reused.num_max_tokens_per_rank == group.size() * 16

    grown = fused_a2a.get_elastic_buffer(group, (group.rank() + 1) * 32, 1024, 2)
    assert grown is not first
    assert grown.num_max_tokens_per_rank == group.size() * 32
    assert capacities == [group.size() * factor for factor in (16, 8, 32)]


@pytest.mark.parametrize("backend", ["deepep", "deepepv2", "hybridep", "ncclep"])
@pytest.mark.parametrize("graph_modules", [[], [CudaGraphModule.moe_preprocess]])
def test_flex_dispatch_retains_router_input(backend, graph_modules):
    """Overlap must retain the router's saved input even outside CUDA graphs."""
    config = SimpleNamespace(
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend=backend,
        cuda_graph_modules=graph_modules,
        fp8=None,
        fp4=None,
    )
    assert not should_free_input("moe_dispatch", True, config, num_local_experts=2)
    assert should_free_input("moe_combine", True, config, num_local_experts=2)


@pytest.mark.skipif(not fused_a2a.HAVE_DEEP_EP_V2, reason="DeepEP v2 is not available")
@pytest.mark.parametrize("uneven_inputs", [False, True])
@pytest.mark.parametrize("async_finish", [False, True])
def test_deepepv2_uneven_dispatch_backward(fresh_nccl_group, uneven_inputs, async_finish):
    """Round-trip tokens and gradients through a fresh group and a reused buffer."""
    group = fresh_nccl_group
    config = TransformerConfig(
        num_layers=1,
        hidden_size=1024,
        num_attention_heads=8,
        num_moe_experts=group.size() * 2,
        moe_router_topk=2,
        moe_router_dtype="fp32",
    )
    manager = _DeepepV2Manager(group, 2, 2, config.num_moe_experts, config)
    initial_buffer = None
    for iteration in range(2):
        # Exercise an empty rank as well as unequal nonempty ranks. The second
        # iteration has smaller inputs and must keep the shared cached capacity.
        tokens = (group.rank() * 16 if uneven_inputs else 64) // (iteration + 1)
        hidden_states = torch.randn(
            tokens, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        indices = torch.arange(tokens, device="cuda") % config.num_moe_experts
        manager.token_indices = torch.stack(
            (indices, (indices + 1) % config.num_moe_experts), dim=-1
        )
        probs = torch.full((tokens, 2), 0.5, device="cuda", requires_grad=True)
        manager.token_probs = probs

        dispatched = manager.dispatch(
            hidden_states, async_finish=async_finish, allocate_on_comm_stream=async_finish
        )
        expected_capacity = (group.size() - 1) * 16 if uneven_inputs else 64
        assert manager.buffer.num_max_tokens_per_rank == expected_capacity
        if initial_buffer is None:
            initial_buffer = manager.buffer
        else:
            assert manager.buffer is initial_buffer
        permuted, permuted_probs = manager.get_permuted_hidden_states_by_experts(dispatched)
        weighted = (permuted * permuted_probs.unsqueeze(-1)).to(hidden_states.dtype)
        restored = manager.get_restored_hidden_states_by_experts(weighted)
        output = manager.combine(
            restored, async_finish=async_finish, allocate_on_comm_stream=async_finish
        )
        torch.testing.assert_close(output, hidden_states)
        output.sum().backward()
        torch.testing.assert_close(hidden_states.grad, torch.ones_like(hidden_states))
        assert probs.grad is not None
        assert torch.isfinite(probs.grad).all()
