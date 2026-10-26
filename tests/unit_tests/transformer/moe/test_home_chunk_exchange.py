# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bounded home migration: cyclic copies, tails, mixed types and stable storage."""

import pytest
import torch
import torch.distributed as dist

from megatron.core.transformer.moe.home_expert_dispatch import HomeExpertDispatch
from megatron.core.transformer.moe.home_expert_state import HomeStateShard, ShardedHomeExpertState
from megatron.core.transformer.moe.replica_nccl_transport import NcclP2PTransport
from megatron.core.transformer.moe.replica_weight_transport import (
    HomeExpertPlacement,
    ReplicaTransportConfig,
)
from tests.unit_tests.transformer.moe.test_replica_nccl_transport import distributed_world, ep_group

pytestmark = pytest.mark.launch_on_gb200


def test_cpu_chunk_adapter_preserves_cycles_and_tail():
    """Independent permutation reference, using a budget smaller than one expert."""
    values = torch.arange(3 * 19, dtype=torch.float32).view(3, 19)
    shards = [HomeStateShard(t, 0, 19) for t in values]
    state = ShardedHomeExpertState(lambda: {"master": shards}, device=torch.device("cpu"))
    (component,) = state.prepare_components()
    before = values.clone()
    pointers = [s.tensor.data_ptr() for s in shards]
    send = torch.empty(3, 4)
    receive = torch.empty_like(send)
    for offset in range(0, 19, 4):
        n = min(4, 19 - offset)
        state.read_chunk(component, (0, 1, 2), offset, send[:, :n])
        receive[:, :n].copy_(send[[2, 0, 1], :n])
        state.write_chunk(component, (0, 1, 2), offset, receive[:, :n])
    torch.testing.assert_close(values, before[[2, 0, 1]], rtol=0, atol=0)
    assert [s.tensor.data_ptr() for s in shards] == pointers


def test_bounded_exchange_cycles_fixed_points_and_mixed_states(ep_group, monkeypatch):
    rank, size = dist.get_rank(ep_group), dist.get_world_size(ep_group)
    device = torch.device("cuda", torch.cuda.current_device())
    cfg = ReplicaTransportConfig(
        group=ep_group,
        device=device,
        world_size=size,
        num_local_home_experts=2,
        num_local_replica_slots=0,
        member_shapes=((1, 19), (1, 53)),
        weight_format="bf16",
        rowwise_scale_shapes=None,
        columnwise_scale_shapes=None,
        grad_dtype=torch.float32,
        num_sms=None,
    )
    transports = [NcclP2PTransport(cfg), NcclP2PTransport(cfg)]
    components = {}
    for name, n, dtype in (
        ("master", 19, torch.float32),
        ("moment", 53, torch.float64),
        ("bytes", 7, torch.uint8),
    ):
        components[name] = [
            HomeStateShard((torch.arange(n, device=device) + (rank * 2 + i) * 7).to(dtype), 0, n)
            for i in range(2)
        ]
    state = ShardedHomeExpertState(lambda: components, device=device)
    pointers = [s.tensor.data_ptr() for shards in components.values() for s in shards]

    def forbid_snapshot():
        raise AssertionError("Production migration must never materialize a full snapshot")

    monkeypatch.setattr(state, "snapshot", forbid_snapshot)
    logical = torch.arange(size * 2)
    permutation = torch.arange(size * 2)
    permutation[: min(size * 2, 3)] = permutation[: min(size * 2, 3)].roll(1)
    plans = [
        permutation,
        torch.arange(size * 2).view(size, 2).flip(1).flatten(),
        torch.arange(size * 2),
    ]
    workspace = None
    try:
        for version, perm in enumerate(plans, 1):
            home = HomeExpertDispatch(buffer_size_bytes=128)
            home.bind_transport(transports[version % 2])
            home.bind_state_adapter(state)
            home.dispatch(None, HomeExpertPlacement(perm.view(size, 2), version), None)
            assert home.step() == version
            logical = logical[perm]
            for shards in components.values():
                for slot, shard in enumerate(shards):
                    expected = (
                        torch.arange(shard.numel, device=device)
                        + logical[rank * 2 + slot].item() * 7
                    ).to(shard.tensor.dtype)
                    torch.testing.assert_close(shard.tensor, expected, rtol=0, atol=0)
            current = home.transport._home_chunk_workspace
            assert current.send.numel() + current.receive.numel() <= 128
            if workspace is not None:
                assert workspace is current
            workspace = current
            assert [
                s.tensor.data_ptr() for shards in components.values() for s in shards
            ] == pointers
        # Reject an undersized budget before modifying any state or committing a map.
        home = HomeExpertDispatch(buffer_size_bytes=8)
        home.bind_transport(transports[0])
        home.bind_state_adapter(state)
        home.dispatch(None, HomeExpertPlacement(plans[0].view(size, 2), 4), None)
        before = [s.tensor.clone() for shards in components.values() for s in shards]
        with pytest.raises(ValueError, match="budget"):
            home.step()
        assert home._completed_version == 0
        with pytest.raises(RuntimeError, match="installation failed"):
            home.assert_healthy()
        for tensor, shard in zip(before, [s for shards in components.values() for s in shards]):
            torch.testing.assert_close(tensor, shard.tensor, rtol=0, atol=0)
    finally:
        for transport in transports:
            transport.destroy()
