# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The non-consuming peek of a GTP weight's gathered buffer.

The virtual-expert weight push needs a weight's gathered bytes before the GEMM that consumes it
(see ``GTPShardedParam._peek_gathered_weight``). A peek must hand out the very buffer the
consume hands out afterwards, must leave the prefetch chain's bookkeeping as the consume expects
it, and must not make the consume gather again, at a chain head or in the middle of the chain.

Run with::

    python -m torch.distributed.run --nproc-per-node 4 -m pytest -q \
        tests/unit_tests/virtual_experts/test_gtp_virtual_experts.py
"""

import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)


from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp
from megatron.core.tensor_parallel import gtp_cuda_graphs
from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTPShardedParam
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)

pytestmark = pytest.mark.internal


@pytest.fixture(scope="module")
def four_rank_gtp_group(_torchrun_dist_init):
    """Keep the exact four-shard oracle on both four- and eight-GPU nodes."""
    groups = [
        dist.new_group(list(range(start, start + 4)))
        for start in range(0, dist.get_world_size(), 4)
    ]
    group = groups[dist.get_rank() // 4]
    yield group
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group(group)


def _full_weight(p):
    """All-gather the shard over its GTP group: the unpadded, unsharded weight."""
    shards = [torch.empty_like(p.data) for _ in range(p.group.size())]
    dist.all_gather(shards, p.data, group=p.group)
    full = torch.cat(shards, dim=0)
    return full[: full.shape[0] - p.pad_length] if p.pad_length else full


def _assert_clean(p):
    """No in-flight gather and no stale hand-off marker after a consume."""
    assert p._prefetch_handle is None
    assert not getattr(p, "_already_ag_drained", False)


@pytest.mark.parametrize("fwd", [True, False], ids=["forward", "backward"])
@pytest.mark.parametrize("readiness", ["missing", "in_flight", "ready"])
def test_peek_returns_ready_current_data(monkeypatch, four_rank_gtp_group, fwd, readiness):
    """Every readiness path exposes current bytes before consume, with exactly one gather.

    Two same-key cache buffers prevent a key-only lookup from identifying the actual AG output.
    The first consume can replace its ticket; subsequent peeks must follow the actual gather.
    """
    check_states = True
    cache = gtp.GTPWeightCache()
    monkeypatch.setattr(gtp, "_GTP_CACHE", cache)
    monkeypatch.setattr(gtp, "_GTP_GROUPED_BUF_PARITY_COUNTER", {})
    monkeypatch.setattr(gtp.GTP_CONFIG, "check_param_states", check_states)
    monkeypatch.setattr(gtp.GTP_CONFIG, "weight_prefetch", True)
    p = GTPShardedParam(torch.zeros(8, 16, dtype=torch.bfloat16, device="cuda"))
    p.group = four_rank_gtp_group
    p.chain_id = "GTP_remat_grouped_fc1_ungraphed"
    held = [cache.reserve(p, p.dtype, fwd=fwd) for _ in range(2)]
    buffers = [cache.get(ticket) for ticket in held]
    for ticket in held:
        cache.release(ticket)
    assert buffers[0].data_ptr() != buffers[1].data_ptr()

    calls = []
    gather = p._all_gather_weight

    def counted_gather(*args, **kwargs):
        calls.append(kwargs["fwd"])
        return gather(*args, **kwargs)

    monkeypatch.setattr(p, "_all_gather_weight", counted_gather)
    peek = p.peek_group_for_forward if fwd else p.peek_group_for_backward
    consume = p.materialize_group_for_forward if fwd else p.materialize_group_for_backward
    pointers = []
    for step in range(3):
        p.data.fill_(16 * step + dist.get_rank())
        expected = _full_weight(p)
        if readiness != "missing":
            _, p._prefetch_handle = p._all_gather_weight(async_op=True, fwd=fwd)
            assert p._prefetch_handle is not None
            if check_states:
                assert p.state == gtp.GTPWeightState.ASYNC_WAIT
        if readiness == "ready":
            # An external drain (or another peek) has finished the gather's host-side work.
            p._wait_param_gather()
            p._already_ag_drained = True
        initialized, previous, following = p.prefetch_initialized, p.prev_w, p.next_w
        peeked = peek()
        # Read immediately: a later consume must not supply the missing completion wait.
        torch.testing.assert_close(peeked, expected, rtol=0, atol=0)
        assert p._prefetch_handle is None and p._already_ag_drained
        if check_states:
            assert p.state == gtp.GTPWeightState.DATA_READY
        assert (p.prefetch_initialized, p.prev_w, p.next_w) == (initialized, previous, following)
        # A second peek on a different stream still observes completion and issues no gather.
        with torch.cuda.stream(torch.cuda.Stream()):
            again = peek()
            torch.testing.assert_close(again, expected, rtol=0, atol=0)
        assert again.data_ptr() == peeked.data_ptr()
        gathered = consume()
        assert gathered.data_ptr() == peeked.data_ptr()
        torch.testing.assert_close(gathered, expected, rtol=0, atol=0)
        assert len(calls) == step + 1
        _assert_clean(p)
        pointers.append(peeked.data_ptr())
    assert pointers[1] == pointers[2], "the established deterministic schedule must reuse storage"


@pytest.mark.parametrize("async_reduction", [False, True], ids=["sync", "async"])
def test_persistent_wgrad_reuse_preserves_every_reduction(
    monkeypatch, four_rank_gtp_group, async_reduction
):
    """Full gradients, padding and exactly-once completion survive shared/repeated writers."""
    group = four_rank_gtp_group
    rank = dist.get_rank(group)
    monkeypatch.setattr(gtp_cuda_graphs, "_GRAPH_WGRAD_RINGS", {})
    monkeypatch.setattr(gtp, "_GTP_GROUPED_BUF_PARITY_COUNTER", {})
    monkeypatch.setattr(gtp.GTP_CONFIG, "async_reduction", async_reduction)
    monkeypatch.setattr(gtp.GTP_CONFIG, "reduce_scatter_with_fp32_accumulation", True)
    monkeypatch.setattr(gtp.GTP_CONFIG, "calculate_per_token_loss", False)
    # Both FC roles deliberately have the same shape; each has two independent experts.
    layers = []
    expected, completions = {}, {}
    for layer in range(3):
        roles = []
        for role in ("fc1", "fc2"):
            weights = [
                gtp.GTPShardedParam(torch.zeros(33, 16, dtype=torch.bfloat16, device="cuda"))
                for _ in range(2)
            ]
            for expert, weight in enumerate(weights):
                weight.group, weight.pad_length, weight.expert_idx = group, 2, expert
                weight.chain_id = f"GTP_remat_grouped_{role}_ungraphed"
                weight.is_routed_expert = True
                weight._debug_name = f"layers.{layer}.{role}.weight{expert}"
                weight.main_grad = torch.full_like(weight, 0.5)
                weight._double_buffer_parity()  # Normally assigned by the forward gathers.
                expected[id(weight)] = weight.main_grad.clone()
                completions[id(weight)] = 0

                def completed(weight=weight):
                    completions[id(weight)] += 1

                weight.register_grad_accum_hook(None, completed)
            weights[0].weight_list = weights
            roles.append(weights)
        layers.append(roles)
    for previous, current in zip(layers, layers[1:]):
        for prev_weights, weights in zip(previous, current):
            prev_weights[0].next_w, weights[0].prev_w = weights[0], prev_weights[0]

    pointers = {}
    try:
        # Repeating the tail before the cascade drains its first RS exercises early reuse.
        for step in range(3):
            for layer_index in (2, 2, 1, 0):
                for role_index in (1, 0):
                    weights = layers[layer_index][role_index]
                    if async_reduction:
                        with torch.cuda.stream(gtp.get_rs_stream(weights[0].chain_id, group)):
                            torch.cuda._sleep(1_000_000)
                    grads = []
                    for weight in weights:
                        grad = weight.get_wgrad_tensor(persistent=True)
                        prior = pointers.setdefault(id(weight), grad.data_ptr())
                        assert grad.data_ptr() == prior
                        if step == 1:
                            # Foreign gradients must copy into the same padded ring storage.
                            grad = torch.empty_like(grad)
                        base = (
                            torch.arange(130 * 16, device="cuda").view(130, 16) % 11 - 5
                        ).float() / 4 + (step + layer_index + weight.expert_idx) / 8
                        grad.copy_(base + rank / 4)
                        mean = torch.nn.functional.pad(base + 3 / 8, (0, 0, 0, 2))
                        # Mean 250.25 + main_grad 0.5 rounds to 251 once, or 250.5 if the
                        # RS output is first rounded to BF16. Padding must still stay zero.
                        grad[:, 0] = 251 if rank == 3 else 250
                        mean[:130, 0] = 250.25
                        expected[id(weight)].add_(mean.chunk(4)[rank])
                        grads.append(grad)
                    weights[0].finalize_group_grads(grads)
            torch.cuda.synchronize()
            for layer in layers:
                for weights in layer:
                    for weight in weights:
                        torch.testing.assert_close(
                            weight.main_grad, expected[id(weight)], rtol=0, atol=0
                        )
                        assert not torch.count_nonzero(
                            weight._gtp_graph_wgrad_ring_slot.tensor[130:]
                        )
                        calls_per_step = 2 if layer is layers[-1] else 1
                        assert completions[id(weight)] == (step + 1) * calls_per_step
            # Two buffers per role/expert, independent of the three-layer model depth.
            assert len(set(pointers.values())) == 8
            assert len(gtp_cuda_graphs._GRAPH_WGRAD_RINGS) == 8
    finally:
        torch.cuda.synchronize()
        for layer in layers:
            for weights in layer:
                weights[0]._wait_reduce_scatter(finalize_grad=True)
