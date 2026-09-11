# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The non-consuming peek of a GTP weight's gathered buffer.

The virtual-expert weight push needs a weight's gathered bytes before the GEMM that consumes it
(see ``GTPShardedParam._peek_gathered_weight``). A peek must hand out the very buffer the
consume hands out afterwards, must leave the prefetch chain's bookkeeping as the consume expects
it, and must not make the consume gather again, at a chain head or in the middle of the chain.

Run with::

    python -m torch.distributed.run --nproc-per-node 4 -m pytest -q \
        tests/unit_tests/generalized_tensor_parallel/test_gtp_peek.py
"""

import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp
from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTPShardedParam
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)
from tests.unit_tests.generalized_tensor_parallel.test_gtp_grad_correctness import (
    BATCH,
    HIDDEN,
    SEQ,
    _make_config,
    _make_stack,
    dtype,
)

pytestmark = pytest.mark.launch_on_gb200


def _chain(stack):
    """The stack's GTP weights in prefetch-chain order, head first."""
    params = [p for p in stack.parameters() if isinstance(p, GTPShardedParam)]
    (head,) = [p for p in params if p.prev_w is None and p.next_w is not None]
    chain, p = [], head
    while p is not None:
        chain.append(p)
        p = p.next_w
    assert len(chain) == len(params)
    return chain


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


def _check_peek_then_consume(p, fwd):
    peeked = p.peek_group_for_forward() if fwd else p.peek_group_for_backward()
    consumed = p.materialize_group_for_forward() if fwd else p.materialize_group_for_backward()
    assert peeked.data_ptr() == consumed.data_ptr(), "the consume must hand out the peeked buffer"
    torch.testing.assert_close(peeked, _full_weight(p), rtol=0, atol=0)
    _assert_clean(p)


def _worker(rank, world_size, port):
    from megatron.core import parallel_state as ps
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=2
    )
    model_parallel_cuda_manual_seed(42)
    pgc = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp', 'gtp_remat'])
    stack = _make_stack(_make_config(), pgc)
    for layer in stack:
        layer.cuda()
    gtp.tag_gtp_params_with_names(stack)

    try:
        # One real forward builds the chain links (in consume order) and leaves no prefetch in
        # flight: a first-iteration consume has no successor to prefetch yet.
        with torch.no_grad():
            out = torch.randn(SEQ, BATCH, HIDDEN, dtype=dtype, device="cuda")
            for layer in stack:
                out, _ = layer(out, attention_mask=None)
        torch.cuda.synchronize()
        chain = _chain(stack)
        assert len(chain) >= 2
        for p in chain:
            _assert_clean(p)

        # Count the collectives each weight issues: the peek must not add gathers on top of the
        # chain's own (one per weight per direction).
        gathers = {id(p): 0 for p in chain}
        original = GTPShardedParam._all_gather_weight

        def counting(self, *args, **kwargs):
            gathers[id(self)] += 1
            return original(self, *args, **kwargs)

        GTPShardedParam._all_gather_weight = counting
        try:
            # Forward: the head has no predecessor, so its peek issues the gather itself; every
            # other weight's peek drains the prefetch its predecessor issued at the consume.
            for p in chain:
                _check_peek_then_consume(p, fwd=True)
            assert all(gathers[id(p)] == 1 for p in chain), gathers
            # MTP replays a block, so a weight can be consumed twice per forward: the second peek
            # has no prefetch and gathers on its own, once.
            _check_peek_then_consume(chain[-1], fwd=True)
            assert gathers[id(chain[-1])] == 2
            # Backward walks the chain in reverse; the tail is the chain head there.
            for p in reversed(chain):
                _check_peek_then_consume(p, fwd=False)
            assert all(gathers[id(p)] == (3 if p is chain[-1] else 2) for p in chain), gathers
            # A consume without a peek still works (the plain GTP path).
            for p in chain:
                p.materialize_group_for_forward()
                _assert_clean(p)
            for p in reversed(chain):
                p.materialize_group_for_backward()
                _assert_clean(p)
        finally:
            GTPShardedParam._all_gather_weight = original
        torch.cuda.synchronize()
    finally:
        ps.destroy_model_parallel()
        GTPShardedParam._chain_state = {}


def test_peek_hands_out_the_buffer_the_consume_hands_out():
    _run_distributed(_worker, 4)


@pytest.mark.parametrize("check_states", [False, True], ids=["states_off", "states_on"])
@pytest.mark.parametrize("fwd", [True, False], ids=["forward", "backward"])
@pytest.mark.parametrize("readiness", ["missing", "in_flight", "ready"])
def test_peek_returns_ready_current_data(monkeypatch, check_states, fwd, readiness):
    """Every readiness path exposes current bytes before consume, with exactly one gather.

    Two same-key cache buffers prevent a key-only lookup from identifying the actual AG output.
    The first consume can replace its ticket; subsequent peeks must follow the actual gather.
    """
    if dist.get_world_size() != 4:
        pytest.skip("requires four ranks")
    cache = gtp.GTPWeightCache()
    monkeypatch.setattr(gtp, "_GTP_CACHE", cache)
    monkeypatch.setattr(gtp, "_GTP_GROUPED_BUF_PARITY_COUNTER", {})
    monkeypatch.setattr(gtp.GTP_CONFIG, "check_param_states", check_states)
    monkeypatch.setattr(gtp.GTP_CONFIG, "weight_prefetch", True)
    p = GTPShardedParam(torch.zeros(8, 16, dtype=torch.bfloat16, device="cuda"))
    p.group = dist.group.WORLD
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


def test_weight_push_rebinds_after_first_consume_changes_gather_ticket(monkeypatch):
    """Follow the actual AG allocation even if chain initialization chooses another buffer."""
    from megatron.core.transformer.moe.virtual_expert_load_balancer import WeightDirection
    from tests.unit_tests.transformer.moe.test_virtual_expert_planner import (
        _build_fc_layer,
        _weight_push,
    )

    if dist.get_world_size() != 4:
        pytest.skip("requires four ranks")
    cache = gtp.GTPWeightCache()
    monkeypatch.setattr(gtp, "_GTP_CACHE", cache)
    monkeypatch.setattr(gtp, "_GTP_GROUPED_BUF_PARITY_COUNTER", {})
    p = GTPShardedParam(torch.ones(32, 128, dtype=torch.bfloat16, device="cuda"))
    p.main_grad = torch.zeros_like(p, dtype=torch.float32)
    p.group = dist.group.WORLD
    p.chain_id = "GTP_remat_grouped_fc1_ungraphed"
    fc_layer = _build_fc_layer((p,), p, None)
    _, push = _weight_push(monkeypatch, fc_layer)
    assert p._ag_ticket_fwd is None and not fc_layer._tables[0]
    first_table = push(WeightDirection.FORWARD)
    first_pointer = p.peek_group_for_forward().data_ptr()
    # Another cache user checks out the pooled gather buffer and returns two choices. The
    # active gather still owns its ticket; first-consume setup can pick the other pooled buffer.
    held = [cache.reserve(p, p.dtype, fwd=True) for _ in range(2)]
    buffers = [cache.get(ticket) for ticket in held]
    assert buffers[0].data_ptr() == first_pointer
    assert buffers[1].data_ptr() != first_pointer
    for ticket in held:
        cache.release(ticket)
    fc_layer.consume(WeightDirection.FORWARD)
    assert cache.get(p._ag_ticket_fwd).data_ptr() != first_pointer
    assert fc_layer.runtime_weights[0][0].data_ptr() == first_pointer
    assert first_table[0].tolist() == [first_pointer]

    tables = {}
    for step in range(3):
        p.data.fill_(step + 2)
        for direction in WeightDirection:
            table = push(direction)
            torch.testing.assert_close(
                fc_layer.runtime_weights[0][0],
                torch.full_like(buffers[0], step + 2),
                rtol=0,
                atol=0,
            )
            fc_layer.consume(direction)
            if direction in tables:
                assert table is tables[direction]
            tables[direction] = table
            if direction == WeightDirection.BACKWARD:
                fc_layer.bind_native_grads(0, None)
    assert tables[WeightDirection.FORWARD] is not first_table
