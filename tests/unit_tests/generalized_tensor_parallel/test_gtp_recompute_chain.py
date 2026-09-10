# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression tests for the GTP recompute-forward prefetch chain.

Weights re-gathered during an activation-recompute forward form their own chain and prefetch one
node ahead, so two adjacent nodes sharing a gather buffer is a data race: the prefetch of node
i+1 overwrites the weight node i is still reading. In training that shows up as silently wrong
recomputed activations, exploding grad norm, then NaN.

The buffer tests are the regression guard. The clobber is a stream race, so a numerical test
only trips when the timing lines up -- removing the parity fails the buffer tests but not the
numerical one at this size. TestGroupedDoubleBuffer asserts cache keys for the same reason.

Test groups
-----------
TestGTPRecomputeChainBuffers         - adjacent recompute nodes never share a gather buffer
TestGTPRecomputeCorrectness          - recompute reproduces the non-recompute dgrad and wgrads
TestGroupedGTPRecomputeChainBuffers  - same invariant per expert on a grouped chain
"""

import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

import megatron.core.tensor_parallel.generalized_tensor_parallelism as gtp_module
from megatron.core.tensor_parallel.random import CheckpointWithoutOutput
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _make_gtp_linear,
    _make_gtp_remat_grouped_linear,
    _requires_multi_gpu,
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)

# Topology matters. Each block is one recomputed square GEMM followed by two NON-recomputed
# GEMMs of a different shape:
#
#   fwd chain:        A0 (H,H) -> B0 (O,H) -> C0 (H,O) -> A1 (H,H) -> ...   heterogeneous
#   recompute chain:  A0 -------------------------------> A1 -> ...         homogeneous
#
# The differently-shaped B/C are what make this a real test: without them the A weights are
# adjacent in the FORWARD chain too, _ensure_no_shared_buffer_with separates them there,
# and the recompute-chain guard becomes redundant -- so the test would pass even when broken.
HIDDEN = 512
OTHER = 256
NUM_LAYERS = 4
DTYPE = torch.bfloat16


def _recompute_buffer_addr(param):
    """Address of the buffer this weight gathers into on the recompute chain, or None.

    Test-only: the production code has no reason to know a buffer's address.
    """
    ticket = getattr(param, "_ag_ticket_recompute", None)
    if ticket is None:  # ticket ids start at 0, so compare against None
        return None
    slot = gtp_module.get_global_GTP_cache()._slots.get(ticket)
    buf = slot.buf if slot is not None else None
    if buf is None:
        return None
    raw = getattr(buf, "_rowwise_data", None)
    if raw is None:
        raw = getattr(buf, "_data", buf)
    return raw.data_ptr()


def _build_layers(world_size):
    """Return (recomputed square layers, non-recomputed differently-shaped spacer pairs)."""
    gtp_remat_group = dist.new_group(list(range(world_size)))

    def linear(in_f, out_f):
        return _make_gtp_linear(in_f, out_f, gtp_remat_group, DTYPE, fuse_wgrad_accumulation=True)

    recomputed = [linear(HIDDEN, HIDDEN) for _ in range(NUM_LAYERS)]
    spacers = [(linear(HIDDEN, OTHER), linear(OTHER, HIDDEN)) for _ in range(NUM_LAYERS)]
    for layer in recomputed + [m for pair in spacers for m in pair]:
        # GTP reduce-scatters wgrad into main_grad, on the local shard shape.
        layer.weight.main_grad = torch.zeros(layer.weight.shape, dtype=DTYPE, device="cuda")
    return recomputed, spacers


def _zero_grads(layers):
    recomputed, spacers = layers
    for layer in recomputed + [m for pair in spacers for m in pair]:
        layer.weight.main_grad.zero_()


def _forward_backward(layers, x, recompute):
    """Run the stack, optionally checkpointing every layer so it is recomputed in backward.

    Mirrors the production pattern: checkpoint the GTP GEMM, let a downstream op consume (and
    save) its output, then hook the recompute on that downstream tensor so it fires during this
    layer's backward.
    """
    recomputed, spacers = layers
    h = x
    for layer, (spacer_down, spacer_up) in zip(recomputed, spacers):
        # te.Linear returns a bare tensor (bias=False), not Megatron's (out, bias) tuple.
        if recompute:
            checkpoint = CheckpointWithoutOutput()
            y = checkpoint.checkpoint(lambda inp, l=layer: l(inp), h)
            # gelu saves y for backward, so discarding y is what makes the recompute necessary.
            h = torch.nn.functional.gelu(y)
            checkpoint.discard_output_and_register_recompute(h)
        else:
            h = torch.nn.functional.gelu(layer(h))
        # Not checkpointed: keeps the fwd chain heterogeneous around each recomputed weight.
        h = torch.nn.functional.gelu(spacer_down(h))
        h = torch.nn.functional.gelu(spacer_up(h))
    loss = h.float().sum()
    loss.backward()
    # Return the INPUT gradient, not the loss: the loss is produced by the forward pass, so it
    # is identical either way and cannot witness anything the recompute got wrong.
    return x.grad.detach().clone()


def _worker_adjacent_nodes_use_distinct_buffers(rank, world_size, port):
    """Every adjacent pair of recompute-chain nodes must gather into different buffers."""
    torch.manual_seed(0)
    gtp_module.reset_gtp_state()
    layers = _build_layers(world_size)
    recomputed = layers[0]

    x = torch.randn(8, HIDDEN, dtype=DTYPE, device="cuda", requires_grad=True)
    dist.broadcast(x, src=0)

    # First pass builds the chain (all gathers on demand); the second uses it.
    for _ in range(2):
        _forward_backward(layers, x, recompute=True)

    chain = []
    node = recomputed[-1].weight
    while node is not None and node._recompute_prev is not None:
        node = node._recompute_prev
    while node is not None:
        chain.append(node)
        node = node._recompute_next

    assert len(chain) == NUM_LAYERS, f"recompute chain has {len(chain)} nodes, want {NUM_LAYERS}"

    addrs = [_recompute_buffer_addr(w) for w in chain]
    assert all(a is not None for a in addrs), f"unallocated recompute buffer: {addrs}"
    shared = [i for i in range(len(addrs) - 1) if addrs[i] == addrs[i + 1]]
    assert not shared, (
        f"recompute-chain nodes {shared} share a gather buffer with their successor "
        f"(addrs={[hex(a) for a in addrs]}); the one-ahead prefetch would clobber the weight "
        "still being read"
    )
    # One-ahead needs exactly two buffers; more would mean the pool stopped being reused.
    assert len(set(addrs)) == 2, f"want 2 alternating buffers, got {len(set(addrs))}: {addrs}"


def _worker_recompute_matches_no_recompute(rank, world_size, port):
    """Recompute is pure rematerialization: same input grad and same weight grads.

    End-to-end sanity check over the real CheckpointWithoutOutput path. It does not reliably
    catch the buffer-sharing bug on its own (see the module docstring); the buffer invariant is
    what guards that.
    """
    torch.manual_seed(0)
    gtp_module.reset_gtp_state()
    layers = _build_layers(world_size)
    recomputed = layers[0]

    x = torch.randn(8, HIDDEN, dtype=DTYPE, device="cuda", requires_grad=True)
    dist.broadcast(x, src=0)

    # Warm up so the chains exist, then measure a steady-state step of each variant.
    _forward_backward(layers, x, recompute=False)
    _forward_backward(layers, x, recompute=True)

    _zero_grads(layers)
    x.grad = None
    dgrad_ref = _forward_backward(layers, x, recompute=False)
    grads_ref = [l.weight.main_grad.clone() for l in recomputed]

    _zero_grads(layers)
    x.grad = None
    dgrad_rc = _forward_backward(layers, x, recompute=True)
    grads_rc = [l.weight.main_grad.clone() for l in recomputed]

    # Both dgrad and wgrad flow through the recomputed weights, so both witness a clobber.
    torch.testing.assert_close(dgrad_rc.float(), dgrad_ref.float(), rtol=1e-3, atol=1e-3)
    for i, (g_rc, g_ref) in enumerate(zip(grads_rc, grads_ref)):
        # A clobbered recompute uses a neighbour's weight, which moves the grad by O(grad),
        # far outside this tolerance.
        torch.testing.assert_close(
            g_rc.float(), g_ref.float(), rtol=1e-3, atol=1e-3, msg=f"layer {i} wgrad mismatch"
        )


class TestGTPRecomputeChainBuffers:
    def test_adjacent_nodes_use_distinct_buffers(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_adjacent_nodes_use_distinct_buffers, 4)


class TestGTPRecomputeCorrectness:
    def test_recompute_matches_no_recompute(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_recompute_matches_no_recompute, 4)


# ---------------------------------------------------------------------------
# Grouped (routed-expert) recompute chains
# ---------------------------------------------------------------------------

NUM_GEMMS = 2
GROUPED_CHAIN = "GTP_remat_grouped_fc1_ungraphed"
# Strict subset: block 1 and 3 are gathered but never recomputed (see the worker docstring).
RECOMPUTED_BLOCKS = {0, 2}


def _worker_grouped_adjacent_nodes_use_distinct_buffers(rank, world_size, port):
    """Same invariant on a grouped one-block-ahead chain, per expert.

    Two details make this a real test rather than a tautology:
      * only a SUBSET of blocks is recomputed. The grouped chain's own _double_buffer_parity is
        drawn in FORWARD order over every block, so recomputing all of them leaves the recompute
        chain alternating by accident and the test passes even when unguarded. Skipping a block
        makes two same-parity weights adjacent on the recompute chain -- the real collision.
      * buffers are compared per EXPERT: grouped weights gather as a batch and the cache key
        carries expert_idx, so expert k of block N collides with expert k of block N+1, not with
        the anchor.
    """
    torch.manual_seed(0)
    gtp_module.reset_gtp_state()

    gtp_remat_group = dist.new_group(list(range(world_size)))
    blocks = [
        _make_gtp_remat_grouped_linear(
            NUM_GEMMS, HIDDEN, HIDDEN, gtp_remat_group, DTYPE, fuse_wgrad_accumulation=True
        )
        for _ in range(NUM_LAYERS)
    ]
    # Production assigns these from the param name in _classify_param_chain; do it by hand so
    # the weights land on the grouped one-block-ahead chain rather than the generic one.
    for block in blocks:
        for w in block.weight0.weight_list:
            w.chain_id = GROUPED_CHAIN
            w.main_grad = torch.zeros(w.shape, dtype=DTYPE, device="cuda")

    tokens = 8 * NUM_GEMMS
    m_splits = [tokens // NUM_GEMMS] * NUM_GEMMS
    x = torch.randn(tokens, HIDDEN, dtype=DTYPE, device="cuda", requires_grad=True)
    dist.broadcast(x, src=0)

    def fwd_bwd():
        h = x
        for i, block in enumerate(blocks):
            call = lambda inp, b=block: b(inp, m_splits=m_splits, is_first_microbatch=True)
            if i in RECOMPUTED_BLOCKS:
                checkpoint = CheckpointWithoutOutput()
                y = checkpoint.checkpoint(call, h)
                h = torch.nn.functional.gelu(y)
                checkpoint.discard_output_and_register_recompute(h)
            else:
                h = torch.nn.functional.gelu(call(h))
        h.float().sum().backward()

    for _ in range(2):  # first pass builds the chain, second uses it
        fwd_bwd()

    chain = [blocks[i].weight0 for i in sorted(RECOMPUTED_BLOCKS)]
    assert all(
        a._recompute_initialized for a in chain
    ), "grouped weights never gathered under recompute -- the chain was not built"

    failures = []
    for expert in range(NUM_GEMMS):
        addrs = [_recompute_buffer_addr(a.weight_list[expert]) for a in chain]
        if any(a is None for a in addrs):
            failures.append(f"expert {expert}: unallocated buffer {addrs}")
            continue
        shared = [i for i in range(len(addrs) - 1) if addrs[i] == addrs[i + 1]]
        if shared:
            failures.append(
                f"expert {expert}: chain nodes {shared} share a buffer with their successor "
                f"({[hex(a) for a in addrs]})"
            )
    assert not failures, "grouped recompute chain collides:\n  " + "\n  ".join(failures)


class TestGroupedGTPRecomputeChainBuffers:
    def test_grouped_adjacent_nodes_use_distinct_buffers(self):
        _requires_multi_gpu(4)
        _run_distributed(_worker_grouped_adjacent_nodes_use_distinct_buffers, 4)
