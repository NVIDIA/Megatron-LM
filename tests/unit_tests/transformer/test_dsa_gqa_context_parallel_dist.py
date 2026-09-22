# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Parity tests for DSA-over-GQA context parallelism, against a real CP process group.

The single-process file next to this one constructs a rank's post-gather state directly. That
covers the position plumbing but deliberately not the collective: whether the gather's backward
reduce-scatters rather than splits is invisible there, and a split would be finite and silent.

These tests run the production helper against a live group, so the collective is under test:

  * the forward all-gather of K and V, and the zigzag reorder that follows it;
  * its backward, which must reduce-scatter dK and dV rather than split them;
  * the learned-K all-gather in ``_project_full_indexer_k``, which runs under ``torch.no_grad``
    and therefore has no backward of its own -- the custom backward owes that reduction;
  * the indexer loss all-reduce.

Every rank computes the single-rank reference itself from the same seed rather than broadcasting
it. The inputs are deterministic, so this costs one extra forward and removes a collective that
could mask the ones being tested.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
    _gather_kv_for_context_parallel,
    _simplified_indexer_norm_spec,
)
from megatron.core.transformer.experimental_attention_variant.dsa_layout import (
    build_zigzag_cp_local_positions,
)
from megatron.core.transformer.experimental_attention_variant.dsa_min_memory import (
    dsa_min_memory_gqa,
)
from tests.unit_tests.test_utilities import Utils

SEQLEN, BATCH, HIDDEN = 16, 2, 12
HEADS, HEAD_DIM, TOPK = 2, 4, 3

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="context-parallel groups need CUDA"
)


def _global_inputs(device):
    """Inputs every rank agrees on. Same seed everywhere, so no broadcast is needed."""
    torch.manual_seed(0)
    q = torch.randn(SEQLEN, BATCH, HEADS, HEAD_DIM, device=device)
    k = torch.randn(SEQLEN, BATCH, 1, HEAD_DIM, device=device)
    v = torch.randn(SEQLEN, BATCH, 1, HEAD_DIM, device=device)
    h = torch.randn(SEQLEN, BATCH, HIDDEN, device=device)
    return q, k, v, h


def _input_norm(device):
    torch.manual_seed(11)
    linear_qkv = SimpleNamespace(
        layer_norm_weight=torch.randn(HIDDEN, device=device),
        layer_norm_bias=None,
        eps=1.0e-5,
        skip_norm_and_all_gather=False,
    )
    norm_config = SimpleNamespace(
        normalization="RMSNorm", layernorm_epsilon=1.0e-5, layernorm_zero_centered_gamma=False
    )
    return _simplified_indexer_norm_spec(linear_qkv, norm_config)


def _launched_world_size() -> int:
    """Rank count torchrun was launched with.

    Read from the environment rather than torch.distributed: these guards run before
    Utils.initialize_model_parallel has created the default process group, and asking
    torch.distributed for a world size before then raises.
    """
    return int(os.environ.get("WORLD_SIZE", "1"))


class _UnitGroup:
    """Stands in for a tensor-parallel group of one."""

    def size(self):
        return 1


def _indexer(device, cp_group):
    """A simplified indexer whose weights are identical on every rank.

    ``pg_collection.cp`` is the live group: the learned-K projection reads it to all-gather K
    across the context-parallel ranks, which is what makes the cached K span the global sequence.
    """
    torch.manual_seed(3)
    linear_q = torch.nn.Linear(HIDDEN, HEAD_DIM, bias=False).to(device)
    linear_k = torch.nn.Linear(HIDDEN, HEAD_DIM, bias=False).to(device)
    return SimpleNamespace(
        index_n_heads=1,
        index_head_dim=HEAD_DIM,
        index_topk=TOPK,
        softmax_scale=HEAD_DIM**-0.5,
        index_rotary_dim=0,
        rotary_pos_emb=None,
        pg_collection=SimpleNamespace(tp=_UnitGroup(), cp=cp_group),
        config=SimpleNamespace(
            dsa_indexer_mode="simplified",
            dsa_simplified_use_learned_k=True,
            rotary_interleaved=False,
        ),
        linear_q=linear_q,
        linear_k=linear_k,
    )


def _run(query, key, value, hidden, indexer, q_chunk, query_positions=None, input_norm=None):
    """Call the kernel the way _forward_min_memory does once K and V are gathered."""
    return dsa_min_memory_gqa(
        query,
        key,
        value,
        hidden.detach(),
        indexer,
        HEAD_DIM**-0.5,
        0.1,
        False,
        query_chunk_size=q_chunk,
        key_chunk_size=SEQLEN,
        # The learned indexer K must be cached so it is projected once for the whole sequence and
        # all-gathered; the streamed projection would read local hidden states at global offsets.
        cache_indexer_k=True,
        use_triton=False,
        simplified_input_norm=input_norm,
        query_positions=query_positions,
    )


def _reference(device, q_chunk):
    """The single-rank result, recomputed identically on every rank."""
    q, k, v, h = _global_inputs(device)
    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)
    indexer = _indexer(device, cp_group=None)
    out, loss = _run(q, k, v, h, indexer, q_chunk, input_norm=_input_norm(device))
    grads = torch.autograd.grad(
        out.float().sum() + loss, (q, k, v, indexer.linear_k.weight, indexer.linear_q.weight)
    )
    return out.detach(), grads


def _sharded(device, cp_group, positions, q_chunk):
    """This rank's result, with the gather run over the live group."""
    q, k, v, h = _global_inputs(device)
    q_local = q[positions].detach().requires_grad_(True)
    k_local = k[positions].detach().requires_grad_(True)
    v_local = v[positions].detach().requires_grad_(True)
    h_local = h[positions]
    indexer = _indexer(device, cp_group)
    dsa_key, dsa_value, query_positions = _gather_kv_for_context_parallel(
        k_local, v_local, cp_group, "all_gather"
    )
    out, loss = _run(
        q_local,
        dsa_key,
        dsa_value,
        h_local,
        indexer,
        q_chunk,
        query_positions=query_positions,
        input_norm=_input_norm(device),
    )
    grads = torch.autograd.grad(
        out.float().sum() + loss,
        (q_local, k_local, v_local, indexer.linear_k.weight, indexer.linear_q.weight),
    )
    return out.detach(), grads


@pytest.mark.parametrize("cp_size", [2, 4])
class TestContextParallelParity:
    """cp>1 must reproduce the single-rank result, collectives included."""

    def _setup(self, cp_size):
        Utils.initialize_model_parallel(1, 1, context_parallel_size=cp_size)
        import megatron.core.parallel_state as ps

        cp_group = ps.get_context_parallel_group()
        device = torch.cuda.current_device()
        # A tile must not straddle a zigzag chunk boundary; one tile per chunk is always legal.
        q_chunk = SEQLEN // (2 * cp_size)
        positions = build_zigzag_cp_local_positions(
            SEQLEN, cp_size, cp_group.rank(), torch.device(device)
        )
        return cp_group, device, q_chunk, positions

    def test_the_gather_reconstructs_the_global_key_and_value(self, cp_size):
        """Isolates the collective from the kernel.

        If this fails, the all-gather or the zigzag reorder is wrong and every parity test
        below inherits it. If it passes, the gathered state is right and a downstream failure
        is the kernel's.
        """
        if _launched_world_size() < cp_size:
            pytest.skip(f"cp={cp_size} needs {cp_size} ranks")
        cp_group, device, _, positions = self._setup(cp_size)
        try:
            q, k, v, _ = _global_inputs(device)
            dsa_key, dsa_value, query_positions = _gather_kv_for_context_parallel(
                k[positions], v[positions], cp_group, "all_gather"
            )
            torch.testing.assert_close(dsa_key, k, atol=0, rtol=0)
            torch.testing.assert_close(dsa_value, v, atol=0, rtol=0)
            # The positions the helper reports must be the ones this rank actually holds.
            torch.testing.assert_close(query_positions, positions, atol=0, rtol=0)
        finally:
            Utils.destroy_model_parallel()

    def test_output_matches_the_single_rank_slice(self, cp_size):
        """A rank's output must equal the reference rows at the positions it holds."""
        if _launched_world_size() < cp_size:
            pytest.skip(f"cp={cp_size} needs {cp_size} ranks")
        cp_group, device, q_chunk, positions = self._setup(cp_size)
        try:
            ref_out, _ = _reference(device, q_chunk)
            out, _ = _sharded(device, cp_group, positions, q_chunk)
            torch.testing.assert_close(out, ref_out[positions], atol=1e-5, rtol=1e-5)
        finally:
            Utils.destroy_model_parallel()

    def test_query_gradient_matches_the_single_rank_slice(self, cp_size):
        """dQ is sharded by query, so a rank's dQ is the reference at its own positions."""
        if _launched_world_size() < cp_size:
            pytest.skip(f"cp={cp_size} needs {cp_size} ranks")
        cp_group, device, q_chunk, positions = self._setup(cp_size)
        try:
            _, ref_grads = _reference(device, q_chunk)
            _, grads = _sharded(device, cp_group, positions, q_chunk)
            torch.testing.assert_close(grads[0], ref_grads[0][positions], atol=1e-5, rtol=1e-5)
        finally:
            Utils.destroy_model_parallel()

    def test_key_and_value_gradients_are_reduce_scattered(self, cp_size):
        """dK and dV must arrive already summed across ranks, not split.

        Every rank's queries attend to every global key, so each contributes to every key's
        gradient. gather_from_sequence_parallel_region's backward reduce-scatters, which sums
        those contributions before scattering each key's gradient to its owner. If it split
        instead, a rank would keep only its own share -- finite, wrong, and silent. Gathering the
        per-rank slices back therefore has to reproduce the single-rank gradient exactly.
        """
        if _launched_world_size() < cp_size:
            pytest.skip(f"cp={cp_size} needs {cp_size} ranks")
        cp_group, device, q_chunk, positions = self._setup(cp_size)
        try:
            _, ref_grads = _reference(device, q_chunk)
            _, grads = _sharded(device, cp_group, positions, q_chunk)
            for local_grad, reference in ((grads[1], ref_grads[1]), (grads[2], ref_grads[2])):
                gathered = torch.zeros_like(reference)
                gathered[positions] = local_grad
                torch.distributed.all_reduce(gathered, group=cp_group)
                torch.testing.assert_close(gathered, reference, atol=1e-5, rtol=1e-5)
        finally:
            Utils.destroy_model_parallel()

    def test_learned_k_weight_gradient_sums_across_ranks(self, cp_size):
        """The learned-K weight gradient must sum to the single-rank gradient.

        Unlike dQ, this weight is a sum over *all* queries for *every* key, so a rank's queries
        scatter gradient into keys other ranks own. The learned-K all-gather runs under
        torch.no_grad, so its reduce-scatter backward never fires; the custom backward owes that
        reduction itself. Without it each rank keeps only the rows for keys it happens to own and
        the sum falls short, while raising nothing.
        """
        if _launched_world_size() < cp_size:
            pytest.skip(f"cp={cp_size} needs {cp_size} ranks")
        cp_group, device, q_chunk, positions = self._setup(cp_size)
        try:
            _, ref_grads = _reference(device, q_chunk)
            _, grads = _sharded(device, cp_group, positions, q_chunk)
            summed = grads[3].clone()
            torch.distributed.all_reduce(summed, group=cp_group)
            torch.testing.assert_close(summed, ref_grads[3], atol=1e-5, rtol=1e-5)
        finally:
            Utils.destroy_model_parallel()

    def test_query_projection_weight_gradient_sums_across_ranks(self, cp_size):
        """Control for the learned-K case: sharded by query, so it should agree either way.

        Holding this while the learned-K test fails localizes a defect to the key axis rather
        than to loss normalization or the CP scale, which would skew both equally.
        """
        if _launched_world_size() < cp_size:
            pytest.skip(f"cp={cp_size} needs {cp_size} ranks")
        cp_group, device, q_chunk, positions = self._setup(cp_size)
        try:
            _, ref_grads = _reference(device, q_chunk)
            _, grads = _sharded(device, cp_group, positions, q_chunk)
            summed = grads[4].clone()
            torch.distributed.all_reduce(summed, group=cp_group)
            torch.testing.assert_close(summed, ref_grads[4], atol=1e-5, rtol=1e-5)
        finally:
            Utils.destroy_model_parallel()
