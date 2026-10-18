# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""moe_ep_invariant_combine: MoE combine results do not depend on the EP size.

Single-GPU tests emulate EP by splitting the experts over ranks; the distributed
tests run the real EP collectives at EP = world size.
"""

from fractions import Fraction

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.transformer.moe import batch_invariant as moe_bi
from megatron.core.transformer.moe import ep_invariant_combine as eic
from tests.unit_tests.test_utilities import Utils

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _bits(t):
    return t.contiguous().view(torch.int32)


def _rows(n, h, gen, spread=24):
    """bf16 values over a wide exponent range, as fp32."""
    v = torch.randn(n, h, device="cuda", generator=gen)
    s = torch.exp2(torch.randint(-spread, 12, (n, h), device="cuda", generator=gen).float())
    return (v * s).to(torch.bfloat16).float()


def _case(tokens, experts, topk, hidden, seed):
    """Routing (top-k slots in ascending expert order) and one bf16 row per (token, slot)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.randn(tokens, experts, device="cuda", generator=gen)
    idx = scores.topk(topk, dim=1).indices.sort(dim=1).values
    vals = _rows(tokens * topk, hidden, gen).view(tokens, topk, hidden)
    vals[:: max(1, tokens // 5), 1] = -vals[:: max(1, tokens // 5), 0]  # exact cancellations
    return idx, vals


def _training_inputs(idx, vals):
    """Permuted rows + Megatron's [2, T, K] (row, expert) batch-invariant inverse map."""
    tokens, topk, hidden = vals.shape
    order = torch.argsort(
        idx.flatten() * tokens + torch.arange(tokens, device="cuda").repeat_interleave(topk)
    )
    rows = vals.reshape(-1, hidden)[order].to(torch.bfloat16)
    row_of_slot = torch.empty(tokens * topk, dtype=torch.long, device="cuda")
    row_of_slot[order] = torch.arange(tokens * topk, device="cuda")
    inverse_map = torch.stack((row_of_slot.view(tokens, topk), idx.long()))
    return rows, inverse_map


def _unpermute(rows, inverse_map, num_experts, ep, monkeypatch, ep_invariant, probs=None):
    monkeypatch.setattr(parallel_state, "get_expert_model_parallel_world_size", lambda: ep)
    kwargs = {} if ep_invariant is None else {"ep_invariant": ep_invariant}
    return moe_bi.unpermute(
        rows,
        torch.Size((inverse_map.shape[1], rows.shape[1])),
        probs=probs,
        num_experts=num_experts,
        inverse_map=inverse_map,
        **kwargs,
    )


@needs_cuda
def test_limbs_exact_and_correctly_rounded():
    gen = torch.Generator(device="cuda").manual_seed(0)
    k, h = 6, 256
    v = _rows(k, h, gen)
    v[:, 0] = -0.0
    limbs = eic.to_limbs(v)
    hi, lo = limbs.chunk(2, dim=-1)
    assert torch.equal(hi, hi.floor()) and torch.equal(lo, lo.round())
    assert torch.equal(hi + lo * 2.0**-48, v.double())  # bf16 values >= 2^-41 are exact
    out = eic.finalize(limbs.sum(0, keepdim=True))[0].cpu()
    vals = v.double().cpu().tolist()
    inf = torch.tensor(float("inf"))
    for j in range(h):  # nearest fp32 to the exact rational sum
        exact = sum(Fraction(vals[i][j]) for i in range(k))
        d = lambda x: abs(Fraction(float(x)) - exact)  # noqa: E731
        assert d(out[j]) <= d(torch.nextafter(out[j], -inf)), j
        assert d(out[j]) <= d(torch.nextafter(out[j], inf)), j
    assert not torch.signbit(out[0])  # an exact zero sum is +0.0


@needs_cuda
def test_any_grouping_gives_the_same_bits():
    gen = torch.Generator(device="cuda").manual_seed(1)
    t, k, h = 32, 8, 512
    limbs = eic.to_limbs(_rows(t * k, h, gen)).view(t, k, 2 * h)
    ref = eic.finalize(limbs.sum(1))
    for cut in range(1, k):
        perm = torch.randperm(k, generator=torch.Generator().manual_seed(cut))
        a, b = limbs[:, perm[:cut]].sum(1), limbs[:, perm[cut:]].sum(1)
        assert torch.equal(_bits(eic.finalize(a + b)), _bits(ref))
        assert torch.equal(_bits(eic.finalize(b + a)), _bits(ref))


def test_check_topk():
    eic.check_topk(32)
    with pytest.raises(ValueError):
        eic.check_topk(33)


@needs_cuda
@pytest.mark.parametrize("with_probs", [False, True])
def test_training_unpermute_same_bits_at_every_ep(monkeypatch, with_probs):
    experts, topk, hidden, tokens = 64, 6, 512, 96
    idx, vals = _case(tokens, experts, topk, hidden, seed=2)
    rows, inverse_map = _training_inputs(idx, vals)
    probs = None
    if with_probs:
        probs = torch.rand(tokens, experts, device="cuda")
    ref = _unpermute(rows, inverse_map, experts, 1, monkeypatch, True, probs)
    old = {}
    for ep in (1, 2, 4, 8):
        out = _unpermute(rows, inverse_map, experts, ep, monkeypatch, True, probs)
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), ep
        old[ep] = _unpermute(rows, inverse_map, experts, ep, monkeypatch, None, probs)
        # Flag off: the existing per-rank tree, unchanged.
        assert torch.equal(
            old[ep], _unpermute(rows, inverse_map, experts, ep, monkeypatch, False, probs)
        )
    if not with_probs:  # (bf16 outputs; the fp32 values are compared below)
        f = lambda ep, on: moe_bi.unpermute(  # noqa: E731
            rows.float(),
            torch.Size((tokens, hidden)),
            probs=None,
            num_experts=experts,
            inverse_map=inverse_map,
            ep_invariant=on,
        )
        monkeypatch.setattr(parallel_state, "get_expert_model_parallel_world_size", lambda: 4)
        old4 = f(4, False)
        monkeypatch.setattr(parallel_state, "get_expert_model_parallel_world_size", lambda: 1)
        assert not torch.equal(old4, f(1, False))  # the old combine depends on EP
        assert torch.equal(_bits(f(1, True)), _bits(eic.finalize(eic.to_limbs(vals).sum(1))))


@needs_cuda
def test_training_unpermute_backward(monkeypatch):
    experts, topk, hidden, tokens = 32, 4, 128, 40
    idx, vals = _case(tokens, experts, topk, hidden, seed=3)
    rows, inverse_map = _training_inputs(idx, vals)
    rows = rows.float()
    probs = torch.rand(tokens, experts, device="cuda")
    grad = torch.randn(tokens, hidden, device="cuda")
    outs = []
    for on in (True, None):
        r = rows.clone().requires_grad_()
        p = probs.clone().requires_grad_()
        out = _unpermute(r, inverse_map, experts, 2, monkeypatch, on, p)
        outs.append(torch.autograd.grad(out, (r, p), grad))
    # The limb sum has the gradient of the plain sum.
    torch.testing.assert_close(outs[0][0], outs[1][0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(outs[0][1], outs[1][1], rtol=1e-5, atol=1e-5)


def _rank_inverse_map(idx, rank, experts_per_rank):
    """[T, E_local] permuted row of each (token, local expert) on one EP rank, -1 if none."""
    tokens = idx.shape[0]
    local = idx - rank * experts_per_rank
    hit = (local >= 0) & (local < experts_per_rank)
    inv = torch.full((tokens, experts_per_rank), -1, dtype=torch.int32, device="cuda")
    t, k = hit.nonzero(as_tuple=True)
    inv[t, local[t, k]] = torch.arange(t.numel(), dtype=torch.int32, device="cuda")
    return inv, (t, k)


@needs_cuda
def test_inference_unpermute_same_bits_at_every_ep(monkeypatch):
    from megatron.core.inference.moe import batch_invariant as inf_bi

    experts, topk, hidden, tokens = 64, 6, 512, 80
    idx, vals = _case(tokens, experts, topk, hidden, seed=4)
    valid = torch.tensor([tokens], dtype=torch.int32, device="cuda")
    rows, inverse_map = _training_inputs(idx, vals)
    train = _unpermute(rows.float(), inverse_map, experts, 2, monkeypatch, True)

    def rank_out(ep, rank, limbs):
        inv, (t, k) = _rank_inverse_map(idx, rank, experts // ep)
        y = vals[t, k].to(torch.bfloat16).contiguous()
        return inf_bi.unpermute_tokens_in_expert_order(y, inv, valid, None, limbs)

    ref = rank_out(1, 0, 2)
    assert torch.equal(_bits(ref), _bits(train))  # engine == learner
    for ep in (2, 4, 8):
        parts = [rank_out(ep, r, 1) for r in range(ep)]
        for order in (parts, parts[::-1]):
            acc = torch.zeros_like(parts[0])
            for p in order:
                acc = acc + p
            assert torch.equal(_bits(eic.finalize(acc)), _bits(ref)), ep
    # Flag off: unchanged (fp64 sum rounded to the fp32 rank partial).
    assert rank_out(1, 0, 0).dtype == torch.float32


@needs_cuda
def test_allgather_limb_partial_same_bits_at_every_ep():
    experts, topk, hidden, tokens = 32, 4, 256, 64
    idx, vals = _case(tokens, experts, topk, hidden, seed=5)
    ref = eic.finalize(eic.to_limbs(vals).sum(1))
    for ep in (1, 2, 4):
        acc = None
        for rank in range(ep):
            _, (t, k) = _rank_inverse_map(idx, rank, experts // ep)
            part = eic.limb_partial(vals[t, k].to(torch.bfloat16), t, tokens)
            acc = part if acc is None else acc + part
        assert torch.equal(_bits(eic.finalize_with_grad(acc)), _bits(ref)), ep
    # Backward: each row gets its token's output gradient.
    _, (t, k) = _rank_inverse_map(idx, 0, experts)
    r = vals[t, k].clone().requires_grad_()
    g = torch.randn(tokens, hidden, device="cuda")
    (dr,) = torch.autograd.grad(eic.finalize_with_grad(eic.limb_partial(r, t, tokens)), r, g)
    assert torch.equal(dr, g[t])


def test_config_requires_batch_invariant_mode():
    from megatron.core.transformer.transformer_config import TransformerConfig

    with pytest.raises(ValueError, match="batch_invariant_mode"):
        TransformerConfig(
            num_layers=1, hidden_size=64, num_attention_heads=4, moe_ep_invariant_combine=True
        )


class TestEPInvariantCollectives:
    """Real EP collectives at EP = world size vs the EP-1 result on one rank."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=Utils.world_size)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _case(self):
        experts = 8 * Utils.world_size
        idx, vals = _case(48 * Utils.world_size, experts, 6, 512, seed=11)
        return experts, idx, vals

    @needs_cuda
    @pytest.mark.skipif(Utils.world_size < 2, reason="needs >= 2 GPUs")
    def test_reduce_scatter_limbs_equals_ep1(self):
        from megatron.core.tensor_parallel import reduce_scatter_to_sequence_parallel_region

        ep, rank = Utils.world_size, torch.distributed.get_rank()
        group = parallel_state.get_expert_model_parallel_group()
        experts, idx, vals = self._case()
        tokens = idx.shape[0]
        _, (t, k) = _rank_inverse_map(idx, rank, experts // ep)
        part = eic.limb_partial(vals[t, k].to(torch.bfloat16), t, tokens)
        out = eic.finalize_with_grad(reduce_scatter_to_sequence_parallel_region(part, group=group))
        ref = eic.finalize(eic.to_limbs(vals).sum(1))
        n = tokens // ep
        assert torch.equal(_bits(out), _bits(ref[rank * n : (rank + 1) * n]))

    @needs_cuda
    @pytest.mark.skipif(Utils.world_size < 2, reason="needs >= 2 GPUs")
    def test_nvls_limbs_reduce_scatter_equals_ep1(self):
        from megatron.core.inference.communication.torch_symm_triton.utils import (
            is_device_nvls_capable,
        )
        from megatron.core.inference.moe import batch_invariant as inf_bi
        from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

        if not is_device_nvls_capable(torch.device("cuda")):
            pytest.skip("needs NVLS")
        ep, rank = Utils.world_size, torch.distributed.get_rank()
        group = parallel_state.get_expert_model_parallel_group()
        experts, idx, vals = self._case()
        tokens, hidden = idx.shape[0], vals.shape[-1]
        counts = [tokens // ep] * ep
        nbytes = tokens * 2 * hidden * 8
        buf = SymmetricMemoryManager.get_buffer(
            "test_ep_inv", process_group=group, size_mb=-(-nbytes // (1 << 20))
        )
        limbs = buf.symm_buffer[:nbytes].view(torch.float64).view(tokens, 2 * hidden)
        inv, (t, k) = _rank_inverse_map(idx, rank, experts // ep)
        valid = torch.tensor([tokens], dtype=torch.int32, device="cuda")
        y = vals[t, k].to(torch.bfloat16).contiguous()
        inf_bi.unpermute_tokens_in_expert_order(y, inv, valid, limbs, 1)
        out = torch.empty(counts[rank], hidden, dtype=torch.float32, device="cuda")
        inf_bi.limbs_reduce_scatter_v(
            out,
            buf.symm_mem_hdl,
            rank_token_offset=torch.tensor([sum(counts[:rank])], dtype=torch.int32, device="cuda"),
            ep_max_tokens=torch.tensor([max(counts)], dtype=torch.int32, device="cuda"),
            per_rank_max_tokens=max(counts),
        )
        torch.cuda.synchronize()
        torch.distributed.barrier()
        ref = eic.finalize(eic.to_limbs(vals).sum(1))
        off = sum(counts[:rank])
        assert torch.equal(_bits(out), _bits(ref[off : off + counts[rank]]))
        SymmetricMemoryManager.destroy("test_ep_inv")
