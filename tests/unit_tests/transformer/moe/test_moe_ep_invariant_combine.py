# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""moe_ep_invariant_combine: MoE combine results do not depend on the EP size.

Single-GPU tests emulate EP by splitting the experts over ranks; the distributed
tests run the real EP collectives at EP = world size.
"""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.transformer.moe import batch_invariant as moe_bi
from megatron.core.transformer.moe import ep_invariant_combine as eic
from tests.unit_tests.test_utilities import Utils

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _pairwise_tree_sum(parts):
    """Reference pairwise tree over a power-of-two number of tensors: (p0 + p1) + (p2 + p3) ..."""
    assert parts and len(parts) & (len(parts) - 1) == 0, "expected a power-of-two count"
    while len(parts) > 1:
        parts = [parts[i] + parts[i + 1] for i in range(0, len(parts), 2)]
    return parts[0]


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
    vals[1, :, 0] = -0.0
    return idx, vals


def _reference(idx, vals, experts, probs=None):
    """The combine tree in plain torch: leaves summed in expert order, then pairwise."""
    tokens, topk, hidden = vals.shape
    num_leaves = eic.num_leaf_blocks(experts)
    leaf_size = experts // num_leaves
    leaves = [torch.zeros(tokens, hidden, device="cuda") for _ in range(num_leaves)]
    for e in range(experts):
        hit = idx == e
        v = (vals * hit[..., None]).sum(1)  # at most one slot per (token, expert)
        if probs is not None:
            v = v * probs[:, e : e + 1]
        leaves[e // leaf_size] = leaves[e // leaf_size] + torch.where(
            hit.any(1, keepdim=True), v, 0.0
        )
    return _pairwise_tree_sum(leaves)


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


def _rank_inverse_map(idx, rank, experts_per_rank):
    """[T, E_local] permuted row of each (token, local expert) on one EP rank, -1 if none."""
    tokens = idx.shape[0]
    local = idx - rank * experts_per_rank
    hit = (local >= 0) & (local < experts_per_rank)
    inv = torch.full((tokens, experts_per_rank), -1, dtype=torch.int32, device="cuda")
    t, k = hit.nonzero(as_tuple=True)
    inv[t, local[t, k]] = torch.arange(t.numel(), dtype=torch.int32, device="cuda")
    return inv, (t, k)


def _rank_partial(idx, vals, experts, ep, rank):
    """One EP rank's fp32 subtree partial, from rows in arbitrary order."""
    inv, (t, k) = _rank_inverse_map(idx, rank, experts // ep)
    rows = vals[t, k].to(torch.bfloat16).contiguous()
    return eic.tree_combine_forward(rows, inv, eic.num_leaf_blocks(experts) // ep)


@needs_cuda
@pytest.mark.parametrize("experts", [64, 96, 8])
@pytest.mark.parametrize("with_probs", [False, True])
def test_kernel_matches_reference_tree(experts, with_probs):
    tokens, topk, hidden = 40, min(6, experts), 1000
    idx, vals = _case(tokens, experts, topk, hidden, seed=0)
    rows, inverse_map = _training_inputs(idx, vals)
    probs = torch.rand(tokens, experts, device="cuda") if with_probs else None
    slots = eic.dense_slots(inverse_map[0], inverse_map[1], experts)
    out = eic.tree_combine_forward(rows, slots, eic.num_leaf_blocks(experts), probs)
    assert torch.equal(_bits(out), _bits(_reference(idx, vals, experts, probs)))
    assert not torch.signbit(out[1, 0])  # -0.0 rows add up to +0.0 at every EP


@needs_cuda
@pytest.mark.parametrize("experts", [64, 96])
def test_rank_partials_same_bits_at_every_ep(experts):
    tokens, topk, hidden = 64, 6, 512
    idx, vals = _case(tokens, experts, topk, hidden, seed=1)
    ref = _reference(idx, vals, experts)
    for ep in (1, 2, 4, 8, 16, 32):
        if experts % ep:
            continue
        parts = [_rank_partial(idx, vals, experts, ep, r) for r in range(ep)]
        assert torch.equal(_bits(_pairwise_tree_sum(parts)), _bits(ref)), ep


def test_check_ep_size():
    eic.check_ep_size(96, 32)
    eic.check_ep_size(128, 1)
    with pytest.raises(ValueError):
        eic.check_ep_size(96, 3)  # not a power of two
    with pytest.raises(ValueError):
        eic.check_ep_size(96, 64)  # does not divide the experts


@needs_cuda
@pytest.mark.parametrize("with_probs", [False, True])
def test_training_unpermute_same_bits_at_every_ep(monkeypatch, with_probs):
    experts, topk, hidden, tokens = 64, 6, 512, 96
    idx, vals = _case(tokens, experts, topk, hidden, seed=2)
    rows, inverse_map = _training_inputs(idx, vals)
    probs = torch.rand(tokens, experts, device="cuda") if with_probs else None
    ref = _unpermute(rows, inverse_map, experts, 1, monkeypatch, True, probs)
    for ep in (1, 2, 4, 8):
        out = _unpermute(rows, inverse_map, experts, ep, monkeypatch, True, probs)
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), ep
        # Flag off: the existing per-rank combine, unchanged.
        assert torch.equal(
            _unpermute(rows, inverse_map, experts, ep, monkeypatch, None, probs),
            _unpermute(rows, inverse_map, experts, ep, monkeypatch, False, probs),
        )
    if not with_probs:  # (bf16 outputs; the fp32 values are compared below)
        f = lambda on: moe_bi.unpermute(  # noqa: E731
            rows.float(),
            torch.Size((tokens, hidden)),
            probs=None,
            num_experts=experts,
            inverse_map=inverse_map,
            ep_invariant=on,
        )
        monkeypatch.setattr(parallel_state, "get_expert_model_parallel_world_size", lambda: 4)
        old4 = f(False)
        monkeypatch.setattr(parallel_state, "get_expert_model_parallel_world_size", lambda: 1)
        assert not torch.equal(old4, f(False))  # the old combine depends on EP
        assert torch.equal(_bits(f(True)), _bits(_reference(idx, vals, experts)))


@needs_cuda
def test_training_unpermute_backward(monkeypatch):
    experts, topk, hidden, tokens = 32, 4, 128, 40
    idx, vals = _case(tokens, experts, topk, hidden, seed=3)
    rows, inverse_map = _training_inputs(idx, vals)
    rows = rows.float()
    probs = torch.rand(tokens, experts, device="cuda")
    grad = torch.randn(tokens, hidden, device="cuda")
    outs = []
    outputs = []
    for on in (True, None, True):
        r = rows.clone().requires_grad_()
        p = probs.clone().requires_grad_()
        out = _unpermute(r, inverse_map, experts, 2, monkeypatch, on, p)
        outputs.append(out.detach().clone())
        outs.append(torch.autograd.grad(out, (r, p), grad))
    # The tree has the gradient of the plain sum.
    torch.testing.assert_close(outs[0][0], outs[1][0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(outs[0][1], outs[1][1], rtol=1e-5, atol=1e-5)
    # Replaying the new tree must preserve every bit of the output and both gradients.
    assert torch.equal(_bits(outputs[0]), _bits(outputs[2]))
    for first, replay in zip(outs[0], outs[2]):
        assert torch.equal(_bits(first), _bits(replay))


@needs_cuda
def test_inference_unpermute_same_bits_at_every_ep(monkeypatch):
    from megatron.core.inference.moe import batch_invariant as inf_bi

    experts, topk, hidden, tokens = 64, 6, 512, 80
    idx, vals = _case(tokens, experts, topk, hidden, seed=4)
    valid = torch.tensor([tokens - 3], dtype=torch.int32, device="cuda")
    rows, inverse_map = _training_inputs(idx, vals)
    train = _unpermute(rows.float(), inverse_map, experts, 2, monkeypatch, True)

    def rank_out(ep, rank, leaves):
        inv, (t, k) = _rank_inverse_map(idx, rank, experts // ep)
        y = vals[t, k].to(torch.bfloat16).contiguous()
        return inf_bi.unpermute_tokens_in_expert_order(y, inv, valid, None, leaves)

    ref = rank_out(1, 0, eic.num_leaf_blocks(experts))
    n = valid.item()
    assert torch.equal(_bits(ref[:n]), _bits(train[:n]))  # engine == learner
    assert not ref[n:].any()  # rows past valid_tokens are zero
    for ep in (2, 4, 8):
        parts = [rank_out(ep, r, eic.num_leaf_blocks(experts) // ep) for r in range(ep)]
        assert torch.equal(_bits(_pairwise_tree_sum(parts)), _bits(ref)), ep
    # Flag off: unchanged (fp64 sum rounded to the fp32 rank partial).
    assert rank_out(1, 0, None).dtype == torch.float32


def test_config_validation():
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
    def test_alltoall_dispatcher_combine_equals_ep1(self):
        """Real AllToAll dispatch + combine at EP = world size give the single-rank bits."""
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
            set_batch_invariant_mode,
        )
        from megatron.core.transformer.enums import AttnBackend
        from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
        from megatron.core.transformer.transformer_config import TransformerConfig

        ep, rank = Utils.world_size, torch.distributed.get_rank()
        experts, topk, hidden, tokens = 8 * ep, 6, 256, 32 * ep
        config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden,
            num_attention_heads=4,
            num_moe_experts=experts,
            moe_router_topk=topk,
            moe_token_dispatcher_type="alltoall",
            expert_model_parallel_size=ep,
            params_dtype=torch.bfloat16,
            bf16=True,
            attention_backend=AttnBackend.flash,
            flash_attention_version=4,
            attention_dropout=0.0,
            batch_invariant_mode=True,
            moe_ep_invariant_combine=True,
        )
        n_local = experts // ep
        disp = MoEAlltoAllTokenDispatcher(
            n_local,
            list(range(rank * n_local, (rank + 1) * n_local)),
            config,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        gen = torch.Generator(device="cuda").manual_seed(7)
        x = _rows(tokens, hidden, gen).to(torch.bfloat16)
        idx = torch.randn(tokens, experts, device="cuda", generator=gen).topk(topk, 1).indices
        probs = torch.zeros(tokens, experts, device="cuda")
        probs.scatter_(1, idx, torch.rand(tokens, topk, device="cuda", generator=gen))
        scale = 1 + torch.arange(experts, device="cuda") / 7  # per-expert "weights"

        def expert(rows, e, p):  # a deterministic stand-in for the experts' bf16 output
            return (rows.float() * scale[e, None] * p[:, None]).to(torch.bfloat16)

        n = tokens // ep
        mine = slice(rank * n, (rank + 1) * n)
        with set_batch_invariant_mode(True):
            h, p = disp.dispatch_preprocess(x[mine], probs[mine] > 0, probs[mine])
            h, p = disp.token_dispatch(h, p)
            h, per_expert, p = disp.dispatch_postprocess(h, p)
            local_e = torch.repeat_interleave(
                torch.arange(n_local, device="cuda"), per_expert.to("cuda"), output_size=h.shape[0]
            )
            y = expert(h, rank * n_local + local_e, p)
            out = disp.combine_postprocess(disp.token_combine(disp.combine_preprocess(y)))

        # Single rank: every (token, expert) row, summed over the whole tree.
        t = torch.arange(tokens, device="cuda").repeat_interleave(topk)
        e = idx.flatten()
        rows = expert(x[t], e, probs[t, e])
        slots = eic.dense_slots(
            torch.arange(tokens * topk, device="cuda").view(tokens, topk), idx, experts
        )
        ref = eic.tree_combine_forward(rows, slots, eic.num_leaf_blocks(experts)).to(torch.bfloat16)
        assert torch.equal(out.view(-1, hidden).view(torch.int16), ref[mine].view(torch.int16))

    @needs_cuda
    @pytest.mark.skipif(Utils.world_size < 2, reason="needs >= 2 GPUs")
    def test_nvls_pairwise_reduce_scatter_equals_ep1(self):
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
        counts = [tokens // ep - 1] * (ep - 1) + [tokens - (tokens // ep - 1) * (ep - 1)]
        buf = SymmetricMemoryManager.get_buffer(
            "test_ep_inv", process_group=group, size_mb=-(-tokens * hidden * 4 // (1 << 20))
        )
        rsv = buf.maybe_get_tensor([tokens, hidden], dtype=torch.float32)
        inv, (t, k) = _rank_inverse_map(idx, rank, experts // ep)
        valid = torch.tensor([tokens], dtype=torch.int32, device="cuda")
        y = vals[t, k].to(torch.bfloat16).contiguous()
        inf_bi.unpermute_tokens_in_expert_order(
            y, inv, valid, rsv["tensor"], eic.num_leaf_blocks(experts) // ep
        )
        out = torch.empty(counts[rank], hidden, dtype=torch.float32, device="cuda")
        inf_bi.ordered_reduce_scatter_v(
            out,
            rsv["tensor"],
            rsv["handle"],
            rank_token_offset=torch.tensor([sum(counts[:rank])], dtype=torch.int32, device="cuda"),
            ep_max_tokens=torch.tensor([max(counts)], dtype=torch.int32, device="cuda"),
            per_rank_max_tokens=max(counts),
            pairwise_tree=True,
        )
        torch.cuda.synchronize()
        torch.distributed.barrier()
        ref = _reference(idx, vals, experts)
        off = sum(counts[:rank])
        assert torch.equal(_bits(out), _bits(ref[off : off + counts[rank]]))
        SymmetricMemoryManager.destroy("test_ep_inv")
