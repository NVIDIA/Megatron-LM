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


def _reference(idx, vals, experts):
    """The combine tree in plain torch: leaves summed in expert order, then pairwise."""
    tokens, topk, hidden = vals.shape
    num_leaves = eic.num_leaf_blocks(experts)
    leaf_size = experts // num_leaves
    leaves = [torch.zeros(tokens, hidden, device="cuda") for _ in range(num_leaves)]
    for e in range(experts):
        hit = idx == e
        v = (vals * hit[..., None]).sum(1)  # at most one slot per (token, expert)
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


def _dense_slots(row_of_slot, expert_of_slot, num_experts):
    """[T, K] (row, expert) top-k slots -> [T, num_experts] row per expert (-1 = none)."""
    valid = (row_of_slot >= 0) & (expert_of_slot >= 0) & (expert_of_slot < num_experts)
    cols = torch.where(valid, expert_of_slot, num_experts).long()  # spare column, dropped
    dense = torch.full((row_of_slot.shape[0], num_experts + 1), -1, dtype=torch.int32)
    dense = dense.to(row_of_slot.device)
    dense.scatter_(1, cols, torch.where(valid, row_of_slot, -1).to(torch.int32))
    return dense[:, :num_experts].contiguous()


def _all(tokens):
    """valid_tokens covering every token."""
    return torch.tensor([tokens], dtype=torch.int32, device="cuda")


def _rank_partial(idx, vals, experts, ep, rank):
    """One EP rank's fp32 subtree partial, from rows in arbitrary order."""
    inv, (t, k) = _rank_inverse_map(idx, rank, experts // ep)
    rows = vals[t, k].to(torch.bfloat16).contiguous()
    return eic.tree_combine_forward(
        rows, inv, eic.num_leaf_blocks(experts) // ep, _all(idx.shape[0])
    )


@needs_cuda
@pytest.mark.parametrize("experts", [64, 96])
def test_kernel_matches_reference_tree(experts):
    tokens, topk, hidden = 40, 6, 1000
    idx, vals = _case(tokens, experts, topk, hidden, seed=0)
    rows, inverse_map = _training_inputs(idx, vals)
    slots = _dense_slots(inverse_map[0], inverse_map[1], experts)
    out = eic.tree_combine_forward(rows, slots, eic.num_leaf_blocks(experts), _all(tokens))
    assert torch.equal(_bits(out), _bits(_reference(idx, vals, experts)))
    assert not torch.signbit(out[1, 0])  # -0.0 rows add up to +0.0 at every EP


@needs_cuda
@pytest.mark.parametrize(
    "experts,topk,hidden,with_probs", [(96, 8, 700, True), (512, 22, 1536, False)]
)
def test_route_kernel_matches_dense_kernel(experts, topk, hidden, with_probs):
    """Training walks each token's sorted routes; it must give the dense kernel's bits,
    including slots that route nowhere (dropped or padded). With probs, the dense kernel
    gets the rows pre-multiplied in fp32 (torch's fp32 product is the kernel's mul_rn)."""
    tokens = 48
    idx, vals = _case(tokens, experts, topk, hidden, seed=4)
    rows, inverse_map = _training_inputs(idx, vals)
    # Top-k slot order, not expert order: the route list sorts it.
    shuffle = torch.rand(tokens, topk, device="cuda").argsort(dim=1)
    row_of_slot = inverse_map[0].gather(1, shuffle)
    expert_of_slot = inverse_map[1].gather(1, shuffle)
    row_of_slot[::5, 0] = -1  # unused slots
    expert_of_slot[1::7, -1] = -1
    probs = torch.rand(tokens, experts, device="cuda") if with_probs else None
    leaves = eic.num_leaf_blocks(experts)
    dense_rows = rows.float()
    if probs is not None:
        used = (row_of_slot >= 0) & (expert_of_slot >= 0)
        t, k = used.nonzero(as_tuple=True)
        row_prob = torch.zeros(rows.shape[0], device="cuda")
        row_prob[row_of_slot[t, k]] = probs[t, expert_of_slot[t, k]]
        dense_rows = dense_rows * row_prob[:, None]
    dense = _dense_slots(row_of_slot, expert_of_slot, experts)
    ref = eic.tree_combine_forward(dense_rows, dense, leaves, _all(tokens))
    cols, slot_rows, slot_probs = eic.sorted_routes(row_of_slot, expert_of_slot, experts, probs)
    out = eic.tree_combine_routes_forward(rows, cols, slot_rows, experts, leaves, slot_probs)
    assert torch.equal(_bits(out), _bits(ref))


@needs_cuda
def test_rank_partials_same_bits_at_every_ep():
    experts, tokens, topk, hidden = 96, 64, 6, 512
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
def test_training_unpermute_same_bits_at_every_ep(monkeypatch):
    experts, topk, hidden, tokens = 64, 6, 512, 96
    idx, vals = _case(tokens, experts, topk, hidden, seed=2)
    rows, inverse_map = _training_inputs(idx, vals)
    probs = None
    ref = _unpermute(rows, inverse_map, experts, 1, monkeypatch, True, probs)
    for ep in (1, 2, 4, 8):
        out = _unpermute(rows, inverse_map, experts, ep, monkeypatch, True, probs)
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), ep
        # Flag off: the existing per-rank combine, unchanged.
        assert torch.equal(
            _unpermute(rows, inverse_map, experts, ep, monkeypatch, None, probs),
            _unpermute(rows, inverse_map, experts, ep, monkeypatch, False, probs),
        )
    # bf16 outputs above; compare the fp32 values too.
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
    gen = torch.Generator(device="cuda").manual_seed(3)
    probs = torch.rand(tokens, experts, device="cuda", generator=gen)
    grad = torch.randn(tokens, hidden, device="cuda", generator=gen)
    outs = []
    outputs = []
    for on in (True, None, True):
        r = rows.clone().requires_grad_()
        p = probs.clone().requires_grad_()
        out = _unpermute(r, inverse_map, experts, 2, monkeypatch, on, p)
        outputs.append(out.detach().clone())
        outs.append(torch.autograd.grad(out, (r, p), grad))
    # The tree has the gradient of the plain sum: d_row = grad * prob exactly, and d_prob is
    # the fp32 dot <grad, row>, checked against fp64 relative to the sum of its |terms|.
    torch.testing.assert_close(outs[0][0], outs[1][0], rtol=1e-6, atol=1e-6)
    row_of_slot, expert_of_slot = inverse_map[0], inverse_map[1]
    terms = grad.double()[:, None, :] * rows.double()[row_of_slot]  # [T, K, H]
    exact = torch.zeros(tokens, experts, dtype=torch.float64, device="cuda")
    exact.scatter_(1, expert_of_slot, terms.sum(-1))
    bound = torch.zeros_like(exact).scatter_(1, expert_of_slot, terms.abs().sum(-1))
    assert ((outs[0][1].double() - exact).abs() <= 1e-6 * bound + 1e-30).all()
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
    for ep in (2, 4, 8):
        parts = [rank_out(ep, r, eic.num_leaf_blocks(experts) // ep)[:n] for r in range(ep)]
        assert torch.equal(_bits(_pairwise_tree_sum(parts)), _bits(ref[:n])), ep
    # Flag off: unchanged (fp64 sum rounded to the fp32 rank partial).
    assert rank_out(1, 0, None).dtype == torch.float32


@needs_cuda
def test_topk_layout_matches_dense_kernel():
    """vLLM's rows (one per top-k slot, slots in top-k order) give the dense kernel's rank
    partials bit for bit at every EP; slots on other ranks and padded tokens are ignored."""
    experts, topk, tokens, hidden = 96, 6, 37, 384  # leaves of 3; top-k padded to 8
    gen = torch.Generator(device="cuda").manual_seed(6)
    routing_map = torch.rand(tokens, experts, device="cuda", generator=gen).topk(topk, 1).indices
    vals = _rows(tokens * topk, hidden, gen).view(tokens, topk, hidden)
    valid = torch.tensor([tokens - 4], dtype=torch.int32, device="cuda")
    n = valid.item()
    for ep in (1, 2, 4, 8):
        local = experts // ep
        leaves = eic.num_leaf_blocks(experts) // ep
        for rank in range(ep):
            start = rank * local
            on_rank = (routing_map >= start) & (routing_map < start + local)
            # vLLM layout: row t * topk + k; rows of other ranks' slots hold garbage.
            rows = vals.to(torch.bfloat16).clone()
            rows[~on_rank] = float("nan")
            sentinel = torch.full((tokens, hidden), 7.0, device="cuda")
            out = eic.tree_combine_topk_forward(
                rows.view(tokens * topk, hidden),
                routing_map.int(),
                start,
                local,
                leaves,
                valid,
                sentinel,
            )
            # Torch layout: dense [T, local] map over the same rows.
            t, k = on_rank.nonzero(as_tuple=True)
            inv = torch.full((tokens, local), -1, dtype=torch.int32, device="cuda")
            inv[t, routing_map[t, k] - start] = (t * topk + k).int()
            ref = eic.tree_combine_forward(
                rows.view(tokens * topk, hidden), inv, leaves, _all(tokens)
            )
            assert torch.equal(_bits(out[:n]), _bits(ref[:n])), (ep, rank)
            assert (out[n:] == 7.0).all()  # padded tokens are not written


def test_config_validation():
    from megatron.core.transformer.transformer_config import TransformerConfig

    with pytest.raises(ValueError, match="batch_invariant_mode"):
        TransformerConfig(
            num_layers=1, hidden_size=64, num_attention_heads=4, moe_ep_invariant_combine=True
        )
    # The torch and vLLM grouped GEMMs hand the combine one row per route; FlashInfer sums
    # a token's routes inside its kernel, so it cannot run the tree.
    inference = dict(
        transformer_impl="inference_optimized",
        normalization="RMSNorm",
        inference_moe_token_dispatcher_type="nvls",
    )
    with pytest.raises(ValueError, match="'torch' or 'vllm'"):
        _ep_invariant_moe_config(
            inference_grouped_gemm_backend="flashinfer",
            inference_moe_nvfp4_recipe="per_token",
            activation_func=torch.nn.functional.silu,
            gated_linear_unit=True,
            **inference,
        )
    _ep_invariant_moe_config(inference_grouped_gemm_backend="vllm", **inference)


def _ep_invariant_moe_config(**overrides):
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import TransformerConfig

    kwargs = dict(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        num_moe_experts=8,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        moe_router_dtype="fp32",
        moe_token_dispatcher_type="alltoall",
        moe_ep_invariant_combine=True,
        batch_invariant_mode=True,
        attention_backend=AttnBackend.flash,
        flash_attention_version=4,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_config_accepts_expert_tensor_parallel_inference():
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import HAVE_DEEPGEMM_BF16

    if not HAVE_DEEPGEMM_BF16:
        pytest.skip("Building a batch-invariant MoE config requires DeepGEMM bf16 kernels.")
    _ep_invariant_moe_config(
        transformer_impl="inference_optimized",
        normalization="RMSNorm",
        inference_grouped_gemm_backend="torch",
        inference_moe_token_dispatcher_type="nvls",
        expert_tensor_parallel_size=2,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
    )


@pytest.mark.parametrize("exact", [False, True])
def test_fold_expert_tp_partials_is_a_rank_order_fold(exact):
    gen = torch.Generator(device="cuda").manual_seed(3)
    parts = (
        torch.randn(4, 300, 96, device="cuda", generator=gen)
        * torch.logspace(-6, 6, 96, device="cuda")
    ).bfloat16()
    acc = parts[0].double() if exact else parts[0].float()
    for r in range(1, 4):
        acc = acc + (parts[r].double() if exact else parts[r].float())
    ref = acc.float().bfloat16()
    out = eic.fold_expert_tp_partials(parts, exact=exact)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


@pytest.mark.parametrize("etp_outer", [False, True], ids=["ep-major", "etp-major"])
def test_inference_expert_tp_combine_matches_training(etp_outer):
    """Expert TP: inference folds each expert row over expert TP, expert-TP rank 0 evaluates
    its subtree and the partners add +0.0; the pairwise reduce-scatter over the tp_ep ranks
    then equals the training tree over expert-TP-reduced rows, for either tp_ep rank order.
    """
    experts, tokens, hidden, expert_tp, ep = 16, 96, 256, 4, 2
    leaves = eic.num_leaf_blocks(experts)
    gen = torch.Generator(device="cuda").manual_seed(7)
    scale = torch.logspace(-3, 3, hidden, device="cuda")
    partials = (
        torch.randn(expert_tp, tokens * experts, hidden, device="cuda", generator=gen) * scale
    ).bfloat16()
    rows = eic.fold_expert_tp_partials(partials)  # training's expert-TP reduction
    t = torch.arange(tokens, device="cuda").unsqueeze(1)

    def slots(lo, hi):
        return (t * experts + torch.arange(lo, hi, device="cuda")).to(torch.int32)

    training = eic.tree_combine_forward(rows, slots(0, experts), leaves, _all(tokens))

    per_rank = experts // ep
    rank_partials = {}
    for ep_rank in range(ep):
        for etp_rank in range(expert_tp):
            if etp_rank == 0:
                rank_partials[ep_rank, etp_rank] = eic.tree_combine_forward(
                    rows,
                    slots(ep_rank * per_rank, (ep_rank + 1) * per_rank),
                    leaves // ep,
                    _all(tokens),
                )
            else:
                rank_partials[ep_rank, etp_rank] = torch.zeros_like(training)
    if etp_outer:
        order = [(e, k) for k in range(expert_tp) for e in range(ep)]
    else:
        order = [(e, k) for e in range(ep) for k in range(expert_tp)]
    level = [rank_partials[key] for key in order]
    while len(level) > 1:  # the reduce-scatter's pairwise tree in tp_ep rank order
        level = [level[i] + level[i + 1] for i in range(0, len(level), 2)]
    assert torch.equal(level[0].view(torch.int32), training.view(torch.int32))


@pytest.mark.skipif(Utils.world_size < 4, reason="needs >= 4 GPUs")
def test_expert_tp_fold_matches_training_reduction():
    """expert_tp_fold on a real expert-TP group: a rank-order fold, and at expert TP 2 the
    same bits as training's stock bf16 reduce-scatter."""
    Utils.initialize_model_parallel(
        1, 1, expert_model_parallel_size=Utils.world_size // 2, expert_tensor_parallel_size=2
    )
    try:
        group = parallel_state.get_expert_tensor_parallel_group()
        assert group.size() == 2
        gen = torch.Generator(device="cuda").manual_seed(11 + torch.distributed.get_rank())
        rows = (
            torch.randn(1024, 512, device="cuda", generator=gen)
            * torch.logspace(-4, 4, 512, device="cuda")
        ).bfloat16()
        folded = eic.expert_tp_fold(rows, group)
        parts = torch.empty((2, *rows.shape), dtype=rows.dtype, device="cuda")
        torch.distributed.all_gather_into_tensor(parts, rows, group=group)
        assert torch.equal(
            folded.view(torch.int16), eic.fold_expert_tp_partials(parts).view(torch.int16)
        )
        shard = torch.empty(512, 512, dtype=rows.dtype, device="cuda")
        torch.distributed.reduce_scatter_tensor(shard, rows, group=group)
        mine = folded[group.rank() * 512 : (group.rank() + 1) * 512]
        assert torch.equal(mine.view(torch.int16), shard.view(torch.int16))
    finally:
        Utils.destroy_model_parallel()


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
        slots = _dense_slots(
            torch.arange(tokens * topk, device="cuda").view(tokens, topk), idx, experts
        )
        ref = eic.tree_combine_forward(rows, slots, eic.num_leaf_blocks(experts), _all(tokens)).to(
            torch.bfloat16
        )
        assert torch.equal(out.view(-1, hidden).view(torch.int16), ref[mine].view(torch.int16))
