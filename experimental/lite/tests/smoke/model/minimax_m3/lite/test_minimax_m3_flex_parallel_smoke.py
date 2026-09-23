# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax-M3 lite (flex MSA backend) under TP / EP / PP / CP vs the single-rank model on the same HF-format weights, bf16, 2 GPUs.

Gates: per-layer rel-to-max, loss rel, top-k flip rate, gradient cosines, bitwise weight export round-trip.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.gpus(2), pytest.mark.env(CUDA_DEVICE_MAX_CONNECTIONS="1")]

B, S = 2, 1024  # 8 KV blocks > top-4 -> the sparse layers are truly sparse; S % (2*tp*cp) == 0
LOSS_REL, LAYER_REL, TOPK_FLIP_BUDGET = 1e-2, 1e-1, 0.2
DENSE_GRAD_COS, EXPERT_GRAD_COS = 0.99, 0.95


def _train_config(ps):
    return SimpleNamespace(
        tp=ps.tp_size, ep=ps.ep_size, etp=ps.etp_size, pp=ps.pp_size, cp=ps.cp_size, vpp=None,
        moe_dispatcher="alltoall", fp8=False, recompute_modules=[], deterministic=True,
    )


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


class _LayerDump:
    def __init__(self, model):
        self.out, self.topk, self._hooks = {}, {}, []
        for local_i, layer in enumerate(model.layers):
            self._hooks.append(layer.register_forward_hook(self._make(model.layer_indices[local_i], layer)))

    def _make(self, gi, layer):
        def hook(_m, _i, out):
            self.out[gi] = out.detach()
            if layer.is_sparse_attention:
                self.topk[gi] = layer.attn.last_block_indices.detach()

        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()


def _gather_seq(x, ps, dist):
    """SP-sharded [S/tp, B, H] -> [S, B, H]."""
    if ps.tp_size <= 1:
        return x
    parts = [torch.empty_like(x) for _ in range(ps.tp_size)]
    dist.all_gather(parts, x.contiguous(), group=ps.tp_group)
    return torch.cat(parts, dim=0)


def _reference_topk_slice(ref_idx, attn):
    """Reference [B, H_idx, S, K] -> the index heads this TP rank holds."""
    idxr = attn.indexer
    if idxr._replicate_heads:
        h = idxr.ps.tp_rank // (idxr.ps.tp_size // idxr.num_heads)
        return ref_idx[:, h : h + 1]
    hl = idxr.num_heads_local
    return ref_idx[:, idxr.ps.tp_rank * hl : (idxr.ps.tp_rank + 1) * hl]


def _run_pipeline_fwd_bwd(model, ps, cfg, ids, labels, dist):
    """Two-stage pipeline with a single microbatch: p2p hidden states and their grads."""
    if ps.pp_is_first:
        h = model(input_ids=ids)["hidden_states"]
        dist.send(h.contiguous(), ps.pp_next_rank)
        grad = torch.empty_like(h)
        dist.recv(grad, ps.pp_next_rank)
        h.backward(grad)
        return None
    h = torch.empty(S // ps.cp_size // ps.tp_size, B, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    dist.recv(h, ps.pp_prev_rank)
    h.requires_grad_(True)
    out = model(hidden_states=h, labels=labels)
    out["loss"].backward()
    dist.send(h.grad.contiguous(), ps.pp_prev_rank)
    return out["loss"]


@pytest.fixture(scope="module")
def reference(flex_cfg, flex_source, dist, grad_tools):
    """Single-rank bf16 forward/backward: layer outputs, top-k selections, loss, gradients, exported weights."""
    from megatron.lite.model.minimax_m3.lite.checkpoint import load_hf_weights
    from megatron.lite.model.minimax_m3.lite.model import MiniMaxM3Model
    from megatron.lite.primitive.parallel import ParallelState

    ps0 = ParallelState()
    torch.manual_seed(7)
    ref = MiniMaxM3Model(flex_cfg, _train_config(ps0), ps0, msa_backend="flex").to(torch.bfloat16).cuda()
    load_hf_weights(ref, flex_source, flex_cfg, ps0)
    torch.manual_seed(1234)
    ids = torch.randint(0, flex_cfg.vocab_size, (B, S), device="cuda")
    labels = torch.randint(0, flex_cfg.vocab_size, (B, S), device="cuda")
    dump = _LayerDump(ref)
    out = ref(input_ids=ids, labels=labels)
    dump.remove()
    out["loss"].backward()
    result = dict(
        ids=ids, labels=labels, loss=out["loss"].detach(), layer_out=dump.out, topk=dump.topk,
        grads=grad_tools.grads_by_hf_name(ref, flex_cfg, ps0, dist), weights=grad_tools.weights_by_hf_name(ref, flex_cfg, ps0),
    )
    del ref
    torch.cuda.empty_cache()
    return result


def _check_case(flex_cfg, flex_source, reference, dist, grad_tools, *, tp=1, ep=1, pp=1, cp=1):
    from megatron.lite.model.minimax_m3.lite.checkpoint import load_hf_weights
    from megatron.lite.model.minimax_m3.lite.model import MiniMaxM3Model
    from megatron.lite.primitive.parallel import init_parallel, zigzag_slice_for_cp
    from megatron.lite.runtime.contracts import ParallelConfig

    world = dist.get_world_size()
    if world % (tp * cp * pp) or world % (ep * pp):
        pytest.skip(f"world={world} incompatible with tp={tp} ep={ep} pp={pp} cp={cp}")
    case = f"tp{tp}_ep{ep}_pp{pp}_cp{cp}"
    ps = init_parallel(ParallelConfig(tp=tp, ep=ep, etp=1, pp=pp, cp=cp))
    torch.manual_seed(7)
    model = MiniMaxM3Model(flex_cfg, _train_config(ps), ps, msa_backend="flex").to(torch.bfloat16).cuda()
    load_hf_weights(model, flex_source, flex_cfg, ps)

    def local(t, seq_dim):  # this rank's zigzag shard of a full-sequence reference tensor
        return zigzag_slice_for_cp(t, ps.cp_rank, ps.cp_size, seq_dim=seq_dim)

    ids, labels = local(reference["ids"], 1).contiguous(), local(reference["labels"], 1).contiguous()
    dump = _LayerDump(model)
    if ps.pp_size == 1:
        loss = model(input_ids=ids, labels=labels)["loss"]
        loss.backward()
    else:
        loss = _run_pipeline_fwd_bwd(model, ps, flex_cfg, ids, labels, dist)
    dump.remove()

    loss_rel = _rel(loss, reference["loss"]) if loss is not None else 0.0  # under CP every rank reports the CP-global mean
    layer_rel = {gi: _rel(_gather_seq(h, ps, dist), local(reference["layer_out"][gi], 0)) for gi, h in dump.out.items()}
    flips = {}
    for local_i, layer in enumerate(model.layers):
        gi = model.layer_indices[local_i]
        if layer.is_sparse_attention:
            got = dump.topk[gi].sort(-1).values
            want = local(_reference_topk_slice(reference["topk"][gi], layer.attn), 2).sort(-1).values
            assert got.shape == want.shape, (got.shape, want.shape)
            flips[gi] = (got != want).any(-1).float().mean().item()
    weights = grad_tools.weights_by_hf_name(model, flex_cfg, ps)
    w_worst = max(_rel(w, reference["weights"][n]) for n, w in weights.items())
    grads = grad_tools.grads_by_hf_name(model, flex_cfg, ps, dist)
    for name, g in grads.items():  # reduce to the single-rank convention
        if grad_tools.is_routed_expert(name):
            # Routed experts see the SP/CP token shards of every rank in their EP group: sum over the expert-DP group
            # (what DDP does), then undo the dp/cp duplication; with tp>1, ep==1 the Experts module already
            # all-reduces its grads over TP.
            if ps.expert_dp_size > 1:
                dist.all_reduce(g, group=ps.ep_dp_group)
            tp_dup = ps.tp_size if (ps.tp_size > 1 and ps.ep_size == 1 and ps.etp_size == 1) else 1
            g.div_(ps.cp_size * ps.dp_size * tp_dup)
        elif ps.cp_size > 1:
            dist.all_reduce(g, group=ps.cp_group)
            g.div_(ps.cp_size)
    assert set(grads) == set(reference["grads"]), (case, set(grads) ^ set(reference["grads"]))
    grad_cos = grad_tools.grad_cosines(grads, reference["grads"])
    (dense_name, dense_cos), (expert_name, expert_cos) = grad_tools.worst(grad_cos)

    # reduce across ranks first so every rank asserts on the same numbers (no collective deadlock)
    stats = torch.tensor([loss_rel, max(layer_rel.values()), max(flips.values()) if flips else 0.0,
                          1 - dense_cos, 1 - expert_cos, w_worst], device="cuda")
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    loss_rel, layer_max, flip_max, dense_gap, expert_gap, w_worst = stats.tolist()
    if dist.get_rank() == 0:
        print(f"\nflex_parallel {case}: loss_rel {loss_rel:.2e} | layer_rel_max {layer_max:.2e} | topk_flips {flip_max:.3%} | "
              f"grads {len(grad_cos)}: worst dense cos {1 - dense_gap:.5f} ({dense_name.split('layers.')[-1]}), "
              f"worst routing-coupled cos {1 - expert_gap:.5f} ({expert_name.split('layers.')[-1]}) | weight export rel {w_worst:.1e}",
              flush=True)
    assert w_worst == 0.0, (case, "weight export round-trip differs", w_worst)
    assert loss_rel < LOSS_REL, (case, loss_rel)
    assert layer_max < LAYER_REL, (case, layer_rel)
    assert flip_max < TOPK_FLIP_BUDGET, (case, flips)
    assert 1 - dense_gap >= DENSE_GRAD_COS, (case, dense_name, 1 - dense_gap)
    assert 1 - expert_gap >= EXPERT_GRAD_COS, (case, expert_name, 1 - expert_gap)
    del model, grads, dump
    torch.cuda.empty_cache()


def test_tp2_matches_single_rank(flex_cfg, flex_source, reference, dist, grad_tools):
    _check_case(flex_cfg, flex_source, reference, dist, grad_tools, tp=2)


def test_ep2_matches_single_rank(flex_cfg, flex_source, reference, dist, grad_tools):
    _check_case(flex_cfg, flex_source, reference, dist, grad_tools, ep=2)


def test_pp2_matches_single_rank(flex_cfg, flex_source, reference, dist, grad_tools):
    _check_case(flex_cfg, flex_source, reference, dist, grad_tools, pp=2)


def test_cp2_matches_single_rank(flex_cfg, flex_source, reference, dist, grad_tools):
    # zigzag chunk of 256 tokens: KV blocks straddle rank boundaries, the global-order gather must fix it up
    _check_case(flex_cfg, flex_source, reference, dist, grad_tools, cp=2)
