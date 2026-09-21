# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax-M3 lite with the production ``magi`` MSA backend (MagiAttention MSA extension + msa_v1), bf16, 2 GPUs.

* magi CP1 vs flex CP1 on the same weights: kernel-level agreement of the two backends, on a single
  4096-token document and on a packed 3-document batch.
* magi CP2 / EP2 / PP2 vs magi CP1: the Magi dispatch and required-K communication path, including a
  packed batch whose total is not a multiple of ``chunk * cp`` (trailing pad document).
* the indexer weights are bitwise unchanged and carry no gradient after forward/backward (frozen selector).

Gates (bf16 kernels): loss rel < 1e-2 (backend) / 5e-3 (layout), per-layer cosine >= 0.999, gradient
cosines >= 0.99 (routing-independent) / 0.95 (router gate, per-layer experts). Numbers are printed as evidence.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.gpus(2, min_architecture="blackwell"), pytest.mark.env(CUDA_DEVICE_MAX_CONNECTIONS="1")]


def _train_config(ps):
    return SimpleNamespace(
        tp=ps.tp_size, ep=ps.ep_size, etp=ps.etp_size, pp=ps.pp_size, cp=ps.cp_size, vpp=None,
        use_deepep=False, fp8=False, recompute_modules=[], deterministic=True,
    )

CHUNK = 512
SINGLE = [4096]  # 32 KV blocks > top-16
PACKED = [1536, 2560, 3840]  # 7936 tokens: not a multiple of 512*2 -> pad doc under CP2
BACKEND_LOSS_REL, LAYOUT_LOSS_REL, LAYER_COS = 1e-2, 5e-3, 0.999
DENSE_GRAD_COS, EXPERT_GRAD_COS = 0.99, 0.95


def _magi_deps():
    pytest.importorskip("magi_attn_extensions.MSA")
    pytest.importorskip("msa_v1")


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _is_routed_expert(name):
    return ".experts." in name and ".shared_experts." not in name


def _is_routing_coupled(name):
    return _is_routed_expert(name) or name.endswith("block_sparse_moe.gate.weight")


def _grads_by_hf_name(model, cfg, ps):
    from megatron.lite.model.minimax_m3.lite.checkpoint import export_hf_weights

    params = list(model.named_parameters())
    saved = [p.data for _, p in params]
    for _, p in params:
        p.data = p.grad if p.grad is not None else torch.zeros_like(p.data)
    try:
        grads = {n: t.detach().clone() for n, t in export_hf_weights(model, cfg, ps)}
    finally:
        for (_, p), data in zip(params, saved):
            p.data = data
    return {n: g for n, g in grads.items() if "e_score_correction_bias" not in n and "index_" not in n}


def _grad_cosines(got, want):
    """Routed experts are compared per MoE layer on the concatenation of all their gradients."""
    out, groups = {}, {}
    for n, g in got.items():
        w = want[n]
        if _is_routed_expert(n):
            a, b = groups.setdefault(n.split(".experts.")[0] + ".experts.all", ([], []))
            a.append(g.float().flatten())
            b.append(w.float().flatten())
        elif w.norm() > 0:
            out[n] = _cos(g, w)
    for n, (a, b) in groups.items():
        out[n] = _cos(torch.cat(a), torch.cat(b))
    return out


def _worst(grad_cos):
    dense = {n: c for n, c in grad_cos.items() if not _is_routing_coupled(n)}
    experts = {n: c for n, c in grad_cos.items() if _is_routing_coupled(n)}
    return min(dense.items(), key=lambda kv: kv[1]), min(experts.items(), key=lambda kv: kv[1])


def _indexer_weights(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if ".indexer." in n}


def _assert_indexer_frozen(model, before):
    for n, w in _indexer_weights(model).items():
        assert torch.equal(w, before[n]), f"indexer weight {n} changed"
    for n, p in model.named_parameters():
        if ".indexer." in n:
            assert p.grad is None and not p.requires_grad, n


class _LayerDump:
    def __init__(self, model):
        self.out = {}
        self._hooks = [layer.register_forward_hook(self._make(model.layer_indices[i])) for i, layer in enumerate(model.layers)]

    def _make(self, gi):
        def hook(_m, _i, out):
            self.out[gi] = out.detach()

        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()


def _make_batch(cfg, seq_lens, seed=1234):
    torch.manual_seed(seed)
    total = sum(seq_lens)
    return torch.randint(0, cfg.vocab_size, (total,), device="cuda"), torch.randint(0, cfg.vocab_size, (total,), device="cuda")


def _build(cfg, ps, src, backend):
    from megatron.lite.model.minimax_m3.lite.checkpoint import load_hf_weights
    from megatron.lite.model.minimax_m3.lite.model import MiniMaxM3Model

    torch.manual_seed(7)
    model = MiniMaxM3Model(cfg, _train_config(ps), ps, msa_backend=backend).to(torch.bfloat16).cuda()
    load_hf_weights(model, src, cfg, ps)
    return model


def _flex_reference(ref_model, seq_lens, ids, labels):
    """Per-document flex forward/backward; token-mean loss over all docs and per-layer outputs in global order."""
    dump = _LayerDump(ref_model)
    total_loss, offset = 0.0, 0
    per_layer = {gi: [] for gi in ref_model.layer_indices}
    for L in seq_lens:
        out = ref_model(input_ids=ids[offset : offset + L][None], labels=labels[offset : offset + L][None])
        (out["loss"] * (L / sum(seq_lens))).backward()
        total_loss += out["loss"].item() * L / sum(seq_lens)
        for gi, h in dump.out.items():
            per_layer[gi].append(h[:, 0])
        offset += L
    dump.remove()
    return total_loss, {gi: torch.cat(v, 0) for gi, v in per_layer.items()}


def _magi_ctx(ps, cfg, seq_lens):
    from megatron.lite.primitive.kernels import magi_msa

    settings = magi_msa.MagiMsaSettings(chunk_size=CHUNK)
    msa_cfg = magi_msa.build_msa_config(settings, head_dim=cfg.head_dim, index_head_dim=cfg.index_head_dim)
    return magi_msa.plan_magi_batch(seq_lens, ps=ps, settings=settings, msa_config=msa_cfg, need_dense_key=True)


def _magi_inputs(ctx, ids, labels):
    from megatron.lite.primitive.kernels import magi_msa

    loss_mask = torch.ones_like(labels, dtype=torch.float32)
    if ctx.pad:
        ids, labels = F.pad(ids, (0, ctx.pad), value=0), F.pad(labels, (0, ctx.pad), value=0)
        loss_mask = F.pad(loss_mask, (0, ctx.pad), value=0.0)
    return tuple(magi_msa.dispatch_tokens(t, ctx).reshape(1, -1) for t in (ids, labels, loss_mask))


def _run_magi(model, ps, ctx, ids, labels, dist):
    """Forward/backward of a magi model (pp == 1 or a two-stage pipeline); returns (loss, layer outputs in global order)."""
    from megatron.lite.primitive.kernels import magi_msa

    d_ids, d_labels, d_mask = _magi_inputs(ctx, ids, labels)
    dump = _LayerDump(model)
    if ps.pp_size == 1:
        loss = model(input_ids=d_ids, labels=d_labels, loss_mask=d_mask, magi_ctx=ctx)["loss"]
        loss.backward()
    elif ps.pp_is_first:
        h = model(input_ids=d_ids, magi_ctx=ctx)["hidden_states"]
        dist.send(h.contiguous(), ps.pp_next_rank)
        grad = torch.empty_like(h)
        dist.recv(grad, ps.pp_next_rank)
        h.backward(grad)
        loss = None
    else:
        h = torch.empty(ctx.local_tokens, 1, model.config.hidden_size, device="cuda", dtype=torch.bfloat16)
        dist.recv(h, ps.pp_prev_rank)
        h.requires_grad_(True)
        loss = model(hidden_states=h, labels=d_labels, loss_mask=d_mask, magi_ctx=ctx)["loss"]
        loss.backward()
        dist.send(h.grad.contiguous(), ps.pp_prev_rank)
    dump.remove()
    real = ctx.cu_seqlens_host[-1] - ctx.pad
    return loss, {gi: magi_msa.undispatch_tokens(h[:, 0].contiguous(), ctx)[:real] for gi, h in dump.out.items()}


@pytest.fixture(scope="module")
def magi_cp1(magi_cfg, magi_source, dist):
    """magi CP1 forward/backward on both batches: the reference for the parallel layouts."""
    from megatron.lite.primitive.parallel import ParallelState

    _magi_deps()
    ps0 = ParallelState()
    results = {}
    for tag, seq_lens in (("single", SINGLE), ("packed", PACKED)):
        model = _build(magi_cfg, ps0, magi_source, "magi")
        w0 = _indexer_weights(model)
        ids, labels = _make_batch(magi_cfg, seq_lens)
        loss, layer_out = _run_magi(model, ps0, _magi_ctx(ps0, magi_cfg, seq_lens), ids, labels, dist)
        _assert_indexer_frozen(model, w0)
        results[tag] = dict(seq_lens=seq_lens, ids=ids, labels=labels, loss=loss.detach(), layer_out=layer_out,
                            grads=_grads_by_hf_name(model, magi_cfg, ps0))
        del model
        torch.cuda.empty_cache()
    return results


@pytest.mark.parametrize("tag", ["single", "packed"])
def test_magi_cp1_matches_flex_cp1(magi_cfg, magi_source, magi_cp1, dist, tag):
    from megatron.lite.primitive.parallel import ParallelState

    ps0 = ParallelState()
    ref_model = _build(magi_cfg, ps0, magi_source, "flex")
    m = magi_cp1[tag]
    ref_loss, ref_layers = _flex_reference(ref_model, m["seq_lens"], m["ids"], m["labels"])
    ref_grads = _grads_by_hf_name(ref_model, magi_cfg, ps0)
    loss_rel = abs(m["loss"].item() - ref_loss) / abs(ref_loss)
    layer_cos = {gi: _cos(m["layer_out"][gi], ref_layers[gi]) for gi in ref_layers}
    (dense_name, dense_cos), (expert_name, expert_cos) = _worst(_grad_cosines(m["grads"], ref_grads))
    if dist.get_rank() == 0:
        print(f"\nmagi_vs_flex {tag}: loss {m['loss'].item():.5f} vs {ref_loss:.5f} (rel {loss_rel:.2e}) | layer cos min "
              f"{min(layer_cos.values()):.6f} | worst dense grad cos {dense_cos:.5f} ({dense_name.split('layers.')[-1]}), "
              f"worst routing-coupled {expert_cos:.5f} ({expert_name.split('layers.')[-1]})", flush=True)
    assert loss_rel < BACKEND_LOSS_REL, (tag, loss_rel)
    assert min(layer_cos.values()) >= LAYER_COS, (tag, layer_cos)
    assert dense_cos >= DENSE_GRAD_COS, (tag, dense_name, dense_cos)
    assert expert_cos >= EXPERT_GRAD_COS, (tag, expert_name, expert_cos)
    del ref_model
    torch.cuda.empty_cache()


def _check_layout(magi_cfg, magi_source, magi_cp1, dist, *, cp=1, ep=1, pp=1, tag="single"):
    from megatron.lite.primitive.parallel import init_parallel
    from megatron.lite.runtime.contracts import ParallelConfig

    world = dist.get_world_size()
    if world % (cp * pp) or world % (ep * pp):
        pytest.skip(f"world={world} incompatible with cp={cp} ep={ep} pp={pp}")
    ps = init_parallel(ParallelConfig(tp=1, ep=ep, pp=pp, cp=cp))
    ref = magi_cp1[tag]
    model = _build(magi_cfg, ps, magi_source, "magi")
    w0 = _indexer_weights(model)
    ctx = _magi_ctx(ps, magi_cfg, ref["seq_lens"])
    loss, layer_out = _run_magi(model, ps, ctx, ref["ids"], ref["labels"], dist)
    _assert_indexer_frozen(model, w0)
    case = f"cp{cp}_ep{ep}_pp{pp}_{tag}"
    loss_rel = _rel(loss, ref["loss"]) if loss is not None else 0.0
    layer_cos = {gi: _cos(h, ref["layer_out"][gi]) for gi, h in layer_out.items()}
    grads = _grads_by_hf_name(model, magi_cfg, ps)
    for name, g in grads.items():  # reduce to the CP1 convention before comparing
        if _is_routed_expert(name):
            if ps.expert_dp_size > 1:
                dist.all_reduce(g, group=ps.ep_dp_group)
            g.div_(ps.cp_size * ps.dp_size)
        elif ps.cp_size > 1:
            dist.all_reduce(g, group=ps.cp_group)
            g.div_(ps.cp_size)
    (dense_name, dense_cos), (expert_name, expert_cos) = _worst(_grad_cosines(grads, ref["grads"]))
    stats = torch.tensor([loss_rel, 1.0 - min(layer_cos.values()), 1.0 - dense_cos, 1.0 - expert_cos], device="cuda")
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    loss_rel, layer_gap, dense_gap, expert_gap = stats.tolist()
    if dist.get_rank() == 0:
        print(f"\nmagi_layout {case}: pad {ctx.pad} local {ctx.local_tokens} | loss_rel {loss_rel:.2e} | layer cos min "
              f"{1 - layer_gap:.6f} | worst dense grad cos {1 - dense_gap:.5f} ({dense_name.split('layers.')[-1]}), "
              f"worst routing-coupled {1 - expert_gap:.5f} ({expert_name.split('layers.')[-1]})", flush=True)
    assert loss_rel < LAYOUT_LOSS_REL, (case, loss_rel)
    assert 1 - layer_gap >= LAYER_COS, (case, layer_cos)
    assert 1 - dense_gap >= DENSE_GRAD_COS, (case, dense_name)
    assert 1 - expert_gap >= EXPERT_GRAD_COS, (case, expert_name)
    del model
    torch.cuda.empty_cache()


def test_magi_cp2_matches_cp1_single(magi_cfg, magi_source, magi_cp1, dist):
    _check_layout(magi_cfg, magi_source, magi_cp1, dist, cp=2, tag="single")


def test_magi_cp2_matches_cp1_packed_with_pad(magi_cfg, magi_source, magi_cp1, dist):
    _check_layout(magi_cfg, magi_source, magi_cp1, dist, cp=2, tag="packed")


def test_magi_ep2_matches_cp1_packed(magi_cfg, magi_source, magi_cp1, dist):
    _check_layout(magi_cfg, magi_source, magi_cp1, dist, ep=2, tag="packed")


def test_magi_pp2_matches_cp1_single(magi_cfg, magi_source, magi_cp1, dist):
    _check_layout(magi_cfg, magi_source, magi_cp1, dist, pp=2, tag="single")
