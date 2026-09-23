# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax-M3 lite with the production ``magi`` MSA backend, bf16, 2 GPUs: magi CP1 vs flex CP1, and magi CP2 / EP2 / PP2
vs magi CP1, on a single document and on a packed batch with a trailing pad document. The indexer stays frozen throughout.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.gpus(2, min_architecture="blackwell"), pytest.mark.env(CUDA_DEVICE_MAX_CONNECTIONS="1"), pytest.mark.optional]

CHUNK = 512
SINGLE = [4096]  # 32 KV blocks > top-16
PACKED = [1536, 2560, 3840]  # 7936 tokens: not a multiple of 512*2 -> pad doc under CP2
BACKEND_LOSS_REL, LAYOUT_LOSS_REL, LAYER_COS = 1e-2, 5e-3, 0.999
DENSE_GRAD_COS, EXPERT_GRAD_COS = 0.99, 0.95


def _train_config(ps):
    return SimpleNamespace(
        tp=ps.tp_size, ep=ps.ep_size, etp=ps.etp_size, pp=ps.pp_size, cp=ps.cp_size, vpp=None,
        moe_dispatcher="alltoall", fp8=False, recompute_modules=[], deterministic=True,
    )


def _magi_deps():
    pytest.importorskip("magi_attn_extensions.MSA")
    pytest.importorskip("msa_v1")


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


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
def magi_cp1(magi_cfg, magi_source, dist, grad_tools):
    """magi CP1 forward/backward on both batches: the reference for the parallel layouts."""
    from megatron.lite.primitive.parallel import ParallelState

    _magi_deps()
    ps0 = ParallelState()
    results = {}
    for tag, seq_lens in (("single", SINGLE), ("packed", PACKED)):
        model = _build(magi_cfg, ps0, magi_source, "magi")
        w0 = grad_tools.indexer_weights(model)
        ids, labels = _make_batch(magi_cfg, seq_lens)
        loss, layer_out = _run_magi(model, ps0, _magi_ctx(ps0, magi_cfg, seq_lens), ids, labels, dist)
        grad_tools.assert_indexer_frozen(model, w0)
        results[tag] = dict(seq_lens=seq_lens, ids=ids, labels=labels, loss=loss.detach(), layer_out=layer_out,
                            grads=grad_tools.grads_by_hf_name(model, magi_cfg, ps0))
        del model
        torch.cuda.empty_cache()
    return results


@pytest.mark.parametrize("tag", ["single", "packed"])
def test_magi_cp1_matches_flex_cp1(magi_cfg, magi_source, magi_cp1, dist, grad_tools, tag):
    from megatron.lite.primitive.parallel import ParallelState

    ps0 = ParallelState()
    ref_model = _build(magi_cfg, ps0, magi_source, "flex")
    m = magi_cp1[tag]
    ref_loss, ref_layers = _flex_reference(ref_model, m["seq_lens"], m["ids"], m["labels"])
    ref_grads = grad_tools.grads_by_hf_name(ref_model, magi_cfg, ps0)
    loss_rel = abs(m["loss"].item() - ref_loss) / abs(ref_loss)
    layer_cos = {gi: _cos(m["layer_out"][gi], ref_layers[gi]) for gi in ref_layers}
    (dense_name, dense_cos), (expert_name, expert_cos) = grad_tools.worst(grad_tools.grad_cosines(m["grads"], ref_grads))
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


def _check_layout(magi_cfg, magi_source, magi_cp1, dist, grad_tools, *, cp=1, ep=1, pp=1, tag="single"):
    from megatron.lite.primitive.parallel import init_parallel
    from megatron.lite.runtime.contracts import ParallelConfig

    world = dist.get_world_size()
    if world % (cp * pp) or world % (ep * pp):
        pytest.skip(f"world={world} incompatible with cp={cp} ep={ep} pp={pp}")
    ps = init_parallel(ParallelConfig(tp=1, ep=ep, pp=pp, cp=cp))
    ref = magi_cp1[tag]
    model = _build(magi_cfg, ps, magi_source, "magi")
    w0 = grad_tools.indexer_weights(model)
    ctx = _magi_ctx(ps, magi_cfg, ref["seq_lens"])
    loss, layer_out = _run_magi(model, ps, ctx, ref["ids"], ref["labels"], dist)
    grad_tools.assert_indexer_frozen(model, w0)
    case = f"cp{cp}_ep{ep}_pp{pp}_{tag}"
    loss_rel = _rel(loss, ref["loss"]) if loss is not None else 0.0
    layer_cos = {gi: _cos(h, ref["layer_out"][gi]) for gi, h in layer_out.items()}
    grads = grad_tools.grads_by_hf_name(model, magi_cfg, ps)
    for name, g in grads.items():  # reduce to the CP1 convention before comparing
        if grad_tools.is_routed_expert(name):
            if ps.expert_dp_size > 1:
                dist.all_reduce(g, group=ps.ep_dp_group)
            g.div_(ps.cp_size * ps.dp_size)
        elif ps.cp_size > 1:
            dist.all_reduce(g, group=ps.cp_group)
            g.div_(ps.cp_size)
    (dense_name, dense_cos), (expert_name, expert_cos) = grad_tools.worst(grad_tools.grad_cosines(grads, ref["grads"]))
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


def test_magi_cp2_matches_cp1_single(magi_cfg, magi_source, magi_cp1, dist, grad_tools):
    _check_layout(magi_cfg, magi_source, magi_cp1, dist, grad_tools, cp=2, tag="single")


def test_magi_cp2_matches_cp1_packed_with_pad(magi_cfg, magi_source, magi_cp1, dist, grad_tools):
    _check_layout(magi_cfg, magi_source, magi_cp1, dist, grad_tools, cp=2, tag="packed")


def test_magi_ep2_matches_cp1_packed(magi_cfg, magi_source, magi_cp1, dist, grad_tools):
    _check_layout(magi_cfg, magi_source, magi_cp1, dist, grad_tools, ep=2, tag="packed")


def test_magi_pp2_matches_cp1_single(magi_cfg, magi_source, magi_cp1, dist, grad_tools):
    _check_layout(magi_cfg, magi_source, magi_cp1, dist, grad_tools, pp=2, tag="single")
