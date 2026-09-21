# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Proxy-M3 with ``msa_backend="magi"`` (MagiAttention MSA extension + msa_v1 kernels), bf16, 8 GPUs.

* magi CP1 vs flex CP1 (same weights, single 4096-token doc): bf16-kernel-level agreement.
* magi CP{2,4,8} (+EP / +PP) vs magi CP1: the Magi dispatch / required-K communication path, incl. a
  packed 3-document batch whose total is not a multiple of chunk*cp (trailing pad doc).
* indexer invariant: the indexer weights are bitwise unchanged after forward/backward (frozen selector).

The msa_v1 kernels are bf16-only, so this is a bf16 comparison (cosine / relative), not the fp32 gate
of ``test_minimax_m3_parallel_smoke.py`` (which stays flex-only).
Run: torchrun --nproc-per-node=8 -m pytest -s <file> with MLITE_TEST_HARNESS=1 and
MAGI_ATTENTION_KERNEL_BACKEND=sdpa_ol (dense layers) on B200.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

_LITE = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
sys.path.insert(0, os.path.join(_LITE, "ref", "minimax_m3"))

pytestmark = [
    pytest.mark.gpus(8, min_architecture="blackwell"),
    pytest.mark.env(CUDA_DEVICE_MAX_CONNECTIONS="1"),
]

CHUNK = 512
SINGLE = [4096]  # 32 KV blocks > top-16
PACKED = [1536, 2560, 3840]  # 7936 tokens: not a multiple of 512*cp for cp >= 2 -> pad doc


def _init_dist_or_skip():
    import torch.distributed as dist

    if not torch.cuda.is_available() or "RANK" not in os.environ:
        pytest.skip("run with torchrun on GPUs")
    pytest.importorskip("magi_attn_extensions.MSA")
    pytest.importorskip("msa_v1")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return dist


def _proxy_config():
    from proxy_config import hf_proxy_text_config_kwargs

    from megatron.lite.model.minimax_m3.config import MiniMaxM3Config

    hf = dict(hf_proxy_text_config_kwargs(magi=True))
    hf["model_type"] = "minimax_m3_vl_text"
    return MiniMaxM3Config._from_hf_dict(hf)


def _train_cfg(ps):
    return SimpleNamespace(
        tp=ps.tp_size, ep=ps.ep_size, etp=ps.etp_size, pp=ps.pp_size, cp=ps.cp_size, vpp=None,
        use_deepep=False, fp8=False, recompute_modules=[], deterministic=True,
    )


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _is_routed_expert(name: str) -> bool:
    return ".experts." in name and ".shared_experts." not in name


def _is_routing_coupled(name: str) -> bool:
    """Routed experts and the router gate: their gradients depend on which tokens are routed where. The
    random-init proxy routes almost uniformly, so a handful of bf16-kernel-induced routing flips moves
    these gradients a lot (P1 measured 5-17% bf16 top-k flips); they get a looser band than the rest."""
    return _is_routed_expert(name) or name.endswith("block_sparse_moe.gate.weight")


def _expert_layer(name: str) -> str:
    return name.split(".experts.")[0]


def _grad_cosines(got: dict[str, torch.Tensor], want: dict[str, torch.Tensor]) -> dict[str, float]:
    """Per-parameter cosines, except routed experts, which are compared per MoE layer on the concatenation of
    all their gradients: a single under-populated expert of the random-init proxy can flip a few tokens' routing
    under bf16 kernel changes and swing its own tiny gradient, while the layer-level expert gradient is stable."""
    out: dict[str, float] = {}
    groups: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    for n, g in got.items():
        w = want[n]
        if _is_routed_expert(n):
            a, b = groups.setdefault(_expert_layer(n) + ".experts[all]", ([], []))
            a.append(g.float().flatten())
            b.append(w.float().flatten())
        elif w.norm() > 0:
            out[n] = _cos(g, w)
    for n, (a, b) in groups.items():
        out[n] = _cos(torch.cat(a), torch.cat(b))
    return out


def _split_worst(grad_cos: dict[str, float]) -> tuple[tuple[str, float], tuple[str, float]]:
    """(worst routing-independent param, worst routing-coupled quantity: router gate or per-layer experts)."""
    dense = {n: c for n, c in grad_cos.items() if not _is_routing_coupled(n)}
    experts = {n: c for n, c in grad_cos.items() if _is_routing_coupled(n)}
    w_d = min(dense.items(), key=lambda kv: kv[1]) if dense else ("", 1.0)
    w_e = min(experts.items(), key=lambda kv: kv[1]) if experts else ("", 1.0)
    return w_d, w_e


DENSE_GRAD_COS, EXPERT_GRAD_COS = 0.99, 0.95


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


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


def _indexer_weights(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if ".indexer." in n}


class _LayerDump:
    def __init__(self, model):
        self.out: dict[int, torch.Tensor] = {}
        self._hooks = [
            layer.register_forward_hook(self._make(model.layer_indices[i])) for i, layer in enumerate(model.layers)
        ]

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
    ids = torch.randint(0, cfg.vocab_size, (total,), device="cuda")
    labels = torch.randint(0, cfg.vocab_size, (total,), device="cuda")
    return ids, labels


def _flex_reference(ref_model, seq_lens, ids, labels):
    """Per-document flex forward/backward; returns token-mean loss over all docs, per-layer global outputs."""
    dump = _LayerDump(ref_model)
    total_loss = 0.0
    per_layer = {gi: [] for gi in ref_model.layer_indices}
    offset = 0
    for L in seq_lens:
        d_ids = ids[offset : offset + L][None]
        d_labels = labels[offset : offset + L][None]
        out = ref_model(input_ids=d_ids, labels=d_labels)
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
        ids = F.pad(ids, (0, ctx.pad), value=0)
        labels = F.pad(labels, (0, ctx.pad), value=0)
        loss_mask = F.pad(loss_mask, (0, ctx.pad), value=0.0)
    return tuple(magi_msa.dispatch_tokens(t, ctx).reshape(1, -1) for t in (ids, labels, loss_mask))


def _run_magi(model, ps, ctx, ids, labels):
    """Forward/backward of a magi model (pp == 1 or a manual 2-stage pipeline); returns (loss, layer outs in global order)."""
    import torch.distributed as dist

    from megatron.lite.primitive.kernels import magi_msa

    d_ids, d_labels, d_mask = _magi_inputs(ctx, ids, labels)
    dump = _LayerDump(model)
    if ps.pp_size == 1:
        out = model(input_ids=d_ids, labels=d_labels, loss_mask=d_mask, magi_ctx=ctx)
        loss = out["loss"]
        loss.backward()
    elif ps.pp_is_first:
        out = model(input_ids=d_ids, magi_ctx=ctx)
        h = out["hidden_states"]
        dist.send(h.contiguous(), ps.pp_next_rank)
        grad = torch.empty_like(h)
        dist.recv(grad, ps.pp_next_rank)
        h.backward(grad)
        loss = None
    else:
        h = torch.empty(ctx.local_tokens, 1, model.config.hidden_size, device="cuda", dtype=torch.bfloat16)
        dist.recv(h, ps.pp_prev_rank)
        h.requires_grad_(True)
        out = model(hidden_states=h, labels=d_labels, loss_mask=d_mask, magi_ctx=ctx)
        loss = out["loss"]
        loss.backward()
        dist.send(h.grad.contiguous(), ps.pp_prev_rank)
    dump.remove()
    real = ctx.cu_seqlens_host[-1] - ctx.pad
    layer_out = {gi: magi_msa.undispatch_tokens(h[:, 0].contiguous(), ctx)[:real] for gi, h in dump.out.items()}
    return loss, layer_out


def _save_source(cfg, tmp_path_factory, dist, tag):
    from megatron.lite.model.minimax_m3.lite.checkpoint import save_hf_weights
    from megatron.lite.model.minimax_m3.lite.model import MiniMaxM3Model
    from megatron.lite.primitive.parallel import ParallelState

    ps0 = ParallelState()
    torch.manual_seed(20260914)
    ref = MiniMaxM3Model(cfg, _train_cfg(ps0), ps0, msa_backend="flex").to(torch.bfloat16).cuda()
    with torch.no_grad():
        for n, p in ref.named_parameters():
            if n.endswith("norm.weight") or n.endswith("layer_norm_weight"):
                p.normal_(std=0.1)
        for layer in ref.layers:
            if layer.moe is not None:
                layer.moe.router.expert_bias.normal_(std=0.05)
    src = [str(tmp_path_factory.mktemp(f"hf_proxy_magi_{tag}")) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(src, src=0)
    save_hf_weights(ref, src[0], cfg, ps0)
    del ref
    return src[0]


def _build(cfg, ps, src, backend):
    from megatron.lite.model.minimax_m3.lite.checkpoint import load_hf_weights
    from megatron.lite.model.minimax_m3.lite.model import MiniMaxM3Model

    torch.manual_seed(7)
    model = MiniMaxM3Model(cfg, _train_cfg(ps), ps, msa_backend=backend).to(torch.bfloat16).cuda()
    load_hf_weights(model, src, cfg, ps)
    return model


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    dist = _init_dist_or_skip()
    cfg = _proxy_config()
    return SimpleNamespace(cfg=cfg, src=_save_source(cfg, tmp_path_factory, dist, "src"))


@pytest.fixture(scope="module")
def magi_cp1(source):
    """magi CP1 forward/backward on both batches: the reference for the CP>1 cases."""
    from megatron.lite.primitive.parallel import ParallelState

    _init_dist_or_skip()
    from megatron.lite.primitive.kernels import magi_msa

    magi_msa.ensure_single_process_group()
    ps0 = ParallelState()
    results = {}
    for tag, seq_lens in (("single", SINGLE), ("packed", PACKED)):
        model = _build(source.cfg, ps0, source.src, "magi")
        w0 = _indexer_weights(model)
        ids, labels = _make_batch(source.cfg, seq_lens)
        ctx = _magi_ctx(ps0, source.cfg, seq_lens)
        loss, layer_out = _run_magi(model, ps0, ctx, ids, labels)
        for n, w in _indexer_weights(model).items():
            assert torch.equal(w, w0[n]), f"indexer weight {n} changed"
        for n, p in model.named_parameters():
            if ".indexer." in n:
                assert p.grad is None and not p.requires_grad, n
        results[tag] = SimpleNamespace(
            seq_lens=seq_lens, ids=ids, labels=labels, loss=loss.detach(), layer_out=layer_out,
            grads=_grads_by_hf_name(model, source.cfg, ps0),
        )
        del model
    return results


def test_magi_cp1_matches_flex_cp1(source, magi_cp1):
    from megatron.lite.primitive.parallel import ParallelState

    dist = _init_dist_or_skip()
    ps0 = ParallelState()
    for tag in ("single", "packed"):
        ref_model = _build(source.cfg, ps0, source.src, "flex")
        m = magi_cp1[tag]
        ref_loss, ref_layers = _flex_reference(ref_model, m.seq_lens, m.ids, m.labels)
        ref_grads = _grads_by_hf_name(ref_model, source.cfg, ps0)
        loss_rel = abs(m.loss.item() - ref_loss) / abs(ref_loss)
        layer_cos = {gi: _cos(m.layer_out[gi], ref_layers[gi]) for gi in ref_layers}
        grad_cos = _grad_cosines(m.grads, ref_grads)
        w_d, w_e = _split_worst(grad_cos)
        if dist.get_rank() == 0:
            print(f"\n[magi vs flex, {tag}] loss {m.loss.item():.5f} vs {ref_loss:.5f} (rel {loss_rel:.2e}); "
                  f"layer cos min {min(layer_cos.values()):.6f}; grads {len(grad_cos)}: worst dense {w_d[0].split('layers.')[-1]} "
                  f"{w_d[1]:.5f}, worst routing-coupled {w_e[0].split('layers.')[-1]} {w_e[1]:.5f}")
        assert loss_rel < 1e-2, (tag, loss_rel)
        assert min(layer_cos.values()) >= 0.999, (tag, layer_cos)
        assert w_d[1] >= DENSE_GRAD_COS, (tag, w_d)
        assert w_e[1] >= EXPERT_GRAD_COS, (tag, w_e)
        del ref_model


def _check_cp(source, magi_cp1, *, cp, ep=1, pp=1, tag="single"):
    from megatron.lite.primitive.parallel import init_parallel
    from megatron.lite.runtime.contracts import ParallelConfig

    dist = _init_dist_or_skip()
    world = dist.get_world_size()
    if world % (cp * pp) or world % (ep * pp):
        pytest.skip(f"world={world} incompatible with cp={cp} ep={ep} pp={pp}")
    ps = init_parallel(ParallelConfig(tp=1, ep=ep, pp=pp, cp=cp))
    ref = magi_cp1[tag]
    model = _build(source.cfg, ps, source.src, "magi")
    w0 = _indexer_weights(model)
    ctx = _magi_ctx(ps, source.cfg, ref.seq_lens)
    loss, layer_out = _run_magi(model, ps, ctx, ref.ids, ref.labels)
    for n, w in _indexer_weights(model).items():
        assert torch.equal(w, w0[n]), f"indexer weight {n} changed"
    case = f"cp{cp}_ep{ep}_pp{pp}_{tag}"
    loss_rel = _rel(loss, ref.loss) if loss is not None else 0.0
    layer_cos = {gi: _cos(h, ref.layer_out[gi]) for gi, h in layer_out.items()}
    grads = _grads_by_hf_name(model, source.cfg, ps)
    for name, g in grads.items():  # reduce to the CP1 convention before comparing
        if _is_routed_expert(name):
            if ps.expert_dp_size > 1:
                dist.all_reduce(g, group=ps.ep_dp_group)
            g.div_(ps.cp_size * ps.dp_size)
        elif ps.cp_size > 1:
            dist.all_reduce(g, group=ps.cp_group)
            g.div_(ps.cp_size)
    grad_cos = _grad_cosines(grads, ref.grads)
    w_d, w_e = _split_worst(grad_cos)
    stats = torch.tensor([loss_rel, 1.0 - min(layer_cos.values()), 1.0 - w_d[1], 1.0 - w_e[1]], device="cuda")
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    loss_rel, layer_gap, dense_gap, expert_gap = stats.tolist()
    if dist.get_rank() == 0:
        print(f"\n[{case}] pad {ctx.pad} local {ctx.local_tokens} | loss_rel {loss_rel:.2e} | layer cos min {1 - layer_gap:.6f} | "
              f"grads {len(grad_cos)}: worst dense cos {1 - dense_gap:.5f} ({w_d[0].split('layers.')[-1]}), "
              f"worst routing-coupled cos {1 - expert_gap:.5f} ({w_e[0].split('layers.')[-1]})")
    assert loss_rel < 5e-3, (case, loss_rel)
    assert 1 - layer_gap >= 0.999, (case, layer_cos)
    assert 1 - dense_gap >= DENSE_GRAD_COS, (case, w_d)
    assert 1 - expert_gap >= EXPERT_GRAD_COS, (case, w_e)
    del model


@pytest.mark.parametrize("cp", [2, 4, 8])
def test_magi_cp_matches_cp1_single(source, magi_cp1, cp):
    _check_cp(source, magi_cp1, cp=cp, tag="single")


@pytest.mark.parametrize("cp", [2, 4])
def test_magi_cp_matches_cp1_packed_with_pad(source, magi_cp1, cp):
    _check_cp(source, magi_cp1, cp=cp, tag="packed")


def test_magi_cp2_ep2(source, magi_cp1):
    _check_cp(source, magi_cp1, cp=2, ep=2, tag="packed")


def test_magi_cp4_ep2(source, magi_cp1):
    _check_cp(source, magi_cp1, cp=4, ep=2, tag="single")


def test_magi_cp2_pp2(source, magi_cp1):
    _check_cp(source, magi_cp1, cp=2, pp=2, tag="single")
