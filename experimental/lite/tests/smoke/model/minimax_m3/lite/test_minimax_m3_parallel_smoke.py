# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""P3: MiniMax-M3 lite Proxy-M3 under TP / EP / PP / CP (and combinations) vs the single-rank model, fp32 gate.

Every rank runs the same batch. The reference is the un-parallelised lite model (``ParallelState()``)
loaded from the same HF-format weights as the parallel model, so the comparison isolates the
parallel code paths (TP shards + SP, index-head replication for tp > H_idx, EP dispatch, PP p2p).

Gate (tools/minimax_m3/thresholds.py, fp32 only, ``NVIDIA_TF32_OVERRIDE=0``):
* per-layer hidden states (CP: on this rank's zigzag shard) and the loss (CP: the CP-global mean): rel < 1e-5
* MSA top-k block sets identical on every query row
* CP gradients are averaged over the CP group (each rank's loss is the mean over its own tokens);
  routed-expert gradients are summed over the expert-DP group and divided by cp_size * dp_size
* every parameter gradient (gathered to HF names through ``export_hf_weights``) rel < 1e-5;

Run with ``tests/run_tests.sh`` or ``torchrun --nproc-per-node=8 -m pytest -s <file>`` with
``MLITE_TEST_HARNESS=1 NVIDIA_TF32_OVERRIDE=0``.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

_LITE = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
sys.path.insert(0, os.path.join(_LITE, "ref", "minimax_m3"))
sys.path.insert(0, os.path.join(_LITE, "tools", "minimax_m3"))

pytestmark = [
    pytest.mark.gpus(8),
    pytest.mark.env(CUDA_DEVICE_MAX_CONNECTIONS="1"),
]

B, S = 2, 1024  # 8 KV blocks > top-4 -> Proxy sparse layers are truly sparse; S % (2*tp) == 0 for tp <= 8


def _init_dist_or_skip():
    import torch.distributed as dist

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("run with torchrun")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return dist


def _proxy_config(all_msa: bool = False):
    from proxy_config import hf_proxy_text_config_kwargs

    from megatron.lite.model.minimax_m3.config import MiniMaxM3Config

    hf = dict(hf_proxy_text_config_kwargs())
    hf["model_type"] = "minimax_m3_vl_text"
    cfg = MiniMaxM3Config._from_hf_dict(hf)
    if all_msa:
        # TE DotProductAttention has no fp32 context-parallel backend, so the fp32 CP gate runs every
        # attention layer through the MSA (flex) path; dense-attention CP is upstream TE behaviour.
        cfg.layer_types = ["minimax_m3_sparse"] * cfg.num_hidden_layers
    return cfg


def _train_cfg(ps):
    return SimpleNamespace(
        tp=ps.tp_size, ep=ps.ep_size, etp=ps.etp_size, pp=ps.pp_size, cp=ps.cp_size, vpp=None,
        use_deepep=False, fp8=False, recompute_modules=[], deterministic=True,
    )


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


def _grads_by_hf_name(model, cfg, ps) -> dict[str, torch.Tensor]:
    """Gather parameter gradients across TP/EP/PP by exporting them through the weight path."""
    import torch.distributed as dist

    from megatron.lite.model.minimax_m3.lite.checkpoint import export_hf_weights

    params = list(model.named_parameters())
    saved = [p.data for _, p in params]
    sp_ids = {id(p) for p in getattr(model, "sp_params", [])}
    for _, p in params:
        g = p.grad if p.grad is not None else torch.zeros_like(p.data)
        if id(p) in sp_ids and ps.tp_size > 1:  # TP-replicated params with SP-sharded input: DDP sums their grads over TP
            g = g.clone()
            dist.all_reduce(g, group=ps.tp_group)
        p.data = g
    try:
        grads = {name: t.detach().clone() for name, t in export_hf_weights(model, cfg, ps)}
    finally:
        for (_, p), data in zip(params, saved):
            p.data = data
    return {n: g for n, g in grads.items() if "e_score_correction_bias" not in n and "index_" not in n}


def _weights_by_hf_name(model, cfg, ps) -> dict[str, torch.Tensor]:
    from megatron.lite.model.minimax_m3.lite.checkpoint import export_hf_weights

    return {name: t.detach().clone() for name, t in export_hf_weights(model, cfg, ps) if "e_score_correction_bias" not in name}


class _LayerDump:
    def __init__(self, model, ps):
        self.out: dict[int, torch.Tensor] = {}
        self.topk: dict[int, torch.Tensor] = {}
        self.ps = ps
        self._hooks = []
        for local_i, layer in enumerate(model.layers):
            gi = model.layer_indices[local_i]
            self._hooks.append(layer.register_forward_hook(self._make(gi, layer)))

    def _make(self, gi, layer):
        def hook(_m, _i, out):
            self.out[gi] = out.detach()
            if getattr(layer, "is_sparse_attention", False):
                self.topk[gi] = layer.attn.last_block_indices.detach()

        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()


def _gather_seq(x: torch.Tensor, ps) -> torch.Tensor:
    """SP-sharded [S/tp, B, H] -> [S, B, H]."""
    import torch.distributed as dist

    if ps.tp_size <= 1:
        return x
    parts = [torch.empty_like(x) for _ in range(ps.tp_size)]
    dist.all_gather(parts, x.contiguous(), group=ps.tp_group)
    return torch.cat(parts, dim=0)


def _reference_topk_slice(ref_idx: torch.Tensor, attn) -> torch.Tensor:
    """Reference [B, H_idx, S, K] -> the index heads this TP rank holds."""
    idxr = attn.indexer
    if idxr._replicate_heads:
        h = idxr.ps.tp_rank // (idxr.ps.tp_size // idxr.num_heads)
        return ref_idx[:, h : h + 1]
    hl = idxr.num_heads_local
    return ref_idx[:, idxr.ps.tp_rank * hl : (idxr.ps.tp_rank + 1) * hl]


def _run_pipeline_fwd_bwd(model, ps, cfg, ids, labels):
    """Manual 1F1B-free two-stage schedule (single microbatch): p2p hidden states and their grads."""
    import torch.distributed as dist

    if ps.pp_is_first:
        out = model(input_ids=ids)
        h = out["hidden_states"]
        dist.send(h.contiguous(), ps.pp_next_rank)
        grad = torch.empty_like(h)
        dist.recv(grad, ps.pp_next_rank)
        h.backward(grad)
        return None
    local_s = S // ps.cp_size // ps.tp_size
    h = torch.empty(local_s, B, cfg.hidden_size, device="cuda", dtype=torch.float32)
    dist.recv(h, ps.pp_prev_rank)
    h.requires_grad_(True)
    if ps.pp_is_last:
        out = model(hidden_states=h, labels=labels)
        out["loss"].backward()
        dist.send(h.grad.contiguous(), ps.pp_prev_rank)
        return out["loss"]
    out = model(hidden_states=h)
    dist.send(out["hidden_states"].contiguous(), ps.pp_next_rank)
    grad = torch.empty_like(out["hidden_states"])
    dist.recv(grad, ps.pp_next_rank)
    out["hidden_states"].backward(grad)
    dist.send(h.grad.contiguous(), ps.pp_prev_rank)
    return None


def _build_reference(cfg, tmp_path_factory, tag):
    """Single-rank fp32 Proxy-M3 + its HF-format weights on disk (rank 0 writes, everyone reads)."""
    from thresholds import assert_fp32_env

    from megatron.lite.model.minimax_m3.lite.checkpoint import save_hf_weights
    from megatron.lite.model.minimax_m3.lite.model import MiniMaxM3Model
    from megatron.lite.primitive.parallel import ParallelState

    dist = _init_dist_or_skip()
    assert_fp32_env()
    ps0 = ParallelState()
    torch.manual_seed(20260910)
    ref = MiniMaxM3Model(cfg, _train_cfg(ps0), ps0, msa_backend="flex").float().cuda()
    with torch.no_grad():  # non-trivial Gemma-norm weights and router bias
        for n, p in ref.named_parameters():
            if n.endswith("norm.weight") or n.endswith("layer_norm_weight"):
                p.normal_(std=0.1)
        for layer in ref.layers:
            if layer.moe is not None:
                layer.moe.router.expert_bias.normal_(std=0.05)
    src = [str(tmp_path_factory.mktemp(f"hf_proxy_ref_{tag}")) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(src, src=0)
    save_hf_weights(ref, src[0], cfg, ps0)  # rank 0 writes; every rank must join (it ends with a barrier)
    torch.manual_seed(1234)
    ids = torch.randint(0, cfg.vocab_size, (B, S), device="cuda")
    labels = torch.randint(0, cfg.vocab_size, (B, S), device="cuda")
    dump = _LayerDump(ref, ps0)
    out = ref(input_ids=ids, labels=labels)
    dump.remove()
    out["loss"].backward()
    grads = _grads_by_hf_name(ref, cfg, ps0)
    weights = _weights_by_hf_name(ref, cfg, ps0)
    return SimpleNamespace(cfg=cfg, src=src[0], ids=ids, labels=labels, loss=out["loss"].detach(), weights=weights,
                           log_probs=out["log_probs"].detach(), layer_out=dump.out, topk=dump.topk, grads=grads)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    return _build_reference(_proxy_config(), tmp_path_factory, "full")


@pytest.fixture(scope="module")
def reference_all_msa(tmp_path_factory):
    return _build_reference(_proxy_config(all_msa=True), tmp_path_factory, "allmsa")


def _check_case(reference, *, tp=1, ep=1, pp=1, cp=1, etp=None):
    from thresholds import FP32

    from megatron.lite.model.minimax_m3.lite.checkpoint import load_hf_weights
    from megatron.lite.model.minimax_m3.lite.model import MiniMaxM3Model
    from megatron.lite.primitive.parallel import init_parallel, zigzag_slice_for_cp
    from megatron.lite.runtime.contracts import ParallelConfig

    dist = _init_dist_or_skip()
    world = dist.get_world_size()
    if world % (tp * cp * pp) or world % ((etp or 1) * ep * pp):
        pytest.skip(f"world={world} incompatible with tp={tp} ep={ep} pp={pp} cp={cp}")
    cfg = reference.cfg
    ps = init_parallel(ParallelConfig(tp=tp, ep=ep, etp=etp, pp=pp, cp=cp))
    torch.manual_seed(7)
    model = MiniMaxM3Model(cfg, _train_cfg(ps), ps, msa_backend="flex").float().cuda()
    load_hf_weights(model, reference.src, cfg, ps)

    def local(t, seq_dim):  # this rank's zigzag shard of a full-sequence reference tensor
        return zigzag_slice_for_cp(t, ps.cp_rank, ps.cp_size, seq_dim=seq_dim)

    ids, labels = local(reference.ids, 1).contiguous(), local(reference.labels, 1).contiguous()
    dump = _LayerDump(model, ps)
    if ps.pp_size == 1:
        out = model(input_ids=ids, labels=labels)
        loss = out["loss"]
        loss.backward()
    else:
        loss = _run_pipeline_fwd_bwd(model, ps, cfg, ids, labels)
    dump.remove()

    report = {"case": f"tp{tp}_ep{ep}_pp{pp}_cp{cp}" + (f"_etp{etp}" if etp else "")}
    # --- loss (last pp stage only; under CP every rank reports the CP-global mean)
    loss_rel = _rel(loss, reference.loss) if loss is not None else 0.0
    # --- per-layer hidden states (gather SP shards)
    layer_rel = {gi: _rel(_gather_seq(h, ps), local(reference.layer_out[gi], 0)) for gi, h in dump.out.items()}
    # --- MSA top-k sets
    flips = {}
    for local_i, layer in enumerate(model.layers):
        gi = model.layer_indices[local_i]
        if layer.is_sparse_attention:
            got = dump.topk[gi].sort(-1).values
            want = local(_reference_topk_slice(reference.topk[gi], layer.attn), 2).sort(-1).values
            assert got.shape == want.shape, (got.shape, want.shape)
            flips[gi] = (got != want).any(-1).float().mean().item()
    # --- gradients (gathered through the export path; experts see ep_size copies of each token)
    # --- weights exported back through the TP/EP/PP gather must equal the source (validates the export merge)
    weights = _weights_by_hf_name(model, cfg, ps)
    w_worst = max(_rel(w, reference.weights[n]) for n, w in weights.items())
    grads = _grads_by_hf_name(model, cfg, ps)
    worst, worst_name = 0.0, ""
    all_rel = {}
    for name, g in grads.items():
        want = reference.grads[name]
        if ".experts." in name and ".shared_experts." not in name:
            # Routed experts see the SP/CP token shards of every rank in their EP group. Summing over the
            # expert-DP group (what DDP does) covers every token dp_size times; each local loss is a mean
            # over S/cp tokens (extra cp factor). With tp>1, ep==1, etp==1 the Experts module already
            # all-reduces its grads over TP (replicated experts), so every TP rank carries the whole TP group.
            if ps.expert_dp_size > 1:
                dist.all_reduce(g, group=ps.ep_dp_group)
            tp_dup = ps.tp_size if (ps.tp_size > 1 and ps.ep_size == 1 and ps.etp_size == 1) else 1
            g = g / (ps.cp_size * ps.dp_size * tp_dup)
        elif ps.cp_size > 1:
            dist.all_reduce(g, group=ps.cp_group)
            g = g / ps.cp_size
        if g.shape != want.shape:
            g = g[: want.shape[0]]
        r = _rel(g, want)
        all_rel[name] = r
        if r > worst:
            worst, worst_name = r, name
    n = len(grads)
    bad_local = sorted(all_rel.items(), key=lambda kv: -kv[1])[:2]
    for name, r in bad_local:  # every rank: where does the worst tensor come from and how big is it
        print(f"[{report['case']}] rank{dist.get_rank()} pp{ps.pp_rank} tp{ps.tp_rank} dp{ps.dp_rank} edp{ps.expert_dp_rank} "
              f"{name.split('layers.')[-1]} rel {r:.2e} |g| {grads[name].norm().item():.4e} |ref| {reference.grads[name].norm().item():.4e}",
              flush=True)
    if dist.get_rank() == 0:
        bad = sorted(all_rel.items(), key=lambda kv: -kv[1])[:8]
        print(f"\n[{report['case']}] weight export worst rel {w_worst:.2e}; worst grads:", [(k, f"{v:.1e}") for k, v in bad])
    # --- reduce across ranks first so every rank asserts on the same numbers (no collective deadlock)
    stats = torch.tensor(
        [loss_rel, max(layer_rel.values()), max(flips.values()) if flips else 0.0, worst, float(n), w_worst], device="cuda"
    )
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    loss_rel, layer_max, flip_max, grad_worst, n_max, w_worst = stats.tolist()
    report.update(
        loss_rel=loss_rel, layer_rel_max=layer_max, layer_rel=layer_rel, topk_flip_rows=flip_max,
        grad_checked=n, grad_worst=grad_worst, grad_worst_local=(worst, worst_name),
    )
    if dist.get_rank() == 0:
        print(
            f"\nP3 {report['case']}: loss_rel {loss_rel:.2e} | layer_rel_max {layer_max:.2e} | "
            f"topk_flips {flip_max:.3%} | grads {n} worst {grad_worst:.2e} (rank0 worst: {worst_name} {worst:.2e})"
        )
    assert n == len(reference.grads) == int(n_max), (n, len(reference.grads), n_max)
    assert w_worst == 0.0, ("weight export round-trip differs", w_worst, report)
    assert loss_rel < FP32.module_rel, report
    assert layer_max < FP32.module_rel, report
    assert flip_max == 0.0, report
    assert grad_worst < FP32.grad_rel, report
    del model, grads, dump
    torch.cuda.empty_cache()


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_tp_matches_single_rank_fp32(reference, tp):
    _check_case(reference, tp=tp)


@pytest.mark.parametrize("ep", [2, 4, 8])
def test_ep_matches_single_rank_fp32(reference, ep):
    _check_case(reference, ep=ep)


def test_pp2_matches_single_rank_fp32(reference):
    _check_case(reference, pp=2)


@pytest.mark.parametrize("tp,ep,pp", [(2, 2, 1), (2, 1, 2), (2, 2, 2), (4, 2, 1)])
def test_combined_matches_single_rank_fp32(reference, tp, ep, pp):
    _check_case(reference, tp=tp, ep=ep, pp=pp)


@pytest.mark.parametrize("cp", [2, 4, 8])
def test_cp_matches_single_rank_fp32(reference_all_msa, cp):
    # cp=8 -> zigzag chunk of 64 tokens: KV blocks straddle rank boundaries (global-order gather must fix it up)
    _check_case(reference_all_msa, cp=cp)


@pytest.mark.parametrize("tp,ep,pp,cp", [(2, 1, 1, 2), (2, 2, 1, 2), (2, 1, 2, 2), (1, 2, 1, 4)])
def test_cp_combined_matches_single_rank_fp32(reference_all_msa, tp, ep, pp, cp):
    _check_case(reference_all_msa, tp=tp, ep=ep, pp=pp, cp=cp)


def test_cp2_full_proxy_bf16_recorded(reference):
    """Dense TE attention + MSA under CP in bf16 (TE's CP has no fp32 backend): numbers are recorded, not gated."""
    from megatron.lite.model.minimax_m3.lite.checkpoint import load_hf_weights
    from megatron.lite.model.minimax_m3.lite.model import MiniMaxM3Model
    from megatron.lite.primitive.parallel import ParallelState, init_parallel, zigzag_slice_for_cp
    from megatron.lite.runtime.contracts import ParallelConfig

    dist = _init_dist_or_skip()
    cfg = reference.cfg
    ps0 = ParallelState()
    ref = MiniMaxM3Model(cfg, _train_cfg(ps0), ps0, msa_backend="flex").to(torch.bfloat16).cuda()
    load_hf_weights(ref, reference.src, cfg, ps0)
    ps = init_parallel(ParallelConfig(tp=1, ep=1, etp=None, pp=1, cp=2))
    model = MiniMaxM3Model(cfg, _train_cfg(ps), ps, msa_backend="flex").to(torch.bfloat16).cuda()
    load_hf_weights(model, reference.src, cfg, ps)
    ids = zigzag_slice_for_cp(reference.ids, ps.cp_rank, 2, seq_dim=1).contiguous()
    labels = zigzag_slice_for_cp(reference.labels, ps.cp_rank, 2, seq_dim=1).contiguous()
    with torch.no_grad():
        ref_out = ref(input_ids=reference.ids, labels=reference.labels)
        out = model(input_ids=ids, labels=labels)
    want_lp = zigzag_slice_for_cp(ref_out["log_probs"], ps.cp_rank, 2, seq_dim=1)
    lp_delta = (out["log_probs"].float() - want_lp.float()).abs()
    stats = torch.stack([lp_delta.mean(), lp_delta.max(), (out["loss"].float() - (-ref_out["log_probs"].float()).mean()).abs()])
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        print(f"\nP3 cp2 full-proxy bf16 (recorded): logprob |delta| mean {stats[0]:.3e} max {stats[1]:.3e} | "
              f"loss |delta| {stats[2]:.3e} | ref loss {(-want_lp.float()).mean():.4f}")
    assert torch.isfinite(stats).all()
