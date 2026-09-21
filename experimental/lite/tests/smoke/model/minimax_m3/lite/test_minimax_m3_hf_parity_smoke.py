# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax-M3 lite vs Hugging Face ``MiniMaxM3VLForCausalLM`` on identical random weights, bf16, one GPU.

The HF model is saved with ``save_pretrained`` and loaded into lite through the real ``load_hf_weights``
path, so structure, config mapping and weight mapping are all under test. Both MSA backends are
checked: ``flex`` (pure torch, Hopper) and the production ``magi`` protocol (msa_v1 kernels, Blackwell).

bf16 is the only precision the msa_v1 kernels support, and bf16 flips a few percent of the
indexer's top-k rows whenever the GEMM order changes; the flipped rows dominate any max-abs metric,
so hidden states are gated on cosine (0.995), logits on cosine (0.995) + KL and gradients on cosine (0.99 / 0.95), while
per-layer rel-to-max, KL and top-1 agreement are printed as evidence.
"""

from __future__ import annotations

import os
import re
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.env(CUDA_DEVICE_MAX_CONNECTIONS="1")
DEV = "cuda"
LAYER_COS = 0.995
LOGITS_COS, LOGITS_KL = 0.995, 5e-2
LOSS_REL = 2e-2
DENSE_GRAD_COS, EXPERT_GRAD_COS = 0.99, 0.95
TOPK_FLIP_BUDGET = 0.25
MAGI_CHUNK = 512


def _hf_available():
    try:
        import transformers.models.minimax_m3_vl.modeling_minimax_m3_vl  # noqa: F401

        return True
    except Exception:
        return False


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _dist():
    from megatron.lite.primitive.kernels import magi_msa

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    magi_msa.ensure_single_process_group()


def _train_config(ps):
    return SimpleNamespace(
        tp=ps.tp_size, ep=ps.ep_size, etp=ps.etp_size, pp=ps.pp_size, cp=ps.cp_size, vpp=None,
        use_deepep=False, fp8=False, recompute_modules=[], deterministic=True,
    )


class _ModuleDump:
    """Forward-hook the given modules; ``out[i]`` is the (first) output tensor of module ``i``."""

    def __init__(self, modules):
        self.out: dict[int, torch.Tensor] = {}
        self._hooks = [m.register_forward_hook(self._make(i)) for i, m in enumerate(modules)]

    def _make(self, i):
        def hook(_m, _i, out):
            self.out[i] = (out[0] if isinstance(out, tuple) else out).detach()

        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()


def _build_hf(hf_kwargs: dict, tmp_path, seed: int):
    """Random HF text model (bf16, eager) with non-trivial Gemma-norm weights and router bias, saved to disk."""
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLForCausalLM

    hf_cfg = MiniMaxM3VLTextConfig(**{k: v for k, v in hf_kwargs.items() if k != "model_type"})
    hf_cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    hf = MiniMaxM3VLForCausalLM(hf_cfg).to(DEV)
    with torch.no_grad():
        for name, p in hf.named_parameters():
            if name.endswith("norm.weight"):
                p.normal_(std=0.1)
        for layer in hf.model.layers:
            if hasattr(layer.mlp, "gate"):
                layer.mlp.gate.e_score_correction_bias.normal_(std=0.05)
    hf = hf.to(torch.bfloat16).eval()
    src = str(tmp_path / "hf_source")
    hf.save_pretrained(src, safe_serialization=True)
    return hf, hf_cfg, src


def _hf_loss(hf_logits, labels):
    # HF's built-in loss shifts labels; lite scores position t against labels[t]. Use the same unshifted CE.
    return F.cross_entropy(hf_logits.reshape(-1, hf_logits.shape[-1]).float(), labels.reshape(-1))


def _hf_grad_lookup(hf_grads: dict, mod: str) -> torch.Tensor:
    g = hf_grads.get(mod)
    if g is not None:
        return g
    em = re.match(r"^(.*\.experts)\.(\d+)\.(w1|w2|w3)\.weight$", mod)
    if em:  # HF packs experts as gate_up_proj[E] / down_proj[E]
        e = int(em.group(2))
        if em.group(3) == "w2":
            return hf_grads[f"{em.group(1)}.down_proj"][e]
        gate, up = hf_grads[f"{em.group(1)}.gate_up_proj"][e].chunk(2, dim=0)
        return gate if em.group(3) == "w1" else up
    fm = re.match(r"^(.*)\.(gate|up)_proj\.weight$", mod)
    assert fm, f"no HF gradient for {mod}"
    gate, up = hf_grads[f"{fm.group(1)}.gate_up_proj.weight"].chunk(2, dim=0)
    return gate if fm.group(2) == "gate" else up


def _lite_grads_by_hf_name(model, cfg, ps) -> dict[str, torch.Tensor]:
    """Lite parameter gradients keyed by the HF *module* name (export path re-used on ``.grad``)."""
    from megatron.lite.model.minimax_m3.lite.checkpoint import disk_to_module_name, export_hf_weights

    params = list(model.named_parameters())
    saved = [p.data for _, p in params]
    for _, p in params:
        p.data = p.grad if p.grad is not None else torch.zeros_like(p.data)
    try:
        grads = {disk_to_module_name(n): t.detach().clone() for n, t in export_hf_weights(model, cfg, ps)}
    finally:
        for (_, p), data in zip(params, saved):
            p.data = data
    return {n: g for n, g in grads.items() if "e_score_correction_bias" not in n and ".indexer." not in n}


def _compare_grads(lite_grads: dict[str, torch.Tensor], hf, tag: str):
    """Per-parameter gradient cosines; routed experts are compared per MoE layer on the concatenation of
    all their gradients (a single under-populated expert of a random-init model can swing its own tiny
    gradient under routing flips while the layer-level expert gradient is stable)."""
    hf_grads = {n: p.grad for n, p in hf.named_parameters() if p.grad is not None}
    dense, experts = {}, {}
    groups: dict[str, tuple[list, list]] = {}
    for mod, g in lite_grads.items():
        want = _hf_grad_lookup(hf_grads, mod).to(DEV)
        if want.shape != g.shape:  # vocab padding
            g = g[: want.shape[0]]
        if mod.endswith("embed_tokens.weight"):
            # The embedding gradient sums the input-gradient rows of every position holding the same token. For a
            # random-init proxy those rows cancel ~200x (the first RMSNorm amplifies the tiny embeddings), so the
            # sum sits below bf16 resolution on both sides; the input gradient itself agrees to cosine 0.9996.
            print(f"{tag} embed_tokens grad cos {_cos(g, want):.5f} (evidence only: below bf16 resolution)")
            continue
        if ".experts." in mod:
            a, b = groups.setdefault(mod.split(".experts.")[0], ([], []))
            a.append(g.float().flatten())
            b.append(want.float().flatten())
        elif mod.endswith("mlp.gate.weight"):
            experts[mod] = _cos(g, want)
        else:
            dense[mod] = _cos(g, want)
    for layer, (a, b) in groups.items():
        experts[layer + ".experts[all]"] = _cos(torch.cat(a), torch.cat(b))
    worst_dense = min(dense.items(), key=lambda kv: kv[1])
    worst_expert = min(experts.items(), key=lambda kv: kv[1])
    print(f"{tag} grads: {len(dense)} dense (worst {worst_dense[0]} cos {worst_dense[1]:.5f}), "
          f"{len(experts)} routing-coupled (worst {worst_expert[0]} cos {worst_expert[1]:.5f})")
    assert len(dense) > 10
    assert worst_dense[1] >= DENSE_GRAD_COS, worst_dense
    assert worst_expert[1] >= EXPERT_GRAD_COS, worst_expert


def _compare_layers_and_logits(tag, lite_layers, hf_layers, lite_logits, hf_logits):
    for i, (got, want) in enumerate(zip(lite_layers, hf_layers, strict=True)):
        rel, cos = _rel(got, want), _cos(got, want)
        print(f"{tag} layer={i} rel_to_max={rel:.3e} cos={cos:.6f}")
        assert cos > LAYER_COS, (i, rel, cos)
    lp_lite, lp_hf = torch.log_softmax(lite_logits.float(), -1), torch.log_softmax(hf_logits.float(), -1)
    kl = (lp_hf.exp() * (lp_hf - lp_lite)).sum(-1).mean().item()
    top1 = (lp_lite.argmax(-1) == lp_hf.argmax(-1)).float().mean().item()
    cos = _cos(lite_logits, hf_logits)
    print(f"{tag} logits cos={cos:.6f} kl(hf||lite)={kl:.3e} top1_agreement={top1:.3%}")
    assert cos > LOGITS_COS and kl < LOGITS_KL, (cos, kl)


# ----------------------------------------------------------------------------------------- flex
@pytest.mark.gpus(1)
@pytest.mark.skipif(not _hf_available(), reason="transformers without minimax_m3_vl")
def test_flex_matches_hf_forward_backward(tmp_path, flex_hf_kwargs):
    from megatron.lite.model.minimax_m3.lite import protocol as P
    from megatron.lite.model.minimax_m3.lite.model import MiniMaxM3Model
    from megatron.lite.primitive.parallel import ParallelState

    _dist()
    hf, hf_cfg, src = _build_hf(flex_hf_kwargs, tmp_path, seed=0)
    cfg = P.build_model_config(hf_cfg.to_dict())
    ps = ParallelState()
    lite = MiniMaxM3Model(cfg, _train_config(ps), ps, msa_backend="flex").to(torch.bfloat16).cuda()
    P.load_hf_weights(lite, src, cfg, ps)
    lite.eval()

    torch.manual_seed(1234)
    B, S = 2, 1024  # 8 KV blocks > top-4: the sparse layers are truly sparse
    ids = torch.randint(0, cfg.vocab_size, (B, S), device=DEV)
    labels = torch.randint(0, cfg.vocab_size, (B, S), device=DEV)
    sparse_layers = [i for i in range(cfg.num_hidden_layers) if cfg.is_sparse_attention_layer(i)]
    hf_dump, lite_dump = _ModuleDump(hf.model.layers), _ModuleDump(lite.layers)
    hf_idx = _ModuleDump([hf.model.layers[i].self_attn.indexer for i in sparse_layers])
    hf_out = hf(input_ids=ids, use_cache=False)
    lite_out = lite(input_ids=ids, labels=labels)
    for dump in (hf_dump, lite_dump, hf_idx):
        dump.remove()
    with torch.no_grad():
        lite_logits = lite.head.gather(lite.head(lite.norm(lite_dump.out[cfg.num_hidden_layers - 1]))).transpose(0, 1)
    _compare_layers_and_logits(
        "flex_vs_hf", [lite_dump.out[i].transpose(0, 1) for i in range(cfg.num_hidden_layers)],
        [hf_dump.out[i] for i in range(cfg.num_hidden_layers)], lite_logits, hf_out.logits,
    )
    for n, i in enumerate(sparse_layers):  # indexer selections as sets; bf16 may flip a few rows
        got = lite.layers[i].attn.last_block_indices.sort(-1).values
        want = hf_idx.out[n].sort(-1).values.to(torch.int32)
        flips = (got != want).any(-1).float().mean().item()
        print(f"flex_vs_hf layer={i} topk_flip_rows={flips:.3%}")
        assert flips < TOPK_FLIP_BUDGET, (i, flips)

    hf_loss = _hf_loss(hf_out.logits, labels)
    loss_rel = abs(lite_out["loss"].item() - hf_loss.item()) / hf_loss.item()
    print(f"flex_vs_hf loss lite={lite_out['loss'].item():.5f} hf={hf_loss.item():.5f} rel={loss_rel:.3e}")
    assert loss_rel < LOSS_REL
    lite_out["loss"].backward()
    hf_loss.backward()
    _compare_grads(_lite_grads_by_hf_name(lite, cfg, ps), hf, "flex_vs_hf")


@pytest.mark.gpus(1)
@pytest.mark.skipif(not _hf_available(), reason="transformers without minimax_m3_vl")
def test_weight_round_trip_is_bitwise(tmp_path, flex_hf_kwargs):
    from safetensors.torch import load_file

    from megatron.lite.model.minimax_m3.lite import protocol as P
    from megatron.lite.model.minimax_m3.lite.checkpoint import disk_to_module_name
    from megatron.lite.model.minimax_m3.lite.model import MiniMaxM3Model
    from megatron.lite.primitive.parallel import ParallelState

    _dist()
    _, hf_cfg, src = _build_hf(flex_hf_kwargs, tmp_path, seed=0)
    cfg = P.build_model_config(hf_cfg.to_dict())
    ps = ParallelState()
    lite = MiniMaxM3Model(cfg, _train_config(ps), ps, msa_backend="flex").to(torch.bfloat16).cuda()
    P.load_hf_weights(lite, src, cfg, ps)

    source = {}
    for f in os.listdir(src):
        if f.endswith(".safetensors"):
            source.update(load_file(os.path.join(src, f)))

    def hf_ref(mod):  # HF packs experts as gate_up_proj[E] / down_proj[E] and fuses dense gate_up_proj
        if mod in source:
            return source[mod]
        em = re.match(r"^(.*\.experts)\.(\d+)\.(w1|w2|w3)\.weight$", mod)
        if em:
            e, kind = int(em.group(2)), em.group(3)
            if kind == "w2":
                return source[f"{em.group(1)}.down_proj"][e]
            gate, up = source[f"{em.group(1)}.gate_up_proj"][e].chunk(2, dim=0)
            return gate if kind == "w1" else up
        fm = re.match(r"^(.*)\.(gate|up)_proj\.weight$", mod)
        assert fm, f"exported tensor {mod} has no HF counterpart"
        gate, up = source[f"{fm.group(1)}.gate_up_proj.weight"].chunk(2, dim=0)
        return gate if fm.group(2) == "gate" else up

    seen = set()
    for hf_name, tensor in P.export_hf_weights([lite], cfg, ps):
        mod = disk_to_module_name(hf_name)
        ref = hf_ref(mod)
        if tensor.shape != ref.shape:  # vocab padding
            tensor = tensor[: ref.shape[0]]
        assert torch.equal(tensor.cpu().to(ref.dtype), ref), hf_name
        seen.add(mod)
    expected = 0
    for k, t in source.items():
        if k.endswith(".experts.gate_up_proj"):
            expected += 2 * t.shape[0]
        elif k.endswith(".experts.down_proj"):
            expected += t.shape[0]
        elif k.endswith(".gate_up_proj.weight"):
            expected += 2
        else:
            expected += 1
    print(f"weight_round_trip: {len(seen)} tensors bitwise equal (expected {expected})")
    assert len(seen) == expected


# ----------------------------------------------------------------------------------------- magi
@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.skipif(not _hf_available(), reason="transformers without minimax_m3_vl")
def test_magi_matches_hf_forward_backward(tmp_path, magi_hf_kwargs):
    pytest.importorskip("magi_attn_extensions.MSA")
    pytest.importorskip("msa_v1")
    from megatron.lite.model.minimax_m3.lite import protocol as P
    from megatron.lite.primitive.kernels import magi_msa
    from megatron.lite.runtime.contracts import ParallelConfig
    from megatron.lite.runtime.contracts.data import PackedBatch

    _dist()
    hf, hf_cfg, src = _build_hf(magi_hf_kwargs, tmp_path, seed=0)
    cfg = P.build_model_config(hf_cfg.to_dict())
    bundle = P.build_model(cfg, impl_cfg=P.ImplConfig(parallel=ParallelConfig(), optimizer=None, magi_chunk_size=MAGI_CHUNK))
    lite, ps = bundle.chunks[0], bundle.parallel_state
    P.load_hf_weights(lite, src, cfg, ps)
    lite.eval()

    torch.manual_seed(1234)
    S = 4096  # 32 KV blocks > top-16
    ids = torch.randint(0, cfg.vocab_size, (S,), device=DEV)
    labels = torch.randint(0, cfg.vocab_size, (S,), device=DEV)
    seq_lens = torch.tensor([S], device=DEV)

    hf_dump = _ModuleDump(hf.model.layers)
    hf_out = hf(input_ids=ids[None], use_cache=False)
    hf_dump.remove()

    logits_batch = PackedBatch(input_ids=ids, labels=None, seq_lens=seq_lens)
    with torch.no_grad():
        lite_logits = P.unpack_forward_output(lite, logits_batch, bundle.forward_step(lite, logits_batch)["logits"]).unbind()[0]

    batch = PackedBatch(input_ids=ids, labels=labels, seq_lens=seq_lens)
    lite_dump = _ModuleDump(lite.layers)
    lite_out = bundle.forward_step(lite, batch)
    lite_dump.remove()
    ctx = P._magi_plan(lite, batch)
    assert ctx.pad == 0
    lite_layers = [magi_msa.undispatch_tokens(lite_dump.out[i][:, 0].contiguous(), ctx)[:S] for i in range(cfg.num_hidden_layers)]
    _compare_layers_and_logits("magi_vs_hf", lite_layers, [hf_dump.out[i][0] for i in range(cfg.num_hidden_layers)], lite_logits, hf_out.logits[0])

    hf_loss = _hf_loss(hf_out.logits, labels)
    loss_rel = abs(lite_out["loss"].item() - hf_loss.item()) / hf_loss.item()
    print(f"magi_vs_hf loss lite={lite_out['loss'].item():.5f} hf={hf_loss.item():.5f} rel={loss_rel:.3e}")
    assert loss_rel < LOSS_REL
    lite_out["loss"].backward()
    hf_loss.backward()
    for name, p in lite.named_parameters():  # frozen selector
        if ".indexer." in name:
            assert p.grad is None and not p.requires_grad, name
    _compare_grads(_lite_grads_by_hf_name(lite, cfg, ps), hf, "magi_vs_hf")
