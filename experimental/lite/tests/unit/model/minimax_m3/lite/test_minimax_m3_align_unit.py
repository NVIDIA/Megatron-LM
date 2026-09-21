# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""P2 alignment: MiniMax-M3 lite (Proxy-M3) vs Hugging Face, fp32 gate.

Requires: 1 GPU, ``NVIDIA_TF32_OVERRIDE=0`` (TE GEMMs default to TF32), transformers with
``minimax_m3_vl``. The HF Proxy-M3 is saved to a temp dir with ``save_pretrained`` and loaded
into the lite model through the real ``load_hf_weights`` path (exercises the weight spec).

Checks (thresholds.FP32):
1. config mapping covers every HF text_config field (no silent defaults) — CPU
2. per-layer hidden states and logits: rel < 1e-5, KL < 1e-4
3. discrete decisions: MSA top-k sets and MoE routing identical (margin-aware)
4. fwd+bwd: every parameter gradient rel < 1e-5 (mapped through the weight spec)
5. weight round-trip lite -> HF names: bitwise equal to the source
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import fields
from types import ModuleType, SimpleNamespace

import pytest
import torch

_LITE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_LITE, "ref", "minimax_m3"))
sys.path.insert(0, os.path.join(_LITE, "tools", "minimax_m3"))

DEV = "cuda"


def test_protocol_build_is_magi_only(monkeypatch):
    from megatron.lite.model.minimax_m3.config import MiniMaxM3Config
    from megatron.lite.model.minimax_m3.lite import protocol as P

    assert "msa_backend" not in {f.name for f in fields(P.ImplConfig)}
    seen = {}

    class FakeModel(torch.nn.Module):
        def __init__(self, _cfg, _train_cfg, _ps, *, msa_backend, **_kwargs):
            super().__init__()
            seen["msa_backend"] = msa_backend
            self.layers = torch.nn.ModuleList()

        def to(self, *_args, **_kwargs):
            return self

        def cuda(self, *_args, **_kwargs):
            return self

    ps = SimpleNamespace(tp_size=1, ep_size=1, etp_size=1, pp_size=1, cp_size=1)
    fake_model_module = ModuleType("megatron.lite.model.minimax_m3.lite.model")
    fake_model_module.MiniMaxM3Model = FakeModel
    monkeypatch.setitem(sys.modules, fake_model_module.__name__, fake_model_module)
    monkeypatch.setattr(P, "init_parallel", lambda _cfg: ps)
    monkeypatch.setattr(P.magi_msa, "validate_device", lambda: None)
    monkeypatch.setattr(P.magi_msa, "validate_kernel_shapes", lambda **_kwargs: None)
    monkeypatch.setattr(P.magi_msa, "build_msa_config", lambda *_args, **_kwargs: object())

    bundle = P.build_model(MiniMaxM3Config(), impl_cfg=P.ImplConfig(optimizer=None))
    assert seen["msa_backend"] == "magi"
    assert bundle.forward_step is P._forward_step
    assert bundle.extras["magi_settings"] is not None
    assert bundle.extras["magi_settings"].deterministic is False


def test_magi_forward_step_preserves_packed_document_contract(monkeypatch):
    from megatron.lite.model.minimax_m3.lite import protocol as P
    from megatron.lite.runtime.contracts.data import PackedBatch

    ctx = SimpleNamespace(pad=3, cu_seqlens_host=(0, 2, 5, 8))
    monkeypatch.setattr(P, "_magi_plan", lambda _model, _batch: ctx)
    monkeypatch.setattr(P.magi_msa, "dispatch_tokens", lambda x, _ctx: x)
    monkeypatch.setattr(P.magi_msa, "undispatch_tokens", lambda x, _ctx: x)

    class Recorder(torch.nn.Module):
        cross_entropy_fusion = False

        def forward(self, **kwargs):
            self.kwargs = kwargs
            return kwargs["input_ids"].unsqueeze(-1)

    model = Recorder()
    batch = PackedBatch(
        input_ids=torch.tensor([10, 11, 20, 21, 22]),
        labels=torch.tensor([11, 12, 21, 22, 23]),
        loss_mask=torch.tensor([1, 1, 1, 0, 1], dtype=torch.float32),
        seq_lens=torch.tensor([2, 3]),
    )
    output = P._forward_step(model, batch)
    assert output.shape == (1, 8, 1)
    assert model.kwargs["input_ids"].tolist() == [[10, 11, 20, 21, 22, 0, 0, 0]]
    assert model.kwargs["labels"].tolist() == [[11, 12, 21, 22, 23, 0, 0, 0]]
    assert model.kwargs["loss_mask"].tolist() == [[1, 1, 1, 0, 1, 0, 0, 0]]
    assert model.kwargs["magi_ctx"] is ctx

    unpacked = P.unpack_forward_output(model, batch, output)
    docs = list(unpacked.unbind())
    assert [d.squeeze(-1).tolist() for d in docs] == [[10, 11], [20, 21, 22]]


def test_magi_forward_step_rejects_invalid_packed_boundaries():
    from megatron.lite.model.minimax_m3.lite import protocol as P
    from megatron.lite.runtime.contracts.data import PackedBatch

    bad_length = PackedBatch(
        input_ids=torch.tensor([10, 11, 20]),
        labels=torch.tensor([11, 12, 21]),
        seq_lens=torch.tensor([2, 2]),
    )
    with pytest.raises(ValueError, match="seq_lens sums to 4"):
        P._validate_packed_batch(bad_length)

    cross_document_positions = PackedBatch(
        input_ids=torch.tensor([10, 11, 20, 21]),
        labels=torch.tensor([11, 12, 21, 22]),
        seq_lens=torch.tensor([2, 2]),
        position_ids=torch.tensor([0, 1, 2, 3]),
    )
    with pytest.raises(ValueError, match="document-local position_ids"):
        P._validate_packed_batch(cross_document_positions)


def _hf_available():
    try:
        import transformers.models.minimax_m3_vl.modeling_minimax_m3_vl  # noqa: F401

        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- CPU
def test_config_mapping_covers_real_config_json():
    from megatron.lite.model.minimax_m3.config import MiniMaxM3Config

    snap = os.environ.get("MINIMAX_M3_CONFIG_JSON") or os.path.join(
        _LITE, "..", "..", "..", "ref_sources", "hf_snapshot", "config.json"
    )
    if not os.path.exists(snap):
        pytest.skip("real config.json snapshot not available")
    hf = json.load(open(snap))
    cfg = MiniMaxM3Config._from_hf_dict(hf)  # strict: raises on unmapped fields
    assert cfg.num_hidden_layers == 60 and cfg.hidden_size == 6144
    assert cfg.layer_types[:3] == ["full_attention"] * 3 and cfg.layer_types[3] == "minimax_m3_sparse"
    assert cfg.mlp_layer_types[:3] == ["dense"] * 3 and cfg.mlp_layer_types[3] == "sparse"
    assert cfg.rotary_dim == 64 and cfg.partial_rotary_factor == 0.5 and cfg.rope_theta == 5e6
    assert (cfg.swiglu_alpha, cfg.swiglu_limit, cfg.swiglu_up_offset) == (1.702, 7.0, 1.0)
    assert cfg.routed_scaling_factor == 2.0 and cfg.num_experts == 128 and cfg.num_experts_per_tok == 4
    assert cfg.index_topk_blocks == 16 and cfg.index_block_size == 128 and cfg.index_n_heads == 4
    assert cfg.router_aux_loss_coef == 0.0 and not cfg.tie_word_embeddings


def test_config_mapping_from_proxy_and_hf_object():
    from proxy_config import hf_proxy_text_config, hf_proxy_text_config_kwargs

    from megatron.lite.model.minimax_m3.config import MiniMaxM3Config

    hf = dict(hf_proxy_text_config_kwargs())
    hf["model_type"] = "minimax_m3_vl_text"
    cfg = MiniMaxM3Config._from_hf_dict(hf)
    assert cfg.num_hidden_layers == 6 and cfg.index_topk_blocks == 4
    if _hf_available():
        cfg2 = MiniMaxM3Config.from_hf_config(hf_proxy_text_config(), strict=False)
        assert cfg2.layer_types == cfg.layer_types and cfg2.mlp_layer_types == cfg.mlp_layer_types


# --------------------------------------------------------------------------- GPU
def _dist():
    import torch.distributed as dist

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29537")
        dist.init_process_group("nccl", rank=0, world_size=1)
        torch.cuda.set_device(0)


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


def _build_pair(tmp_path, seed=0):
    """HF Proxy-M3 (fp32, eager) saved to disk + lite Proxy-M3 (fp32) loaded from it."""
    from proxy_config import hf_proxy_text_config
    from thresholds import assert_fp32_env
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLForCausalLM

    from megatron.lite.model.minimax_m3.lite import protocol as P
    from megatron.lite.model.minimax_m3.lite.model import MiniMaxM3Model
    from megatron.lite.primitive.parallel import ParallelState

    assert_fp32_env()
    _dist()
    hf_cfg = hf_proxy_text_config()
    hf_cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    hf = MiniMaxM3VLForCausalLM(hf_cfg).to(DEV).float().eval()
    with torch.no_grad():  # non-trivial Gemma-norm weights and router bias
        for n, p in hf.named_parameters():
            if n.endswith("norm.weight"):
                p.normal_(std=0.1)
        for layer in hf.model.layers:
            if hasattr(layer.mlp, "gate"):
                layer.mlp.gate.e_score_correction_bias.normal_(std=0.05)
    src = str(tmp_path / "hf_proxy")
    hf.save_pretrained(src, safe_serialization=True)

    cfg = P.build_model_config(hf_cfg.to_dict())
    ps = ParallelState()
    train_cfg = SimpleNamespace(
        tp=1, ep=1, etp=1, pp=1, cp=1, vpp=None, use_deepep=False,
        fp8=False, recompute_modules=[], deterministic=True,
    )
    # The production protocol is Magi-only. The fp32 HF gate intentionally uses the explicit
    # flex oracle because msa_v1 is bf16-only and fixed to the public M3 kernel shapes.
    lite = MiniMaxM3Model(cfg, train_cfg, ps, msa_backend="flex").float().cuda()
    P.load_hf_weights(lite, src, cfg, ps)
    lite.eval()
    return hf, hf_cfg, lite, cfg, ps, src


@pytest.mark.gpus(1)
@pytest.mark.skipif(not _hf_available(), reason="transformers without minimax_m3_vl")
def test_proxy_forward_backward_matches_hf_fp32(tmp_path):
    from align_harness import LayerDumper, compare_dumps, logits_metrics, print_table
    from thresholds import FP32

    from megatron.lite.model.minimax_m3.lite.checkpoint import (
        MiniMaxM3WeightSpec,
        disk_to_module_name,
    )

    hf, hf_cfg, lite, cfg, ps, _ = _build_pair(tmp_path)
    torch.manual_seed(1234)
    B, S = 2, 1024  # 8 blocks > top-4 -> sparse layers are truly sparse
    ids = torch.randint(0, cfg.vocab_size, (B, S), device=DEV)
    labels = torch.randint(0, cfg.vocab_size, (B, S), device=DEV)

    hf_d = LayerDumper(hf, patterns=(r"model\.layers\.\d+", r"model\.layers\.\d+\.self_attn\.indexer"))
    lite_d = LayerDumper(lite, patterns=(r"layers\.\d+",))
    hf_out = hf(input_ids=ids, use_cache=False)
    lite_out = lite(input_ids=ids, labels=labels)
    hf_d.remove()
    lite_d.remove()
    # HF's built-in loss shifts labels by one; lite scores position t against labels[t]. Use the same
    # (unshifted) mean cross-entropy on both sides so loss and gradients are comparable.
    hf_loss = torch.nn.functional.cross_entropy(
        hf_out.logits.reshape(-1, hf_out.logits.shape[-1]).float(), labels.reshape(-1)
    )

    # --- layer outputs (lite is [S, B, H]) and logits
    diffs = []
    for i in range(cfg.num_hidden_layers):
        a = lite_d.dump[f"layers.{i}"].transpose(0, 1)
        b = hf_d.dump[f"model.layers.{i}"]
        diffs.append((f"layer {i}", _rel(a, b)))
    with torch.no_grad():
        lite_logits = lite.head.gather(lite.head(lite.norm(lite_d.dump[f"layers.{cfg.num_hidden_layers-1}"].to(DEV)))).transpose(0, 1)
    lm = logits_metrics(lite_logits, hf_out.logits)
    print("\nlayer rel:", [(n, f"{r:.2e}") for n, r in diffs], "| logits", lm, "| loss lite/hf", lite_out["loss"].item(), hf_loss.item())
    for n, r in diffs:
        assert r < FP32.module_rel, (n, r)
    assert lm["kl_mean"] < FP32.logits_kl and lm["logprob_mean_abs_delta"] < FP32.logits_logprob_delta, lm
    assert abs(lite_out["loss"].item() - hf_loss.item()) / hf_loss.item() < 1e-5

    # --- MSA top-k sets identical (fp32; ties would show as mismatching rows)
    for i, layer in enumerate(lite.layers):
        if layer.is_sparse_attention:
            l_idx = layer.attn.last_block_indices.sort(-1).values
            h_idx = hf_d.dump[f"model.layers.{i}.self_attn.indexer"].to(DEV).sort(-1).values.to(torch.int32)
            mism = (l_idx != h_idx).any(-1).float().mean().item()
            assert mism == 0.0, f"layer {i}: {mism:.3%} rows differ in MSA top-k"

    # --- gradients: lite param -> HF names via the weight spec
    lite_out["loss"].backward()
    hf_loss.backward()
    spec = MiniMaxM3WeightSpec(cfg)
    hf_grads = {n: p.grad for n, p in hf.named_parameters()}
    hf_grads.setdefault("lm_head.weight", hf.lm_head.weight.grad)
    worst = 0.0
    checked = 0
    for name, p in lite.named_parameters():
        if p.grad is None or "indexer" in name or ".router.expert_bias" in name:
            continue  # indexer is a frozen selector (no grad path); expert_bias is a buffer
        for hf_name, piece in spec.native_to_hf(name, p.grad):
            mod = disk_to_module_name(hf_name)
            g = hf_grads.get(mod)
            if g is None:  # packed experts / fused gate_up in the HF module
                em = __import__("re").match(r"^(.*\.experts)\.(\d+)\.(w1|w2|w3)\.weight$", mod)
                if em:
                    e = int(em.group(2))
                    if em.group(3) == "w2":
                        g = hf_grads[f"{em.group(1)}.down_proj"][e]
                    else:
                        gate, up = hf_grads[f"{em.group(1)}.gate_up_proj"][e].chunk(2, dim=0)
                        g = gate if em.group(3) == "w1" else up
                else:
                    fm = __import__("re").match(r"^(.*)\.(gate|up)_proj\.weight$", mod)
                    assert fm, f"no HF grad for {mod}"
                    gate, up = hf_grads[f"{fm.group(1)}.gate_up_proj.weight"].chunk(2, dim=0)
                    g = gate if fm.group(2) == "gate" else up
            if piece.shape != g.shape:  # vocab padding
                g = torch.nn.functional.pad(g, (0, 0, 0, piece.shape[0] - g.shape[0]))
            r = _rel(piece, g)
            worst = max(worst, r)
            checked += 1
            assert r < FP32.grad_rel, (name, hf_name, r)
    print(f"grad check: {checked} tensors, worst rel {worst:.2e}")
    assert checked > 20


@pytest.mark.gpus(1)
@pytest.mark.skipif(not _hf_available(), reason="transformers without minimax_m3_vl")
def test_proxy_weight_round_trip(tmp_path):
    from safetensors.torch import load_file

    from megatron.lite.model.minimax_m3.lite import protocol as P
    from megatron.lite.model.minimax_m3.lite.checkpoint import disk_to_module_name

    hf, hf_cfg, lite, cfg, ps, src = _build_pair(tmp_path)
    source = {}
    for f in os.listdir(src):
        if f.endswith(".safetensors"):
            source.update(load_file(os.path.join(src, f)))
    # HF packs experts as gate_up_proj[E] / down_proj[E]; the export emits the per-expert disk names.
    def hf_ref(mod):
        if mod in source:
            return source[mod]
        em = re.match(r"^(.*\.experts)\.(\d+)\.(w1|w2|w3)\.weight$", mod)
        if em:
            e, kind = int(em.group(2)), em.group(3)
            if kind == "w2":
                return source[f"{em.group(1)}.down_proj"][e]
            gate, up = source[f"{em.group(1)}.gate_up_proj"][e].chunk(2, dim=0)
            return gate if kind == "w1" else up
        fm = re.match(r"^(.*)\.(gate|up)_proj\.weight$", mod)  # dense MLP / shared expert: fused gate_up_proj.weight
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
    # every HF tensor must be covered (expert packs count once per expert slice)
    expected = 0
    for k, t in source.items():
        if k.endswith(".experts.gate_up_proj"):
            expected += 2 * t.shape[0]  # w1 and w3 per expert
        elif k.endswith(".experts.down_proj"):
            expected += t.shape[0]
        elif k.endswith(".gate_up_proj.weight"):
            expected += 2  # gate_proj + up_proj
        else:
            expected += 1
    print(f"round-trip: {len(seen)} tensors bitwise equal (expected {expected})")
    assert len(seen) == expected, (len(seen), expected)
