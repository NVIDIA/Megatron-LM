# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Numerical alignment of the Megatron DeepSeek-V4.1 port against the official reference code.

Builds a small randomly initialised model with the *official* ``inference/model.py`` (from the
Hugging Face snapshot) and the same architecture on Megatron-Core (``hybrid_dsv41_stack_spec``),
copies the weights across, feeds identical tokens and reports full-vocabulary logit agreement
(mean / p95 / max KL, top-1 agreement, logit cosine) plus per-layer residual-stream cosines.

The official code fake-quantises activations (FP8 window KV, FP4 compressed KV and indexer
q/k) even in bf16 mode; ``--no-fake-quant`` patches those calls to the identity so the two
implementations are compared on identical arithmetic (the "reference-alignment" baseline),
``--fake-quant`` keeps them to measure the quantisation gap the training path does not model.

Usage (GPU, inside the training container, CP 1)::

    python tools/dsv41/align_with_reference.py --reference-dir $HF_SNAPSHOT/inference \
        --seq-len 128 --batch 2 --no-fake-quant

The weight mapping implemented here is the parameter-name contract later reused by the
Megatron-Bridge importer (milestone M2). Independent implementation.
"""

import argparse
import json
import os
import sys
from typing import Optional

import torch

# ---------------------------------------------------------------------------------------------
# Shared tiny architecture
# ---------------------------------------------------------------------------------------------

TINY = dict(
    vocab=1024,
    dim=256,
    n_layers=6,
    n_heads=64,  # FlashMLA sparse kernel geometry (fused variant); shared by both models
    head_dim=512,
    rope_dim=64,
    q_lora=64,
    o_groups=8,
    o_lora=64,
    window=32,
    ratios=[0, 0, 2, 2, 1, 1],
    kv_sources=[2, 4],
    index_sources=[2, 3, 4, 5],
    candidate_source=4,
    candidate_blocks=4,  # 4 x 2 >= index_topk: the official indexer otherwise back-fills
    candidate_block_size=2,  # its top-k with non-candidate positions
    index_heads=4,
    index_head_dim=128,
    index_topk=8,
    experts=8,
    experts_topk=2,
    moe_inter=256,
    route_scale=1.5,
    swiglu_limit=10.0,
    hc=4,
    sinkhorn=20,
    engram_layers=[1, 2],
    engram_rows=[131072, 131072],
    engram_ngram=4,
    engram_bucket=5000,
    engram_heads=4,
    engram_head_dim=32,
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    rope_factor=16,
    original_seq_len=256,
    norm_eps=1e-6,
    beta_fast=32,
    beta_slow=1,
    pad_id=2,
    engram_compressed_vocab=1024,  # identity token map for the random tiny model
)


def spec_from_text_config(cfg: dict, seq_len: int, token_map_path: str) -> dict:
    """The released architecture (``config.json`` ``text_config``, possibly truncated by
    ``real_weights.truncated_text_config``) in the shape of ``TINY``."""
    rope = cfg["rope_scaling"]
    cand = cfg.get("candidate_source_layer_id", -1)
    return dict(
        vocab=cfg["vocab_size"],
        dim=cfg["hidden_size"],
        n_layers=cfg["num_hidden_layers"],
        n_heads=cfg["num_attention_heads"],
        head_dim=cfg["head_dim"],
        rope_dim=cfg["qk_rope_head_dim"],
        q_lora=cfg["q_lora_rank"],
        o_groups=cfg["o_groups"],
        o_lora=cfg["o_lora_rank"],
        window=cfg["sliding_window"],
        ratios=list(cfg["compress_ratios"]),
        kv_sources=list(cfg["kv_source_layer_ids"]),
        index_sources=list(cfg["index_source_layer_ids"]),
        candidate_source=cand if cand >= 0 else None,
        candidate_blocks=cfg["candidate_topk_blocks"] if cand >= 0 else 0,
        candidate_block_size=cfg["candidate_block_size"] if cand >= 0 else 0,
        index_heads=cfg["index_n_heads"],
        index_head_dim=cfg["index_head_dim"],
        index_topk=cfg["index_topk"],
        experts=cfg["n_routed_experts"],
        experts_topk=cfg["num_experts_per_tok"],
        moe_inter=cfg["moe_intermediate_size"],
        route_scale=cfg["routed_scaling_factor"],
        swiglu_limit=cfg["swiglu_limit"],
        hc=cfg["hc_mult"],
        sinkhorn=cfg["hc_sinkhorn_iters"],
        engram_layers=list(cfg["engram_layer_ids"]),
        engram_rows=list(cfg["engram_num_embeddings"]),
        engram_ngram=cfg["engram_max_ngram_size"],
        engram_bucket=cfg["engram_vocab_size"],
        engram_heads=cfg["engram_n_heads"],
        engram_head_dim=cfg["engram_head_dim"],
        rope_theta=float(cfg["rope_theta"]),
        compress_rope_theta=float(cfg["compress_rope_theta"]),
        rope_factor=rope["factor"],
        original_seq_len=rope["original_max_position_embeddings"],
        norm_eps=cfg["rms_norm_eps"],
        beta_fast=rope["beta_fast"],
        beta_slow=rope["beta_slow"],
        pad_id=cfg["engram_pad_token_id"],
        engram_compressed_vocab=cfg["engram_compressed_vocab_size"],
        engram_token_map_path=token_map_path,
        grouped_gemm=True,
        max_seq_len=seq_len,
    )


def torch_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Plain-PyTorch equivalent of the official TileLang ``sparse_attn_kernel``.

    ``q`` ``[b, s, h, d]`` bf16, ``kv`` ``[b, n, d]`` bf16, ``topk_idxs`` ``[b, s, k]`` int32
    with ``-1`` for empty slots. Follows the kernel: fp32 scores, running max initialised at
    ``-1e30`` (a row without valid indices yields zeros), probabilities cast to bf16 before the
    PV product, sink added to the denominator.
    """
    b, s, h, d = q.shape
    # The official indexer may hand back indices on a different device than q.
    idx = topk_idxs.to(device=q.device, dtype=torch.long)
    valid = idx >= 0
    gathered = kv[torch.arange(b, device=q.device).view(b, 1, 1), idx.clamp_min(0)]  # [b,s,k,d]
    scores = torch.einsum("bshd,bskd->bshk", q.float(), gathered.float()) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    row_max = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
    probs = torch.exp(scores - row_max).to(torch.bfloat16).float()
    denom = probs.sum(dim=-1, keepdim=True) + torch.exp(
        attn_sink.float().view(1, 1, h, 1) - row_max
    )
    out = torch.einsum("bshk,bskd->bshd", probs, gathered.float()) / denom
    return out.to(q.dtype)


def torch_hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Plain-PyTorch equivalent of the official TileLang ``hc_split_sinkhorn_kernel``.

    ``mixes`` ``[b, s, (2 + hc) * hc]`` fp32 -> ``(pre [b,s,hc], post [b,s,hc], comb [b,s,hc,hc])``
    with the kernel's operation order: sigmoid gates, row softmax + eps, one column
    normalisation, then ``sinkhorn_iters - 1`` row/column normalisation pairs.
    """
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (m[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).view(*m.shape[:-1], hc, hc)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def build_official(
    reference_dir: str,
    fake_quant: bool,
    max_seq_len: int,
    torch_kernels: bool,
    tokenizer=None,
    max_batch_size: int = 4,
    random_init: bool = True,
):
    """Instantiate the official Transformer in bf16 (random weights unless ``random_init`` is
    off; the released weights are then loaded by ``real_weights.load_official``)."""
    sys.path.insert(0, reference_dir)
    import engram as ref_engram  # noqa: E402
    import model as ref  # noqa: E402

    if tokenizer is None:
        # Identity token map (the tokenizer only feeds the Engram compressed-id map).
        ref_engram.build_compressed_token_map = lambda tokenizer: (
            list(range(TINY["vocab"])),
            TINY["vocab"],
        )
    if not fake_quant:
        ref.act_quant = lambda x, *a, **k: x
        ref.fp4_act_quant = lambda x, *a, **k: x
    if torch_kernels:
        # The container's TileLang 0.1.8 cannot lower any kernel (NestedLoopChecker construction
        # fails inside TVM FFI, 2026-09-12); the two kernels bf16 inference needs are replaced by
        # their PyTorch transcriptions above. Fake quantisation still needs the TileLang kernels.
        if fake_quant:
            raise SystemExit("--fake-quant needs the official TileLang kernels (act_quant)")
        ref.sparse_attn = torch_sparse_attn
        ref.hc_split_sinkhorn = torch_hc_split_sinkhorn

    args = ref.ModelArgs(
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        dtype="bf16",
        expert_dtype=None,
        vocab_size=TINY["vocab"],
        dim=TINY["dim"],
        moe_inter_dim=TINY["moe_inter"],
        n_layers=TINY["n_layers"],
        n_mtp_layers=0,
        n_heads=TINY["n_heads"],
        n_routed_experts=TINY["experts"],
        n_shared_experts=1,
        n_activated_experts=TINY["experts_topk"],
        score_func="sqrtsoftplus",
        norm_topk_prob=True,
        route_scale=TINY["route_scale"],
        swiglu_limit=TINY["swiglu_limit"],
        q_lora_rank=TINY["q_lora"],
        head_dim=TINY["head_dim"],
        rope_head_dim=TINY["rope_dim"],
        norm_eps=TINY["norm_eps"],
        o_groups=TINY["o_groups"],
        o_lora_rank=TINY["o_lora"],
        window_size=TINY["window"],
        compress_ratios=tuple(TINY["ratios"]),
        kv_source_layers=tuple(TINY["kv_sources"]),
        index_source_layers=tuple(TINY["index_sources"]),
        compress_rope_theta=TINY["compress_rope_theta"],
        original_seq_len=TINY["original_seq_len"],
        rope_theta=TINY["rope_theta"],
        rope_factor=TINY["rope_factor"],
        index_n_heads=TINY["index_heads"],
        index_head_dim=TINY["index_head_dim"],
        index_topk=TINY["index_topk"],
        candidate_source_layer=(
            TINY["candidate_source"] if TINY["candidate_source"] is not None else -1
        ),
        candidate_topk_blocks=TINY["candidate_blocks"],
        candidate_block_size=TINY["candidate_block_size"],
        hc_mult=TINY["hc"],
        hc_sinkhorn_iters=TINY["sinkhorn"],
        engram_layer_ids=tuple(TINY["engram_layers"]),
        engram_num_embeddings=tuple(TINY["engram_rows"]),
        engram_max_ngram_size=TINY["engram_ngram"],
        engram_vocab_size=TINY["engram_bucket"],
        engram_n_heads=TINY["engram_heads"],
        engram_head_dim=TINY["engram_head_dim"],
        engram_pad_id=TINY["pad_id"],
        engram_compressed_vocab_size=TINY["engram_compressed_vocab"],
        beta_fast=TINY["beta_fast"],
        beta_slow=TINY["beta_slow"],
    )
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        model = ref.Transformer(args, tokenizer=tokenizer)
    torch.set_default_dtype(torch.float32)
    if random_init:
        _random_init(model)
    return ref, model, args


@torch.no_grad()
def _random_init(model: torch.nn.Module) -> None:
    """Deterministic small random init for every parameter (norm weights around 1)."""
    gen = torch.Generator(device="cuda").manual_seed(2026)
    for name, p in model.named_parameters():
        if p.dtype == torch.float8_e4m3fn:
            vals = torch.randn(p.shape, device=p.device, generator=gen) * 0.5
            p.copy_(vals.to(p.dtype))
        elif p.dtype == torch.float8_e8m0fnu:
            p.copy_(torch.ones(p.shape, device=p.device).to(p.dtype))  # scale 1
        elif name.endswith(("norm.weight", "q_norm.weight", "kv_norm.weight", "k_norm.weight")):
            p.copy_(1.0 + 0.1 * torch.randn(p.shape, device=p.device, generator=gen))
        elif "hc_" in name and name.endswith("_scale"):
            p.copy_(0.05 * torch.ones_like(p))
        elif "hc_" in name and name.endswith("_base"):
            p.copy_(0.1 * torch.randn(p.shape, device=p.device, generator=gen))
        elif name.endswith(("q_weight", "k_weight")):
            p.copy_(1.0 + 0.1 * torch.randn(p.shape, device=p.device, generator=gen))
        elif name.endswith("attn_sink"):
            p.copy_(0.1 * torch.randn(p.shape, device=p.device, generator=gen))
        elif name.endswith("gate.bias"):
            p.copy_(0.05 * torch.randn(p.shape, device=p.device, generator=gen))
        else:
            std = 0.02 if p.dim() >= 2 else 0.05
            p.copy_((std * torch.randn(p.shape, device=p.device, generator=gen)).to(p.dtype))


# ---------------------------------------------------------------------------------------------
# Megatron model
# ---------------------------------------------------------------------------------------------


def build_megatron(
    sparse_impl: str,
    ep_size: int = 1,
    seq_len: int = None,
    indexer_impl: str = "reference",
    fused_mhc: bool = False,
):
    """Megatron HybridModel of the shared architecture (TP 1, expert parallel ``ep_size``).

    ``indexer_impl`` / ``fused_mhc`` select the training-time kernel paths (cuDNN indexer scoring
    plus the Triton candidate-set kernel; merged fused mHC aggregation / residual kernels plus
    the Triton projection + RMS kernel) so their effect on the logits can be measured against
    the official reference."""
    from megatron.core.models.deepseek_v41.layer_specs import (
        build_dsv41_hybrid_layer_pattern,
        dsv41_config_kwargs_from_model_layers,
        hybrid_dsv41_stack_spec,
    )
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    config = MLATransformerConfig(
        hidden_size=TINY["dim"],
        num_attention_heads=TINY["n_heads"],
        kv_channels=TINY["head_dim"],
        use_cpu_initialization=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        q_lora_rank=TINY["q_lora"],
        qk_pos_emb_head_dim=TINY["rope_dim"],
        v_head_dim=TINY["head_dim"],
        o_groups=TINY["o_groups"],
        o_lora_rank=TINY["o_lora"],
        rope_type="rope",
        rotary_base=TINY["rope_theta"],
        rotary_scaling_factor=TINY["rope_factor"],
        original_max_position_embeddings=TINY["original_seq_len"],
        beta_fast=TINY["beta_fast"],
        beta_slow=TINY["beta_slow"],
        mscale=1.0,
        mscale_all_dim=1.0,
        multi_latent_attention=True,
        qk_layernorm=True,
        normalization="RMSNorm",
        layernorm_epsilon=TINY["norm_eps"],
        experimental_attention_variant="dsv4_hybrid",
        dsv4_version="v4.1",
        enable_hyper_connections=True,
        num_residual_streams=TINY["hc"],
        mhc_sinkhorn_iterations=TINY["sinkhorn"],
        csa_window_size=TINY["window"],
        csa_compress_rotary_base=TINY["compress_rope_theta"],
        dsa_indexer_n_heads=TINY["index_heads"],
        dsa_indexer_head_dim=TINY["index_head_dim"],
        dsa_indexer_topk=TINY["index_topk"],
        dsa_indexer_loss_coeff=0.0,
        dsa_kernel_backend="none",
        csa2_kv_source_layers=TINY["kv_sources"],
        csa2_index_source_layers=TINY["index_sources"],
        csa2_candidate_source_layer=TINY["candidate_source"],
        csa2_candidate_topk_blocks=TINY["candidate_blocks"],
        csa2_candidate_block_size=TINY["candidate_block_size"],
        csa2_sparse_attention_impl=sparse_impl,
        csa2_indexer_impl=indexer_impl,
        use_fused_mhc=fused_mhc,
        engram_layer_ids=TINY["engram_layers"],
        engram_num_embeddings=TINY["engram_rows"],
        engram_max_ngram_size=TINY["engram_ngram"],
        engram_bucket_size=TINY["engram_bucket"],
        engram_n_heads=TINY["engram_heads"],
        engram_head_dim=TINY["engram_head_dim"],
        engram_pad_token_id=TINY["pad_id"],
        engram_compressed_vocab_size=TINY["engram_compressed_vocab"],
        engram_token_map_path=TINY.get("engram_token_map_path"),
        engram_shard_group="ep" if ep_size > 1 else "none",
        expert_model_parallel_size=ep_size,
        num_moe_experts=TINY["experts"],
        moe_ffn_hidden_size=TINY["moe_inter"],
        moe_shared_expert_intermediate_size=TINY["moe_inter"],
        moe_router_topk=TINY["experts_topk"],
        moe_router_score_function="sqrtsoftplus",
        moe_router_enable_expert_bias=True,
        moe_router_topk_scaling_factor=TINY["route_scale"],
        moe_router_dtype="fp32",
        moe_grouped_gemm=TINY.get("grouped_gemm", False),
        moe_token_dispatcher_type="alltoall",
        activation_func=torch.nn.functional.silu,  # official experts are SwiGLU
        activation_func_clamp_value=TINY["swiglu_limit"],
        gated_linear_unit=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        init_method_std=0.02,
        **dsv41_config_kwargs_from_model_layers(TINY["ratios"]),
    )
    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_dsv41_stack_spec(config),
        vocab_size=TINY["vocab"],
        max_sequence_length=seq_len or TINY["original_seq_len"],
        hybrid_layer_pattern=build_dsv41_hybrid_layer_pattern(TINY["ratios"]),
        position_embedding_type="none",
    ).cuda()
    return model, config


# ---------------------------------------------------------------------------------------------
# Weight mapping (official -> Megatron), the contract reused by the Bridge importer
# ---------------------------------------------------------------------------------------------


def _dequant_engram_table(weight_fp8: torch.Tensor, scale: torch.Tensor, block: int = 32):
    rows = weight_fp8.float().unflatten(-1, (-1, block)) * scale.float().unsqueeze(-1)
    return rows.flatten(-2)


@torch.no_grad()
def copy_weights(ref_model, mg_model) -> None:
    src = dict(ref_model.named_parameters())
    src.update({k: v for k, v in ref_model.named_buffers()})
    dst = dict(mg_model.named_parameters())
    buffers = dict(mg_model.named_buffers())  # e.g. the router expert bias
    used = set()

    def put(dst_name: str, tensor: torch.Tensor):
        if dst_name in dst:
            target = dst[dst_name]
        elif dst_name in buffers:
            target = buffers[dst_name]
        else:
            import difflib

            close = difflib.get_close_matches(dst_name, list(dst) + list(buffers), n=5, cutoff=0.6)
            raise KeyError(f"Megatron parameter missing: {dst_name}; closest names: {close}")
        if target.shape != tensor.shape:
            raise ValueError(
                f"{dst_name}: shape {tuple(target.shape)} vs source {tuple(tensor.shape)}"
            )
        target.copy_(tensor.to(target.dtype))
        used.add(dst_name)

    def get(name: str) -> torch.Tensor:
        return src[name].detach()

    put("embedding.word_embeddings.weight", get("embed.weight"))
    put("output_layer.weight", get("head.weight"))
    put("decoder.final_norm.weight", get("norm.weight"))

    n = TINY["n_layers"]
    for i in range(n):
        attn = f"decoder.layers.{2 * i}"
        ffn = f"decoder.layers.{2 * i + 1}"
        inner_a = f"{attn}.inner_layer"
        inner_f = f"{ffn}.inner_layer"
        # hyper-connections: mixes layout [pre(n) | post(n) | comb(n*n)] == Megatron mapping_proj
        for prefix, mg in ((f"layers.{i}.hc_attn", attn), (f"layers.{i}.hc_ffn", ffn)):
            put(f"{mg}.hyper_connection.mapping_proj.weight", get(f"{prefix}_fn"))
            put(f"{mg}.hyper_connection.bias", get(f"{prefix}_base"))
            scale = get(f"{prefix}_scale")
            put(f"{mg}.hyper_connection.alpha_pre", scale[0:1])
            put(f"{mg}.hyper_connection.alpha_post", scale[1:2])
            put(f"{mg}.hyper_connection.alpha_res", scale[2:3])
        put(f"{inner_a}.input_layernorm.weight", get(f"layers.{i}.attn_norm.weight"))
        put(f"{inner_f}.pre_mlp_layernorm.weight", get(f"layers.{i}.ffn_norm.weight"))

        sa = f"{inner_a}.self_attention"
        put(f"{sa}.linear_q_down_proj.weight", get(f"layers.{i}.attn.wq_a.weight"))
        put(f"{sa}.q_layernorm.weight", get(f"layers.{i}.attn.q_norm.weight"))
        put(f"{sa}.linear_q_up_proj.weight", get(f"layers.{i}.attn.wq_b.weight"))
        put(f"{sa}.linear_kv_proj.weight", get(f"layers.{i}.attn.wkv.weight"))
        put(f"{sa}.kv_layernorm.weight", get(f"layers.{i}.attn.kv_norm.weight"))
        put(f"{sa}.linear_o_group_proj", get(f"layers.{i}.attn.wo_a.weight"))
        put(f"{sa}.linear_proj.weight", get(f"layers.{i}.attn.wo_b.weight"))
        core = f"{sa}.core_attention"
        put(f"{core}.attn_sink", get(f"layers.{i}.attn.attn_sink"))
        if f"layers.{i}.attn.compressor.wkv.weight" in src:
            comp = f"layers.{i}.attn.compressor"
            put(f"{core}.compressor.linear_wkv.weight", get(f"{comp}.wkv.weight"))
            put(f"{core}.compressor.norm.weight", get(f"{comp}.norm.weight"))
            if f"{comp}.wgate.weight" in src:
                put(f"{core}.compressor.linear_wgate.weight", get(f"{comp}.wgate.weight"))
        if f"layers.{i}.attn.indexer.wq_b.weight" in src:
            idx = f"layers.{i}.attn.indexer"
            put(f"{core}.indexer.linear_wq_b.weight", get(f"{idx}.wq_b.weight"))
            put(f"{core}.indexer.linear_weights_proj.weight", get(f"{idx}.weights_proj.weight"))
            if f"layers.{i}.attn.indexer.wk.weight" in src:
                put(f"{core}.indexer.linear_wk.weight", get(f"layers.{i}.attn.indexer.wk.weight"))
                put(f"{core}.indexer.k_norm.weight", get(f"layers.{i}.attn.indexer.k_norm.weight"))

        mlp = f"{inner_f}.mlp"
        put(f"{mlp}.router.weight", get(f"layers.{i}.ffn.gate.weight"))
        put(f"{mlp}.router.expert_bias", get(f"layers.{i}.ffn.gate.bias"))
        # SequentialMLP names experts ``local_experts.{e}.linear_fc1.weight``; the grouped
        # (TEGroupedMLP) layout uses ``linear_fc1.weight{e}``.
        grouped = f"{mlp}.experts.linear_fc1.weight0" in dst
        for e in range(TINY["experts"]):
            w1, w2, w3 = (get(f"layers.{i}.ffn.experts.{e}.w{j}.weight") for j in (1, 2, 3))
            if grouped:
                put(f"{mlp}.experts.linear_fc1.weight{e}", torch.cat([w1, w3], dim=0))
                put(f"{mlp}.experts.linear_fc2.weight{e}", w2)
            else:
                put(
                    f"{mlp}.experts.local_experts.{e}.linear_fc1.weight", torch.cat([w1, w3], dim=0)
                )
                put(f"{mlp}.experts.local_experts.{e}.linear_fc2.weight", w2)
        w1, w2, w3 = (get(f"layers.{i}.ffn.shared_experts.w{j}.weight") for j in (1, 2, 3))
        put(f"{mlp}.shared_experts.linear_fc1.weight", torch.cat([w1, w3], dim=0))
        put(f"{mlp}.shared_experts.linear_fc2.weight", w2)

        if f"layers.{i}.engram.wkv.weight" in src:
            eg = f"{attn}.engram"
            table = _dequant_engram_table(
                get(f"layers.{i}.engram.embed.weight"), get(f"layers.{i}.engram.embed.scale")
            )
            put(f"{eg}.embedding_rows", table)
            put(f"{eg}.linear_wkv.weight", get(f"layers.{i}.engram.wkv.weight"))
            put(f"{eg}.q_weight", get(f"layers.{i}.engram.q_weight"))
            put(f"{eg}.k_weight", get(f"layers.{i}.engram.k_weight"))

    missing = [k for k in dst if k not in used]
    if missing:
        raise RuntimeError(f"Megatron parameters not covered by the mapping: {missing[:20]}")


# ---------------------------------------------------------------------------------------------
# Forward passes and metrics
# ---------------------------------------------------------------------------------------------


@torch.no_grad()
def official_forward(ref, model, input_ids: torch.Tensor):
    """Official Transformer.forward with full-sequence logits and per-layer stream snapshots.

    Runs under the CUDA default device like the official ``generate.py``: the window index
    helper builds its tensors on the default device.
    """
    with torch.device("cuda"):
        return _official_forward(ref, model, input_ids)


def _official_forward(ref, model, input_ids: torch.Tensor):
    hashes = model.engram_hash(input_ids, 0, None) if model.engram_hash is not None else None
    h = model.embed(input_ids)
    h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    pre_mix = ref.make_identity_pre_mix(h, model.hc_mult)
    streams = []
    for i, layer in enumerate(model.layers):
        if layer.engram is not None:
            h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index, :], None)
        h, pre_mix = layer(h, 0, pre_mix, None)
        streams.append(h.float().clone())
    h = layer.hc_pre(h, pre_mix)
    logits = model.head(model.norm(h), full_logits=True)
    return logits.float(), streams


def _packed_params(batch: int, seq_len: int, device):
    """THD metadata for ``batch`` equal-length segments (the training layout; the fused sparse
    attention exists on the THD path only)."""
    from megatron.core.packed_seq_params import PackedSeqParams

    cu = torch.arange(0, (batch + 1) * seq_len, seq_len, dtype=torch.int32, device=device)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=seq_len,
        max_seqlen_kv=seq_len,
        cp_partition_mode="contiguous",
    )


def _streams_to_bshd(hidden: torch.Tensor, batch: int, n: int) -> torch.Tensor:
    """``[b*s, 1, n*C]`` packed streams -> ``[b, s, n, C]`` fp32."""
    total, _, nc = hidden.shape
    return hidden.float().view(batch, total // batch, n, nc // n).clone()


@torch.no_grad()
def megatron_forward(mg_model, input_ids: torch.Tensor):
    """HybridModel forward on the packed (THD) layout with per-model-layer stream snapshots
    (after the MoE sub-layer). Returns ``[b, s, vocab]`` logits and ``[b, s, n, C]`` streams."""
    snapshots = []
    b, s = input_ids.shape
    n = mg_model.config.num_residual_streams

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        snapshots.append(_streams_to_bshd(hidden, b, n))

    handles = [layer.register_forward_hook(hook) for layer in mg_model.decoder.layers[1::2]]
    flat_ids = input_ids.reshape(1, b * s)
    pos = torch.arange(s, device=input_ids.device).repeat(b).unsqueeze(0)
    logits = mg_model(
        flat_ids, pos, attention_mask=None, packed_seq_params=_packed_params(b, s, flat_ids.device)
    )  # [1, b*s, vocab]
    for h in handles:
        h.remove()
    return logits.float().view(b, s, -1), snapshots


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


class _NoHooks:
    """Stand-in for an absent model side (two-phase runs): attribute access returns itself and
    hook registration is a no-op, so the probe code below needs no per-side branches."""

    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self

    def register_forward_hook(self, *args, **kwargs):
        return self

    def register_forward_pre_hook(self, *args, **kwargs):
        return self

    def remove(self):
        return None


@torch.no_grad()
def probe_layer(
    ref,
    ref_model,
    mg_model,
    input_ids: torch.Tensor,
    layer_id: int,
    sharded: bool = False,
    preloaded_ref: Optional[dict] = None,
):
    """Compare the sub-layer intermediates of model layer ``layer_id`` between the two models.

    Both models run a full forward; hooks capture, for that layer, the attention input (after
    the pre-norm), the attention branch output, the streams after the attention sub-layer, the
    MoE input, the MoE branch output and the mHC coefficients computed from the layer input.
    Either model may be ``None`` (two-phase runs): the official captures can then be supplied
    as ``preloaded_ref`` (the dict returned by an earlier official-only run).
    Returns ``(rows, got_ref, got_mg)``; ``rows`` = ``[(name, cosine, max_rel_diff, note)]`` in
    execution order, empty unless both sides are available.
    """
    b, s = input_ids.shape
    n = mg_model.config.num_residual_streams if mg_model is not None else TINY["hc"]
    ref_layer = ref_model.layers[layer_id] if ref_model is not None else _NoHooks()
    mg_attn = mg_model.decoder.layers[2 * layer_id] if mg_model is not None else _NoHooks()
    mg_ffn = mg_model.decoder.layers[2 * layer_id + 1] if mg_model is not None else _NoHooks()
    got_ref, got_mg, handles = {}, {}, []

    def capture(store, name, transform=lambda t: t):
        def hook(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            store[name] = transform(out).float().clone()

        return hook

    def capture_input(store, name, transform=lambda t: t):
        # Megatron layers are called with keyword arguments (hidden_states=...), the official
        # modules positionally; register with kwargs so both forms are seen.
        def hook(module, args, kwargs):
            first = args[0] if args else kwargs["hidden_states"]
            store[name] = transform(first).float().clone()

        return hook

    def mixes_ref(x):  # x: [b, s, hc, d]
        pre, post, comb = ref_layer.hc_mixes(
            x, ref_layer.hc_attn_fn, ref_layer.hc_attn_scale, ref_layer.hc_attn_base
        )
        return torch.cat([pre, post, comb.flatten(-2)], dim=-1)

    def mixes_mg(hidden):  # hidden: [b*s, 1, n*C]
        pre, post, res = mg_attn.compute_mappings_fp32(hidden)
        cat = torch.cat([pre, post, res.flatten(-2)], dim=-1)  # [b*s, 1, n^2+2n]
        return cat.view(b, s, -1)

    sbd = lambda t: t.reshape(b, s, -1)  # Megatron [b*s, (1,) d] -> [b, s, d]
    streams = lambda t: _streams_to_bshd(t, b, n)
    heads, head_dim, rope_dim = TINY["n_heads"], TINY["head_dim"], TINY["rope_dim"]
    # Attention output before the inverse RoPE: the rotary tail is laid out differently on the
    # two sides (Megatron de-interleaves the rotated pairs), so only the content part compares.
    core_nope = lambda t: t.reshape(b, s, heads, head_dim)[..., : head_dim - rope_dim].reshape(
        b, s, -1
    )
    sa = mg_attn.inner_layer.self_attention

    # official side
    handles.append(
        ref_layer.register_forward_pre_hook(
            with_kwargs=True, hook=capture_input(got_ref, "attn_mixes", mixes_ref)
        )
    )
    if (
        getattr(ref_layer, "engram", None) is not None
        and getattr(mg_attn, "engram", None) is not None
    ):
        # Streams after the Engram injection (official: [b, s, hc, d]; Megatron: [b*s, 1, n*C]).
        handles.append(ref_layer.engram.register_forward_hook(capture(got_ref, "engram_out")))
        handles.append(mg_attn.engram.register_forward_hook(capture(got_mg, "engram_out", streams)))
    handles.append(ref_layer.attn_norm.register_forward_hook(capture(got_ref, "attn_in")))
    handles.append(ref_layer.attn.wq_b.register_forward_hook(capture(got_ref, "q_up")))  # pre-RoPE
    handles.append(
        ref_layer.attn.kv_norm.register_forward_hook(capture(got_ref, "kv_norm"))
    )  # pre-RoPE
    handles.append(
        ref_layer.attn.wo_b.register_forward_pre_hook(
            with_kwargs=True, hook=capture_input(got_ref, "wo_b_in")
        )
    )
    handles.append(ref_layer.attn.register_forward_hook(capture(got_ref, "attn_out")))
    handles.append(ref_layer.ffn_norm.register_forward_hook(capture(got_ref, "ffn_in")))
    handles.append(ref_layer.ffn.register_forward_hook(capture(got_ref, "ffn_out")))
    handles.append(ref_layer.register_forward_hook(capture(got_ref, "layer_out")))
    n_experts = TINY["experts"]

    def gate_hook(module, inputs, output):  # (weights [n, k], indices [n, k]) -> dense [n, E]
        weights, indices = output
        dense = torch.zeros(weights.size(0), n_experts, device=weights.device, dtype=torch.float32)
        got_ref["router"] = dense.scatter(1, indices.long(), weights.float())

    def router_hook(module, inputs, output):  # (probs [n, E], routing_map [n, E])
        probs = output[0]
        got_mg["router"] = probs.float().reshape(-1, n_experts).clone()

    handles.append(ref_layer.ffn.gate.register_forward_hook(gate_hook))
    shared_ref = ref_layer.ffn.shared_experts
    handles.append(
        shared_ref.register_forward_pre_hook(
            with_kwargs=True, hook=capture_input(got_ref, "shared_in")
        )
    )
    handles.append(shared_ref.w1.register_forward_hook(capture(got_ref, "shared_w1")))
    handles.append(shared_ref.w3.register_forward_hook(capture(got_ref, "shared_w3")))
    handles.append(shared_ref.register_forward_hook(capture(got_ref, "shared_out")))
    core_calls = []
    orig_sparse_attn = ref.sparse_attn if ref is not None else None

    def recording_sparse_attn(*a, **k):
        out = orig_sparse_attn(*a, **k)
        core_calls.append(out.float().clone())  # cloned before the in-place inverse RoPE
        return out

    if ref is not None:
        ref.sparse_attn = recording_sparse_attn
    # Megatron side. The official Engram runs before the layer, so the official layer input (and
    # its mixes) is post-Engram; Megatron injects Engram inside the layer, so on Engram layers
    # the mixes must be taken from the Engram output rather than from the layer input.
    if getattr(mg_attn, "engram", None) is not None:
        handles.append(
            mg_attn.engram.register_forward_hook(capture(got_mg, "attn_mixes", mixes_mg))
        )
    else:
        handles.append(
            mg_attn.register_forward_pre_hook(
                with_kwargs=True, hook=capture_input(got_mg, "attn_mixes", mixes_mg)
            )
        )
    handles.append(
        sa.register_forward_pre_hook(with_kwargs=True, hook=capture_input(got_mg, "attn_in", sbd))
    )
    handles.append(sa.linear_q_up_proj.register_forward_hook(capture(got_mg, "q_up", sbd)))
    handles.append(sa.kv_layernorm.register_forward_hook(capture(got_mg, "kv_norm", sbd)))
    handles.append(
        sa.core_attention.register_forward_hook(capture(got_mg, "core_out_nope", core_nope))
    )
    handles.append(
        sa.linear_proj.register_forward_pre_hook(
            with_kwargs=True, hook=capture_input(got_mg, "wo_b_in", sbd)
        )
    )
    handles.append(sa.register_forward_hook(capture(got_mg, "attn_out", sbd)))
    handles.append(
        mg_ffn.inner_layer.mlp.register_forward_pre_hook(
            with_kwargs=True, hook=capture_input(got_mg, "ffn_in", sbd)
        )
    )
    handles.append(mg_ffn.inner_layer.mlp.register_forward_hook(capture(got_mg, "ffn_out", sbd)))
    handles.append(mg_ffn.inner_layer.mlp.router.register_forward_hook(router_hook))
    shared_mg = mg_ffn.inner_layer.mlp.shared_experts
    handles.append(
        shared_mg.register_forward_pre_hook(
            with_kwargs=True, hook=capture_input(got_mg, "shared_in", sbd)
        )
    )
    handles.append(shared_mg.linear_fc1.register_forward_hook(capture(got_mg, "shared_fc1", sbd)))
    handles.append(
        shared_mg.linear_fc2.register_forward_pre_hook(
            with_kwargs=True, hook=capture_input(got_mg, "shared_act", sbd)
        )
    )
    handles.append(shared_mg.register_forward_hook(capture(got_mg, "shared_out", sbd)))
    if mg_model is not None:
        got_mg["shared_w2_weight"] = shared_mg.linear_fc2.weight.detach().float().clone()
    if ref_model is not None:
        got_ref["shared_w2_weight"] = shared_ref.w2.weight.detach().float().clone()
    handles.append(mg_ffn.register_forward_hook(capture(got_mg, "layer_out", streams)))
    try:
        if ref_model is not None:
            official_forward(ref, ref_model, input_ids)
        if mg_model is not None:
            megatron_forward(mg_model, input_ids)
    finally:
        if ref is not None:
            ref.sparse_attn = orig_sparse_attn
        for h in handles:
            h.remove()
    if ref_model is not None:
        if not sharded:
            got_ref["core_out_nope"] = core_nope(core_calls[layer_id])
        for key in ("shared_in", "shared_out"):
            got_ref[key] = got_ref[key].reshape(b, s, -1)
        # Megatron fc1 = [gate (w1) | up (w3)] halves; the activation recomputed from the
        # official projections with the official clamp rule (gate from above, up on both sides).
        w1, w3 = got_ref.pop("shared_w1"), got_ref.pop("shared_w3")
        got_ref["shared_fc1"] = torch.cat([w1, w3], dim=-1).reshape(b, s, -1)
        limit = TINY["swiglu_limit"]
        act = torch.nn.functional.silu(w1.clamp(max=limit)) * w3.clamp(-limit, limit)
        got_ref["shared_act"] = act.to(torch.bfloat16).float().reshape(b, s, -1)
    elif preloaded_ref is not None:
        got_ref = {k: v.to(input_ids.device) for k, v in preloaded_ref.items()}
    rows = []
    if not got_ref or not got_mg:
        return rows, got_ref, got_mg
    if sharded:
        # The official model holds 1/world of the heads / output groups: the per-head
        # projections and the pre-projection attention output are not comparable rank-locally.
        for key in ("q_up", "core_out_nope", "wo_b_in"):
            got_ref.pop(key, None)
            got_mg.pop(key, None)
    if got_ref["router"].shape == got_mg["router"].shape:
        chosen_ref, chosen_mg = got_ref["router"] > 0, got_mg["router"] > 0
        flips = (chosen_ref != chosen_mg).any(dim=1).float().mean().item()
        rows.append(
            ("router_sets", 1.0 - flips, 0.0, "fraction of tokens with identical expert sets")
        )
    for name in (
        "engram_out",
        "attn_mixes",
        "attn_in",
        "q_up",
        "kv_norm",
        "core_out_nope",
        "wo_b_in",
        "attn_out",
        "ffn_in",
        "router",
        "shared_in",
        "shared_fc1",
        "shared_act",
        "shared_w2_weight",
        "shared_out",
        "ffn_out",
        "layer_out",
    ):
        if name not in got_ref or name not in got_mg:
            continue
        a, m = got_ref[name], got_mg[name]
        if a.shape != m.shape:
            rows.append(
                (name, float("nan"), float("nan"), f"shape {tuple(a.shape)} vs {tuple(m.shape)}")
            )
            continue
        rel = ((a - m).abs().max() / a.abs().max().clamp_min(1e-6)).item()
        rows.append((name, _cos(a, m), rel, ""))
    return rows, got_ref, got_mg


def kl_metrics(ref_logits: torch.Tensor, test_logits: torch.Tensor) -> dict:
    p = torch.log_softmax(ref_logits, dim=-1)
    q = torch.log_softmax(test_logits, dim=-1)
    kl = (p.exp() * (p - q)).sum(-1).flatten()
    cos = torch.nn.functional.cosine_similarity(ref_logits.flatten(), test_logits.flatten(), dim=0)
    ref_top1 = ref_logits.argmax(-1).flatten()
    test_top1 = test_logits.argmax(-1).flatten()
    ref_top1_prob = p.exp().flatten(0, -2).gather(1, ref_top1.view(-1, 1)).squeeze(1)
    n = kl.numel()
    # Heavy-tail diagnostics: where the mean KL comes from (a few positions with large KL vs a
    # uniform shift), by position range and as an explicit list of the worst positions.
    sorted_kl = kl.sort(descending=True).values
    buckets = {}
    for lo, hi in ((0, 128), (128, 1024), (1024, n)):
        if lo < n:
            buckets[f"{lo}-{min(hi, n) - 1}"] = kl[lo:hi].mean().item()
    worst = kl.topk(min(32, n)).indices
    worst_rows = [
        {
            "pos": int(i),
            "kl": round(kl[i].item(), 4),
            "ref_top1_prob": round(ref_top1_prob[i].item(), 4),
            "top1_equal": bool(ref_top1[i] == test_top1[i]),
        }
        for i in worst.tolist()
    ]
    return {
        "kl_mean": kl.mean().item(),
        "kl_median": kl.median().item(),
        "kl_p95": kl.quantile(0.95).item(),
        "kl_p99": kl.quantile(0.99).item(),
        "kl_max": kl.max().item(),
        "kl_mean_trimmed16": sorted_kl[16:].mean().item() if n > 16 else kl.mean().item(),
        "kl_count_gt_0.5": int((kl > 0.5).sum().item()),
        "kl_count_gt_0.1": int((kl > 0.1).sum().item()),
        "kl_mean_by_position": buckets,
        "kl_worst_positions": worst_rows,
        "top1_agreement": (ref_top1 == test_top1).float().mean().item(),
        "logit_cosine": cos.item(),
        "logit_max_abs_diff": (ref_logits - test_logits).abs().max().item(),
    }


def _load_parquet_tokens(path: str, row: int, seq_len: int) -> torch.Tensor:
    """First ``seq_len`` token ids of one packed row (real token statistics for the Engram)."""
    import pyarrow.parquet as pq

    table = pq.ParquetFile(path)
    ids = None
    for i, b in enumerate(table.iter_batches(batch_size=1, columns=["input_ids"])):
        if i == row:
            ids = b.column("input_ids")[0].as_py()
            break
    if ids is None or len(ids) < seq_len:
        raise SystemExit(f"parquet row {row} missing or shorter than {seq_len}")
    return torch.tensor(ids[:seq_len], dtype=torch.long).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, help="official inference/ directory")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--fake-quant", dest="fake_quant", action="store_true")
    parser.add_argument("--no-fake-quant", dest="fake_quant", action="store_false")
    parser.set_defaults(fake_quant=False)
    parser.add_argument("--sparse-impl", choices=["reference", "fused"], default="reference")
    parser.add_argument(
        "--indexer-impl",
        choices=["reference", "fused"],
        default="reference",
        help="Megatron indexer scoring: 'fused' = cuDNN dense scoring for the candidate source "
        "plus the Triton candidate-set kernel for the reindex layers (--csa2-indexer-impl fused "
        "in training; the training default is 'reference')",
    )
    parser.add_argument(
        "--fused-mhc",
        action="store_true",
        help="Megatron mHC through the merged fused aggregation / residual kernels and the Triton "
        "projection + RMS kernel (--use-fused-mhc in training)",
    )
    parser.add_argument(
        "--dense-routing",
        action="store_true",
        help="route every token to all experts on both sides (removes top-k flips from the "
        "comparison so remaining differences are semantic)",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "official", "megatron"],
        default="both",
        help="'official': run only the official model and save its logits, stream snapshots and "
        "probe captures to --reference-out; 'megatron': run only the Megatron model and compare "
        "against --reference-in. The official model shards over world_size and its indexer / "
        "output-group dimensions cap world_size at 8, while the full Megatron model needs more "
        "GPUs, so the 40-layer real-weight alignment runs in these two phases.",
    )
    parser.add_argument("--reference-out", default=None, help="torch.save target (--mode official)")
    parser.add_argument("--reference-in", default=None, help="pack from --mode official")
    parser.add_argument(
        "--probe-layer",
        type=int,
        default=None,
        help="also compare the sub-layer intermediates of this model layer (0-based)",
    )
    parser.add_argument(
        "--torch-kernels",
        action="store_true",
        help="replace the official TileLang sparse_attn / hc_split_sinkhorn kernels by PyTorch "
        "transcriptions (needed where TileLang cannot compile)",
    )
    parser.add_argument(
        "--real-weights",
        default=None,
        help="Hugging Face snapshot directory: run the released architecture (first "
        "--num-layers layers) with its dequantised weights instead of the random tiny model; "
        "launch with torchrun, world size = expert parallel size",
    )
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument(
        "--engram-token-map", default=None, help=".pt from build_engram_token_map.py"
    )
    parser.add_argument("--input-parquet", default=None, help="packed parquet with input_ids")
    parser.add_argument("--parquet-row", type=int, default=0)
    parser.add_argument("--output", default=None, help="write metrics as JSON")
    args = parser.parse_args()

    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29577")
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(local_rank)
    log = print if rank == 0 else (lambda *a, **k: None)

    tokenizer = None
    if args.real_weights:
        import real_weights as rw  # noqa: E402  (same directory)

        snap = rw.Snapshot.open(args.real_weights)
        cfg = rw.truncated_text_config(snap.text_config, args.num_layers)
        TINY.clear()
        TINY.update(spec_from_text_config(cfg, args.seq_len, args.engram_token_map))
        if cfg["engram_layer_ids"]:
            if args.engram_token_map is None:
                raise SystemExit("--engram-token-map is required when Engram layers are loaded")
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.real_weights)
    if args.dense_routing:
        TINY["experts_topk"] = TINY["experts"]

    parallel_state.initialize_model_parallel(1, 1, expert_model_parallel_size=world)
    model_parallel_cuda_manual_seed(1)

    mode = args.mode
    if mode != "both" and not args.real_weights:
        raise SystemExit(
            "--mode official / megatron requires --real-weights (random weights "
            "cannot be reproduced across two runs)"
        )
    if mode == "official" and not args.reference_out:
        raise SystemExit("--mode official requires --reference-out")
    if mode == "megatron" and not args.reference_in:
        raise SystemExit("--mode megatron requires --reference-in")

    ref = ref_model = mg_model = None
    if mode in ("both", "official"):
        ref, ref_model, _ = build_official(
            args.reference_dir,
            args.fake_quant,
            max(args.seq_len, 8),
            args.torch_kernels,
            tokenizer=tokenizer,
            max_batch_size=max(args.batch, 1),
            random_init=not args.real_weights,
        )
        ref_model.eval()
    if mode in ("both", "megatron"):
        mg_model, _ = build_megatron(
            args.sparse_impl,
            ep_size=world,
            seq_len=max(args.seq_len, 8),
            indexer_impl=args.indexer_impl,
            fused_mhc=args.fused_mhc,
        )
        mg_model.eval()
    if args.real_weights:
        if ref_model is not None:
            rw.load_official(ref_model, snap, args.num_layers, rank, world, log=print)
        if mg_model is not None:
            rw.load_megatron(mg_model, snap, args.num_layers, rank, world, log=print)
        torch.distributed.barrier()
    else:
        copy_weights(ref_model, mg_model)

    if args.input_parquet:
        input_ids = _load_parquet_tokens(args.input_parquet, args.parquet_row, args.seq_len)
        input_ids = input_ids.repeat(args.batch, 1).cuda()
    else:
        gen = torch.Generator(device="cuda").manual_seed(11)
        input_ids = torch.randint(
            3, TINY["vocab"], (args.batch, args.seq_len), device="cuda", generator=gen
        )

    # Two-phase runs: the official pack (logits, stream snapshots, probe captures) is produced
    # by an official-only run and consumed on rank 0 of the Megatron-only run.
    ref_pack = None
    ref_sharded = world > 1
    if mode == "megatron":
        if rank == 0:
            ref_pack = torch.load(args.reference_in, map_location="cpu", weights_only=False)
            if not torch.equal(ref_pack["input_ids"], input_ids.cpu()):
                raise SystemExit("reference pack was produced for different input ids")
            if ref_pack["num_layers"] != TINY["n_layers"]:
                raise SystemExit(
                    f"reference pack has {ref_pack['num_layers']} layers, run has {TINY['n_layers']}"
                )
            ref_sharded = ref_pack.get("official_world", 1) > 1
            log(
                f"loaded reference pack {args.reference_in}: official world "
                f"{ref_pack.get('official_world')}, {len(ref_pack['streams'])} stream snapshots"
            )

    got_ref_probe = {}
    if args.probe_layer is not None:
        preloaded = None
        if ref_pack is not None:
            if ref_pack.get("probe_layer") != args.probe_layer:
                raise SystemExit(
                    f"reference pack probed layer {ref_pack.get('probe_layer')}, "
                    f"run asks for {args.probe_layer}"
                )
            preloaded = ref_pack["probe"]
        rows, got_ref_probe, _ = probe_layer(
            ref,
            ref_model,
            mg_model,
            input_ids,
            args.probe_layer,
            sharded=ref_sharded,
            preloaded_ref=preloaded,
        )
        for name, cos, rel, note in rows:
            log(
                f"probe layer {args.probe_layer} {name:>10s}: cosine {cos:.5f} max_rel {rel:.4f} {note}"
            )

    if mode == "official":
        ref_logits, ref_streams = official_forward(ref, ref_model, input_ids)
        if rank == 0:
            pack = {
                "input_ids": input_ids.cpu(),
                "logits": ref_logits.cpu(),
                "streams": [s.cpu() for s in ref_streams],
                "probe": {k: v.cpu() for k, v in got_ref_probe.items()},
                "probe_layer": args.probe_layer,
                "official_world": world,
                "num_layers": TINY["n_layers"],
                "seq_len": args.seq_len,
                "fake_quant": args.fake_quant,
                "torch_kernels": args.torch_kernels,
            }
            torch.save(pack, args.reference_out)
            print(
                f"saved reference pack to {args.reference_out}: logits {tuple(ref_logits.shape)}, "
                f"{len(ref_streams)} stream snapshots, {len(pack['probe'])} probe tensors"
            )
        torch.distributed.barrier()
        return

    if mode == "both":
        ref_logits, ref_streams = official_forward(ref, ref_model, input_ids)
    mg_logits, mg_streams = megatron_forward(mg_model, input_ids)
    if mode == "megatron":
        if rank != 0:
            torch.distributed.barrier()
            return
        ref_logits = ref_pack["logits"].to(mg_logits.device)
        ref_streams = [s.to(mg_logits.device) for s in ref_pack["streams"]]

    metrics = kl_metrics(ref_logits, mg_logits)
    per_layer = []
    for i, (a, b) in enumerate(zip(ref_streams, mg_streams)):
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        rel = ((a - b).abs().max() / a.abs().max().clamp_min(1e-6)).item()
        per_layer.append({"model_layer": i, "stream_cosine": cos, "max_rel_diff": rel})
    metrics["per_layer"] = per_layer
    metrics["fake_quant"] = args.fake_quant
    metrics["sparse_impl"] = args.sparse_impl
    metrics["indexer_impl"] = args.indexer_impl
    metrics["fused_mhc"] = args.fused_mhc
    metrics["torch_kernels"] = args.torch_kernels
    metrics["dense_routing"] = args.dense_routing
    metrics["real_weights"] = args.real_weights
    metrics["num_layers"] = TINY["n_layers"]
    metrics["seq_len"] = args.seq_len
    metrics["positions"] = int(input_ids.numel())
    if rank == 0:
        print(json.dumps(metrics, indent=2))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(metrics, f, indent=2)
    torch.distributed.barrier()


if __name__ == "__main__":
    main()
