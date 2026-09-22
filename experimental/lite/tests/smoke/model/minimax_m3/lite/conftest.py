# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fixtures shared by the MiniMax-M3 smoke tests: tiny HF-spelled configs (layers 0-1 dense, layers 2-3 MSA + MoE),
the torchrun process group, one random bf16 HF-format source per config, and gradient comparison by HF disk name."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

_NUM_LAYERS = 4
_LAYER_IS_MOE = [0, 0, 1, 1]


def _hf_kwargs(*, hidden_size: int, num_attention_heads: int, num_key_value_heads: int, topk_blocks: int) -> dict:
    return dict(
        model_type="minimax_m3_vl_text",
        vocab_size=256,
        hidden_size=hidden_size,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=128,
        dense_intermediate_size=2 * hidden_size,
        intermediate_size=hidden_size // 2,
        shared_intermediate_size=hidden_size // 2,
        n_shared_experts=1,
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_layer_freq=list(_LAYER_IS_MOE),
        scoring_func="sigmoid",
        use_routing_bias=True,
        routed_scaling_factor=2.0,
        hidden_act="swigluoai",
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        rms_norm_eps=1e-6,
        use_gemma_norm=True,
        use_qk_norm=True,
        qk_norm_type="per_head",
        rope_theta=5_000_000,
        rotary_dim=64,
        partial_rotary_factor=0.5,
        max_position_embeddings=65536,
        tie_word_embeddings=False,
        attention_output_gate=False,
        sparse_attention_config=dict(
            use_sparse_attention=True,
            sparse_index_dim=128,
            sparse_num_index_heads=num_key_value_heads,
            sparse_topk_blocks=topk_blocks,
            sparse_block_size=128,
            sparse_score_type="max",
            sparse_init_block=0,
            sparse_local_block=1,
            sparse_disable_index_value=list(_LAYER_IS_MOE),
            sparse_attention_freq=list(_LAYER_IS_MOE),
        ),
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


@pytest.fixture(scope="module")
def flex_hf_kwargs() -> dict:
    return _hf_kwargs(hidden_size=256, num_attention_heads=8, num_key_value_heads=2, topk_blocks=4)


@pytest.fixture(scope="module")
def magi_hf_kwargs() -> dict:
    return _hf_kwargs(hidden_size=512, num_attention_heads=64, num_key_value_heads=4, topk_blocks=16)


@pytest.fixture(scope="module")
def flex_cfg(flex_hf_kwargs):
    from megatron.lite.model.minimax_m3.config import MiniMaxM3Config

    return MiniMaxM3Config._from_hf_dict(flex_hf_kwargs)


@pytest.fixture(scope="module")
def magi_cfg(magi_hf_kwargs):
    from megatron.lite.model.minimax_m3.config import MiniMaxM3Config

    return MiniMaxM3Config._from_hf_dict(magi_hf_kwargs)


def _train_config(ps):
    return SimpleNamespace(
        tp=ps.tp_size, ep=ps.ep_size, etp=ps.etp_size, pp=ps.pp_size, cp=ps.cp_size, vpp=None,
        use_deepep=False, fp8=False, recompute_modules=[], deterministic=True,
    )


@pytest.fixture(scope="module")
def dist():
    """torch.distributed (NCCL) for torchrun-launched GPU smoke tests."""
    import torch.distributed as dist

    if not torch.cuda.is_available() or "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("run with torchrun on GPUs")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    yield dist
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def save_random_source(cfg, dist, tmp_path_factory, tag: str, *, seed: int) -> str:
    """Random bf16 flex model with non-trivial Gemma-norm weights and router bias, saved in HF format."""
    from megatron.lite.model.minimax_m3.lite.checkpoint import save_hf_weights
    from megatron.lite.model.minimax_m3.lite.model import MiniMaxM3Model
    from megatron.lite.primitive.parallel import ParallelState

    ps0 = ParallelState()
    torch.manual_seed(seed)
    model = MiniMaxM3Model(cfg, _train_config(ps0), ps0, msa_backend="flex").to(torch.bfloat16).cuda()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or name.endswith("layer_norm_weight"):
                p.normal_(std=0.1)
        for layer in model.layers:
            if layer.moe is not None:
                layer.moe.router.expert_bias.normal_(std=0.05)
    src = [str(tmp_path_factory.mktemp(f"minimax_m3_{tag}")) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(src, src=0)
    save_hf_weights(model, src[0], cfg, ps0)  # rank 0 writes; every rank must join (ends with a barrier)
    del model
    torch.cuda.empty_cache()
    return src[0]


@pytest.fixture(scope="module")
def flex_source(flex_cfg, dist, tmp_path_factory) -> str:
    return save_random_source(flex_cfg, dist, tmp_path_factory, "flex", seed=20260910)


@pytest.fixture(scope="module")
def magi_source(magi_cfg, dist, tmp_path_factory) -> str:
    return save_random_source(magi_cfg, dist, tmp_path_factory, "magi", seed=20260914)


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _is_routed_expert(name):
    return ".experts." in name and ".shared_experts." not in name


def _is_routing_coupled(name):
    return _is_routed_expert(name) or name.endswith("block_sparse_moe.gate.weight")


def _grads_by_hf_name(model, cfg, ps, dist=None):
    """Parameter gradients gathered across TP/EP/PP through the weight-export path, keyed by HF disk name."""
    from megatron.lite.model.minimax_m3.lite.checkpoint import export_hf_weights

    params = list(model.named_parameters())
    saved = [p.data for _, p in params]
    sp_ids = {id(p) for p in getattr(model, "sp_params", [])}
    for _, p in params:
        g = p.grad if p.grad is not None else torch.zeros_like(p.data)
        if dist is not None and id(p) in sp_ids and ps.tp_size > 1:  # SP-sharded input: DDP sums these grads over TP
            g = g.clone()
            dist.all_reduce(g, group=ps.tp_group)
        p.data = g
    try:
        grads = {n: t.detach().clone() for n, t in export_hf_weights(model, cfg, ps)}
    finally:
        for (_, p), data in zip(params, saved):
            p.data = data
    return {n: g for n, g in grads.items() if "e_score_correction_bias" not in n and "index_" not in n}


def _weights_by_hf_name(model, cfg, ps):
    from megatron.lite.model.minimax_m3.lite.checkpoint import export_hf_weights

    return {n: t.detach().clone() for n, t in export_hf_weights(model, cfg, ps) if "e_score_correction_bias" not in n}


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
    """(worst routing-independent, worst routing-coupled) as (name, cosine)."""
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


@pytest.fixture(scope="module")
def grad_tools():
    """Gradient / weight comparison helpers shared by the parity smokes."""
    return SimpleNamespace(
        grads_by_hf_name=_grads_by_hf_name, weights_by_hf_name=_weights_by_hf_name, grad_cosines=_grad_cosines,
        worst=_worst, is_routed_expert=_is_routed_expert, is_routing_coupled=_is_routing_coupled,
        indexer_weights=_indexer_weights, assert_indexer_frozen=_assert_indexer_frozen,
    )
