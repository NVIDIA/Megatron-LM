# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fixtures shared by the MiniMax-M3 smoke tests.

Tiny configurations use the on-disk HF ``config.json`` spelling so ``MiniMaxM3Config._from_hf_dict``
is exercised. Layers 0-1 are dense attention + dense MLP, layers 2-3 are MSA + MoE (the real model's
L0-2 / L3-59 split). ``flex_cfg`` uses small heads (8/2, top-4; sparse past 512 tokens); ``magi_cfg``
uses the msa_v1 kernel shapes (64/4 heads, 4x128 index heads, top-16; sparse past 2048 tokens).

``*_source`` fixtures save one randomly initialised bf16 model in HF format (rank 0 writes, everyone
reads) so every parallel configuration in a module starts from identical weights through the real
``load_hf_weights`` sharding path.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

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
    return dist


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
