# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax-M3 context parallelism: CP2 matches CP1 on a tiny hybrid model.

Layer 0 is dense attention with a dense FFN and layer 1 is MSA with an MoE FFN, so both attention
kinds run on the MSA dispatch. Every rank holds the same weights: the CP1 model computes the whole
packed row, the CP2 model this rank's dispatched half. The CP2 loss and gradients summed over the
two ranks must match the CP1 ones up to reduction-order rounding.

The MSA kernels need Blackwell and the Magi-MSA ``magi_attn_extensions.MSA`` backend.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

pytestmark = [
    pytest.mark.gpus(2, min_architecture="blackwell"),
    pytest.mark.optional,
    pytest.mark.env(CUDA_DEVICE_MAX_CONNECTIONS="1"),
]

# Two documents of more than ``topk_blocks * block_size`` = 2048 tokens, so attention is sparse.
SEQ_LENS = (3000, 5192)
LOSS_RTOL = 1e-4
GRAD_REL_L2 = 5e-2


def _init_dist_or_skip():
    pytest.importorskip(
        "magi_attn_extensions.MSA", reason="MiniMax-M3 MSA needs the Magi-MSA backend."
    )
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the MiniMax-M3 CP smoke.")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun so CP ranks are available.")

    from megatron.lite.primitive.deterministic import set_deterministic

    set_deterministic(2026)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    if dist.get_world_size() != 2:
        pytest.skip("The MiniMax-M3 CP smoke requires exactly 2 ranks.")


def _tiny_config():
    from megatron.lite.model.minimax_m3.config import MiniMaxM3Config

    # Attention shapes are fixed by the MSA kernels (64 heads, 4 KV heads, head_dim 128). Every
    # token goes to all experts: with a random router, top-k choices sit on near-ties that CP1 and
    # CP2 rounding resolve differently, moving whole tokens between experts.
    return MiniMaxM3Config(
        num_hidden_layers=2,
        hidden_size=512,
        dense_intermediate_size=1024,
        moe_intermediate_size=256,
        shared_intermediate_size=256,
        n_routed_experts=8,
        num_experts_per_tok=8,
        moe_layer_freq=[0, 1],
        sparse_attention_freq=[0, 1],
        vocab_size=512,
        max_position_embeddings=8192,
        sparse_kl_loss_coeff=0.1,
    )


def _build_chunk(model_cfg, cp: int):
    from megatron.lite.model.minimax_m3.lite import protocol
    from megatron.lite.primitive.parallel import init_parallel
    from megatron.lite.runtime.contracts import ParallelConfig

    impl_cfg = protocol.ImplConfig(
        parallel=ParallelConfig(cp=cp), optimizer=None, deterministic=True
    )
    ps = init_parallel(impl_cfg.parallel)
    return protocol.build_chunk(model_cfg, impl_cfg, ps), ps


def _packed_batch(vocab_size: int):
    from megatron.lite.runtime.contracts.data import PackedBatch

    total = sum(SEQ_LENS)
    gen = torch.Generator().manual_seed(1234)
    ids = torch.randint(0, vocab_size, (total,), generator=gen).cuda()
    return PackedBatch(
        input_ids=ids,
        labels=ids.clone(),
        seq_lens=torch.tensor(SEQ_LENS, dtype=torch.int64),
        loss_mask=torch.ones(total, device="cuda"),
    )


def _loss_and_grads(model, batch, cp_group=None):
    """Loss sum, token count and FP32 gradients, summed over ``cp_group`` when given."""
    from megatron.lite.model.minimax_m3.lite import protocol

    model.train()
    out = model(**protocol.prepare_forward_kwargs(model, batch))
    out["loss_sum"].backward()
    loss = out["loss_sum"].detach().float()
    tokens = out["num_tokens"].detach().float()
    grads = {}
    for name, param in model.named_parameters():
        grad = torch.zeros(param.shape, dtype=torch.float32, device=param.device)
        if param.grad is not None:
            grad.add_(param.grad)
        grads[name] = grad
    if cp_group is not None:
        for tensor in (loss, tokens, *grads.values()):
            dist.all_reduce(tensor, group=cp_group)
    return loss, tokens, grads


def test_minimax_m3_tiny_model_cp2_matches_cp1_forward_and_grad():
    _init_dist_or_skip()
    model_cfg = _tiny_config()

    torch.manual_seed(1234)
    cp1_model, _ = _build_chunk(model_cfg, cp=1)
    cp2_model, cp2_ps = _build_chunk(model_cfg, cp=2)
    cp2_model.load_state_dict(cp1_model.state_dict())
    assert cp2_ps.cp_size == 2

    loss1, tokens1, grads1 = _loss_and_grads(cp1_model, _packed_batch(model_cfg.vocab_size))
    loss2, tokens2, grads2 = _loss_and_grads(
        cp2_model, _packed_batch(model_cfg.vocab_size), cp2_ps.cp_group
    )

    # Every document's last token has no next-token label.
    assert int(tokens1) == int(tokens2) == sum(SEQ_LENS) - len(SEQ_LENS)
    assert grads1.keys() == grads2.keys()
    for name, expected in grads1.items():
        norm = expected.norm()
        assert torch.isfinite(norm) and norm > 0, f"{name}: gradient norm {float(norm)}"

    loss_rel = float((loss2 - loss1).abs() / loss1.abs())
    rel_l2 = {
        name: float((grads2[name] - expected).norm() / expected.norm())
        for name, expected in grads1.items()
    }
    worst = sorted(rel_l2, key=rel_l2.get, reverse=True)[:3]
    if dist.get_rank() == 0:
        print(f"CP2 vs CP1: loss rel {loss_rel:.3e}; largest grad rel-L2:")
        for name in worst:
            print(f"  {rel_l2[name]:.3e} {name}")
    assert loss_rel <= LOSS_RTOL, f"loss rel {loss_rel:.3e}"
    assert rel_l2[worst[0]] <= GRAD_REL_L2, f"{worst[0]}: rel-L2 {rel_l2[worst[0]]:.3e}"
