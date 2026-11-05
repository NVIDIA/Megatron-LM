# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax-M3 training smoke on CP2 x PP2 x DP2 x EP2 with the distributed optimizer.

Four layers (dense attention + dense FFN, then MSA layers, the last two with MoE) split ``Ett|ttL``
over two pipeline stages, 1F1B over four microbatches per data-parallel rank, per-token loss
normalisation over DP x CP. The documents follow a fixed bigram chain, so the next token is
learnable. Steps on the same data must start within 25% of ``ln(vocab)``, lower the loss every
step, and keep replicas identical: once the next forward has waited for the overlapped parameter
all-gather, every DP x CP (experts: expert-DP) replica holds the same values bit for bit.

The MSA kernels need Blackwell and the Magi-MSA ``magi_attn_extensions.MSA`` backend.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytestmark = [
    pytest.mark.gpus(8, min_architecture="blackwell"),
    pytest.mark.optional,
    pytest.mark.env(CUDA_DEVICE_MAX_CONNECTIONS="1"),
]

SEQ_LENS = (3000, 5192)
MICROBATCHES = 4
STEPS = 3


def _init_dist_or_skip():
    pytest.importorskip(
        "magi_attn_extensions.MSA", reason="MiniMax-M3 MSA needs the Magi-MSA backend."
    )
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the MiniMax-M3 parallel training smoke.")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun so all parallel ranks are available.")

    from megatron.lite.primitive.deterministic import set_deterministic

    set_deterministic(2026)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    if dist.get_world_size() != 8:
        pytest.skip("The MiniMax-M3 CP2 x PP2 x DP2 x EP2 smoke requires exactly 8 GPUs.")


def _tiny_config():
    from megatron.lite.model.minimax_m3.config import MiniMaxM3Config

    return MiniMaxM3Config(
        num_hidden_layers=4,
        hidden_size=512,
        dense_intermediate_size=1024,
        moe_intermediate_size=256,
        shared_intermediate_size=256,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_layer_freq=[0, 0, 1, 1],
        sparse_attention_freq=[0, 1, 1, 1],
        vocab_size=512,
        max_position_embeddings=8192,
        sparse_kl_loss_coeff=0.1,
    )


def _build_handle(model_cfg):
    from megatron.lite.model.minimax_m3.lite import protocol
    from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig
    from megatron.lite.runtime.contracts.handle import ModelHandle

    torch.manual_seed(4242)
    torch.cuda.manual_seed_all(4242)
    parallel = ParallelConfig(cp=2, pp=2, ep=2, pp_layout="Ett|ttL")
    impl_cfg = protocol.ImplConfig(
        parallel=parallel,
        optimizer="dist_opt",
        optimizer_config=OptimizerConfig(
            optimizer="adam", lr=1.0e-3, weight_decay=0.0, clip_grad=1.0
        ),
        deterministic=True,
    )
    bundle = protocol.build_model(model_cfg, impl_cfg=impl_cfg)
    extras = dict(bundle.extras)
    extras.update(
        {
            "model_chunks": bundle.chunks,
            "forward_step": bundle.forward_step,
            "finalize_grads": bundle.finalize_grads,
            "protocol": protocol,
        }
    )
    # As MegatronLiteRuntime.build_model: a single chunk is the model itself.
    model = bundle.chunks[0] if len(bundle.chunks) == 1 else bundle.chunks
    return ModelHandle(
        model=model,
        optimizer=bundle.optimizer,
        parallel_state=bundle.parallel_state,
        config=SimpleNamespace(parallel=parallel),
        _extras=extras,
    )


def _microbatches(vocab_size: int, dp_rank: int):
    """This data-parallel rank's microbatches; every CP and PP rank of it gets the same rows."""
    from megatron.lite.runtime.contracts.data import PackedBatch

    total = sum(SEQ_LENS)
    # Tokens walk one fixed random cycle through the vocabulary, from a random start per row.
    cycle = torch.randperm(vocab_size, generator=torch.Generator().manual_seed(7))
    gen = torch.Generator().manual_seed(1000 + dp_rank)
    starts = torch.randint(0, vocab_size, (MICROBATCHES,), generator=gen)
    rows = [cycle[(start + torch.arange(total)) % vocab_size] for start in starts]
    return [
        PackedBatch(
            input_ids=row.cuda(),
            labels=row.cuda(),
            seq_lens=torch.tensor(SEQ_LENS, dtype=torch.int64),
            loss_mask=torch.ones(total, device="cuda"),
        )
        for row in rows
    ]


def _assert_replicas_identical(handle) -> None:
    """Each parameter equals its copy on all its replicas (dense: DP x CP, experts: expert-DP)."""
    from megatron.lite.model.minimax_m3.lite import protocol

    ps = handle._parallel_state
    for chunk in handle._extras["model_chunks"]:
        for name, param in chunk.named_parameters():
            group = ps.ep_dp_group if protocol.is_expert_param(name) else ps.dp_cp_group
            size = dist.get_world_size(group)
            local = param.detach().reshape(-1).contiguous()
            gathered = torch.empty(size * local.numel(), dtype=local.dtype, device=local.device)
            dist.all_gather_into_tensor(gathered, local, group=group)
            for copy in gathered.chunk(size):
                assert torch.equal(copy, local), f"{name} differs across replicas"


def test_minimax_m3_cp2_pp2_dp2_ep2_training_steps_smoke():
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime

    _init_dist_or_skip()
    model_cfg = _tiny_config()
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    handle = _build_handle(model_cfg)
    ps = handle._parallel_state
    assert (ps.cp_size, ps.pp_size, ps.dp_size, ps.ep_size) == (2, 2, 2, 2)

    sums: list[tuple[torch.Tensor, torch.Tensor]] = []
    forward_step = handle._extras["forward_step"]

    def recording_step(model, batch):
        out = forward_step(model, batch)
        if "loss_sum" in out:
            sums.append((out["loss_sum"].detach().double(), out["num_tokens"].detach().double()))
        return out

    handle._extras["forward_step"] = recording_step
    batches = _microbatches(model_cfg.vocab_size, ps.dp_rank)

    losses = []
    for step in range(STEPS):
        sums.clear()
        runtime.zero_grad(handle)
        runtime.forward_backward(handle, iter(batches), None, num_microbatches=MICROBATCHES)
        if step > 0:
            # The forward has waited for every bucket of the previous step's parameter all-gather.
            _assert_replicas_identical(handle)
        update_ok, grad_norm, _ = runtime.optimizer_step(handle)
        assert update_ok and math.isfinite(grad_norm) and grad_norm > 0

        # Only last-stage ranks hold losses; summing over the world counts each DP x CP shard once.
        totals = torch.zeros(2, dtype=torch.float64, device="cuda")
        for loss_sum, num_tokens in sums:
            totals[0] += loss_sum
            totals[1] += num_tokens
        dist.all_reduce(totals)
        assert int(totals[1]) == ps.dp_size * MICROBATCHES * (sum(SEQ_LENS) - len(SEQ_LENS))
        losses.append(float(totals[0] / totals[1]))

    if dist.get_rank() == 0:
        print(f"CP2 x PP2 x DP2 x EP2 losses: {losses}")
    assert all(math.isfinite(loss) for loss in losses)
    # A random model is close to uniform; a DP or CP normalisation error is a factor of 2 or more.
    assert abs(losses[0] / math.log(model_cfg.vocab_size) - 1) < 0.25
    assert all(prev > cur for prev, cur in zip(losses, losses[1:])), losses
