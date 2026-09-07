# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.gpus(2)


def _init_dist_or_skip():
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GLM5 FSDP2 smoke.")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun so FSDP2 ranks are available.")
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("GLM5 FSDP2 smoke requires at least 2 ranks.")
    if int(os.environ.get("WORLD_SIZE", "1")) > 8:
        pytest.skip("Megatron Lite smoke tests are capped at single-node 8 GPUs.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    created_pg = False
    if not dist.is_initialized():
        dist.init_process_group("nccl")
        created_pg = True
    yield torch.device("cuda", local_rank)
    if created_pg and dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def cuda_dist():
    yield from _init_dist_or_skip()


def _tiny_config_kwargs():
    return dict(
        num_hidden_layers=2,
        hidden_size=128,
        num_attention_heads=64,
        num_key_value_heads=64,
        head_dim=256,
        vocab_size=32,
        max_position_embeddings=512,
        initializer_range=0.002,
        q_lora_rank=16,
        kv_lora_rank=512,
        qk_head_dim=256,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        index_head_dim=128,
        index_n_heads=32,
        index_topk=512,
        intermediate_size=20,
        moe_intermediate_size=6,
        first_k_dense_replace=1,
        n_routed_experts=3,
        n_shared_experts=1,
        num_experts_per_tok=3,
    )


def test_glm5_tiny_model_builds_and_steps_with_fsdp2_backend(cuda_dist):
    import torch

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite import protocol
    from megatron.lite.primitive.optimizers.fsdp2 import FSDP2Optimizer, fsdp2_available
    from megatron.lite.runtime.contracts import OptimizerConfig, ParallelConfig

    if not fsdp2_available():
        pytest.skip("Installed PyTorch does not expose FSDP2 fully_shard.")

    device = cuda_dist
    cfg = Glm5Config(**_tiny_config_kwargs())
    impl_cfg = protocol.ImplConfig(
        parallel=ParallelConfig(),
        optimizer="fsdp2",
        optimizer_config=OptimizerConfig(
            optimizer="adam", lr=1.0e-3, weight_decay=0.0, clip_grad=1.0, offload_fraction=0.0
        ),
        deterministic=True,
    )

    torch.manual_seed(20260612)
    torch.cuda.manual_seed_all(20260612)
    bundle = protocol.build_model(cfg, impl_cfg=impl_cfg)
    assert bundle.extras["optimizer_backend"] == "fsdp2"
    assert bundle.optimizer is None
    assert callable(bundle.extras["post_model_load_hook"])

    model = bundle.chunks[0]
    updates = bundle.extras["post_model_load_hook"]()
    bundle.optimizer = updates["optimizer"]
    assert isinstance(bundle.optimizer, FSDP2Optimizer)

    model.train()
    batch, seq = 1, 512
    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    bundle.optimizer.zero_grad()
    output = model(input_ids=input_ids)
    assert output["logits"].shape == (batch, seq, cfg.vocab_size)
    loss = output["logits"].float().square().mean()
    assert torch.isfinite(loss)
    loss.backward()

    success, grad_norm, _ = bundle.optimizer.step()
    assert success
    assert torch.isfinite(torch.tensor(grad_norm, device=device))
