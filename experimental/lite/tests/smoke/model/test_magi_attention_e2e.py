# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real-kernel MagiAttention E2E coverage for Megatron Lite."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

# Requires a real MagiAttention install (optional dependency): excluded from
# the standard workflow and driven by tests/run_magi_attention_e2e.sh.
pytestmark = [pytest.mark.gpus(4), pytest.mark.optional]


@pytest.fixture(scope="module", autouse=True)
def _single_node_cuda_dist():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the MagiAttention E2E test.")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in {2, 4}:
        pytest.skip("The MagiAttention E2E test requires torchrun with 2 or 4 ranks.")

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29532")
    os.environ.pop("MAGI_ATTENTION_SDPA_BACKEND", None)
    os.environ.pop("MAGI_ATTENTION_FA4_BACKEND", None)
    # FA4 (flash_attn_cute) is the Blackwell kernel backend; FFA is native sm90.
    default_backend = "fa4" if torch.cuda.get_device_capability()[0] >= 10 else "ffa"
    os.environ.setdefault("MAGI_ATTENTION_KERNEL_BACKEND", default_backend)

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    created_pg = False
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        created_pg = True
    yield
    if created_pg and dist.is_initialized():
        dist.destroy_process_group()


def _tiny_magi_qwen3_config():
    from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig

    return Qwen3MoEConfig(
        num_hidden_layers=1,
        hidden_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        vocab_size=256,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=64,
        max_position_embeddings=256,
        layer_types=["full_attention"],
    )


def _parallel_state():
    from megatron.lite.primitive.parallel import init_parallel
    from megatron.lite.runtime.contracts.config import ParallelConfig

    world_size = dist.get_world_size()
    return init_parallel(ParallelConfig(tp=1, etp=1, ep=1, pp=1, cp=world_size))


def _packed_batch(vocab_size: int):
    from megatron.lite.runtime.contracts.data import PackedBatch

    # The same unsharded packed batch is present on every CP rank. MagiAttention
    # owns the load-balanced permutation and dispatch that follows.
    seq_lens = torch.tensor([256, 192, 128, 64], dtype=torch.int32, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(20260710)
    total_tokens = int(seq_lens.sum().item())
    input_ids = torch.randint(0, vocab_size, (total_tokens,), generator=generator, device="cuda")
    return PackedBatch(
        input_ids=input_ids,
        labels=input_ids.clone(),
        loss_mask=torch.ones(total_tokens, dtype=torch.float32, device="cuda"),
        seq_lens=seq_lens,
    )


def test_qwen3_moe_magi_attention_forward_backward_and_undispatch():
    pytest.importorskip("magi_attention")
    if os.environ.get("MAGI_ATTENTION_KERNEL_BACKEND", "ffa").lower() == "fa4":
        pytest.importorskip("flash_attn_cute")

    from megatron.lite.model.protocol_utils import (
        pack_magi_forward_kwargs,
        unpack_magi_forward_output,
    )
    from megatron.lite.model.qwen3_moe.lite.model import Qwen3MoEModel
    from megatron.lite.runtime.backends.mlite.runtime import _apply_attention_backend_env

    torch.manual_seed(1234)
    _apply_attention_backend_env("magi", tag="magi-e2e")
    config = _tiny_magi_qwen3_config()
    model = Qwen3MoEModel(
        config,
        _parallel_state(),
        use_deepep=False,
        use_thd=True,
        attention_backend="magi",
    ).cuda()
    model = model.to(torch.bfloat16)
    batch = _packed_batch(config.vocab_size)

    forward_kwargs = pack_magi_forward_kwargs(model, batch)
    runtime_key = forward_kwargs["packed_seq_params"].magi_runtime_key
    assert int(runtime_key.pad_size) == 0
    assert forward_kwargs["input_ids"].numel() == batch.total_tokens // dist.get_world_size()

    output = model(**forward_kwargs)
    loss = output["loss"]
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()

    grads = [
        parameter.grad.detach().float()
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)
    local_grad_norm = torch.stack([grad.norm() for grad in grads]).sum()
    assert local_grad_norm > 0

    # Exercise the reverse layout path used by protocol consumers as well as
    # the forward/backward kernel path above.
    restored = unpack_magi_forward_output(model, batch, output["hidden_states"].detach())
    restored_lengths = torch.tensor(
        [piece.size(0) for piece in restored.unbind()], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(restored_lengths, batch.seq_lens)

    # --- hot swap: magi -> te -> magi on the same built model, same weights ---
    from megatron.lite.model.protocol_utils import pack_thd_forward_kwargs

    def _split_keys(module):
        keys = set(module.state_dict().keys())
        # TE modules register backend-internal ``_extra_state`` metadata
        # entries (fp8 history; empty of trainable content in bf16). The
        # hot-swap contract is exact invariance of parameters and buffers,
        # while _extra_state keys may appear/disappear with the te backend.
        return keys, {k for k in keys if not k.endswith("_extra_state")}

    keys_before, trainable_before = _split_keys(model)
    loss_magi = float(loss.detach().item())

    model.set_attention_backend("te")
    assert model.attention_backend == "te"
    keys_te, trainable_te = _split_keys(model)
    assert trainable_te == trainable_before, "swap must not touch parameters/buffers"
    assert all(
        k.endswith("_extra_state") for k in keys_te.symmetric_difference(keys_before)
    ), "swap may only change TE _extra_state metadata keys"
    with torch.no_grad():
        te_output = model(**pack_thd_forward_kwargs(model, batch))
    assert torch.isfinite(te_output["loss"])

    model.set_attention_backend("magi")
    keys_back, trainable_back = _split_keys(model)
    assert trainable_back == trainable_before
    assert keys_back == keys_before, "magi round trip must restore the exact key set"
    with torch.no_grad():
        magi_output_again = model(**pack_magi_forward_kwargs(model, batch))
    loss_magi_again = float(magi_output_again["loss"].item())
    # Same backend, same weights, same batch: the round trip must reproduce
    # the original loss up to nondeterministic-accumulation noise.
    assert abs(loss_magi_again - loss_magi) <= max(2e-2 * abs(loss_magi), 2e-2)

    # Make any rank-local assertion failure visible to the entire torchrun job.
    dist.all_reduce(local_grad_norm, group=model.ps.cp_group)
    assert torch.isfinite(local_grad_norm) and local_grad_norm > 0
