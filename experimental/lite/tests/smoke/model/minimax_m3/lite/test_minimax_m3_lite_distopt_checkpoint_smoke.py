# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax-M3 dist_opt checkpoint: a restored model continues bitwise like the original.

DP2 x EP2 with the distributed optimizer's default overlapped parameter all-gather, so after a
step each rank owns half of the updated parameters; the save must gather the other half first.

The MSA kernels need Blackwell and the Magi-MSA ``magi_attn_extensions.MSA`` backend.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytestmark = [
    pytest.mark.gpus(2, min_architecture="blackwell"),
    pytest.mark.optional,
    pytest.mark.env(CUDA_DEVICE_MAX_CONNECTIONS="1"),
]

SEQ_LENS = (3000, 5192)


@pytest.fixture(scope="module", autouse=True)
def _single_node_cuda_dist():
    pytest.importorskip(
        "magi_attn_extensions.MSA", reason="MiniMax-M3 MSA needs the Magi-MSA backend."
    )
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the MiniMax-M3 dist_opt checkpoint smoke.")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29543")

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    created_pg = False
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        created_pg = True
    yield
    try:
        from megatron.core import parallel_state as mpu

        if mpu.is_initialized():
            mpu.destroy_model_parallel()
    finally:
        if created_pg and dist.is_initialized():
            dist.destroy_process_group()


def _tiny_config():
    from megatron.lite.model.minimax_m3.config import MiniMaxM3Config

    return MiniMaxM3Config(
        num_hidden_layers=2,
        hidden_size=512,
        dense_intermediate_size=1024,
        moe_intermediate_size=256,
        shared_intermediate_size=256,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_layer_freq=[0, 1],
        sparse_attention_freq=[0, 1],
        vocab_size=512,
        max_position_embeddings=8192,
        sparse_kl_loss_coeff=0.1,
    )


def _build_handle(model_seed: int):
    from megatron.lite.model.minimax_m3.lite import protocol
    from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig
    from megatron.lite.runtime.contracts.handle import ModelHandle

    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    parallel = ParallelConfig(ep=2)
    impl_cfg = protocol.ImplConfig(
        parallel=parallel,
        optimizer="dist_opt",
        optimizer_config=OptimizerConfig(
            optimizer="adam", lr=1.0e-3, weight_decay=0.0, clip_grad=1.0
        ),
        deterministic=True,
    )
    bundle = protocol.build_model(_tiny_config(), impl_cfg=impl_cfg)
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


def _shared_tmp_path(tmp_path) -> str:
    payload = [str(tmp_path) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _random_batch(vocab_size: int):
    from megatron.lite.runtime.contracts.data import PackedBatch

    total = sum(SEQ_LENS)
    ids = torch.randint(0, vocab_size, (total,), device="cuda")
    return PackedBatch(
        input_ids=ids,
        labels=ids.clone(),
        seq_lens=torch.tensor(SEQ_LENS, dtype=torch.int64),
        loss_mask=torch.ones(total, device="cuda"),
    )


def _clone_batch(batch):
    from megatron.lite.runtime.contracts.data import PackedBatch

    return PackedBatch(
        input_ids=batch.input_ids.detach().clone(),
        labels=batch.labels.detach().clone(),
        seq_lens=batch.seq_lens.detach().clone(),
        loss_mask=batch.loss_mask.detach().clone(),
    )


def _train_step(runtime, handle, batch) -> None:
    runtime.zero_grad(handle)
    runtime.forward_backward(handle, iter([batch]), None, num_microbatches=1)
    update_ok, grad_norm, _ = runtime.optimizer_step(handle)
    assert update_ok and torch.isfinite(torch.tensor(grad_norm))
    runtime.zero_grad(handle)


def _local_named_params(handle) -> dict[str, torch.Tensor]:
    from megatron.lite.primitive.optimizers.megatron_wrap import sync_overlapped_params

    # Complete the deferred all-gather so every rank holds all of its updated parameters.
    sync_overlapped_params(handle._extras["model_chunks"])
    params: dict[str, torch.Tensor] = {}
    for chunk_idx, chunk in enumerate(handle._extras["model_chunks"]):
        for name, param in chunk.named_parameters():
            params[f"{chunk_idx}.{name}"] = param.detach().cpu().float().clone()
    return params


def _assert_params_bitwise_equal(lhs, rhs) -> None:
    lhs_params = _local_named_params(lhs)
    rhs_params = _local_named_params(rhs)
    assert lhs_params.keys() == rhs_params.keys()
    for name in lhs_params:
        torch.testing.assert_close(lhs_params[name], rhs_params[name], atol=0.0, rtol=0.0)


def test_minimax_m3_dist_opt_checkpoint_restores_and_continues_bitwise_dp2_ep2(tmp_path):
    from megatron.lite.primitive.deterministic import set_deterministic
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime

    if dist.get_world_size() != 2:
        pytest.skip("The MiniMax-M3 DP2 x EP2 checkpoint smoke requires exactly 2 GPUs.")

    set_deterministic(2026)
    vocab_size = _tiny_config().vocab_size
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    model_for_ckpt = _build_handle(model_seed=4242)
    direct_model = _build_handle(model_seed=4242)
    loaded_model = _build_handle(model_seed=4242)

    torch.manual_seed(1357 + dist.get_rank())
    torch.cuda.manual_seed_all(1357 + dist.get_rank())
    step0_batch = _random_batch(vocab_size)

    cpu_rng_before_step0 = torch.get_rng_state()
    cuda_rng_before_step0 = torch.cuda.get_rng_state()
    _train_step(runtime, model_for_ckpt, step0_batch)

    torch.set_rng_state(cpu_rng_before_step0)
    torch.cuda.set_rng_state(cuda_rng_before_step0)
    _train_step(runtime, direct_model, _clone_batch(step0_batch))

    # Save before anything else touches the pending all-gather: the save itself must complete it.
    checkpoint_dir = _shared_tmp_path(tmp_path)
    runtime.save_checkpoint(model_for_ckpt, checkpoint_dir, step=1)
    _assert_params_bitwise_equal(model_for_ckpt, direct_model)

    direct_step1_batch = _random_batch(vocab_size)
    expected_step1_batch = _clone_batch(direct_step1_batch)
    _train_step(runtime, direct_model, direct_step1_batch)

    assert runtime.load_checkpoint(loaded_model, checkpoint_dir) == 1
    loaded_step1_batch = _random_batch(vocab_size)
    assert torch.equal(loaded_step1_batch.input_ids, expected_step1_batch.input_ids)
    _train_step(runtime, loaded_model, loaded_step1_batch)

    _assert_params_bitwise_equal(direct_model, loaded_model)
