from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist

from megatron.lite.primitive.deterministic import set_deterministic
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig
from megatron.lite.runtime.contracts.handle import ModelHandle

pytestmark = [pytest.mark.mlite, pytest.mark.smoke, pytest.mark.gpu, pytest.mark.distributed]


def _qwen3_moe_symbols():
    te = pytest.importorskip(
        "transformer_engine.pytorch",
        reason="Qwen3MoE distopt checkpoint smoke requires real Transformer Engine.",
    )
    assert hasattr(te, "Linear"), "Qwen3MoE smoke requires real Transformer Engine Linear."
    from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
    from megatron.lite.model.qwen3_moe.lite import protocol

    return Qwen3MoEConfig, protocol


@pytest.fixture(scope="module", autouse=True)
def _single_node_cuda_dist():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Qwen3MoE distopt checkpoint smoke tests.")
    if int(os.environ.get("WORLD_SIZE", "1")) > 8:
        pytest.skip("Megatron Lite smoke tests are capped at single-node 8 GPUs.")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29541")

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


def _topology() -> ParallelConfig:
    return ParallelConfig(tp=2, ep=2, etp=1, pp=2, cp=1)


def _tiny_qwen3_moe_config():
    Qwen3MoEConfig, _protocol = _qwen3_moe_symbols()
    return Qwen3MoEConfig(
        num_hidden_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=64,
        num_experts=4,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        max_position_embeddings=16,
        layer_types=["full_attention", "full_attention"],
    )


def _build_handle(model_seed: int) -> ModelHandle:
    _Qwen3MoEConfig, protocol = _qwen3_moe_symbols()
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)

    parallel = _topology()
    model_cfg = _tiny_qwen3_moe_config()
    impl_cfg = protocol.ImplConfig(
        parallel=parallel,
        optimizer="mc",
        optimizer_config=OptimizerConfig(
            optimizer="adam", lr=1.0e-3, weight_decay=0.0, clip_grad=1.0
        ),
        use_deepep=False,
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
    return ModelHandle(
        model=bundle.chunks,
        optimizer=bundle.optimizer,
        parallel_state=bundle.parallel_state,
        config=SimpleNamespace(parallel=parallel),
        _extras=extras,
    )


def _shared_tmp_path(tmp_path) -> str:
    payload = [str(tmp_path) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _random_batch(vocab_size: int) -> dict[str, torch.Tensor | bool]:
    return {
        "input_ids": torch.randint(0, vocab_size, (2, 4), device="cuda"),
        "labels": torch.randint(0, vocab_size, (2, 4), device="cuda"),
        "return_log_probs": False,
    }


def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _assert_batch_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert actual.keys() == expected.keys()
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if torch.is_tensor(expected_value):
            assert torch.equal(actual_value, expected_value), key
        else:
            assert actual_value == expected_value


def _train_step(runtime: MegatronLiteRuntime, handle: ModelHandle, batch: dict[str, Any]) -> None:
    runtime.zero_grad(handle)
    runtime.forward_backward(handle, iter([batch]), None, num_microbatches=1)
    runtime.optimizer_step(handle)
    runtime.zero_grad(handle)


def _local_named_params(handle: ModelHandle) -> dict[str, torch.Tensor]:
    params: dict[str, torch.Tensor] = {}
    for chunk_idx, chunk in enumerate(handle._extras["model_chunks"]):
        for name, param in chunk.named_parameters():
            params[f"{chunk_idx}.{name}"] = param.detach().cpu().float().clone()
    return params


def _assert_params_bitwise_equal(lhs: ModelHandle, rhs: ModelHandle) -> None:
    lhs_params = _local_named_params(lhs)
    rhs_params = _local_named_params(rhs)
    assert lhs_params.keys() == rhs_params.keys()
    for name in lhs_params:
        torch.testing.assert_close(lhs_params[name], rhs_params[name], atol=0.0, rtol=0.0)


def test_qwen3_moe_distopt_checkpoint_restores_rng_and_continues_bitwise_tp2_pp2_ep2(tmp_path):
    if dist.get_world_size() != 8:
        pytest.skip("Qwen3MoE tp2/pp2/ep2 distopt checkpoint smoke requires exactly 8 GPUs.")

    set_deterministic(2026)
    model_cfg = _tiny_qwen3_moe_config()
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    model_for_ckpt = _build_handle(model_seed=4242)
    direct_model = _build_handle(model_seed=4242)
    loaded_model = _build_handle(model_seed=4242)

    torch.manual_seed(1357 + dist.get_rank())
    torch.cuda.manual_seed_all(1357 + dist.get_rank())
    step0_batch = _random_batch(model_cfg.vocab_size)

    cpu_rng_before_step0 = torch.get_rng_state()
    cuda_rng_before_step0 = torch.cuda.get_rng_state()
    _train_step(runtime, model_for_ckpt, step0_batch)

    torch.set_rng_state(cpu_rng_before_step0)
    torch.cuda.set_rng_state(cuda_rng_before_step0)
    _train_step(runtime, direct_model, _clone_batch(step0_batch))
    _assert_params_bitwise_equal(model_for_ckpt, direct_model)

    checkpoint_dir = _shared_tmp_path(tmp_path)
    runtime.save_checkpoint(model_for_ckpt, checkpoint_dir, step=1)

    direct_step1_batch = _random_batch(model_cfg.vocab_size)
    expected_step1_batch = _clone_batch(direct_step1_batch)
    _train_step(runtime, direct_model, direct_step1_batch)

    assert runtime.load_checkpoint(loaded_model, checkpoint_dir) == 1
    loaded_step1_batch = _random_batch(model_cfg.vocab_size)
    _assert_batch_equal(loaded_step1_batch, expected_step1_batch)
    _train_step(runtime, loaded_model, loaded_step1_batch)

    _assert_params_bitwise_equal(direct_model, loaded_model)
