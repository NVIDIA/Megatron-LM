# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import os
import shutil
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytestmark = [
    pytest.mark.gpus(1),
    pytest.mark.env(CUDA_DEVICE_MAX_CONNECTIONS="1"),
]


def _qwen3_symbols():
    pytest.importorskip("transformer_engine.pytorch")
    from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
    from megatron.lite.model.qwen3_moe.lite.model import Qwen3MoEModel

    return Qwen3MoEConfig, Qwen3MoEModel


def _qwen35_symbols():
    pytest.importorskip("transformer_engine.pytorch")
    from megatron.lite.model.qwen3_5.config import Qwen35Config
    from megatron.lite.model.qwen3_5.lite.model import Qwen35Model

    return Qwen35Config, Qwen35Model


@pytest.fixture(scope="module", autouse=True)
def _single_node_cuda_dist():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Qwen lite model smoke tests.")
    if int(os.environ.get("WORLD_SIZE", "1")) > 8:
        pytest.skip("Megatron Lite smoke tests are capped at single-node 8 GPUs.")

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    created_pg = False
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        created_pg = True
    yield
    if created_pg and dist.is_initialized():
        dist.destroy_process_group()


def _tiny_qwen3_config():
    Qwen3MoEConfig, _Qwen3MoEModel = _qwen3_symbols()
    return Qwen3MoEConfig(
        num_hidden_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=64,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        max_position_embeddings=16,
        layer_types=["full_attention"],
    )


def _tiny_qwen35_config():
    Qwen35Config, _Qwen35Model = _qwen35_symbols()
    return Qwen35Config(
        num_hidden_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=64,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_num_value_heads=2,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=2,
        max_position_embeddings=16,
        partial_rotary_factor=1.0,
        mrope_section=[1, 1, 0],
        layer_types=["linear_attention"],
    )


def _parallel_state():
    from megatron.lite.primitive.parallel import init_parallel
    from megatron.lite.runtime.contracts.config import ParallelConfig

    return init_parallel(ParallelConfig(tp=1, etp=1, ep=1, pp=1, cp=1))


def _token_batch(vocab_size: int):
    torch.manual_seed(9876 + dist.get_rank())
    input_ids = torch.randint(0, vocab_size, (2, 4), device="cuda")
    labels = torch.randint(0, vocab_size, (2, 4), device="cuda")
    return input_ids, labels


def _assert_loss_and_backward(output: dict, model: torch.nn.Module):
    loss = output["loss"]
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    grad_params = [
        param for param in model.parameters() if param.requires_grad and param.grad is not None
    ]
    assert grad_params
    assert all(
        torch.isfinite(param.grad.detach().float()).all() for param in grad_params
    )


def test_qwen3_moe_lite_tiny_forward_backward_smoke():
    _Qwen3MoEConfig, Qwen3MoEModel = _qwen3_symbols()
    config = _tiny_qwen3_config()
    model = (
        Qwen3MoEModel(config, _parallel_state(), use_deepep=False)
        .cuda()
        .to(torch.bfloat16)
    )
    input_ids, labels = _token_batch(config.vocab_size)

    output = model(input_ids=input_ids, labels=labels, return_log_probs=True)

    assert output["hidden_states"].shape[-1] == config.hidden_size
    assert output["log_probs"].shape == labels.shape
    _assert_loss_and_backward(output, model)


def test_qwen35_lite_tiny_forward_backward_smoke():
    _Qwen35Config, Qwen35Model = _qwen35_symbols()
    config = _tiny_qwen35_config()
    train_config = SimpleNamespace(
        tp=1,
        ep=1,
        etp=1,
        pp=1,
        cp=1,
        vpp=None,
        use_deepep=False,
        fp8=False,
        recompute_modules=[],
        deterministic=True,
    )
    model = (
        Qwen35Model(config, train_config, _parallel_state()).cuda().to(torch.bfloat16)
    )
    input_ids, labels = _token_batch(config.vocab_size)

    output = model(input_ids=input_ids, labels=labels)

    assert output["hidden_states"].shape[-1] == config.hidden_size
    assert output["log_probs"].shape == labels.shape
    _assert_loss_and_backward(output, model)


def test_qwen35_tp2_tp4_mixed_attention_parity_and_backward():
    if dist.get_world_size() < 4 or dist.get_world_size() % 4 != 0:
        pytest.skip("Qwen3.5 TP replication smoke requires a world size divisible by 4.")

    Qwen35Config, Qwen35Model = _qwen35_symbols()
    from megatron.lite.model.qwen3_5.lite.checkpoint import (
        load_hf_weights,
        save_hf_weights,
    )
    from megatron.lite.primitive.parallel import init_parallel

    config = Qwen35Config(
        num_hidden_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        # Production Qwen3.5 needs no TP vocab padding. Keep the proxy aligned
        # so HF save/load does not intentionally zero synthetic padding rows.
        vocab_size=128,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_num_value_heads=2,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=2,
        max_position_embeddings=16,
        partial_rotary_factor=1.0,
        mrope_section=[1, 1, 0],
        layer_types=["linear_attention", "full_attention"],
    )

    path_box = [
        tempfile.mkdtemp(prefix="mlite-qwen35-tp-replication-")
        if dist.get_rank() == 0
        else None
    ]
    dist.broadcast_object_list(path_box, src=0)
    checkpoint_dir = path_box[0]

    def build(tp: int):
        ps = init_parallel(SimpleNamespace(tp=tp, ep=1, etp=1, pp=1, cp=1))
        train_config = SimpleNamespace(
            tp=tp,
            ep=1,
            etp=1,
            pp=1,
            cp=1,
            vpp=None,
            use_deepep=False,
            fp8=False,
            recompute_modules=[],
            deterministic=True,
        )
        model = Qwen35Model(config, train_config, ps).cuda().to(torch.bfloat16)
        return model, ps

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    tp2_model, tp2_ps = build(2)
    save_hf_weights(tp2_model, checkpoint_dir, config, tp2_ps)
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    labels = torch.tensor([[2, 3, 4, 5]], device="cuda")
    with torch.no_grad():
        tp2_logits = tp2_model(input_ids=input_ids)["logits"].float()
        tp2_loss = tp2_model(input_ids=input_ids, labels=labels)["loss"].float()
    del tp2_model
    torch.cuda.empty_cache()
    dist.barrier()

    torch.manual_seed(4321)
    torch.cuda.manual_seed_all(4321)
    tp4_model, tp4_ps = build(4)
    load_hf_weights(tp4_model, checkpoint_dir, config, tp4_ps)
    tp4_logits = tp4_model(input_ids=input_ids)["logits"].float()
    tp4_output = tp4_model(input_ids=input_ids, labels=labels)

    if dist.get_rank() == 0:
        logits_max_abs = (tp4_logits - tp2_logits).abs().max().item()
        loss_abs = (tp4_output["loss"].float() - tp2_loss).abs().item()
        print(f"QWEN35_TP_PARITY logits_max_abs={logits_max_abs:.8g} loss_abs={loss_abs:.8g}")
    torch.testing.assert_close(tp4_logits, tp2_logits, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(tp4_output["loss"].float(), tp2_loss, atol=1e-3, rtol=1e-3)
    tp4_output["loss"].backward()

    gdn = tp4_model.layers[0].linear_attn
    assert gdn is not None
    for state in (gdn.dt_bias, gdn.A_log):
        assert state.grad is not None
        gathered = [torch.empty_like(state.grad) for _ in range(tp4_ps.tp_size)]
        dist.all_gather(gathered, state.grad, group=tp4_ps.tp_group)
        for replica_grad in gathered[1:]:
            torch.testing.assert_close(replica_grad, gathered[0], atol=0, rtol=0)

    dist.barrier()
    if dist.get_rank() == 0:
        shutil.rmtree(checkpoint_dir)
