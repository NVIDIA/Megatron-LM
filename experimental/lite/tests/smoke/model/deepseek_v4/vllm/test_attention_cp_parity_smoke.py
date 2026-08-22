from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.gpus(2)


def _grads(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    }


def _normalized_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    error = torch.linalg.vector_norm(reference.float() - candidate.float())
    norm = torch.linalg.vector_norm(reference.float()).clamp_min(1.0e-12)
    return float(error / norm)


def test_vllm_attention_cp2_matches_cp1_forward_backward_and_saves_memory() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.model.deepseek_v4.vllm.primitive.attention.module import (
        VLLMAttention,
    )
    from megatron.lite.model.deepseek_v4.vllm.primitive.attention.metadata import (
        AttentionMetadataBuilder,
        _build_rope,
    )
    from megatron.lite.primitive.parallel import ParallelState
    from megatron.lite.primitive.utils.packed_seq import PackedSeqParams
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance

    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("nccl")
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    assert dist.get_world_size(group) == 2
    init_batch_invariance()

    config = DeepseekV4Config(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=64,
        head_dim=512,
        q_lora_rank=128,
        o_lora_rank=128,
        o_groups=8,
        index_head_dim=128,
        index_n_heads=64,
        index_topk=512,
        compress_ratios=[1, 4],
        sliding_window=128,
        max_position_embeddings=4096,
    )
    cp1 = ParallelState(cp_size=1, cp_rank=0, cp_group=None)
    attention = VLLMAttention(config, ps=cp1, layer_idx=1).cuda().bfloat16()
    torch.manual_seed(7701)
    for name, parameter in attention.named_parameters():
        if name.endswith(("q_norm.weight", "kv_norm.weight", "norm.weight")):
            parameter.data.fill_(1)
        elif name.endswith("sinks"):
            parameter.data.zero_()
        else:
            parameter.data.normal_(0, 0.02)

    seq_len = 1024
    positions = torch.arange(seq_len, device="cuda", dtype=torch.int64)
    cu = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)
    packed = PackedSeqParams.from_cu_seqlens(cu, max_seqlen=seq_len)
    hf_config = SimpleNamespace(
        rope_theta=10_000.0,
        compress_rope_theta=160_000.0,
        rope_parameters={
            "rope_type": "deepseek_yarn",
            "factor": 16.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
    )
    builder = AttentionMetadataBuilder(
        config,
        layer_idx=1,
        cos_sin_cache=_build_rope(
            hf_config,
            config,
            compress_ratio=4,
            device=torch.device("cuda"),
        ),
    )
    generator = torch.Generator(device="cuda").manual_seed(7711)
    full = torch.randn(
        seq_len, config.hidden_size, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    cotangent = torch.randn(
        seq_len, config.hidden_size, generator=generator, device="cuda", dtype=torch.bfloat16
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    full_input = full.detach().clone().requires_grad_(True)
    cp1_output = attention(full_input, metadata=builder.build(positions, packed))
    (cp1_output * cotangent).sum().backward()
    cp1_peak = torch.cuda.max_memory_allocated() - baseline
    cp1_forward = cp1_output.detach().clone()
    cp1_input_grad = full_input.grad.detach().clone()
    cp1_grads = _grads(attention)

    attention.zero_grad(set_to_none=True)
    del cp1_output, full_input
    torch.cuda.empty_cache()
    dist.barrier(group=group)

    local_rows = seq_len // 2
    start = rank * local_rows
    attention.ps = ParallelState(
        cp_size=2, cp_rank=rank, cp_group=group, cp_global_ranks=[0, 1]
    )
    local_input = full[start : start + local_rows].detach().clone().requires_grad_(True)
    local_positions = positions[start : start + local_rows]
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    cp2_output = attention(
        local_input, metadata=builder.build(local_positions, packed)
    )
    (cp2_output * cotangent[start : start + local_rows]).sum().backward()
    cp2_peak = torch.cuda.max_memory_allocated() - baseline
    gathered_output = [torch.empty_like(cp2_output) for _ in range(2)]
    gathered_input_grad = [torch.empty_like(local_input.grad) for _ in range(2)]
    dist.all_gather(gathered_output, cp2_output.detach(), group=group)
    dist.all_gather(gathered_input_grad, local_input.grad.detach(), group=group)
    for parameter in attention.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, group=group)
    cp2_grads = _grads(attention)

    assert torch.equal(cp1_forward, torch.cat(gathered_output))
    assert _normalized_l2(cp1_input_grad, torch.cat(gathered_input_grad)) <= 2.9e-3
    assert cp1_grads.keys() == cp2_grads.keys()
    assert max(
        _normalized_l2(cp1_grads[name], cp2_grads[name]) for name in cp1_grads
    ) <= 4.5e-3
    assert cp2_peak < cp1_peak

    if created_group:
        dist.destroy_process_group()
