from __future__ import annotations

import importlib.util
import os

import pytest
import torch
import torch.distributed as dist

from megatron.lite.model.deepseek_v4.vllm.primitive.moe.communication import (
    VLLMAlignedNormalDeepEPDispatcher,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.moe.grouped import (
    VLLMGroupedMoEWithBF16Backward,
)
from megatron.lite.primitive.parallel import ParallelState


def _skip_reason() -> str | None:
    missing = [
        package
        for package in ("deep_ep", "deep_gemm", "vllm")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        return f"requires compiled packages: {', '.join(missing)}"
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return "requires two visible CUDA GPUs"
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        return "requires torchrun --standalone --nproc-per-node=2"
    return None


def _ep8_skip_reason() -> str | None:
    missing = [
        package
        for package in ("deep_ep", "vllm")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        return f"requires compiled packages: {', '.join(missing)}"
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        return "requires eight visible CUDA GPUs"
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        return "requires torchrun --standalone --nproc-per-node=8"
    return None


def _run_microbatch(
    dispatcher: VLLMAlignedNormalDeepEPDispatcher,
    weights: tuple[torch.nn.Parameter, ...],
    *,
    rank: int,
    tokens: int,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    for weight in weights:
        weight.grad = None
    generator = torch.Generator(device="cuda").manual_seed(seed + rank)
    hidden = torch.randn(
        tokens,
        256,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    rows = torch.arange(tokens, device="cuda", dtype=torch.int64)
    topk_ids = torch.stack(
        ((rows + rank) % 4, (rows * 2 + rank + 1) % 4), dim=1
    )
    topk_weights = torch.tensor(
        [0.25, 0.75], device="cuda", dtype=torch.float32
    ).expand(tokens, -1).contiguous()

    expert_input, actual_counts, _route_weights = dispatcher.dispatch(
        hidden, topk_weights, topk_ids
    )
    assert actual_counts.device.type == "cpu"
    assert actual_counts.numel() == 2
    assert int(actual_counts.sum()) == expert_input.shape[0]
    assert any(0 < int(count) < 128 for count in actual_counts)
    expert_output = VLLMGroupedMoEWithBF16Backward.apply(
        expert_input,
        actual_counts,
        0.0,
        *weights,
    )
    output = dispatcher.combine(expert_output)
    output.float().square().mean().backward()
    assert hidden.grad is not None
    assert all(weight.grad is not None for weight in weights)
    return (
        output.detach().clone(),
        hidden.grad.detach().clone(),
        *(weight.grad.detach().clone() for weight in weights),
    )


def _run_duplicate_destination_route_gate(
    dispatcher: VLLMAlignedNormalDeepEPDispatcher,
    *,
    rank: int,
) -> None:
    """Compare normal DeepEP route-slot semantics with a direct reference."""
    tokens = 3 + rank
    rows = torch.arange(tokens, device="cuda", dtype=torch.int64)
    hidden = (
        torch.arange(tokens * 256, device="cuda", dtype=torch.float32)
        .reshape(tokens, 256)
        .add_(rank * 17)
        .remainder_(127)
        .sub_(63)
        .to(torch.bfloat16)
    )
    # Experts [0, 1] live on rank 0 and [2, 3] on rank 1.  Every token has
    # multiple distinct top-k slots for the same destination rank, which is
    # exactly the route identity lost by rank-deduplicating normal dispatch.
    topk_ids = torch.stack(
        (
            rows.remainder(2),
            1 - rows.remainder(2),
            2 + rows.remainder(2),
            3 - rows.remainder(2),
        ),
        dim=1,
    )
    topk_weights = torch.tensor(
        [0.125, 0.25, 0.275, 0.35], device="cuda", dtype=torch.float32
    ).expand(tokens, -1).contiguous()

    expert_input, actual_counts, _ = dispatcher.dispatch(
        hidden, topk_weights, topk_ids
    )
    assert int(actual_counts.sum()) == expert_input.shape[0]

    expert_output = expert_input.clone()
    offset = 0
    local_experts = actual_counts.numel()
    for local_expert, count_tensor in enumerate(actual_counts):
        count = int(count_tensor)
        global_expert = rank * local_experts + local_expert
        expert_output[offset : offset + count].mul_(global_expert + 1)
        offset += count

    actual = dispatcher.combine(expert_output)
    expected_fp32 = torch.zeros_like(hidden, dtype=torch.float32)
    for slot in range(topk_ids.shape[1]):
        expected_fp32.add_(
            hidden.float()
            * topk_weights[:, slot : slot + 1]
            * (topk_ids[:, slot : slot + 1].to(torch.float32) + 1)
        )
    expected = expected_fp32.to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpus(2)
def test_normal_deepep_contiguous_moe_is_shape_invariant_across_microbatches() -> None:
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)

    from vllm.model_executor.layers.batch_invariant import init_batch_invariance
    from vllm.utils.deep_gemm import get_num_sms

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("nccl")
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    init_batch_invariance()
    get_num_sms()

    dispatcher = VLLMAlignedNormalDeepEPDispatcher(
        num_experts=4,
        hidden_size=256,
        ps=ParallelState(
            ep_size=2,
            ep_rank=rank,
            ep_group=group,
            tp_ep_group=group,
        ),
        use_deepep=True,
    )
    generator = torch.Generator(device="cuda").manual_seed(1701 + rank)
    weights = tuple(
        torch.nn.Parameter(
            torch.randn(
                shape,
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            / 32
        )
        for shape in (
            (256, 256),
            (256, 256),
            (256, 128),
            (256, 128),
        )
    )

    try:
        first = _run_microbatch(
            dispatcher, weights, rank=rank, tokens=5 + 2 * rank, seed=41
        )
        _run_microbatch(
            dispatcher, weights, rank=rank, tokens=9 - 3 * rank, seed=73
        )
        repeated = _run_microbatch(
            dispatcher, weights, rank=rank, tokens=5 + 2 * rank, seed=41
        )
        for actual, expected in zip(repeated, first, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        dist.barrier(group=group)
    finally:
        if created_group:
            dist.destroy_process_group()


@pytest.mark.gpus(2)
def test_normal_deepep_preserves_multiple_route_slots_to_same_rank() -> None:
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("nccl")
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    dispatcher = VLLMAlignedNormalDeepEPDispatcher(
        num_experts=4,
        hidden_size=256,
        ps=ParallelState(
            ep_size=2,
            ep_rank=rank,
            ep_group=group,
            tp_ep_group=group,
        ),
        use_deepep=True,
    )
    try:
        _run_duplicate_destination_route_gate(dispatcher, rank=rank)
        dist.barrier(group=group)
    finally:
        if created_group:
            dist.destroy_process_group()


@pytest.mark.gpus(8)
def test_normal_deepep_ep8_full_model_dispatch_geometry() -> None:
    """Cover the EP8/hidden-7168/topk-8 geometry used by full-model RL."""
    reason = _ep8_skip_reason()
    if reason:
        pytest.skip(reason)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("nccl")
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    num_experts = 256
    hidden_size = 7168
    topk = 8
    tokens = 4096 - 128 * rank
    dispatcher = VLLMAlignedNormalDeepEPDispatcher(
        num_experts=num_experts,
        hidden_size=hidden_size,
        ps=ParallelState(
            ep_size=8,
            ep_rank=rank,
            ep_group=group,
            tp_ep_group=group,
        ),
        use_deepep=True,
    )
    try:
        generator = torch.Generator(device="cuda").manual_seed(2903 + rank)
        hidden = torch.randn(
            tokens,
            hidden_size,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        rows = torch.arange(tokens, device="cuda", dtype=torch.int64).unsqueeze(1)
        slots = torch.arange(topk, device="cuda", dtype=torch.int64).unsqueeze(0)
        topk_ids = (rows * 17 + slots * 31 + rank * 13).remainder(num_experts)
        topk_weights = torch.full(
            (tokens, topk),
            1.0 / topk,
            device="cuda",
            dtype=torch.float32,
        )

        expert_input, actual_counts, _ = dispatcher.dispatch(
            hidden, topk_weights, topk_ids
        )
        assert not torch.utils.deterministic.fill_uninitialized_memory
        assert actual_counts.numel() == num_experts // 8
        assert int(actual_counts.sum()) == expert_input.shape[0]
        output = dispatcher.combine(expert_input)
        torch.testing.assert_close(output, hidden, rtol=0, atol=0)
        dist.barrier(group=group)
    finally:
        if created_group:
            dist.destroy_process_group()
