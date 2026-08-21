from __future__ import annotations

import importlib.util
import os

import pytest
import torch
import torch.distributed as dist

from megatron.lite.model.deepseek_v4.vllm.dispatcher import (
    VLLMAlignedNormalDeepEPDispatcher,
)
from megatron.lite.model.deepseek_v4.vllm.grouped_moe import (
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

    expert_input, actual_counts, route_weights = dispatcher.dispatch(
        hidden, topk_weights, topk_ids
    )
    assert actual_counts.numel() == 2
    assert int(actual_counts.sum()) == expert_input.shape[0]
    assert any(0 < int(count) < 128 for count in actual_counts)
    expert_output = VLLMGroupedMoEWithBF16Backward.apply(
        expert_input,
        actual_counts,
        route_weights,
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
        import megatron.lite.primitive.modules.dispatcher as dispatcher_module

        buffer = dispatcher_module._deepep_buffer
        if buffer is not None:
            buffer.destroy()
            dispatcher_module._deepep_buffer = None
        if created_group:
            dist.destroy_process_group()
