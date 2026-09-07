"""Real FP8 MoE parity gate: HybridEP versus aligned normal DeepEP."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.vllm.primitive.moe.module import (
    DeepseekV4MoE,
)
from megatron.lite.primitive.parallel import ParallelState


_HIDDEN = 4096
_INTERMEDIATE = 2048
_LOCAL_EXPERTS = 2
_TOKENS = 128
_TOPK = 6
_SEED = 20260823


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("deepep", "hybridep"))
    parser.add_argument("--golden-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument(
        "--route-pattern",
        choices=("regular", "hash-duplicate"),
        default="regular",
    )
    parser.add_argument("--local-experts", type=int, default=_LOCAL_EXPERTS)
    parser.add_argument("--tokens", type=int, default=_TOKENS)
    parser.add_argument("--rank-token-step", type=int, default=0)
    return parser.parse_args()


def _capsule(
    rank: int,
    world_size: int,
    route_pattern: str,
    local_experts: int,
    tokens: int,
):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(_SEED + rank)
    hidden = (
        torch.randn(
            tokens,
            _HIDDEN,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        / 16
    )
    rows = torch.arange(tokens, device="cuda").unsqueeze(1)
    slots = torch.arange(_TOPK, device="cuda").unsqueeze(0)
    if route_pattern == "hash-duplicate":
        indices = (
            rows * 13 + rank + torch.div(slots, 2, rounding_mode="floor") * 17
        ).remainder(world_size * local_experts)
    else:
        indices = (
            rank * local_experts + rows + slots * 3
        ).remainder(world_size * local_experts)
    raw_weights = torch.arange(
        1, _TOPK + 1, dtype=torch.float32, device="cuda"
    )
    weights = (raw_weights / raw_weights.sum()).unsqueeze(0).expand(
        tokens, -1
    )
    return hidden, weights.contiguous(), indices.contiguous()


def _build_moe(
    backend: str,
    group: dist.ProcessGroup,
    rank: int,
    world_size: int,
    local_experts: int,
    tokens: int,
) -> DeepseekV4MoE:
    torch.manual_seed(_SEED + 1000 + rank)
    moe = DeepseekV4MoE(
        DeepseekV4Config(
            hidden_size=_HIDDEN,
            moe_intermediate_size=_INTERMEDIATE,
            n_routed_experts=world_size * local_experts,
            n_shared_experts=0,
            num_experts_per_tok=_TOPK,
            num_hash_layers=0,
        ),
        ParallelState(
            ep_size=world_size,
            ep_rank=rank,
            ep_group=group,
            tp_ep_group=group,
        ),
        layer_idx=0,
        moe_token_dispatcher_type=backend,
        hybridep_max_tokens_per_rank=(
            tokens * _TOPK if backend == "hybridep" else None
        ),
    ).cuda()
    # Standalone construction does not run the training framework's weight
    # initialization callback. Fill BF16 masters explicitly so the FP8 scale
    # audit never observes allocator contents.
    with torch.no_grad():
        for parameter in moe.experts.parameters():
            parameter.normal_(mean=0.0, std=0.01)
    return moe


def _fixed_route_forward(
    moe: DeepseekV4MoE,
    hidden: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    dispatched, counts, probs = moe.dispatcher.dispatch(
        hidden, weights, indices
    )
    moe.dispatcher.wait_dispatch_event()
    expert_output = moe.experts(
        dispatched,
        counts,
        probs,
        tokens_per_expert_list=getattr(
            moe.dispatcher, "_local_tpe_list", None
        ),
    )
    return moe.dispatcher.combine(expert_output)


def main() -> None:
    args = _arguments()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", local_rank)
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.new_group(list(range(world_size)), backend="nccl")
    rank_tokens = args.tokens + rank * args.rank_token_step
    hidden, weights, indices = _capsule(
        rank,
        world_size,
        args.route_pattern,
        args.local_experts,
        rank_tokens,
    )

    from vllm.config import VllmConfig, set_current_vllm_config
    import vllm.utils.deep_gemm as deep_gemm

    # vLLM normally performs this lazy binding during worker startup. This
    # standalone gate must bind DeepGEMM before freezing the scale-format
    # oracle, otherwise SM100 is incorrectly cached as FLOAT32.
    deep_gemm._lazy_init()
    deep_gemm.DeepGemmQuantScaleFMT.init_oracle_cache()
    with set_current_vllm_config(VllmConfig()):
        moe = _build_moe(
            args.mode,
            group,
            rank,
            world_size,
            args.local_experts,
            args.tokens + (world_size - 1) * args.rank_token_step,
        )

        def operation() -> torch.Tensor:
            with torch.no_grad():
                return _fixed_route_forward(
                    moe, hidden, weights, indices
                )

        for _ in range(args.warmup):
            output = operation()
        torch.cuda.synchronize()
        dist.barrier()
        started = time.perf_counter()
        for _ in range(args.iterations):
            output = operation()
        torch.cuda.synchronize()
        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000 / args.iterations

    args.golden_dir.mkdir(parents=True, exist_ok=True)
    golden_file = (
        args.golden_dir
        / (
            f"clean-normal-deepep-fp8-{args.route_pattern}-"
            f"le{args.local_experts}-t{args.tokens}-s{args.rank_token_step}-"
            f"ep{world_size}-rank{rank}.pt"
        )
    )
    if args.mode == "deepep":
        torch.save(output.cpu(), golden_file)
        bitwise = True
        max_abs = 0.0
    else:
        if not golden_file.is_file():
            raise FileNotFoundError(golden_file)
        golden = torch.load(
            golden_file, map_location=output.device, weights_only=True
        )
        bitwise = torch.equal(output, golden)
        max_abs = float(
            (output.float() - golden.float()).abs().max().item()
        )

    metrics = torch.tensor(
        [int(bitwise), max_abs, elapsed_ms],
        device="cuda",
        dtype=torch.float64,
    )
    dist.all_reduce(metrics[0], op=dist.ReduceOp.MIN)
    dist.all_reduce(metrics[1:], op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "reference": "clean-aligned-normal-deepep-fp8-moe",
                    "world_size": world_size,
                    "route_pattern": args.route_pattern,
                    "local_experts": args.local_experts,
                    "tokens": args.tokens,
                    "rank_token_step": args.rank_token_step,
                    "candidate_vs_reference_bitwise": bool(
                        metrics[0].item()
                    ),
                    "max_abs_all_ranks": float(metrics[1].item()),
                    "forward_ms_max_rank": float(metrics[2].item()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not bool(metrics[0].item()):
        raise AssertionError(
            f"HybridEP FP8 MoE differs from normal DeepEP: {max_abs}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
