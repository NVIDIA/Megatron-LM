# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real VERL-mLite native-FP8 load/export parity gate.

Run with four ranks.  The topology and export contract intentionally match the
four-layer RL gate: PP2, CP2, EP2, FSDP2, and block-FP8 resync.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from safetensors import safe_open


VERL_EXAMPLE_ROOT = Path(__file__).resolve().parents[3] / "examples" / "verl"
if str(VERL_EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_EXAMPLE_ROOT))


def _tensor_equal(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    actual = actual.detach().cpu().contiguous()
    expected = expected.detach().cpu().contiguous()
    if actual.dtype == torch.bfloat16 and expected.dtype == torch.float32:
        expected = expected.to(torch.bfloat16)
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        return False
    if actual.dtype in {torch.float8_e4m3fn, torch.float8_e5m2}:
        return torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    return torch.equal(actual, expected)


def _describe_difference(actual: torch.Tensor, expected: torch.Tensor) -> str:
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        return (
            f"actual={actual.dtype}{tuple(actual.shape)} "
            f"expected={expected.dtype}{tuple(expected.shape)}"
        )
    actual = actual.detach().cpu().contiguous()
    expected = expected.detach().cpu().contiguous()
    if actual.dtype in {torch.float8_e4m3fn, torch.float8_e5m2}:
        changed = int(
            (actual.view(torch.uint8) != expected.view(torch.uint8)).sum().item()
        )
        return f"changed_bytes={changed}/{actual.numel()}"
    if actual.dtype.is_floating_point:
        maximum = float((actual.float() - expected.float()).abs().max().item())
        changed = int((actual != expected).sum().item())
        return f"changed={changed}/{actual.numel()} max_abs_diff={maximum:.9g}"
    changed = int((actual != expected).sum().item())
    return f"changed={changed}/{actual.numel()}"


def _optimizer_config() -> SimpleNamespace:
    return SimpleNamespace(
        optimizer="adam",
        lr=1.0e-6,
        min_lr=0.0,
        min_lr_ratio=None,
        clip_grad=1.0,
        weight_decay=0.1,
        lr_warmup_steps_ratio=0.0,
        total_training_steps=2,
        lr_warmup_steps=0,
        lr_warmup_init=0.0,
        lr_decay_steps=None,
        lr_decay_style="constant",
        weight_decay_incr_style="constant",
        lr_wsd_decay_style="exponential",
        lr_wsd_decay_steps=None,
        use_checkpoint_opt_param_scheduler=False,
        betas=(0.9, 0.95),
        override_optimizer_config={
            "offload_fraction": 1.0,
            "use_precision_aware_optimizer": True,
            "decoupled_weight_decay": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    model_path = args.model.resolve()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    # VERL imports the colocated vLLM worker before it initializes the actor's
    # NCCL groups.  Preserve that lifecycle here: vLLM's platform plugin uses
    # NVML during first import and must resolve CUDA before process-group init.
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        raise RuntimeError(f"vLLM did not resolve CUDA: {current_platform!r}")
    from verl_mlite.engine.config import MegatronLiteEngineConfig
    from verl_mlite.engine.mlite_engine import MegatronLiteEngine

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    if dist.get_world_size() != 4:
        raise RuntimeError("native-FP8 export gate requires exactly four ranks")

    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    engine = MegatronLiteEngine(
        model_config=SimpleNamespace(
            local_path=str(model_path),
            hf_config=config,
            mtp=None,
        ),
        engine_config=MegatronLiteEngineConfig(
            model_name="deepseek_v4",
            impl="vllm",
            tp=1,
            etp=1,
            ep=2,
            pp=2,
            cp=2,
            vpp=1,
            param_offload=True,
            optimizer_offload=True,
            attention_backend_override=None,
            cross_entropy_fusion=True,
            resync_format="block_fp8",
            resync_config={"expert_dtype": "fp8"},
            load_hf_weights=True,
            full_determinism=True,
            impl_cfg={
                "optimizer": "fsdp2",
                "recompute": "full",
                "deterministic": True,
                "mtp_enable": False,
                "mtp_enable_train": False,
            },
        ),
        optimizer_config=_optimizer_config(),
        checkpoint_config={},
    )
    engine.initialize()

    index = json.loads(
        (model_path / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    from megatron.lite.model.deepseek_v4.lite.checkpoint import (
        _export_source_scales,
    )

    chunks = engine.handle._extras.get("model_chunks", [engine.handle._model])
    source_scales = _export_source_scales(
        chunks,
        engine.handle._extras["model_cfg"],
        engine.handle._parallel_state,
    )
    expected_scaled_weights = {
        name.removesuffix(".scale") + ".weight"
        for name in index
        if name.endswith(".scale")
    }
    missing_source_scales = sorted(expected_scaled_weights - set(source_scales))
    print(
        "DS4_NATIVE_FP8_SOURCE_REGISTRY "
        f"rank={dist.get_rank()} gathered={len(source_scales)} "
        f"expected={len(expected_scaled_weights)} missing={len(missing_source_scales)} "
        f"first_missing={missing_source_scales[:8]}",
        flush=True,
    )
    if missing_source_scales:
        raise AssertionError(
            f"native FP8 source-scale registry is incomplete: {missing_source_scales[:32]}"
        )

    local_seen: set[str] = set()
    local_errors: list[str] = []
    weights, metadata = engine.get_per_tensor_param()
    if metadata is not None:
        raise RuntimeError(f"unexpected export metadata: {metadata!r}")

    with ExitStack() as stack:
        shards = {
            shard: stack.enter_context(
                safe_open(model_path / shard, framework="pt", device="cpu")
            )
            for shard in sorted(set(index.values()))
        }
        for name, actual in weights:
            if name in local_seen:
                local_errors.append(f"duplicate export tensor: {name}")
                continue
            local_seen.add(name)
            shard = index.get(name)
            if shard is None:
                local_errors.append(f"export tensor absent from checkpoint: {name}")
                continue
            expected = shards[shard].get_tensor(name)
            if not _tensor_equal(actual, expected):
                local_errors.append(f"{name}: {_describe_difference(actual, expected)}")
            del actual, expected

    reports: list[dict[str, object] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(
        reports,
        {
            "rank": dist.get_rank(),
            "seen": sorted(local_seen),
            "errors": local_errors[:32],
            "error_count": len(local_errors),
        },
    )
    if dist.get_rank() == 0:
        producer = max(reports, key=lambda report: len(report["seen"]))
        missing = sorted(set(index) - set(producer["seen"]))
        errors = [
            f"rank{report['rank']}: {message}"
            for report in reports
            for message in report["errors"]
        ]
        total_errors = sum(int(report["error_count"]) for report in reports)
        print(
            "DS4_NATIVE_FP8_ENGINE_EXPORT "
            f"producer_rank={producer['rank']} exported={len(producer['seen'])} "
            f"expected={len(index)} missing={len(missing)} errors={total_errors}",
            flush=True,
        )
        if missing:
            errors.append(f"missing checkpoint tensors: {missing[:32]}")
        if errors:
            raise AssertionError("\n".join(errors[:64]))
        print("DS4_NATIVE_FP8_ENGINE_EXPORT=PASS", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
