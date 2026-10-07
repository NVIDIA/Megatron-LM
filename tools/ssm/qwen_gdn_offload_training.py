# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compare complete BF16 Qwen3.5 text training from local Hugging Face weights.

Run each arm in a separate torchrun process. Correctness fingerprints and timed
measurements use separate runs so snapshots do not affect memory/runtime claims.
Megatron Bridge imports the VL checkpoint; training keeps its complete language
model and excludes vision and MTP.
"""

import argparse
import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.dict_utils import dict_list_map_outplace
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import PipelineOffloadManager
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import get_attr_wrapped_model, get_batch_on_this_cp_rank, get_model_config


def _arguments() -> argparse.Namespace:
    """Validate the external run configuration before building a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="JSONL with a messages field")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="CPU configuration/tokenizer preflight"
    )
    parser.add_argument(
        "--check-state", action="store_true", help="Fingerprint every training step"
    )
    parser.add_argument("--fraction", type=float, choices=[0.0, 0.5, 1.0], default=None)
    parser.add_argument("--recompute-norm", action="store_true")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--sp", action="store_true")
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--vp", type=int, default=None)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max-documents", type=int, default=128)
    parser.add_argument("--min-offloaded-tensor-size", type=int, default=1024 * 1024)
    args = parser.parse_args()
    for name in (
        "tp",
        "cp",
        "pp",
        "seq_length",
        "microbatches",
        "warmup",
        "iterations",
        "max_documents",
        "min_offloaded_tensor_size",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.sp and args.tp == 1:
        parser.error("--sp requires --tp greater than one")
    if args.vp is not None and (args.vp < 1 or args.pp == 1):
        parser.error("--vp requires a positive chunk count and --pp greater than one")
    if args.cp > 1 and args.seq_length % (2 * args.cp):
        parser.error("--seq-length must be divisible by twice --cp for zigzag partitioning")
    if not args.dry_run and int(os.environ.get("WORLD_SIZE", "1")) % (args.tp * args.cp * args.pp):
        parser.error("WORLD_SIZE must be divisible by TP * CP * PP")
    return args


def _provider(args: argparse.Namespace) -> Any:
    """Configure the existing Qwen VL import and the BF16/FLA language-model option."""
    from megatron.bridge import AutoBridge  # type: ignore[import-untyped]

    metadata = json.loads((args.weights / "config.json").read_text())
    if (
        metadata.get("model_type") != "qwen3_5"
        or metadata.get("text_config", {}).get("model_type") != "qwen3_5_text"
    ):
        raise ValueError("Use a dense Qwen3.5 VL checkpoint, such as Qwen3.5-0.8B.")
    bridge = AutoBridge.from_hf_pretrained(str(args.weights), local_files_only=True)
    provider = bridge.to_megatron_provider(load_weights=True)
    provider.apply_overrides_and_finalize(
        dtype=torch.bfloat16,
        overrides={
            "tensor_model_parallel_size": args.tp,
            "sequence_parallel": args.sp,
            "context_parallel_size": args.cp,
            "pipeline_model_parallel_size": args.pp,
            "virtual_pipeline_model_parallel_size": args.vp,
            "pipeline_dtype": torch.bfloat16,
            # PP=2/VPP sends both directions to one peer; native unbatched P2P orders them.
            "batch_p2p_comm": False,
            "use_cpu_initialization": False,
            "gradient_accumulation_fusion": False,
            # The VL import also constructs vision with the default auto backend;
            # TE backend settings must agree across both models in one process.
            "attention_backend": AttnBackend.auto,
            "attention_dropout": 0.0,
            "hidden_dropout": 0.0,
            "mtp_num_layers": 0,
            "freeze_vision_model": True,
            "freeze_vision_projection": True,
            "vision_recompute_granularity": None,
            "language_max_sequence_length": args.seq_length,
            "deterministic_mode": False,
            "fine_grained_activation_offloading": args.fraction is not None,
            "offload_modules": ["gdn_core_attn"] if args.fraction is not None else [],
            "activation_offload_fraction": args.fraction if args.fraction is not None else 1.0,
            "min_offloaded_tensor_size": args.min_offloaded_tensor_size,
            "recompute_granularity": "selective" if args.recompute_norm else None,
            "recompute_modules": ["gdn_norm_out"] if args.recompute_norm else [],
        },
    )
    return provider


def _tokenize(args: argparse.Namespace) -> torch.Tensor:
    """Render existing text conversations and concatenate them for causal-LM training."""
    tokenizer = AutoTokenizer.from_pretrained(str(args.weights), local_files_only=True)
    tokens = []
    with args.data.open() as source:
        for index, line in enumerate(source):
            if index == args.max_documents:
                break
            messages = json.loads(line)["messages"]
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{args.data}:{index + 1}: messages must be a nonempty list")
            for message in messages:
                if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                    raise ValueError(f"{args.data}:{index + 1}: expected text message objects")
                if message.get("role") not in ("system", "user", "assistant"):
                    raise ValueError(f"{args.data}:{index + 1}: unsupported message role")
            tokens.extend(
                tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False, return_dict=False
                )
            )
    if len(tokens) < 2:
        raise ValueError("The selected dataset must contain at least two tokens")
    return torch.tensor(tokens, dtype=torch.long)


def _batches(
    token_ids: torch.Tensor, args: argparse.Namespace, step: int, groups: ProcessGroupCollection
) -> list[dict[str, torch.Tensor]]:
    """Give each DP replica distinct cyclic text windows shared by its TP/CP/PP ranks."""
    batches = []
    for microbatch in range(args.microbatches):
        sample = (step * args.microbatches + microbatch) * groups.dp.size() + groups.dp.rank()
        indices = torch.arange(args.seq_length + 1) + sample * args.seq_length
        tokens = token_ids[indices % len(token_ids)].unsqueeze(0).cuda()
        batch = {"tokens": tokens[:, :-1].contiguous(), "labels": tokens[:, 1:].contiguous()}
        batch = get_batch_on_this_cp_rank(batch, False, cp_group=groups.cp)
        # Qwen MRoPE partitions its output by CP rank, so give it global positions.
        # Text uses the same positions on all three axes.
        batch["position_ids"] = (
            torch.arange(args.seq_length, device="cuda").view(1, 1, -1).expand(3, 1, -1)
        )
        batches.append(batch)
    return batches


def _forward_step(
    batches: Iterator[dict[str, torch.Tensor]], model: torch.nn.Module
) -> tuple[torch.Tensor, Any]:
    batch = next(batches)
    # Bridge's Qwen forward overrides GPTModel.forward without its offload setup.
    if get_model_config(model).fine_grained_activation_offloading:
        get_attr_wrapped_model(model, "preprocess_for_fine_grained_offloading")()
    output = model(batch["tokens"], batch["position_ids"], None, labels=batch["labels"])

    def loss_func(
        losses: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        loss = losses.float().sum()
        num_tokens = torch.tensor(losses.numel(), device=losses.device, dtype=torch.int64)
        return loss, num_tokens, {"loss": (loss / num_tokens).detach()}

    return output, loss_func


def _fingerprint(value: Any) -> Any:
    """Hash a tensor's raw bytes on CPU without retaining another GPU model copy."""
    if isinstance(value, torch.Tensor):
        data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    return value


def _train(args: argparse.Namespace, provider: Any, token_ids: torch.Tensor) -> dict[str, Any]:
    """Run native gradient accumulation, finalization and BF16 Adam updates."""
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl")
    parallel_state.initialize_model_parallel(
        args.tp, args.pp, args.vp, context_parallel_size=args.cp
    )
    groups = ProcessGroupCollection.use_mpu_process_groups()
    torch.manual_seed(31)
    model_parallel_cuda_manual_seed(31)
    off_interface.reset_instance()
    try:
        # Import with Bridge's existing VL mappings, then release the frozen
        # vision modules and train the native language model with ordinary DDP.
        parents = provider.provide_distributed_model(
            wrap_with_ddp=False, mixed_precision_wrapper=None, pg_collection=groups
        )
        models = [
            DistributedDataParallel(
                provider,
                DistributedDataParallelConfig(grad_reduce_in_fp32=True),
                Float16Module(provider, parent.language_model),
                pg_collection=groups,
            )
            for parent in parents
        ]
        del parents
        torch.cuda.empty_cache()
        optimizer = get_megatron_optimizer(
            OptimizerConfig(bf16=True, lr=1e-4, clip_grad=1.0),
            models,
            use_gloo_process_groups=False,
            pg_collection=groups,
        )
        for model in models:
            config = get_model_config(model)
            config.grad_scale_func = optimizer.scale_loss
            config.finalize_model_grads_func = partial(finalize_model_grads, pg_collection=groups)
        schedule = get_forward_backward_func(pp_size=args.pp, vp_size=args.vp)
        manager = PipelineOffloadManager.get_instance()
        samples, states = [], []
        for step in range(args.warmup + args.iterations):
            batches = _batches(token_ids, args, step, groups)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            optimizer.zero_grad()
            for model in models:
                model.zero_grad_buffer()
            losses = schedule(
                forward_step_func=_forward_step,
                data_iterator=[iter(batches) for _ in models],
                model=models,
                num_microbatches=args.microbatches,
                seq_length=args.seq_length,
                micro_batch_size=1,
                forward_only=False,
                p2p_communicator=P2PCommunicator(groups.pp, get_model_config(models[0])),
                pg_collection=groups,
            )
            if args.check_state:
                gradients = {
                    f"{stage}.{name}": _fingerprint(parameter.main_grad)
                    for stage, model in enumerate(models)
                    for name, parameter in model.named_parameters()
                }
            success, grad_norm, _ = optimizer.step()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            assert success and math.isfinite(grad_norm) and grad_norm > 0
            assert manager.cpu_tensor_pool.get_pool_status()["global_stats"]["current_in_use"] == 0
            if args.fraction is not None:
                assert not manager._is_warmup
            selected = [
                group
                for chunk in manager._cached_chunks_forward
                for group in chunk.offload_groups
                if group.offload
            ]
            offload_status = {
                "selected_group_calls": len(selected),
                "selected_transfer_bytes": sum(group.total_offload_bytes for group in selected),
                "offload_summary_bytes": (
                    manager.offload_summary_total_bytes if args.fraction is not None else 0
                ),
            }
            if args.check_state:
                states.append(
                    {
                        "step": step,
                        "losses": [float(item["loss"]) for item in losses],
                        "grad_norm": float(grad_norm),
                        "grads": gradients,
                        "weights": {
                            f"{stage}.{name}": _fingerprint(parameter)
                            for stage, model in enumerate(models)
                            for name, parameter in model.named_parameters()
                        },
                        "optimizer": dict_list_map_outplace(_fingerprint, optimizer.state_dict()),
                    }
                )
            elif step >= args.warmup:
                samples.append(
                    {
                        "step": step,
                        "seconds": seconds,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        **offload_status,
                    }
                )
            logging.info(
                "Rank %d step %d: losses=%s grad_norm=%s selected_transfer_bytes=%d",
                torch.distributed.get_rank(),
                step,
                [float(item["loss"]) for item in losses],
                grad_norm,
                offload_status["selected_transfer_bytes"],
            )
        return {
            "rank": torch.distributed.get_rank(),
            "gpu": torch.cuda.get_device_name(),
            "samples": samples,
            "states": states,
            "offload_status": offload_status,
        }
    finally:
        torch.cuda.synchronize()
        off_interface.reset_instance()
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def main() -> None:
    """Prepare a local checkpoint/data run or write one rank's training evidence."""
    logging.basicConfig(level=logging.INFO)
    args = _arguments()
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    provider = _provider(args)
    token_ids = _tokenize(args)
    result = {
        "arguments": vars(args),
        "text_layers": provider.num_layers,
        "text_hidden_size": provider.hidden_size,
        "dataset_tokens": len(token_ids),
        "pipeline_batch_p2p_comm": provider.batch_p2p_comm,
        "pipeline_deallocate_outputs": provider.deallocate_pipeline_outputs,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_training_run": not args.dry_run,
    }
    if not args.dry_run:
        result.update(_train(args, provider, token_ids))
    rank = int(os.environ.get("RANK", "0"))
    output = args.output.with_name(f"{args.output.stem}.rank{rank}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    logging.info("Wrote %s", output)


if __name__ == "__main__":
    main()
