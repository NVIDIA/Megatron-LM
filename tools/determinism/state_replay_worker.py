# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Small, explicit training recipes for fresh-process and resume state checks.

The CPU recipe validates the harness. The GPU recipe trains a tiny MCore GPT
with TP only, FP32 state, Torch AdamW and StepLR. It is not a substitute for the
Megatron training-loop/distributed-optimizer/FP8/PP/FSDP recipe adapters.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import platform
import random
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch

from tools.determinism.training_state import (
    capture_configuration,
    checkpoint_path,
    complete_rank,
    file_sha256,
    record_checkpoint,
    write_snapshot,
)

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


class PilotData:
    """Own a dedicated data RNG and a checkpointed sample cursor."""

    def __init__(self, backend: str, device: torch.device) -> None:
        self.backend = backend
        self.device = device
        self.generator = torch.Generator().manual_seed(739)
        self.position = 0

    def next_batch(self) -> tuple:
        """Consume data and global RNGs, making missing restores observable."""
        self.position += 4
        shift = random.randrange(4) + int(np.random.randint(0, 4))
        if self.backend == "cpu":
            inputs = torch.randn(4, 8, generator=self.generator) + shift / 8
            target = torch.randn(4, 4, generator=self.generator)
            return inputs, target
        tokens = (torch.randint(0, 128, (4, 32), generator=self.generator) + shift) % 128
        return {
            "input_ids": tokens.to(self.device),
            "position_ids": torch.arange(32, device=self.device).expand(4, -1),
            "attention_mask": None,
        }, None

    def state_dict(self) -> dict:
        """Return all state that determines subsequent pilot batches."""
        return {
            "generator": self.generator.get_state(),
            "position": self.position,
            "dataset": "generated-pilot-v1",
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore the cursor and generator, not a newly seeded approximation."""
        self.generator.set_state(state["generator"])
        self.position = state["position"]


def _rng_state(gpu: bool) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if gpu else None,
        "model_parallel": None,
    }
    if gpu:
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

        state["model_parallel"] = get_cuda_rng_tracker().get_states()
    return state


def _restore_rng(state: dict, gpu: bool) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if gpu:
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

        torch.cuda.set_rng_state(state["torch_cuda"])
        get_cuda_rng_tracker().set_states(state["model_parallel"])


def _initialize(backend: str, world_size: int):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    if backend == "cpu":
        device = torch.device("cpu")
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(16, 4)
        )
        return model, device

    # Configure before importing training/Core, which can initialize CUDA.
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    from megatron.determinism import configure_determinism

    configure_determinism({"deterministic_mode": True})
    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.distributed.init_process_group("nccl")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)
    model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=8,
        tensor_model_parallel_size=world_size,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        pipeline_dtype=torch.float32,
        bf16=False,
        fp16=False,
        fp8=None,
        fp4=None,
        enable_autocast=False,
        hidden_dropout=0.2,
        attention_dropout=0.2,
        deterministic_mode=True,
        gradient_accumulation_fusion=False,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=128,
        max_sequence_length=32,
        parallel_output=True,
        position_embedding_type="rope",
    ).to(device)
    return model, device


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _provenance(backend: str, world_size: int, device: torch.device, *, capture: dict) -> dict:
    def command(*args):
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()

    environment = {
        "NCCL_ALGO",
        "NCCL_PROTO",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "MAMBA_DETERMINISTIC",
        "CAUSAL_CONV1D_DETERMINISTIC",
        "TRITON_CACHE_AUTOTUNING",
        "TRITON_CACHE_DIR",
    }
    environment.update(key for key in os.environ if key.startswith("TRITON_AUTOTUNE_BLOCK_"))
    hardware: dict = {
        "system": platform.system(),
        "machine": platform.machine(),
        "device": str(device),
    }
    if backend != "cpu":
        hardware.update(
            gpu=torch.cuda.get_device_name(device),
            capability=list(torch.cuda.get_device_capability(device)),
            devices_and_driver=command(
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,driver_version",
                "--format=csv,noheader",
            ),
            topology=command("nvidia-smi", "topo", "-m"),
        )
    return {
        "source_revision": command("git", "rev-parse", "HEAD"),
        "source_dirty": bool(command("git", "status", "--porcelain")),
        "hardware": hardware,
        "software": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if backend != "cpu" else None,
            "transformer_engine": (
                _package_version("transformer-engine") if backend != "cpu" else None
            ),
            "environment": {key: os.environ.get(key) for key in sorted(environment)},
            "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        },
        "recipe": {
            "id": f"{backend}_training_state_pilot_v1",
            "capture": capture,
            "seed": 123,
            "data_seed": 739,
            "world_size": world_size,
            "TP": world_size,
            "PP": 1,
            "DP": 1,
            "precision": "fp32",
            "optimizer": "torch.optim.AdamW(foreach=False,fused=False)",
            "scheduler": "StepLR(step_size=2,gamma=0.9)",
            "loss": "MSE" if backend == "cpu" else "mean_squared_local_logits",
        },
    }


def _save_checkpoint(root, step, rank, run_id, model, optimizer, scheduler, data, gpu):
    rng = _rng_state(gpu)
    # Keep the checkpoint load weights-only: NumPy RNG arrays become a list.
    numpy_state = rng["numpy"]
    rng["numpy"] = (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:])
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "dataloader": data.state_dict(),
        "rng": rng,
        "step": step,
        "run_id": run_id,
    }
    path = checkpoint_path(root, step, rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(payload, stream)
    record_checkpoint(root, step=step, rank=rank, run_id=run_id)


def run_worker(args: argparse.Namespace) -> None:
    """Train the full schedule up to the requested boundary and capture state."""
    capture = capture_configuration(args.steps, args.checkpoint_step, args.stop_step)
    last_step = args.steps if args.stop_step is None else args.stop_step
    gpu = args.backend != "cpu"
    rank = int(os.environ.get("RANK", "0")) if gpu else 0
    world_size = int(os.environ.get("WORLD_SIZE", "1")) if gpu else 1
    if gpu and world_size not in (1, 2, 4, 8):
        raise ValueError("The tiny GPT pilot supports TP=1,2,4,8 and no DP/PP/FSDP")
    model, device = _initialize(args.backend, world_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    data = PilotData(args.backend, device)
    provenance = _provenance(args.backend, world_size, device, capture=capture)
    captured_steps = []
    resumed = None
    start = 0
    try:
        if args.resume is not None:
            path = checkpoint_path(args.resume, args.checkpoint_step, rank)
            resumed = json.loads(path.with_suffix(".json").read_text())
            if resumed["checkpoint_sha256"] != file_sha256(path):
                raise ValueError("Checkpoint differs from its recorded identity")
            # BytesIO is TE's extra-state container, not an arbitrary pickle class.
            with torch.serialization.safe_globals([io.BytesIO]):
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint["run_id"] != resumed["run_id"] or checkpoint["step"] != resumed["step"]:
                raise ValueError("Checkpoint contents do not match the resume record")
            model.load_state_dict(checkpoint["model"])
            if args.omit_restore != "optimizer":
                optimizer.load_state_dict(checkpoint["optimizer"])
            if args.omit_restore != "scheduler":
                scheduler.load_state_dict(checkpoint["scheduler"])
            if args.omit_restore != "dataloader":
                data.load_state_dict(checkpoint["dataloader"])
            rng = checkpoint["rng"]
            numpy_state = rng["numpy"]
            rng["numpy"] = (
                numpy_state[0],
                np.asarray(numpy_state[1], dtype=np.uint32),
                *numpy_state[2:],
            )
            # Restore RNG last: model/optimizer construction and loading may consume it.
            if args.omit_restore != "rng":
                _restore_rng(rng, gpu)
            start = checkpoint["step"]
        if start >= last_step:
            raise ValueError("Resume must execute at least one new training step")
        model.train()
        for step in range(start + 1, last_step + 1):
            optimizer.zero_grad(set_to_none=True)
            inputs, target = data.next_batch()
            output = model(**inputs) if gpu else model(inputs)
            loss = (
                output.float().square().mean()
                if gpu
                else torch.nn.functional.mse_loss(output, target)
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            if args.stop_step is not None and step != args.stop_step:
                if step == args.checkpoint_step and resumed is None:
                    _save_checkpoint(
                        args.output, step, rank, args.run_id, model, optimizer, scheduler, data, gpu
                    )
                # No diagnostic sync, CPU copies, state enumeration or loss.item().
                continue
            if gpu:
                torch.cuda.synchronize(device)
            state = {
                "model": {"chunk0": model.state_dict()},
                "gradients": {
                    name: {
                        "grad": parameter.grad,
                        "main_grad": getattr(parameter, "main_grad", None),
                    }
                    for name, parameter in model.named_parameters()
                },
                "optimizer": optimizer.state_dict(),
                "precision": {
                    "parameters": "fp32",
                    "autocast": False,
                    "loss_scaler": "disabled",
                    "quantizers": "disabled",
                    "master_parameters": "same_as_model_parameters",
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                },
                "rng": _rng_state(gpu),
                "scheduler": scheduler.state_dict(),
                "dataloader": data.state_dict(),
            }
            write_snapshot(
                args.output,
                state,
                step=step,
                rank=rank,
                world_size=world_size,
                run_id=args.run_id,
                provenance=provenance,
                resume_from=resumed,
            )
            captured_steps.append(step)
            logger.info("Captured step=%d rank=%d loss=%s", step, rank, loss.detach().item())
            if step == args.checkpoint_step and resumed is None:
                _save_checkpoint(
                    args.output, step, rank, args.run_id, model, optimizer, scheduler, data, gpu
                )
        if _provenance(args.backend, world_size, device, capture=capture) != provenance:
            raise ValueError("Source/runtime provenance changed during training")
        complete_rank(
            args.output,
            rank=rank,
            world_size=world_size,
            run_id=args.run_id,
            steps=captured_steps,
            provenance=provenance,
        )
    finally:
        if gpu and torch.distributed.is_initialized():
            from megatron.core import parallel_state

            parallel_state.destroy_model_parallel()
            torch.distributed.destroy_process_group()


def main() -> None:
    """Entry point for the isolated worker; normally launched by run_state_replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "mcore_gpt"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--checkpoint-step", type=int, default=2)
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--omit-restore", choices=("rng", "optimizer", "scheduler", "dataloader"))
    args = parser.parse_args()
    try:
        capture_configuration(args.steps, args.checkpoint_step, args.stop_step)
    except ValueError as error:
        parser.error(str(error))
    if args.omit_restore and args.resume is None:
        parser.error("An omitted-restore control requires a checkpoint resume")
    logging.basicConfig(level=logging.INFO)
    logger.info("Worker arguments: %s", vars(args))
    run_worker(args)


if __name__ == "__main__":
    main()
