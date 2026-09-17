# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Observe the real pretrain_gpt loop and synchronous distributed checkpoints.

Hooks are confined to this diagnostic process. The training entrypoint, forward
and backward schedules, optimizer, data iterator and save/load implementation
are the existing Megatron code. No alternate optimizer or checkpoint is used.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import logging
import os
import runpy
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from tools.determinism.megatron_state import (
    capture_model,
    capture_optimizer,
    capture_single_pass_loader,
)
from tools.determinism.state_replay_worker import ROOT, _provenance, _rng_state
from tools.determinism.training_state import (
    UnverifiedState,
    capture_configuration,
    complete_rank,
    read_checkpoint_record,
    record_checkpoint_directory,
    write_snapshot,
)


def recipe_arguments(
    world_size: int,
    steps: int,
    stop_step: int | None = None,
    *,
    pipeline_size: int = 1,
    virtual_pipeline_size: int = 1,
    optimizer_mode: str = "standard",
) -> list[str]:
    """Freeze BF16 distributed Adam with TP=2, PP=1/2 and optional two-chunk VPP."""
    if world_size not in (4, 8) or type(pipeline_size) is not int or pipeline_size not in (1, 2):
        raise ValueError("The Megatron training recipe requires four/eight ranks and PP=1/2")
    if (
        type(virtual_pipeline_size) is not int
        or virtual_pipeline_size not in (1, 2)
        or (virtual_pipeline_size == 2 and pipeline_size != 2)
    ):
        raise ValueError("The two-chunk virtual pipeline requires PP=2")
    if optimizer_mode not in ("standard", "precision_aware_fp16") or (
        optimizer_mode != "standard" and (pipeline_size != 1 or virtual_pipeline_size != 1)
    ):
        raise ValueError("Precision-aware FP16 moments require the PP=1 non-virtual recipe")
    arguments = [
        "--num-layers",
        str(2 * virtual_pipeline_size),
        "--hidden-size",
        "128",
        "--ffn-hidden-size",
        "256",
        "--num-attention-heads",
        "8",
        "--seq-length",
        "32",
        "--max-position-embeddings",
        "32",
        "--micro-batch-size",
        "1",
        "--global-batch-size",
        str(world_size),
        "--train-iters",
        str(steps),
        "--lr-decay-iters",
        str(steps),
        "--lr",
        "0.001",
        "--min-lr",
        "0.0001",
        "--lr-decay-style",
        "cosine",
        "--weight-decay",
        "0.01",
        "--clip-grad",
        "1.0",
        "--seed",
        "123",
        "--tensor-model-parallel-size",
        "2",
        "--pipeline-model-parallel-size",
        str(pipeline_size),
        "--context-parallel-size",
        "1",
        "--use-distributed-optimizer",
        "--bf16",
        "--transformer-impl",
        "transformer_engine",
        "--attention-backend",
        "unfused",
        "--hidden-dropout",
        "0.2",
        "--attention-dropout",
        "0.2",
        "--no-gradient-accumulation-fusion",
        "--deterministic-mode",
        "--mock-data",
        "--tokenizer-type",
        "NullTokenizer",
        "--vocab-size",
        "256",
        "--make-vocab-size-divisible-by",
        "8",
        "--split",
        "100,0,0",
        "--dataloader-type",
        "single",
        "--num-workers",
        "0",
        "--rerun-mode",
        "disabled",
        "--eval-iters",
        "0",
        "--eval-interval",
        str(steps + 1),
        "--log-interval",
        "1",
        "--ckpt-format",
        "torch_dist",
        "--use-checkpoint-opt-param-scheduler",
    ]
    if virtual_pipeline_size == 2:
        # Megatron requires P2P overlap for an interleaved two-stage pipeline.
        arguments += ["--num-virtual-stages-per-pipeline-rank", "2"]
    if optimizer_mode == "precision_aware_fp16":
        arguments += [
            "--use-precision-aware-optimizer",
            "--main-params-dtype",
            "fp32",
            "--main-grads-dtype",
            "fp32",
            "--exp-avg-dtype",
            "fp16",
            "--exp-avg-sq-dtype",
            "fp16",
        ]
    if stop_step is not None:
        # train-iters also controls data indexing and scheduler construction.
        arguments += ["--exit-interval", str(stop_step)]
    return arguments


class TrainingCapture:
    """Observe fixed step boundaries without replacing Megatron's training logic."""

    def __init__(self, args, training, checkpointing):
        self.args = args
        self.training = training
        self.checkpointing = checkpointing
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.loaders: list = []
        self.iterators: list = []
        self.chunks: list = []
        self.provenance = None
        self.resumed = None
        self.captured_steps: list[int] = []
        self.train_completed = False
        self.capture_config = capture_configuration(
            args.steps, args.checkpoint_step, args.stop_step
        )
        self.stop_requested = False
        self.expected_exit: SystemExit | None = None

    def runtime_provenance(self) -> dict:
        """Record actual runtime plus the complete, path-independent CLI recipe."""
        from megatron.core import parallel_state

        record = _provenance(
            "megatron_gpt",
            self.world_size,
            torch.device("cuda", int(os.environ["LOCAL_RANK"])),
            capture=self.capture_config,
        )
        record["recipe"] = {
            "id": "megatron_pretrain_gpt_dist_optimizer_bf16_v1",
            "entrypoint": "pretrain_gpt.py",
            "arguments": recipe_arguments(
                self.world_size,
                self.args.steps,
                self.args.stop_step,
                pipeline_size=self.args.pipeline_size,
                virtual_pipeline_size=self.args.virtual_pipeline_size,
                optimizer_mode=self.args.optimizer_mode,
            ),
            "optimizer_mode": self.args.optimizer_mode,
            "capture": self.capture_config,
            "checkpoint_step": self.args.checkpoint_step,
            "mock_documents": 512,
            "mock_max_sequence_length": 64,
            "capture_boundary": "post_training_step_callbacks_before_checkpoint",
        }
        record["rank_layout"] = {
            "global_rank": torch.distributed.get_rank(),
            "world_size": torch.distributed.get_world_size(),
            "sizes": {
                "TP": parallel_state.get_tensor_model_parallel_world_size(),
                "PP": parallel_state.get_pipeline_model_parallel_world_size(),
                "DP": parallel_state.get_data_parallel_world_size(),
                "CP": parallel_state.get_context_parallel_world_size(),
            },
            "TP": parallel_state.get_tensor_model_parallel_rank(),
            "PP": parallel_state.get_pipeline_model_parallel_rank(),
            "DP": parallel_state.get_data_parallel_rank(),
            "CP": parallel_state.get_context_parallel_rank(),
            "VPP": parallel_state.get_virtual_pipeline_model_parallel_world_size(),
            "owns_loader": any(self.loader_owners()),
        }
        if self.args.virtual_pipeline_size == 2:
            record["rank_layout"]["chunks"] = self.chunk_layout()
        return record

    def loader_owners(self) -> list[bool]:
        """Require one correctly paired loader/iterator slot per local chunk."""
        count = self.args.virtual_pipeline_size
        if len(self.loaders) != count or len(self.iterators) != count:
            raise UnverifiedState("Missing or extra pipeline loader/iterator slot")
        owners = []
        for loader, iterator in zip(self.loaders, self.iterators):
            if (loader is None) != (iterator is None):
                raise UnverifiedState("Pipeline loader and iterator ownership disagree")
            owners.append(loader is not None)
        return owners

    def chunk_layout(self) -> list[dict]:
        """Read actual virtual chunk identity, global layers, endpoints and P2P policy."""
        owners = self.loader_owners()
        if len(self.chunks) != len(owners):
            raise UnverifiedState("Model and loader chunk counts disagree")
        return [
            {
                "VP": chunk.vp_stage,
                "pre_process": chunk.pre_process,
                "post_process": chunk.post_process,
                "layers": [layer.layer_number for layer in chunk.decoder.layers],
                "owns_loader": owners[index],
                "p2p": {
                    key: getattr(chunk.config, key)
                    for key in (
                        "overlap_p2p_comm",
                        "batch_p2p_comm",
                        "deallocate_pipeline_outputs",
                        "overlap_p2p_comm_warmup_flush",
                    )
                },
            }
            for index, chunk in enumerate(self.chunks)
        ]

    def validate_configuration(self) -> None:
        """Reject unimplemented precision, overlap, loader and checkpoint variants."""
        args = self.training.get_args()
        if not (args.bf16 and args.use_distributed_optimizer and args.deterministic_mode):
            raise UnverifiedState("Expected BF16 distributed-optimizer deterministic training")
        if (
            args.ckpt_format != "torch_dist"
            or args.dataloader_type != "single"
            or args.num_workers != 0
        ):
            raise UnverifiedState("Unsupported checkpoint/loader configuration")
        for name in (
            "fp16",
            "fp8",
            "fp4",
            "optimizer_cpu_offload",
            "use_megatron_fsdp",
            "use_torch_fsdp2",
            "overlap_param_gather",
            "overlap_grad_reduce",
            "reuse_grad_buf_for_mxfp8_param_ag",
            "async_save",
            "no_save_optim",
            "no_save_rng",
            "skip_train",
            "perform_rl_step",
            "optimizer_cuda_graph",
            "defer_embedding_wgrad_compute",
            "overlap_p2p_comm_warmup_flush",
        ):
            if getattr(args, name, None):
                raise UnverifiedState(f"State adapter does not cover {name}")
        precision_aware = self.args.optimizer_mode == "precision_aware_fp16"
        if bool(getattr(args, "use_precision_aware_optimizer", False)) != precision_aware:
            raise UnverifiedState("Requested and actual optimizer modes disagree")
        if precision_aware and (
            self.args.pipeline_size != 1
            or self.args.virtual_pipeline_size != 1
            or args.main_params_dtype != torch.float32
            or args.main_grads_dtype != torch.float32
            or args.exp_avg_dtype != torch.float16
            or args.exp_avg_sq_dtype != torch.float16
        ):
            raise UnverifiedState("Unsupported precision-aware optimizer recipe")
        if (
            args.tensor_model_parallel_size != 2
            or args.pipeline_model_parallel_size != self.args.pipeline_size
            or args.context_parallel_size != 1
            or args.virtual_pipeline_model_parallel_size
            != (2 if self.args.virtual_pipeline_size == 2 else None)
            or bool(getattr(args, "overlap_p2p_comm", False))
            != (self.args.virtual_pipeline_size == 2)
        ):
            raise UnverifiedState("Expected TP=2, requested PP/VPP, CP=1 and matching P2P overlap")
        if self.args.stop_step is not None:
            if (
                args.train_iters != self.args.steps
                or args.lr_decay_iters != self.args.steps
                or args.exit_interval != self.args.stop_step
                or args.exit_duration_in_mins
                or args.exit_signal_handler
                or args.phase_transition_iterations
            ):
                raise UnverifiedState("Stop-point training horizon or exit policy changed")

    def wrap_load(self, original):
        """Verify checkpoint files on both sides of Megatron's actual load call."""

        @functools.wraps(original)
        def load(*positional, **keywords):
            if self.args.resume is None:
                raise UnverifiedState("Fresh training unexpectedly tried loading a checkpoint")
            record = read_checkpoint_record(
                self.args.resume, step=self.args.checkpoint_step, rank=self.rank
            )
            args = self.training.get_args()
            expected = Path(args.load) / f"iter_{self.args.checkpoint_step:07d}"
            if expected.resolve() != (self.args.resume / record["checkpoint_directory"]).resolve():
                raise UnverifiedState("Megatron load path differs from the recorded checkpoint")
            previous = args.no_load_rng
            try:
                if self.args.omit_restore == "rng":
                    args.no_load_rng = True
                result = original(*positional, **keywords)
            finally:
                args.no_load_rng = previous
            if result[0] != self.args.checkpoint_step:
                raise UnverifiedState("Megatron did not restore the requested iteration")
            if (
                read_checkpoint_record(
                    self.args.resume, step=self.args.checkpoint_step, rank=self.rank
                )
                != record
            ):
                raise UnverifiedState("Checkpoint changed during load")
            self.resumed = record
            return result

        return load

    def wrap_save(self, original):
        """Fingerprint the actual distributed checkpoint after synchronous save."""

        @functools.wraps(original)
        def save(*positional, **keywords):
            bound = inspect.signature(original).bind(*positional, **keywords)
            result = original(*positional, **keywords)
            step = bound.arguments["iteration"]
            if step == self.args.checkpoint_step and self.args.resume is None:
                args = self.training.get_args()
                if args.async_save:
                    raise UnverifiedState(
                        "Checkpoint capture requires completed synchronous writes"
                    )
                torch.distributed.barrier()
                directory = Path(
                    self.checkpointing.get_checkpoint_name(args.save, step, return_base_dir=True)
                )
                record_checkpoint_directory(
                    self.args.output,
                    checkpoint_directory=directory,
                    step=step,
                    rank=self.rank,
                    run_id=self.args.run_id,
                )
            return result

        return save

    def wrap_loaders(self, original):
        """Retain every actual loader in the same order as virtual chunk construction."""

        @functools.wraps(original)
        def loaders(*positional, **keywords):
            result = original(*positional, **keywords)
            self.loaders.append(result[0])
            return result

        return loaders

    def wrap_train(self, original):
        """Require normal completion or a successful, observed target-step exit."""

        @functools.wraps(original)
        def train(*positional, **keywords):
            bound = inspect.signature(original).bind(*positional, **keywords)
            iterator = bound.arguments["train_data_iterator"]
            if self.args.virtual_pipeline_size == 2:
                if not isinstance(iterator, list):
                    raise UnverifiedState("Virtual pipeline requires a list of data iterators")
                self.iterators = iterator
            else:
                self.iterators = [iterator]
            self.chunks = self.training.unwrap_model(bound.arguments["model"])
            self.validate_configuration()
            if (self.args.resume is not None) != (self.resumed is not None):
                raise UnverifiedState("Missing or unexpected Megatron checkpoint load")
            self.provenance = self.runtime_provenance()
            try:
                result = original(*positional, **keywords)
            except SystemExit as error:
                if (
                    self.args.stop_step is None
                    or error.code not in (None, 0)
                    or not self.stop_requested
                    or self.captured_steps != [self.args.stop_step]
                ):
                    raise UnverifiedState(
                        "Megatron exited without completing the stop point"
                    ) from error
                self.train_completed = True
                self.expected_exit = error
                raise
            if self.args.stop_step is not None:
                raise UnverifiedState("Megatron did not exit at the requested stop point")
            if result[0] != self.args.steps:
                raise UnverifiedState("Megatron training stopped before the requested step")
            self.train_completed = True
            return result

        return train

    def wrap_decide_exit(self, original):
        """Observe the existing checkpoint/exit decision, without overriding it."""

        @functools.wraps(original)
        def decide(*positional, **keywords):
            bound = inspect.signature(original).bind(*positional, **keywords)
            result = original(*positional, **keywords)
            if self.args.stop_step is not None:
                step = bound.arguments["iteration"]
                if (result and step != self.args.stop_step) or (
                    step >= self.args.stop_step and not result
                ):
                    raise UnverifiedState("Unexpected Megatron stop-point exit decision")
                if result:
                    self.stop_requested = True
            return result

        return decide

    def capture(self, model, optimizer, scheduler, step: int) -> None:
        """Snapshot live state after updates and consumed-sample bookkeeping."""
        if self.provenance is None:
            raise UnverifiedState("State capture occurred before training initialization")
        torch.cuda.synchronize()
        args = self.training.get_args()
        chunks = self.training.unwrap_model(model)
        self.validate_partition(chunks)
        model_state, gradients = capture_model(chunks)
        optimizer_state, precision = capture_optimizer(optimizer, model_chunks=chunks)
        precision.update(
            mode=(
                "bf16_precision_aware_fp16_moments"
                if self.args.optimizer_mode == "precision_aware_fp16"
                else "bf16_with_fp32_master_parameters"
            ),
            fp8="disabled",
            fp4="disabled",
            autocast=False,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
        )
        state = {
            "model": model_state,
            "gradients": gradients,
            "optimizer": optimizer_state,
            "precision": precision,
            "rng": _rng_state(True),
            "scheduler": scheduler.state_dict(),
            "dataloader": self.capture_dataloaders(args.consumed_train_samples),
        }
        write_snapshot(
            self.args.output,
            state,
            step=step,
            rank=self.rank,
            world_size=self.world_size,
            run_id=self.args.run_id,
            provenance=self.provenance,
            resume_from=self.resumed,
        )
        self.captured_steps.append(step)

    def capture_dataloaders(self, consumed_samples: int) -> dict:
        """Capture each actual iterator, including explicit loader-free virtual chunks."""
        self.loader_owners()
        states = {
            str(index): (
                {"mode": "no_local_loader_for_virtual_chunk"}
                if self.args.virtual_pipeline_size == 2 and loader is None
                else capture_single_pass_loader(loader, iterator, consumed_samples=consumed_samples)
            )
            for index, (loader, iterator) in enumerate(zip(self.loaders, self.iterators))
        }
        if self.args.virtual_pipeline_size == 1:
            return states["0"]
        return {"mode": "virtual_pipeline_chunks", "chunks": states}

    def validate_partition(self, chunks: list) -> None:
        """Require the expected local layer count and pipeline endpoint ownership."""
        pipeline_size = self.args.pipeline_size
        pipeline_rank = self.provenance["rank_layout"]["PP"]
        if self.args.virtual_pipeline_size == 2:
            if len(chunks) != 2 or any(
                chunk.vp_stage != index
                or chunk.pre_process != (pipeline_rank == 0 and index == 0)
                or chunk.post_process != (pipeline_rank == 1 and index == 1)
                or [layer.layer_number for layer in chunk.decoder.layers]
                != [index * pipeline_size + pipeline_rank + 1]
                or not chunk.config.overlap_p2p_comm
                or chunk.config.batch_p2p_comm
                or not chunk.config.deallocate_pipeline_outputs
                or chunk.config.overlap_p2p_comm_warmup_flush
                for index, chunk in enumerate(chunks)
            ):
                raise UnverifiedState("Model partition or P2P policy differs from the VPP recipe")
        elif len(chunks) != 1 or (
            chunks[0].pre_process != (pipeline_rank == 0)
            or chunks[0].post_process != (pipeline_rank == pipeline_size - 1)
            or len(chunks[0].decoder.layers) != 2 // pipeline_size
        ):
            raise UnverifiedState("Model partition does not match the declared GPT pipeline")

    def wrap_post_step(self, original):
        """Observe after the existing callbacks, before checkpoint save or zero_grad."""

        @functools.wraps(original)
        def post_step(model, optimizer, opt_param_scheduler, iteration, *positional, **keywords):
            result = original(
                model, optimizer, opt_param_scheduler, iteration, *positional, **keywords
            )
            if self.args.stop_step is None or iteration == self.args.stop_step:
                self.capture(model, optimizer, opt_param_scheduler, iteration)
            return result

        return post_step

    def finish(self) -> None:
        """Only publish evidence after the entrypoint returns successfully."""
        start = self.args.checkpoint_step if self.args.resume else 0
        expected = (
            [self.args.stop_step]
            if self.args.stop_step is not None
            else list(range(start + 1, self.args.steps + 1))
        )
        if not self.train_completed or self.captured_steps != expected:
            raise UnverifiedState("Missing training completion or required step captures")
        if self.runtime_provenance() != self.provenance:
            raise UnverifiedState("Source/runtime changed during training")
        if self.args.resume is None:
            read_checkpoint_record(self.args.output, step=self.args.checkpoint_step, rank=self.rank)
        complete_rank(
            self.args.output,
            rank=self.rank,
            world_size=self.world_size,
            run_id=self.args.run_id,
            steps=self.captured_steps,
            provenance=self.provenance,
        )


def run_worker(args: argparse.Namespace) -> None:
    """Run pretrain_gpt with diagnostic hooks and retain its real artifacts."""
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    from megatron.determinism import configure_determinism

    configure_determinism({"deterministic_mode": True})
    from megatron.core.datasets.gpt_dataset import MockGPTLowLevelDataset
    from megatron.training import checkpointing, training

    capture = TrainingCapture(args, training, checkpointing)
    command = [
        str(ROOT / "pretrain_gpt.py"),
        *recipe_arguments(
            capture.world_size,
            args.steps,
            args.stop_step,
            pipeline_size=args.pipeline_size,
            virtual_pipeline_size=args.virtual_pipeline_size,
            optimizer_mode=args.optimizer_mode,
        ),
    ]
    command += [
        "--save",
        str(args.output / "megatron-checkpoints"),
        "--save-interval",
        str(args.checkpoint_step),
    ]
    if args.resume:
        command += [
            "--load",
            str(args.resume / "megatron-checkpoints"),
            "--ckpt-step",
            str(args.checkpoint_step),
        ]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"training-command-rank-{capture.rank:06d}.json").write_text(
        json.dumps(command, indent=2) + "\n"
    )
    with ExitStack() as stack:
        # Keep the real MockGPT indexing code, with explicitly recorded data size.
        stack.enter_context(patch.object(MockGPTLowLevelDataset, "size", 512))
        stack.enter_context(patch.object(MockGPTLowLevelDataset, "max_sequence_length", 64))
        for name, wrapper in (
            ("load_checkpoint", capture.wrap_load),
            ("save_checkpoint", capture.wrap_save),
            ("build_train_valid_test_data_loaders", capture.wrap_loaders),
            ("train", capture.wrap_train),
            ("post_training_step_callbacks", capture.wrap_post_step),
            ("checkpoint_and_decide_exit", capture.wrap_decide_exit),
        ):
            stack.enter_context(patch.object(training, name, wrapper(getattr(training, name))))
        stack.enter_context(patch.object(sys, "argv", command))
        try:
            runpy.run_path(str(ROOT / "pretrain_gpt.py"), run_name="__main__")
        except SystemExit as error:
            if error is not capture.expected_exit:
                raise
    capture.finish()


def main() -> None:
    """Entry point used by the shared four-launch coordinator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("megatron_gpt",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pipeline-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--virtual-pipeline-size", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--optimizer-mode", choices=("standard", "precision_aware_fp16"), default="standard"
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--checkpoint-step", type=int, default=2)
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--omit-restore", choices=("rng",))
    args = parser.parse_args()
    try:
        capture_configuration(args.steps, args.checkpoint_step, args.stop_step)
    except ValueError as error:
        parser.error(str(error))
    if not 0 < args.checkpoint_step < args.steps or (args.omit_restore and not args.resume):
        parser.error("Require a real resume interval and a checkpoint for the control")
    logging.basicConfig(level=logging.INFO)
    run_worker(args)


if __name__ == "__main__":
    main()
