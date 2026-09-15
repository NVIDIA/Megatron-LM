# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Hybrid training entry point with experiment-local MTP scheduling and audit records."""

import argparse
import json
import os
from functools import partial
from pathlib import Path
from typing import Any

import torch

import pretrain_hybrid
from examples.engram.audit_data import audit_training_indices
from examples.engram.events import EventJournal
from examples.engram.parameter_manifest import (
    assert_expected_parameters,
    collect_parameter_manifest,
    local_parameter_records,
)
from examples.engram.recipe import (
    PaddedEvaluationDataset,
    Schedule,
    archive_invocation_manifest,
    recipe_manifest,
    set_mtp_weight,
)
from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.enums import ModelType
from megatron.core.optimizer_param_scheduler import get_canonical_lr_for_logging
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_attr_wrapped_model
from megatron.training import get_args, get_tensorboard_writer, get_wandb_writer, pretrain
from megatron.training.argument_utils import (
    hybrid_config_from_args,
    pretrain_cfg_container_from_args,
)
from megatron.training.arguments import parse_and_validate_args
from megatron.training.initialize import initialize_megatron


def add_recipe_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add experiment controls without extending Megatron's core configuration."""
    group = parser.add_argument_group("FineWeb comparison recipe")
    group.add_argument("--recipe-artifacts", type=Path, required=True)
    group.add_argument("--recipe-audit-data", action="store_true")
    group.add_argument("--recipe-full-eval-label", choices=("valid", "test"))
    group.add_argument("--recipe-validate-only", action="store_true")
    group.add_argument("--recipe-parameters-only", action="store_true")
    return parser


def write_metrics(metrics: dict[str, float], step: int) -> None:
    """Write identical scalar names and values through Megatron's existing logger rank."""
    tensorboard = get_tensorboard_writer()
    wandb = get_wandb_writer()
    if tensorboard:
        for name, value in metrics.items():
            tensorboard.add_scalar(name, value, step)
    if wandb:
        wandb.log(metrics, step=step)


def write_final_metric(name: str, value: float, checkpoint_step: int) -> None:
    """Keep checkpoint coordinates without rewinding a resumed W&B history stream."""
    if name not in ("final/valid_ce", "final/test_ce"):
        raise ValueError(f"Unsupported final evaluation metric: {name}")
    tensorboard = get_tensorboard_writer()
    wandb = get_wandb_writer()
    if tensorboard:
        tensorboard.add_scalar(name, value, checkpoint_step)
    if wandb:
        wandb.define_metric("final/checkpoint_step")
        wandb.define_metric("final/*", step_metric="final/checkpoint_step")
        # W&B's internal history may already have advanced beyond the saved
        # checkpoint. Let it advance normally; the chart uses the explicit axis.
        wandb.log({"final/checkpoint_step": checkpoint_step, name: value})


class FullEvaluationGPTDataset(pretrain_hybrid.GPTDataset):
    """Retain a partial holdout tail when the tokenizer has no padding token.

    The negative sentinel is only a dataset padding marker, never a tokenizer
    vocabulary entry. Native GPTDataset replaces it with token 0 and masks its
    loss before model input. Training datasets and tokenizer definitions stay unchanged.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self._pad_token_id is None:
            self._pad_token_id = -1

    def __getitem__(self, index: int | None) -> dict:
        was_cached = self.masks_and_position_ids_are_cached
        sample = super().__getitem__(index)
        if self.masks_and_position_ids_are_cacheable and not was_cached:
            # GPTDataset initially aliases its cached mask with the returned mask.
            # A shuffled partial first sample must not poison later full samples.
            # Cacheable masks have no EOD/position/attention reset and start at one.
            self.cached_loss_mask = torch.ones_like(self.cached_loss_mask)
        return sample


class Recipe:
    """Wrap native data and forward functions; model math and optimizer scheduling stay native."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.artifacts = args.recipe_artifacts
        self.audit_data = args.recipe_audit_data
        self.full_eval_label = args.recipe_full_eval_label
        self.schedule = Schedule()
        self.last_logged_step = None
        self.recorded_models = set()
        self.eval_numerator = None
        self.eval_denominator = None
        self.eval_calls = 0
        self.is_distributed = True
        self.events = EventJournal(
            self.artifacts,
            Path(args.save) if args.save else None,
            get_tensorboard_writer,
            get_wandb_writer,
        )
        self.started = False

    def datasets(self, sizes: list, **kwargs: Any) -> tuple:
        """Build native datasets, audit the training stream, or pad full holdout evaluation."""
        if self.full_eval_label:
            config = pretrain_hybrid.core_gpt_dataset_config_from_args(get_args())
            config.drop_last_partial_validation_sequence = False
            train, valid, test = BlendedMegatronDatasetBuilder(
                FullEvaluationGPTDataset,
                [0, None, 0],
                partial(pretrain_hybrid.is_dataset_built_on_rank, **kwargs),
                config,
            ).build()
        else:
            train, valid, test = pretrain_hybrid.train_valid_test_datasets_provider(sizes, **kwargs)
            # Native end-of-training short test reuses the validation metric tag.
            # This recipe evaluates test data only through its independent full-test entry.
            test = None
        if self.audit_data and train is not None and torch.distributed.get_rank() == 0:
            audit = audit_training_indices(train, 36754 * 64)
            target = self.artifacts / "data_audit.json"
            if target.exists() and json.loads(target.read_text()) != audit:
                raise RuntimeError("Dataset indices changed relative to this run's frozen audit")
            target.write_text(json.dumps(audit, indent=2) + "\n")
        if self.full_eval_label and valid is not None:
            valid = PaddedEvaluationDataset(valid, mpu.get_data_parallel_world_size())
        return train, valid, test

    def forward(self, data_iterator: Any, model: Any) -> tuple:
        """Set MTP weight before each forward, including every PP/VPP chunk and microbatch."""
        args = get_args()
        completed = getattr(args, "curr_iteration", args.iteration)
        weight = self.schedule.mtp_weight(completed)
        set_mtp_weight(model, weight)
        args.mtp_loss_scaling_factor = weight
        if id(model) not in self.recorded_models:
            self.recorded_models.add(id(model))
            self._record_model(model)
        if model.training and completed != self.last_logged_step:
            if not self.started:
                self.started = True
                self.events.emit(
                    "checkpoint_loaded" if args.iteration else "training_started",
                    args.iteration,
                    load=args.load,
                    completed_samples=args.consumed_train_samples,
                )
            if args.save_interval and completed % args.save_interval == 0:
                self.events.observe_checkpoints()
            if completed == self.schedule.train_steps - self.schedule.decay_steps:
                self.events.emit("mtp_weight_changed", completed + 1, weight=weight)
            self.last_logged_step = completed
            config = get_attr_wrapped_model(model, "config")
            scale_loss = getattr(config, "grad_scale_func", None)
            optimizer = getattr(scale_loss, "__self__", None)
            metrics = {
                "recipe/mtp_weight": weight,
                "recipe/phase": {"warmup": 0, "stable": 1, "decay": 2}[
                    self.schedule.phase(completed)
                ],
                "recipe/completed_samples_before_update": float(args.consumed_train_samples),
                "recipe/completed_tokens_before_update": float(
                    args.consumed_train_samples * args.seq_length
                ),
            }
            if optimizer is None:
                raise RuntimeError("Cannot verify update LR: model config has no bound optimizer")
            groups = optimizer.param_groups
            ordinary_lr = get_canonical_lr_for_logging(groups)
            if ordinary_lr is None:
                raise RuntimeError("Cannot identify the ordinary optimizer learning rate")
            metrics["recipe/update_lr"] = float(ordinary_lr)
            sparse = [group for group in groups if group.get("lr_mult", 1.0) == 5.0]
            if sparse:
                metrics["recipe/engram_update_lr"] = float(sparse[0]["lr"])
            write_metrics(metrics, completed + 1)
        elif not model.training:
            self.events.emit(
                "validation_started", args.consumed_train_samples // args.global_batch_size
            )
        output, loss = pretrain_hybrid.forward_step(data_iterator, model)
        if not self.full_eval_label:
            return output, loss

        def full_eval_loss(output_tensor: torch.Tensor) -> tuple:
            result = loss(output_tensor)
            pair = result[2]["lm loss"]
            self.eval_numerator = (
                pair[0].double() if self.eval_numerator is None else self.eval_numerator + pair[0]
            )
            self.eval_denominator = (
                pair[1].double()
                if self.eval_denominator is None
                else self.eval_denominator + pair[1]
            )
            self.eval_calls += 1
            if self.eval_calls == args.eval_iters:
                totals = torch.stack([self.eval_numerator, self.eval_denominator])
                collection = get_attr_wrapped_model(model, "pg_collection")
                torch.distributed.all_reduce(totals, group=collection.dp_cp)
                metric = f"final/{self.full_eval_label}_ce"
                write_final_metric(metric, (totals[0] / totals[1]).item(), args.iteration)
                if get_tensorboard_writer():
                    record = {
                        "step": args.iteration,
                        "loss_sum": totals[0].item(),
                        "valid_main_tokens": totals[1].item(),
                        "ce": (totals[0] / totals[1]).item(),
                    }
                    (self.artifacts / f"full_{self.full_eval_label}.json").write_text(
                        json.dumps(record, indent=2) + "\n"
                    )
            return result

        return output, full_eval_loss

    def _record_model(self, model: Any) -> None:
        rank = torch.distributed.get_rank()
        stage = get_attr_wrapped_model(model, "vp_stage", allow_none=True)
        records = local_parameter_records(model)
        (self.artifacts / f"parameters_rank{rank}_chunk{stage}.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
        if len(self.recorded_models) == 1 and get_tensorboard_writer():
            metadata = {
                "recipe": recipe_manifest(bool(get_args().engram_layer_ids)),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "wandb_mode": os.environ.get("WANDB_MODE", "online"),
                "wandb_run_id": os.environ.get("WANDB_RUN_ID"),
            }
            audit_path = self.artifacts / "data_audit.json"
            if audit_path.exists():
                metadata["data_audit"] = json.loads(audit_path.read_text())
            get_tensorboard_writer().add_text(
                "recipe/manifest", json.dumps(metadata, indent=2), get_args().iteration
            )
            if get_wandb_writer():
                get_wandb_writer().config.update(metadata, allow_val_change=True)


def main() -> None:
    """Run the native pretraining driver with the frozen experiment recipe."""
    args = parse_and_validate_args(extra_args_provider=add_recipe_args)
    args.recipe_artifacts.mkdir(parents=True, exist_ok=True)
    recipe = Recipe(args)
    if int(os.environ.get("RANK", "0")) == 0:
        manifest = recipe_manifest(bool(args.engram_layer_ids))
        manifest["execution"] = {
            "exit_interval": args.exit_interval,
            "eval_interval": args.eval_interval,
            "eval_iters": args.eval_iters,
            "save_interval": args.save_interval,
            "full_eval_label": args.recipe_full_eval_label,
            "micro_batch_size": args.micro_batch_size,
            "world_size": args.world_size,
            "tensor_parallel": args.tensor_model_parallel_size,
            "pipeline_parallel": args.pipeline_model_parallel_size,
            "context_parallel": args.context_parallel_size,
            "expert_parallel": args.expert_model_parallel_size,
        }
        archive_invocation_manifest(args.recipe_artifacts, manifest)
    config = pretrain_cfg_container_from_args(args, hybrid_config_from_args(args))
    if args.recipe_parameters_only:
        initialize_megatron(skip_dependency_compilation=True)
        builder = config.model.get_builder_cls()(config.model)
        models = builder.build_distributed_models(
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(), wrap_with_ddp=False
        )
        actual = collect_parameter_manifest(models)
        expected = recipe_manifest(bool(args.engram_layer_ids))["expected_parameters"]
        assert_expected_parameters(
            actual, {key: value for key, value in expected.items() if key in actual["totals"]}
        )
        if torch.distributed.get_rank() == 0:
            (args.recipe_artifacts / "actual_parameters.json").write_text(
                json.dumps(actual, indent=2) + "\n"
            )
            print(json.dumps(actual["totals"], indent=2))
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        return
    if args.recipe_validate_only:
        print(
            json.dumps(
                {
                    "hybrid_pattern": args.hybrid_layer_pattern,
                    "hidden_size": args.hidden_size,
                    "ffn_hidden_size": args.ffn_hidden_size,
                    "moe_ffn_hidden_size": args.moe_ffn_hidden_size,
                    "num_experts": args.num_experts,
                    "mtp_num_layers": args.mtp_num_layers,
                    "train_iters": args.train_iters,
                    "lr": args.lr,
                    "min_lr": args.min_lr,
                    "seed": args.seed,
                    "engram_hash_table_min_sizes": args.engram_hash_table_min_sizes,
                },
                indent=2,
            )
        )
        return

    # Bound methods cannot carry attributes. The closure preserves the native
    # distributed-provider marker and the signature accepted by the PP/VPP driver.
    def datasets(
        sizes: list, vp_stage: int | None = None, requires_token_ids: bool = False
    ) -> tuple:
        return recipe.datasets(sizes, vp_stage=vp_stage, requires_token_ids=requires_token_ids)

    setattr(datasets, "is_distributed", True)
    outcome, error_type = "completed", None
    try:
        pretrain(config, datasets, ModelType.encoder_or_decoder, recipe.forward)
    except BaseException as error:
        outcome = "exited" if isinstance(error, SystemExit) and not error.code else "failed"
        error_type = type(error).__name__
        raise
    finally:
        completed = args.consumed_train_samples // args.global_batch_size
        recipe.events.finish(completed, outcome, error_type)


if __name__ == "__main__":
    main()
