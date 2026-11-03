# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Native Energon loaders on heterogeneous MIMO sample lanes.

Applications supply ordinary TaskEncoders and an encoder-batch merger. Data
formats, tokenization, targets and modality layouts do not belong to this bridge.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from megatron.energon import SavableDataLoader, TaskEncoder


@dataclass(frozen=True)
class EnergonDataConfig:
    """Application hooks and native options, selected by ``--energon-data-config``.

    ``create_task_encoder(*, split)`` returns a fresh, role-independent native
    TaskEncoder. Its batches must follow the existing MIMO model-input contract.
    ``merge_encoder_batches(batches, *, lane_ids, module_name)`` assembles ordered
    native batches into encoder inputs. It runs outside workers and must be pure:
    no sampling, RNG, loader advancement or persistent state. The application
    owns modality-specific concatenation, lengths and optional lazy decoding.

    ``checkpoint_identity`` identifies application configuration not captured by
    native ``config()``; change it when task behavior changes. Native dataset and
    loader options are forwarded, except bridge-owned placement/batching keys.
    """

    create_task_encoder: Callable[..., TaskEncoder]
    checkpoint_identity: str
    merge_encoder_batches: Callable[..., dict[str, Any]] | None = None
    dataset_kwargs: Mapping[str, Any] = field(default_factory=dict)
    loader_kwargs: Mapping[str, Any] = field(default_factory=dict)


def add_energon_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register application selection and checkpoint roots; keep data policy in the task."""
    group = parser.add_argument_group("MIMO Energon data", "Used with --dataset-provider energon.")
    group.add_argument(
        "--energon-data-config",
        metavar="MODULE:FUNCTION",
        help="Importable function called with parsed args to return EnergonDataConfig, "
        "for example my_project.data:make_config. Task encoders define the data and model batch.",
    )
    group.add_argument(
        "--energon-dataloader-load",
        metavar="ROOT",
        help="Dataloader checkpoint root containing iter_XXXXXXX/mp_rank_00 directories; "
        "defaults to --load and requires a model --load to resolve the iteration. "
        "Pass the root, not an iteration directory or a .pt file.",
    )
    group.add_argument(
        "--dataloader-save",
        metavar="ROOT",
        help="Dataloader checkpoint output root; defaults to --save. "
        "Only language sample lanes write authoritative state.",
    )
    return parser


def validate_energon_data_args(args: argparse.Namespace) -> None:
    """Reject unsupported topology before initialization, without importing an application."""
    if args.dataset_provider != "energon":
        return
    module, separator, function = (args.energon_data_config or "").partition(":")
    if not module or not separator or not function or ":" in function:
        raise ValueError("--energon-data-config must be MODULE:FUNCTION")
    if not args.data_path or len(args.data_path) != 1:
        raise ValueError("Energon requires one --data-path dataset or recipe; blend in the recipe")
    if getattr(args, "energon_dataloader_load", None) and not getattr(args, "load", None):
        raise ValueError("--energon-dataloader-load requires model --load to resolve the iteration")
    for name in ("mimo_llm_tp", "mimo_llm_pp", "mimo_llm_cp", "mimo_encoder_tp"):
        if getattr(args, name) != 1:
            raise ValueError(f"initial Energon MIMO support requires {name}=1")
    if getattr(args, "gtp_weight_remat_size", 1) != 1:
        raise ValueError("Energon MIMO GTP rematerialization is not supported yet")
    if args.micro_batch_size < 1 or type(args.num_workers) is not int or args.num_workers < 0:
        raise ValueError("Energon requires positive micro-batch size and nonnegative workers")
    if args.eval_iters != 0:
        raise ValueError("initial Energon MIMO support is train-only; use --eval-iters 0")
    if args.mimo_encoder_prefetch or args.mimo_encoder_ddp_overlap:
        raise ValueError("Energon MIMO does not support encoder prefetch or encoder DDP overlap")
    if not args.mimo_llm_only:
        if args.mimo_encoder_dp <= 0 or args.mimo_llm_dp % args.mimo_encoder_dp:
            raise ValueError("encoder DP must be positive and divide the language sample lanes")
        if args.num_workers == 0 and args.mimo_llm_dp != args.mimo_encoder_dp:
            raise ValueError(
                "unequal-DP Energon loading requires positive --num-workers: "
                "native zero-worker RNG state is shared across lane replicas"
            )
    args.dataloader_save = args.dataloader_save or args.save


def _close_loader(loader: SavableDataLoader) -> None:
    close = getattr(loader, "close", None)
    if close is not None:
        close()
    else:
        # The existing 7.4.1 dependency has no close API. Terminate its owned
        # persistent PyTorch iterator explicitly, rather than waiting for GC.
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()


class EnergonMimoIterator:
    """Deliver native lane batches and delegate authoritative state to language lanes."""

    def __init__(
        self,
        loaders: list[SavableDataLoader],
        contracts: list[dict[str, Any]],
        *,
        lane_ids: Sequence[int],
        module_name: str | None,
        merge_encoder_batches: Callable[..., dict[str, Any]] | None,
    ) -> None:
        self._loaders = loaders
        self._contracts = contracts
        self._iterators = []
        self.lane_ids = tuple(lane_ids)
        self.module_name = module_name
        self._merge_encoder_batches = merge_encoder_batches
        self._closed = False

    def __iter__(self) -> EnergonMimoIterator:
        return self

    def __next__(self) -> dict[str, Any]:
        if self._closed:
            raise StopIteration
        try:
            if not self._iterators:
                for loader in self._loaders:
                    self._iterators.append(iter(loader))
            batches = [next(iterator) for iterator in self._iterators]
            expected_rows = self._contracts[0]["micro_batch_size"]
            for batch in batches:
                if not isinstance(batch, dict) or not isinstance(
                    batch.get("input_ids"), torch.Tensor
                ):
                    raise ValueError("TaskEncoder must produce a MIMO dict with tensor input_ids")
                if batch["input_ids"].ndim != 2 or batch["input_ids"].shape[0] != expected_rows:
                    raise ValueError("TaskEncoder input_ids must have micro-batch-size rows")
            if self.module_name is None:
                result = dict(batches[0])
                result.pop("modality_inputs", None)
            else:
                result = self._merge_encoder_batches(
                    batches, lane_ids=self.lane_ids, module_name=self.module_name
                )
                if (
                    not isinstance(result, dict)
                    or not isinstance(result.get("input_ids"), torch.Tensor)
                    or result["input_ids"].ndim != 2
                    or result["input_ids"].shape[0] != expected_rows * len(batches)
                ):
                    raise ValueError("encoder merger must preserve all lane-major input_ids rows")
                modalities = result.get("modality_inputs")
                if modalities and self.module_name not in modalities:
                    raise ValueError("modality_inputs must be keyed by the encoder module name")
                result = dict(result)
            result.pop("__restore_key__", None)
            return result
        except BaseException:
            # A failed partial read/merge cannot be retried without losing lane alignment.
            self.close()
            raise

    def save_state(self) -> dict | None:
        """Use the stock external-iterator hook; encoder replicas write no state."""
        if self._closed:
            raise RuntimeError("cannot checkpoint a closed Energon iterator")
        if self.module_name is not None:
            return None
        return {"contract": self._contracts[0], "energon": self._loaders[0].save_state_rank()}

    def close(self) -> None:
        """Close every generator and loader, including partially constructed iterations."""
        if self._closed:
            return
        self._closed = True
        with ExitStack() as cleanup:
            for loader in self._loaders:
                cleanup.callback(_close_loader, loader)
            for iterator in self._iterators:
                cleanup.callback(iterator.close)
            self._iterators.clear()
            self._loaders.clear()


def build_energon_iterator(
    args: argparse.Namespace,
    *,
    lane_ids: Sequence[int],
    language_lanes: int,
    module_name: str | None,
    checkpoint_paths: Mapping[int, Path] | None = None,
    allow_missing_checkpoint: bool = False,
) -> EnergonMimoIterator:
    """Build native loaders with explicit sample identities, restoring before iteration.

    ``module_name=None`` denotes a single language lane. Other modules consume
    ordered language-lane replicas through the application's encoder merger.
    ``checkpoint_paths`` maps lane IDs to canonical .pt files, not directories.
    """
    from megatron.energon import WorkerConfig, __version__, get_savable_loader, get_train_dataset
    from megatron.energon.errors import reraise_exception

    if (
        not lane_ids
        or len(set(lane_ids)) != len(lane_ids)
        or any(not 0 <= lane < language_lanes for lane in lane_ids)
        or (module_name is None and len(lane_ids) != 1)
    ):
        raise ValueError("invalid ordered language sample lanes")
    if len(lane_ids) > 1 and args.num_workers == 0:
        raise ValueError("multiple Energon lane replicas require positive num_workers")
    module, function = args.energon_data_config.split(":", 1)
    config = getattr(importlib.import_module(module), function)(args)
    if not isinstance(config, EnergonDataConfig):
        raise TypeError("--energon-data-config must return EnergonDataConfig")
    if (
        not callable(config.create_task_encoder)
        or not isinstance(config.checkpoint_identity, str)
        or not config.checkpoint_identity
    ):
        raise ValueError("EnergonDataConfig needs a task-encoder factory and checkpoint_identity")
    if module_name is not None and not callable(config.merge_encoder_batches):
        raise ValueError("encoder data requires merge_encoder_batches")
    dataset_kwargs = dict(config.dataset_kwargs)
    loader_kwargs = dict(config.loader_kwargs)
    if {"path", "worker_config", "task_encoder", "batch_size"} & dataset_kwargs.keys():
        raise ValueError("dataset_kwargs cannot override bridge-owned placement or batching")
    if {"dataset", "worker_config"} & loader_kwargs.keys():
        raise ValueError("loader_kwargs cannot override bridge-owned dataset or worker config")
    options = {"shuffle_buffer_size": None, "max_samples_per_sequence": None, **dataset_kwargs}
    loaders, contracts = [], []
    with ExitStack() as cleanup:
        for lane in lane_ids:
            workers = WorkerConfig(
                rank=lane,
                world_size=language_lanes,
                num_workers=args.num_workers,
                seed_offset=args.seed,
                data_parallel_group=None,
                global_error_handler=reraise_exception,
                restore_error_handler=reraise_exception,
            )
            dataset = get_train_dataset(
                args.data_path[0],
                worker_config=workers,
                batch_size=args.micro_batch_size,
                task_encoder=config.create_task_encoder(split="train"),
                **options,
            )
            loader = get_savable_loader(dataset, **loader_kwargs)
            cleanup.callback(_close_loader, loader)
            loaders.append(loader)
            native_config = json.dumps(loader.config(), sort_keys=True).encode("utf-8")
            contract = {
                "application": args.energon_data_config,
                "checkpoint_identity": config.checkpoint_identity,
                "native_config_sha256": hashlib.sha256(native_config).hexdigest(),
                "energon_version": __version__,
                "lane": lane,
                "language_lanes": language_lanes,
                "num_workers": args.num_workers,
                "micro_batch_size": args.micro_batch_size,
                "seed": args.seed,
            }
            contracts.append(contract)
            if checkpoint_paths is not None:
                path = checkpoint_paths[lane]
                if path.is_file():
                    state = torch.load(path, map_location="cpu", weights_only=False)[
                        "dataloader_state_dict"
                    ]
                    if not isinstance(state, dict) or state.get("contract") != contract:
                        raise ValueError(f"incompatible Energon dataloader state: {path}")
                    loader.restore_state_rank(state["energon"])
                elif not allow_missing_checkpoint:
                    raise FileNotFoundError(f"missing Energon dataloader state: {path}")
        iterator = EnergonMimoIterator(
            loaders,
            contracts,
            lane_ids=lane_ids,
            module_name=module_name,
            merge_encoder_batches=config.merge_encoder_batches,
        )
        cleanup.pop_all()
    return iterator
