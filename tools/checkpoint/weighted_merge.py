# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Weighted averaging utility for Megatron distributed checkpoints.

This tool is intentionally kept under ``tools/checkpoint``. It derives the merge
structure entirely from each source checkpoint's public DCP metadata (FQN,
global shape, dtype, chunk layout, byte/object ``_extra_state``): no Megatron
model or sharded-state template is ever constructed. Floating model tensors are
accumulated in fp32 on CPU; Transformer Engine ``_extra_state`` entries are
copied from one source checkpoint instead of averaged. Because no model is
built, it imports no CUDA-only kernels and runs CPU-only for every model family
(GPT/Mamba/hybrid/TE).
"""

import argparse
import copy
import io
import math
import os
import random
import re
import resource
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as torch_dcp
from torch.distributed.checkpoint import CheckpointException, FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.default_planner import create_default_global_save_plan
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    SavePlan,
    SavePlanner,
    TensorWriteData,
    WriteItem,
    WriteItemType,
)

from megatron.core import dist_checkpointing
from megatron.core._rank_utils import safe_get_rank
from megatron.core.dist_checkpointing.core import maybe_load_config
from megatron.core.dist_checkpointing.mapping import (
    ShardedObject,
    ShardedStateDict,
    ShardedTensor,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistLoadShardedStrategy,
)
from megatron.core.dist_checkpointing.utils import (
    force_all_tensors_to_non_fp8,
)
from megatron.core.dist_checkpointing.validation import (
    StrictHandling,
)

ITERATION_RE = re.compile(r"^iter_(\d+)$")
LATEST_CHECKPOINTED_ITERATION = "latest_checkpointed_iteration.txt"
SAVE_DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
VALID_MODIFIERS = ("reverse", "scramble")
BYTE_ACCOUNTING_MODES = ("none", "rank0", "all")
RESOURCE_LOG_MODES = ("none", "rank0", "all")

# The sole merge path derives the merge structure from public DCP metadata only
# (no model/template construction, no CUDA kernels), so it merges any model
# family on a GPU-less CPU node.
METADATA_SAME_LAYOUT_MODE = "dcp-metadata-same-layout"

METADATA_SAME_LAYOUT_MODEL_PREFIXES = ("model.", "model0.", "model1.")
METADATA_SAME_LAYOUT_UNPREFIXED_MODEL_ROOTS = (
    "decoder.",
    "embedding.",
    "output_layer.",
    "mtp.",
)


class WeightedMergeError(ValueError):
    """Raised when weighted checkpoint merge inputs are invalid."""


@dataclass(frozen=True)
class MergeTimings:
    """Wall-clock timing split for a checkpoint merge."""

    discovery: float = 0.0
    model_init: float = 0.0
    load: float = 0.0
    accumulation: float = 0.0
    save: float = 0.0
    verification: float = 0.0
    total: float = 0.0


@dataclass(frozen=True)
class MergeMemoryEstimate:
    """Local-rank tensor residency estimate for the whole-shard merge path."""

    mergeable_tensors: int = 0
    extra_state_entries: int = 0
    loaded_checkpoint_bytes: int = 0
    accumulator_bytes: int = 0
    output_tensor_bytes: int = 0
    extra_state_tensor_bytes: int = 0
    projected_cpu_peak_bytes: int = 0
    template_devices: tuple[str, ...] = ()


@dataclass(frozen=True)
class MergeResult:
    """Result metadata returned after a successful merge."""

    output_dir: Path
    input_dirs: tuple[Path, ...]
    weights: tuple[float, ...]
    timings: MergeTimings
    averaged_tensors: int
    copied_extra_states: int
    bytes_read: int = 0
    bytes_written: int = 0
    backend: str = ""
    verified_load: bool = False
    host_peak_bytes: int = 0
    max_host_peak_bytes: int = 0
    max_host_peak_rank: int = 0
    world_size: int = 1
    rank: int = 0
    implementation_mode: str = METADATA_SAME_LAYOUT_MODE
    memory_estimate: MergeMemoryEstimate = MergeMemoryEstimate()
    preflight_only: bool = False
    balance_rank_work: bool = False
    plan_tensor_bytes_by_rank: tuple[int, ...] = ()
    plan_tensor_chunks_by_rank: tuple[int, ...] = ()


@dataclass(frozen=True)
class _DirectDcpWriteSpec:
    path: tuple[Union[str, int], ...]
    load_path: tuple[Union[str, int], ...]
    template_leaf: Any
    sharded_key: str
    global_shape: tuple[int, ...]
    global_offsets: tuple[int, ...]
    chunk_shape: tuple[int, ...]
    target_dtype: torch.dtype
    chunk_dim: int
    chunk_start: int
    chunk_length: int
    source_dtype: Optional[torch.dtype] = None
    source_shape: Optional[tuple[int, ...]] = None
    is_extra_state: bool = False


@dataclass(frozen=True)
class _DcpMetadataByteSpec:
    fqn: str
    path: tuple[Union[str, int], ...]
    template_leaf: ShardedObject


@dataclass(frozen=True)
class _DcpMetadataTensorLayout:
    global_shape: tuple[int, ...]
    dtype: torch.dtype
    chunks: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]


@dataclass(frozen=True)
class _DcpMetadataSnapshot:
    tensor_metadata: dict[str, TensorStorageMetadata]
    byte_extra_state_keys: tuple[str, ...]


@dataclass(frozen=True)
class _MetadataSameLayoutWorkPlan:
    write_specs: list[_DirectDcpWriteSpec]
    byte_specs: list[_DcpMetadataByteSpec]
    merge_keys: int
    extra_state_keys: int
    tensor_bytes_by_rank: tuple[int, ...]
    tensor_chunks_by_rank: tuple[int, ...]


class _WeightedMergeDirectOutputSavePlanner(SavePlanner):
    """Public DCP planner that resolves exactly one merged output chunk per item."""

    def __init__(
        self,
        *,
        write_specs: list[_DirectDcpWriteSpec],
        byte_specs: Optional[list[_DcpMetadataByteSpec]] = None,
        resolved_input_dirs: list[Path],
        weights: list[float],
        load_strategies: dict[Path, TorchDistLoadShardedStrategy],
        extra_state_source_index: int,
    ) -> None:
        self.write_specs = write_specs
        self.byte_specs = byte_specs or []
        self.resolved_input_dirs = resolved_input_dirs
        self.weights = weights
        self.load_strategies = load_strategies
        self.extra_state_source_index = extra_state_source_index
        self.load_time = 0.0
        self.accumulation_time = 0.0
        self.resolved_tensor_count = 0
        self.max_resolved_output_tensor_bytes = 0
        self._write_spec_by_index = {
            (spec.sharded_key, spec.global_offsets, spec.chunk_shape): spec
            for spec in write_specs
        }
        self._byte_spec_by_fqn = {spec.fqn: spec for spec in self.byte_specs}
        self._plan = SavePlan([])

    def set_up_planner(
        self,
        state_dict: ShardedStateDict,
        storage_meta: Any = None,
        is_coordinator: bool = False,
    ) -> None:
        self._state_dict = state_dict
        self._storage_meta = storage_meta
        self._is_coordinator = is_coordinator

    def create_local_plan(self) -> SavePlan:
        items = [
            self._write_item(spec, index)
            for index, spec in enumerate(self.write_specs)
        ]
        items.extend(self._byte_write_item(spec) for spec in self.byte_specs)
        self._plan = SavePlan(items)
        return self._plan

    def create_global_plan(self, all_plans: list[SavePlan]) -> tuple[list[SavePlan], Any]:
        return create_default_global_save_plan(all_plans)

    def finish_plan(self, new_plan: SavePlan) -> SavePlan:
        self._plan = new_plan
        return self._plan

    def resolve_data(self, write_item: WriteItem) -> Union[torch.Tensor, io.BytesIO]:
        if write_item.type == WriteItemType.BYTE_IO:
            return self._resolve_byte_object(write_item)
        if write_item.type != WriteItemType.SHARD or write_item.tensor_data is None:
            raise WeightedMergeError(
                "direct-dcp-streaming emits only tensor shard or byte-object write items."
            )
        chunk = write_item.tensor_data.chunk
        spec_key = (
            write_item.index.fqn,
            tuple(int(offset) for offset in chunk.offsets),
            tuple(int(size) for size in chunk.sizes),
        )
        try:
            spec = self._write_spec_by_index[spec_key]
        except KeyError as exc:
            raise WeightedMergeError(
                "direct-dcp-streaming could not resolve public DCP write item "
                f"for '{write_item.index.fqn}' at offsets={tuple(chunk.offsets)} "
                f"sizes={tuple(chunk.sizes)}."
            ) from exc
        if spec.is_extra_state:
            output = self._resolve_extra_state(spec)
        else:
            output = self._resolve_merged_chunk(spec)
        self.resolved_tensor_count += 1
        self.max_resolved_output_tensor_bytes = max(
            self.max_resolved_output_tensor_bytes,
            int(output.numel() * output.element_size()),
        )
        return output

    @staticmethod
    def _write_item(spec: _DirectDcpWriteSpec, index: int) -> WriteItem:
        fake_tensor = torch.empty((), dtype=spec.target_dtype, device="cpu")
        offsets = torch.Size(spec.global_offsets)
        sizes = torch.Size(spec.chunk_shape)
        return WriteItem(
            index=MetadataIndex(spec.sharded_key, offsets, index),
            type=WriteItemType.SHARD,
            tensor_data=TensorWriteData(
                chunk=ChunkStorageMetadata(offsets=offsets, sizes=sizes),
                properties=TensorProperties.create_from_tensor(fake_tensor),
                size=torch.Size(spec.global_shape),
            ),
        )

    @staticmethod
    def _byte_write_item(spec: _DcpMetadataByteSpec) -> WriteItem:
        return WriteItem(
            index=MetadataIndex(spec.fqn),
            type=WriteItemType.BYTE_IO,
            tensor_data=None,
        )

    def _resolve_merged_chunk(self, spec: _DirectDcpWriteSpec) -> torch.Tensor:
        if spec.source_dtype is None or spec.source_shape is None:
            raise WeightedMergeError(
                f"Direct DCP tensor plan for '{_path_label(spec.path, spec.template_leaf)}' "
                "is missing source metadata."
            )
        load_leaf = _streaming_chunk_leaf(
            spec.template_leaf,
            chunk_dim=spec.chunk_dim,
            chunk_start=spec.chunk_start,
            chunk_length=spec.chunk_length,
            dtype=spec.source_dtype,
            global_shape=spec.source_shape,
        )
        path_leaves = [(spec.load_path, load_leaf)]
        accumulator = torch.zeros(spec.chunk_shape, dtype=torch.float32, device="cpu")
        for checkpoint_dir, weight in zip(self.resolved_input_dirs, self.weights):
            load_start = time.perf_counter()
            loaded_by_path = _load_tensor_path_group_fast(
                checkpoint_dir,
                path_leaves,
                self.load_strategies[checkpoint_dir],
            )
            self.load_time += time.perf_counter() - load_start
            tensor = _as_tensor(loaded_by_path[spec.load_path])
            if tensor is None:
                raise WeightedMergeError(
                    f"Checkpoint {checkpoint_dir} key '{_path_label(spec.path)}' "
                    "is not a tensor."
                )
            if not tensor.is_floating_point():
                raise WeightedMergeError(
                    f"Checkpoint {checkpoint_dir} key '{_path_label(spec.path)}' "
                    f"has non-floating dtype {tensor.dtype}; weighted averaging is "
                    "only supported for floating tensors."
                )
            if tuple(tensor.shape) != spec.chunk_shape:
                if tensor.numel() != accumulator.numel():
                    raise WeightedMergeError(
                        f"Checkpoint {checkpoint_dir} key '{_path_label(spec.path)}' "
                        f"loaded chunk shape {tuple(tensor.shape)} cannot be reshaped "
                        f"to {spec.chunk_shape}."
                    )
                tensor = tensor.reshape(spec.chunk_shape)
            accumulation_start = time.perf_counter()
            # Accumulation invariant: floating tensors are accumulated in fp32 on CPU,
            # applying the per-checkpoint weight via add_(alpha=weight).
            source_tensor = tensor.detach()
            if source_tensor.device.type != "cpu":
                source_tensor = source_tensor.to(device="cpu")
            accumulator.add_(source_tensor, alpha=weight)
            self.accumulation_time += time.perf_counter() - accumulation_start
            del loaded_by_path, tensor, source_tensor
        return accumulator.to(dtype=spec.target_dtype).contiguous()

    def _resolve_extra_state(self, spec: _DirectDcpWriteSpec) -> torch.Tensor:
        load_leaf = copy.deepcopy(spec.template_leaf)
        template_tensor = _as_tensor(load_leaf)
        if template_tensor is not None:
            _assign_leaf_data(load_leaf, torch.empty_like(template_tensor, device="cpu"))
        load_start = time.perf_counter()
        loaded = _load_path_group(
            self.resolved_input_dirs[self.extra_state_source_index],
            [(spec.path, load_leaf)],
            strict=StrictHandling.ASSUME_OK_UNEXPECTED,
        )
        self.load_time += time.perf_counter() - load_start
        value = _copy_loaded_value(loaded[spec.path])
        tensor = _as_tensor(value)
        if tensor is None:
            raise WeightedMergeError(
                "direct-dcp-streaming currently supports tensor _extra_state leaves only; "
                f"'{_path_label(spec.path, spec.template_leaf)}' loaded as {type(value).__name__}."
            )
        return tensor.detach().cpu().contiguous()

    def _resolve_byte_object(self, write_item: WriteItem) -> io.BytesIO:
        try:
            spec = self._byte_spec_by_fqn[write_item.index.fqn]
        except KeyError as exc:
            raise WeightedMergeError(
                "direct-dcp-streaming could not resolve public DCP byte-object write item "
                f"for '{write_item.index.fqn}'."
            ) from exc
        load_start = time.perf_counter()
        loaded = _load_tensor_path_group_fast(
            self.resolved_input_dirs[self.extra_state_source_index],
            [(spec.path, spec.template_leaf)],
            self.load_strategies[self.resolved_input_dirs[self.extra_state_source_index]],
        )
        self.load_time += time.perf_counter() - load_start
        value = loaded[spec.path]
        if isinstance(value, io.BytesIO):
            value = io.BytesIO(value.getvalue())
        elif isinstance(value, (bytes, bytearray)):
            value = io.BytesIO(value)
        else:
            value = copy.deepcopy(value)
        serialized = io.BytesIO()
        torch.save([value], serialized)
        serialized.seek(0)
        return serialized


def is_rank_0() -> bool:
    return safe_get_rank() == 0


def print_rank_0(*args: Any, **kwargs: Any) -> None:
    if is_rank_0():
        print(*args, **kwargs)


def ensure_process_group() -> None:
    """Initialize a gloo process group when one is not already active."""

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available.")
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    dist.init_process_group(
        backend="gloo", rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"])
    )


def iteration_dir_name(iteration: int) -> str:
    if iteration < 0:
        raise WeightedMergeError(f"Iteration must be non-negative, got {iteration}.")
    return f"iter_{iteration:07d}"


def _schedule_linear_decay(x: float) -> float:
    if x < 0:
        raise WeightedMergeError(f"Schedule position must be non-negative, got {x}.")
    return 1 - x


def _schedule_minus_sqrt_decay(x: float) -> float:
    if x < 0:
        raise WeightedMergeError(f"Schedule position must be non-negative, got {x}.")
    return 1 - math.sqrt(x)


SCHEDULES: dict[str, Callable[[float], float]] = {
    "linear": _schedule_linear_decay,
    "minus-sqrt": _schedule_minus_sqrt_decay,
}


def parse_schedule_style(style: str) -> tuple[str, Optional[str]]:
    """Parse ``base__modifier`` coefficient style strings."""

    if "__" not in style:
        return style, None
    base_schedule, modifier = style.split("__", 1)
    if modifier not in VALID_MODIFIERS:
        raise WeightedMergeError(
            f"Unknown coefficient modifier '{modifier}'. "
            f"Valid modifiers are: {', '.join(VALID_MODIFIERS)}."
        )
    return base_schedule, modifier


def get_valid_styles() -> list[str]:
    styles = list(SCHEDULES)
    for schedule in SCHEDULES:
        styles.extend(f"{schedule}__{modifier}" for modifier in VALID_MODIFIERS)
    return styles


def schedule_to_merge_coefficients(
    schedule_fn: Callable[[float], float], n_checkpoints: int
) -> list[float]:
    """Convert a decay schedule into discrete checkpoint merge coefficients."""

    if n_checkpoints < 0:
        raise WeightedMergeError(
            f"Number of checkpoints must be non-negative, got {n_checkpoints}."
        )
    if n_checkpoints <= 1:
        return [1.0] * n_checkpoints

    decay_schedule = [schedule_fn(index / n_checkpoints) for index in range(n_checkpoints)]
    coefficients = [0.0 for _ in range(n_checkpoints)]
    coefficients[-1] = decay_schedule[-1]
    for index in range(1, n_checkpoints - 1):
        coefficients[index] = decay_schedule[index] - decay_schedule[index + 1]
    coefficients[0] = 1 - sum(coefficients)
    return coefficients


def apply_modifier(
    coefficients: list[float], modifier: Optional[str], seed: Optional[int] = 0
) -> list[float]:
    """Apply a deterministic coefficient modifier."""

    if modifier is None:
        return coefficients
    if modifier == "reverse":
        return list(reversed(coefficients))
    if modifier == "scramble":
        shuffled = list(coefficients)
        random.Random(seed).shuffle(shuffled)
        return shuffled
    raise WeightedMergeError(f"Unknown coefficient modifier '{modifier}'.")


def checkpoint_coefficients(
    checkpoints: list[int], schedule: str, seed: Optional[int] = 0
) -> dict[int, float]:
    """Return ``iteration -> coefficient`` for the checkpoints in input order."""

    base_schedule, modifier = parse_schedule_style(schedule)
    if base_schedule not in SCHEDULES:
        raise WeightedMergeError(
            f"Unknown coefficient schedule '{base_schedule}'. "
            f"Valid schedules are: {', '.join(SCHEDULES)}."
        )

    coefficients = schedule_to_merge_coefficients(SCHEDULES[base_schedule], len(checkpoints))
    coefficients = apply_modifier(coefficients, modifier, seed)
    return dict(zip(checkpoints, coefficients))


def normalize_weights(weights: Iterable[float]) -> list[float]:
    weights = [float(weight) for weight in weights]
    if any(not math.isfinite(weight) for weight in weights):
        raise WeightedMergeError(f"Weights must be finite, got {weights}.")
    total = sum(weights)
    if not math.isfinite(total) or total <= 0:
        raise WeightedMergeError(f"Weight sum must be positive, got {total}.")
    return [weight / total for weight in weights]


def validate_weights(weights: Iterable[float]) -> list[float]:
    weights = [float(weight) for weight in weights]
    if any(not math.isfinite(weight) for weight in weights):
        raise WeightedMergeError(f"Weights must be finite, got {weights}.")
    return weights


def _manual_weight_warnings(weights: Iterable[float], *, normalize: bool) -> list[str]:
    weights = [float(weight) for weight in weights]
    messages: list[str] = []
    if any(weight < 0 for weight in weights):
        messages.append(
            "WARNING: manual merge weights include negative values; this is allowed for "
            "subtractive merges but can produce outputs outside the input checkpoint range."
        )
    if not normalize:
        total = sum(weights)
        if math.isfinite(total) and not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
            messages.append(
                f"WARNING: manual merge weights sum to {total:.12g} without --normalize; "
                "the merged tensors will be scaled by that total."
            )
    return messages


def warn_manual_weight_policy(weights: Iterable[float], *, normalize: bool) -> None:
    for message in _manual_weight_warnings(weights, normalize=normalize):
        print_rank_0(message, flush=True)


def parse_weighted_inputs(specs: Iterable[str]) -> tuple[list[Path], list[float]]:
    """Parse manual ``PATH:WEIGHT`` input specifications."""

    paths: list[Path] = []
    weights: list[float] = []
    for spec in specs:
        if ":" not in spec:
            raise WeightedMergeError(f"Input must be PATH:WEIGHT, got '{spec}'.")
        path, weight = spec.rsplit(":", 1)
        paths.append(Path(path))
        try:
            weights.append(float(weight))
        except ValueError as exc:
            raise WeightedMergeError(f"Invalid weight in '{spec}'.") from exc
    return paths, weights


def discover_checkpoint_iterations(checkpoint_root: Union[str, Path]) -> list[int]:
    """Discover sorted ``iter_*`` checkpoint directories under ``checkpoint_root``."""

    root = Path(checkpoint_root)
    if not root.exists():
        raise WeightedMergeError(f"Checkpoint root does not exist: {root}.")
    if not root.is_dir():
        raise WeightedMergeError(f"Checkpoint root is not a directory: {root}.")

    iterations = []
    for child in root.iterdir():
        match = ITERATION_RE.match(child.name)
        if child.is_dir() and match:
            iterations.append(int(match.group(1)))
    return sorted(iterations)


def filter_checkpoints_by_interval(
    checkpoints: list[int], min_iteration_interval: Optional[int]
) -> list[int]:
    """Greedily keep checkpoints at least ``min_iteration_interval`` apart.

    Filtering walks backward from the target checkpoint, so the last checkpoint
    in the input list is always preserved.
    """

    if not checkpoints or min_iteration_interval is None or min_iteration_interval <= 0:
        return list(checkpoints)

    filtered = []
    last_selected = None
    for checkpoint in reversed(checkpoints):
        if last_selected is None or last_selected - checkpoint >= min_iteration_interval:
            filtered.append(checkpoint)
            last_selected = checkpoint
    return list(reversed(filtered))


def derive_start_iteration_from_token_window(
    end_iteration: int, token_window_btok: int, seq_length: int, global_batch_size: int
) -> int:
    tokens_per_iteration = seq_length * global_batch_size
    if tokens_per_iteration <= 0:
        raise WeightedMergeError(
            f"Tokens per iteration must be positive, got {tokens_per_iteration}."
        )
    if token_window_btok <= 0:
        raise WeightedMergeError(f"Token window must be positive, got {token_window_btok}.")

    window_tokens = token_window_btok * 1_000_000_000
    iterations = math.ceil(window_tokens / tokens_per_iteration)
    return max(end_iteration - iterations, 0)


def select_checkpoints_in_window(
    checkpoint_root: Union[str, Path],
    *,
    start_iteration: Optional[int],
    end_iteration: int,
    token_window_btok: Optional[int] = None,
    seq_length: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    min_iteration_interval: Optional[int] = None,
) -> list[int]:
    """Select sorted checkpoint iterations for range or token-window merging."""

    if token_window_btok is not None:
        if seq_length is None or global_batch_size is None:
            raise WeightedMergeError(
                "Token-window selection requires seq_length and global_batch_size."
            )
        start_iteration = derive_start_iteration_from_token_window(
            end_iteration, token_window_btok, seq_length, global_batch_size
        )
    if start_iteration is None:
        raise WeightedMergeError("start_iteration is required for checkpoint selection.")
    if start_iteration > end_iteration:
        raise WeightedMergeError(
            f"start_iteration ({start_iteration}) must be <= end_iteration ({end_iteration})."
        )

    available = discover_checkpoint_iterations(checkpoint_root)
    if end_iteration not in available:
        raise WeightedMergeError(
            f"Target iteration {end_iteration} is not present under {checkpoint_root}."
        )

    selected = [
        iteration for iteration in available if start_iteration <= iteration <= end_iteration
    ]
    selected = filter_checkpoints_by_interval(selected, min_iteration_interval)
    if not selected or selected[-1] != end_iteration:
        raise WeightedMergeError(
            f"Checkpoint selection did not preserve target iteration {end_iteration}."
        )
    return selected


def checkpoint_paths_for_iterations(
    checkpoint_root: Union[str, Path], iterations: Iterable[int]
) -> list[Path]:
    root = Path(checkpoint_root)
    return [root / iteration_dir_name(iteration) for iteration in iterations]


def validate_min_checkpoints(num_checkpoints: int, min_checkpoints: Optional[int]) -> None:
    """Validate an optional minimum input checkpoint count."""

    if min_checkpoints is None:
        return
    if min_checkpoints < 1:
        raise WeightedMergeError(f"min_checkpoints must be positive, got {min_checkpoints}.")
    if num_checkpoints < min_checkpoints:
        raise WeightedMergeError(
            f"Selected {num_checkpoints} checkpoints, but at least {min_checkpoints} are required."
        )


def _read_latest_checkpointed_iteration(path: Path) -> Union[int, str]:
    tracker = path / LATEST_CHECKPOINTED_ITERATION
    if not tracker.exists():
        raise WeightedMergeError(f"Missing {LATEST_CHECKPOINTED_ITERATION} under {path}.")
    value = tracker.read_text(encoding="utf-8").strip()
    if value == "release":
        return value
    try:
        return int(value)
    except ValueError as exc:
        raise WeightedMergeError(
            f"Invalid latest checkpoint marker '{value}' in {tracker}."
        ) from exc


def resolve_checkpoint_dir(path: Union[str, Path]) -> Path:
    """Resolve direct, release, or latest-marker checkpoint paths."""

    checkpoint = Path(path)
    if (checkpoint / "metadata.json").exists():
        return checkpoint

    if (checkpoint / LATEST_CHECKPOINTED_ITERATION).exists():
        latest = _read_latest_checkpointed_iteration(checkpoint)
        if latest == "release":
            resolved = checkpoint / "release"
        else:
            resolved = checkpoint / iteration_dir_name(latest)
        if not (resolved / "metadata.json").exists():
            raise WeightedMergeError(
                f"{checkpoint} points to {resolved}, but that is not a distributed checkpoint."
            )
        return resolved

    raise WeightedMergeError(
        f"{checkpoint} is not a distributed checkpoint and has no "
        f"{LATEST_CHECKPOINTED_ITERATION} marker."
    )


def output_checkpoint_dir(output_root: Union[str, Path], output_iteration: Optional[int]) -> Path:
    """Return the concrete directory to pass to dist_checkpointing.save."""

    root = Path(output_root)
    if output_iteration is None:
        return root

    expected_name = iteration_dir_name(output_iteration)
    if root.name == expected_name:
        return root
    if ITERATION_RE.match(root.name):
        raise WeightedMergeError(
            f"Output directory {root} does not match requested iteration {output_iteration}."
        )
    return root / expected_name


def write_latest_checkpointed_iteration(checkpoint_dir: Union[str, Path], iteration: int) -> None:
    """Write Megatron's latest-checkpoint marker for an iteration checkpoint."""

    checkpoint_dir = Path(checkpoint_dir)
    if not (checkpoint_dir / "metadata.json").exists():
        raise WeightedMergeError(
            f"Refusing to write {LATEST_CHECKPOINTED_ITERATION} because {checkpoint_dir} "
            "does not contain distributed checkpoint metadata."
        )
    parent = checkpoint_dir.parent if ITERATION_RE.match(checkpoint_dir.name) else checkpoint_dir
    tracker = parent / LATEST_CHECKPOINTED_ITERATION
    tracker.parent.mkdir(parents=True, exist_ok=True)
    temporary_tracker = tracker.with_name(f".{tracker.name}.tmp.{os.getpid()}")
    temporary_tracker.write_text(f"{iteration}\n", encoding="utf-8")
    _best_effort_fsync_path(temporary_tracker, "latest checkpoint marker temporary file")
    os.replace(temporary_tracker, tracker)
    _best_effort_fsync_path(tracker.parent, "latest checkpoint marker directory")


def _checkpoint_format(checkpoint_dir: Path) -> str:
    config = maybe_load_config(str(checkpoint_dir))
    if config is None:
        raise WeightedMergeError(
            f"Missing distributed checkpoint metadata.json in {checkpoint_dir}."
        )
    return config.sharded_backend


def _directory_size(path: Union[str, Path]) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _directory_size_for_accounting(path: Union[str, Path], byte_accounting: str) -> int:
    if byte_accounting == "none":
        return 0
    if byte_accounting == "rank0" and not is_rank_0():
        return 0
    return _directory_size(path)


def _host_peak_memory_bytes() -> int:
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(peak_rss)
    return int(peak_rss) * 1024


def _distributed_memory_peaks(host_peak_bytes: int) -> tuple[int, int, int, int]:
    """Return local rank/world size plus max host peak across ranks."""

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    local_record = {
        "rank": rank,
        "host_peak_bytes": int(host_peak_bytes),
    }
    if world_size == 1:
        return rank, world_size, rank, int(host_peak_bytes)

    records: list[Optional[dict[str, int]]] = [None for _ in range(world_size)]
    dist.all_gather_object(records, local_record)
    valid_records = [record for record in records if record is not None]
    host_record = max(
        valid_records,
        key=lambda record: (int(record["host_peak_bytes"]), -int(record["rank"])),
    )
    return (
        rank,
        world_size,
        int(host_record["rank"]),
        int(host_record["host_peak_bytes"]),
    )


def _get_path(root: Any, path: tuple[Union[str, int], ...]) -> Any:
    value = root
    for key in path:
        value = value[key]
    return value


def _single_path_state_dict(path: tuple[Union[str, int], ...], leaf: Any) -> ShardedStateDict:
    if not path:
        raise WeightedMergeError("Cannot build a single-path state dict for an empty path.")
    value: Any = leaf
    for key in reversed(path):
        if isinstance(key, int):
            items: list[Any] = [None for _ in range(key + 1)]
            items[key] = value
            value = items
        else:
            value = {key: value}
    return value


def _merge_state_dict_containers(target: Any, source: Any) -> Any:
    if target is None:
        return source
    if isinstance(target, dict) and isinstance(source, dict):
        for key, value in source.items():
            target[key] = _merge_state_dict_containers(target.get(key), value)
        return target
    if isinstance(target, list) and isinstance(source, list):
        if len(target) < len(source):
            target.extend([None] * (len(source) - len(target)))
        for index, value in enumerate(source):
            if value is not None:
                target[index] = _merge_state_dict_containers(target[index], value)
        return target
    raise WeightedMergeError("Conflicting container structure while building partial state dict.")


def _multi_path_state_dict(
    path_leaves: Iterable[tuple[tuple[Union[str, int], ...], Any]],
) -> ShardedStateDict:
    state_dict: Any = None
    for path, leaf in path_leaves:
        state_dict = _merge_state_dict_containers(state_dict, _single_path_state_dict(path, leaf))
    if state_dict is None:
        raise WeightedMergeError("Cannot build a partial state dict with no paths.")
    return state_dict


def _path_label(path: tuple[Union[str, int], ...], leaf: Any = None) -> str:
    key = getattr(leaf, "key", None)
    if key:
        return str(key)
    return ".".join(str(part) for part in path)


def _is_extra_state(path: tuple[Union[str, int], ...], leaf: Any) -> bool:
    label = _path_label(path, leaf)
    return (
        label == "_extra_state"
        or label.endswith("._extra_state")
        or any(str(part) == "_extra_state" or str(part).endswith("._extra_state") for part in path)
    )


def _as_tensor(value: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(value):
        return value
    data = getattr(value, "data", None)
    if torch.is_tensor(data):
        return data
    return None


def _copy_loaded_value(value: Any) -> Any:
    tensor = _as_tensor(value)
    if tensor is not None:
        return tensor.detach().cpu().clone()
    return copy.deepcopy(value)


def _assign_leaf_data(leaf: Any, value: Any) -> None:
    if hasattr(leaf, "data"):
        leaf.data = value
    else:
        raise WeightedMergeError(f"Cannot assign merged value to non-sharded leaf {leaf!r}.")


def _broadcast_rank0_value(value: Any) -> Any:
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return value
    objects = [value if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(objects, src=0)
    return objects[0]


def _run_rank0_filesystem_op(description: str, operation: Callable[[], None]) -> None:
    """Run a filesystem publication step on rank 0 and raise consistently on all ranks."""

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        operation()
        return

    error_messages: list[Optional[str]] = [None]
    if dist.get_rank() == 0:
        try:
            operation()
        except Exception as exc:
            error_messages[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(error_messages, src=0)
    if error_messages[0] is not None:
        raise WeightedMergeError(f"{description} failed on rank 0: {error_messages[0]}")


def _direct_dcp_save_uses_no_dist() -> bool:
    """Return whether direct DCP output should bypass distributed save orchestration."""

    return not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1


def _git_revision() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    revision = result.stdout.strip()
    return revision or None


def _checkpoint_iteration_from_dir(checkpoint_dir: Path) -> Optional[int]:
    match = ITERATION_RE.match(checkpoint_dir.name)
    if match:
        return int(match.group(1))
    return None


def _add_merge_provenance(
    common_state: dict[str, Any],
    *,
    input_dirs: list[Path],
    weights: list[float],
    normalize: bool,
    save_dtype: str,
    output_iteration: Optional[int],
    extra_state_source_index: int,
    strict: StrictHandling,
    merge_style: Optional[str],
    execution_mode: str = METADATA_SAME_LAYOUT_MODE,
    balance_rank_work: bool = False,
) -> None:
    common_state["weighted_merge_provenance"] = {
        "format_version": 1,
        "input_paths": [str(path) for path in input_dirs],
        "source_iterations": [_checkpoint_iteration_from_dir(path) for path in input_dirs],
        "weights": [float(weight) for weight in weights],
        "normalize": bool(normalize),
        "merge_style": merge_style if merge_style is not None else "manual",
        "output_iteration": output_iteration,
        "output_dtype": save_dtype,
        "extra_state_source_index": extra_state_source_index,
        "extra_state_source_path": str(input_dirs[extra_state_source_index]),
        "implementation_mode": execution_mode,
        "balance_rank_work": bool(balance_rank_work),
        "strict": strict.value,
        "code_revision": _git_revision(),
        "optimizer_merged": False,
        "rng_merged": False,
    }


def _prepare_temporary_output_dir(output_dir: Path, overwrite_output: bool) -> Path:
    if output_dir.exists() and not overwrite_output:
        raise WeightedMergeError(
            f"Output directory already exists: {output_dir}. "
            "Use --overwrite-merge-output to replace it."
        )
    temporary_name = _broadcast_rank0_value(f".{output_dir.name}.tmp-{uuid.uuid4().hex}")
    temporary_dir = output_dir.parent / temporary_name

    def prepare_parent() -> None:
        if temporary_dir.exists():
            raise WeightedMergeError(f"Temporary output directory already exists: {temporary_dir}.")
        temporary_dir.parent.mkdir(parents=True, exist_ok=True)

    _run_rank0_filesystem_op("Preparing atomic merge output directory", prepare_parent)
    return temporary_dir


def _reject_existing_atomic_overwrite(
    output_dir: Path,
    *,
    overwrite_output: bool,
    atomic_output: bool,
) -> None:
    if not (atomic_output and overwrite_output and output_dir.exists()):
        return
    raise WeightedMergeError(
        f"Refusing to replace existing output directory {output_dir} with atomic "
        "publication: non-empty checkpoint directory overwrite cannot be made "
        "crash-atomic with normal filesystem rename semantics. Remove the existing "
        "checkpoint explicitly or write to a new output path/iteration."
    )


def _require_publishable_checkpoint_dir(checkpoint_dir: Path) -> None:
    if not (checkpoint_dir / "metadata.json").exists():
        raise WeightedMergeError(
            f"Refusing to publish {checkpoint_dir} because distributed checkpoint metadata is missing."
        )
    try:
        FileSystemReader(checkpoint_dir).read_metadata()
    except (Exception, CheckpointException) as exc:
        raise WeightedMergeError(
            f"Refusing to publish {checkpoint_dir} because DCP metadata is missing or unreadable."
        ) from exc


def _best_effort_fsync_path(path: Path, description: str) -> None:
    if os.name != "posix":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as exc:
        print_rank_0(f"WARNING: Could not open {description} {path} for fsync: {exc}", flush=True)
        return
    try:
        os.fsync(fd)
    except OSError as exc:
        print_rank_0(f"WARNING: Could not fsync {description} {path}: {exc}", flush=True)
    finally:
        os.close(fd)


def _best_effort_fsync_checkpoint_metadata(checkpoint_dir: Path) -> None:
    for name in ("metadata.json", "common.pt", ".metadata"):
        path = checkpoint_dir / name
        if path.exists():
            _best_effort_fsync_path(path, f"checkpoint sidecar {name}")
    _best_effort_fsync_path(checkpoint_dir, "checkpoint directory")


def _publish_temporary_output_dir(
    temporary_dir: Path, output_dir: Path, *, overwrite_output: bool
) -> None:
    def publish() -> None:
        _require_publishable_checkpoint_dir(temporary_dir)
        _best_effort_fsync_checkpoint_metadata(temporary_dir)
        if output_dir.exists():
            if not overwrite_output:
                raise WeightedMergeError(
                    f"Output directory already exists: {output_dir}. "
                    "Use --overwrite-merge-output to replace it."
                )
            backup_dir = output_dir.with_name(f".{output_dir.name}.old-{uuid.uuid4().hex}")
            output_dir.rename(backup_dir)
            _best_effort_fsync_path(output_dir.parent, "checkpoint parent directory")
            try:
                temporary_dir.rename(output_dir)
                _best_effort_fsync_checkpoint_metadata(output_dir)
                _best_effort_fsync_path(output_dir.parent, "checkpoint parent directory")
            except Exception:
                if not output_dir.exists() and backup_dir.exists():
                    backup_dir.rename(output_dir)
                    _best_effort_fsync_path(output_dir.parent, "checkpoint parent directory")
                raise
            try:
                shutil.rmtree(backup_dir)
                _best_effort_fsync_path(output_dir.parent, "checkpoint parent directory")
            except Exception as exc:
                print_rank_0(
                    f"WARNING: Published merged checkpoint to {output_dir}, but failed to "
                    f"remove overwritten checkpoint backup {backup_dir}: {exc}",
                    flush=True,
                )
            return
        temporary_dir.rename(output_dir)
        _best_effort_fsync_checkpoint_metadata(output_dir)
        _best_effort_fsync_path(output_dir.parent, "checkpoint parent directory")

    _run_rank0_filesystem_op(f"Publishing merged checkpoint to {output_dir}", publish)


def _streaming_chunk_leaf(
    leaf: Any,
    *,
    chunk_dim: int,
    chunk_start: int,
    chunk_length: int,
    dtype: torch.dtype,
    global_shape: Iterable[int],
) -> Any:
    tensor = _as_tensor(leaf)
    if tensor is None:
        raise WeightedMergeError(f"Cannot create a tensor chunk for non-tensor leaf {leaf!r}.")

    prepended_axis_num = int(getattr(leaf, "prepend_axis_num", 0))
    local_shape = list(tensor.shape)
    if chunk_dim >= 0:
        local_shape[chunk_dim] = chunk_length
    data_shape = (1,) * prepended_axis_num + tuple(local_shape)
    data = torch.empty(data_shape, dtype=dtype, device="cpu")
    global_offset = list(getattr(leaf, "global_offset"))
    if chunk_dim >= 0:
        global_offset[prepended_axis_num + chunk_dim] += chunk_start
    return replace(
        leaf,
        data=data,
        dtype=dtype,
        local_shape=data_shape,
        global_shape=tuple(int(dim) for dim in global_shape),
        global_offset=tuple(global_offset),
        prepend_axis_num=0,
        axis_fragmentations=None,
    )


def _prepare_common_state(
    common_state: dict[str, Any], output_iteration: Optional[int]
) -> dict[str, Any]:
    common_state = copy.deepcopy(common_state)
    if output_iteration is None:
        return common_state

    common_state["iteration"] = output_iteration
    checkpoint_args = common_state.get("args")
    if checkpoint_args is not None and hasattr(checkpoint_args, "iteration"):
        checkpoint_args.iteration = output_iteration
    return common_state


def _load_path_group(
    checkpoint_dir: Path,
    path_leaves: list[tuple[tuple[Union[str, int], ...], Any]],
    *,
    strict: StrictHandling,
) -> dict[tuple[Union[str, int], ...], Any]:
    loaded = dist_checkpointing.load(
        _multi_path_state_dict(path_leaves),
        str(checkpoint_dir),
        validate_access_integrity=False,
        strict=strict,
    )
    return {path: _get_path(loaded, path) for path, _ in path_leaves}


def _load_tensor_path_group_fast(
    checkpoint_dir: Path,
    path_leaves: list[tuple[tuple[Union[str, int], ...], Any]],
    sharded_strategy: TorchDistLoadShardedStrategy,
) -> dict[tuple[Union[str, int], ...], Any]:
    """Load sharded tensor chunks without reloading common checkpoint state."""

    partial_state_dict = _multi_path_state_dict(path_leaves)
    force_all_tensors_to_non_fp8(partial_state_dict)
    loaded = sharded_strategy.load(partial_state_dict, checkpoint_dir, async_strategy="mcore")
    return {path: _get_path(loaded, path) for path, _ in path_leaves}


def _metadata_same_layout_is_model_key(
    fqn: str, model_key_prefixes: tuple[str, ...]
) -> bool:
    return any(fqn.startswith(prefix) for prefix in model_key_prefixes) or any(
        fqn.startswith(root) for root in METADATA_SAME_LAYOUT_UNPREFIXED_MODEL_ROOTS
    )


def _metadata_same_layout_is_extra_state_key(fqn: str) -> bool:
    return (
        fqn == "_extra_state"
        or fqn.endswith("._extra_state")
        or fqn.startswith("_extra_state/")
        or "._extra_state/" in fqn
        or "._extra_state." in fqn
    )


def _metadata_same_layout_path(fqn: str) -> tuple[Union[str, int], ...]:
    return tuple(fqn.split("."))


def _metadata_same_layout_tensor_dtype(
    fqn: str, metadata_entry: TensorStorageMetadata
) -> torch.dtype:
    dtype = getattr(getattr(metadata_entry, "properties", None), "dtype", None)
    if not isinstance(dtype, torch.dtype):
        raise WeightedMergeError(
            f"DCP metadata for '{fqn}' does not expose a torch dtype."
        )
    return dtype


def _metadata_same_layout_tensor_layout(
    fqn: str, metadata_entry: TensorStorageMetadata
) -> _DcpMetadataTensorLayout:
    chunks = tuple(
        (
            tuple(int(offset) for offset in chunk.offsets),
            tuple(int(size) for size in chunk.sizes),
        )
        for chunk in getattr(metadata_entry, "chunks", ()) or ()
    )
    if not chunks:
        raise WeightedMergeError(f"DCP metadata for '{fqn}' has no tensor chunks.")
    return _DcpMetadataTensorLayout(
        global_shape=tuple(int(dim) for dim in metadata_entry.size),
        dtype=_metadata_same_layout_tensor_dtype(fqn, metadata_entry),
        chunks=chunks,
    )


def _read_public_dcp_metadata(
    checkpoint_dir: Path, *, model_key_prefixes: tuple[str, ...]
) -> _DcpMetadataSnapshot:
    try:
        metadata = FileSystemReader(checkpoint_dir).read_metadata()
    except (Exception, CheckpointException) as exc:
        raise WeightedMergeError(
            f"Could not read public DCP metadata from {checkpoint_dir}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    tensor_metadata: dict[str, TensorStorageMetadata] = {}
    non_tensor_model_keys: list[str] = []
    byte_extra_state_keys: list[str] = []
    non_model_byte_keys: list[str] = []
    for fqn, metadata_entry in metadata.state_dict_metadata.items():
        fqn = str(fqn)
        if isinstance(metadata_entry, TensorStorageMetadata):
            tensor_metadata[fqn] = metadata_entry
        elif _metadata_same_layout_is_model_key(fqn, model_key_prefixes):
            if isinstance(metadata_entry, BytesStorageMetadata) and (
                _metadata_same_layout_is_extra_state_key(fqn)
            ):
                byte_extra_state_keys.append(fqn)
            else:
                non_tensor_model_keys.append(fqn)
        elif isinstance(metadata_entry, BytesStorageMetadata):
            non_model_byte_keys.append(fqn)

    if non_model_byte_keys:
        sample = ", ".join(sorted(non_model_byte_keys)[:5])
        raise WeightedMergeError(
            "metadata-same-layout refuses byte/object DCP entries outside model roots "
            f"in {checkpoint_dir}: {sample}. Allowed prefixes are {model_key_prefixes}; "
            f"allowed unprefixed roots are {METADATA_SAME_LAYOUT_UNPREFIXED_MODEL_ROOTS}."
        )
    if non_tensor_model_keys:
        sample = ", ".join(sorted(non_tensor_model_keys)[:5])
        raise WeightedMergeError(
            "metadata-same-layout currently supports tensor DCP entries only; "
            f"found non-tensor model metadata in {checkpoint_dir}: {sample}."
        )
    return _DcpMetadataSnapshot(
        tensor_metadata=tensor_metadata,
        byte_extra_state_keys=tuple(sorted(byte_extra_state_keys)),
    )


def _metadata_same_layout_model_keys(
    tensor_metadata: dict[str, TensorStorageMetadata],
    *,
    checkpoint_dir: Path,
    model_key_prefixes: tuple[str, ...],
) -> tuple[str, ...]:
    unsupported = sorted(
        fqn
        for fqn in tensor_metadata
        if not _metadata_same_layout_is_model_key(fqn, model_key_prefixes)
    )
    if unsupported:
        sample = ", ".join(unsupported[:5])
        raise WeightedMergeError(
            "metadata-same-layout refuses to infer merge policy for non-model "
            f"DCP tensor keys in {checkpoint_dir}: {sample}. "
            f"Allowed prefixes are {model_key_prefixes}; allowed unprefixed roots are "
            f"{METADATA_SAME_LAYOUT_UNPREFIXED_MODEL_ROOTS}."
        )
    model_keys = tuple(sorted(tensor_metadata))
    if not model_keys:
        raise WeightedMergeError(
            f"metadata-same-layout found no model tensor keys in {checkpoint_dir}."
        )
    return model_keys


def _validate_metadata_same_layout(
    *,
    tensor_metadata_by_checkpoint: list[dict[str, TensorStorageMetadata]],
    byte_extra_state_keys_by_checkpoint: list[tuple[str, ...]],
    resolved_input_dirs: list[Path],
    model_key_prefixes: tuple[str, ...],
    save_dtype: str,
    require_matching_chunks: bool = True,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, _DcpMetadataTensorLayout]]:
    first_keys = _metadata_same_layout_model_keys(
        tensor_metadata_by_checkpoint[0],
        checkpoint_dir=resolved_input_dirs[0],
        model_key_prefixes=model_key_prefixes,
    )
    expected_key_set = set(first_keys)
    for checkpoint_dir, tensor_metadata in zip(
        resolved_input_dirs[1:], tensor_metadata_by_checkpoint[1:]
    ):
        model_keys = _metadata_same_layout_model_keys(
            tensor_metadata,
            checkpoint_dir=checkpoint_dir,
            model_key_prefixes=model_key_prefixes,
        )
        key_set = set(model_keys)
        if key_set != expected_key_set:
            missing = sorted(expected_key_set - key_set)
            unexpected = sorted(key_set - expected_key_set)
            raise WeightedMergeError(
                "metadata-same-layout requires identical model tensor key sets; "
                f"{checkpoint_dir} is missing {missing[:5]} and has unexpected "
                f"{unexpected[:5]}."
            )

    first_byte_keys = byte_extra_state_keys_by_checkpoint[0]
    expected_byte_key_set = set(first_byte_keys)
    for checkpoint_dir, byte_extra_state_keys in zip(
        resolved_input_dirs[1:], byte_extra_state_keys_by_checkpoint[1:]
    ):
        byte_key_set = set(byte_extra_state_keys)
        if byte_key_set != expected_byte_key_set:
            missing = sorted(expected_byte_key_set - byte_key_set)
            unexpected = sorted(byte_key_set - expected_byte_key_set)
            raise WeightedMergeError(
                "metadata-same-layout requires identical byte/object _extra_state key sets; "
                f"{checkpoint_dir} is missing {missing[:5]} and has unexpected "
                f"{unexpected[:5]}."
            )

    first_layouts = {
        fqn: _metadata_same_layout_tensor_layout(
            fqn, tensor_metadata_by_checkpoint[0][fqn]
        )
        for fqn in first_keys
    }
    for fqn in first_keys:
        expected = first_layouts[fqn]
        for checkpoint_dir, tensor_metadata in zip(
            resolved_input_dirs[1:], tensor_metadata_by_checkpoint[1:]
        ):
            actual = _metadata_same_layout_tensor_layout(fqn, tensor_metadata[fqn])
            if actual.global_shape != expected.global_shape:
                raise WeightedMergeError(
                    f"Shape mismatch for '{fqn}' in metadata-same-layout: "
                    f"expected {expected.global_shape}, got {actual.global_shape} "
                    f"in {checkpoint_dir}."
                )
            if require_matching_chunks and actual.chunks != expected.chunks:
                raise WeightedMergeError(
                    f"Chunk layout mismatch for '{fqn}' in metadata-same-layout "
                    f"for {checkpoint_dir}."
                )
            if save_dtype == "same" and actual.dtype != expected.dtype:
                raise WeightedMergeError(
                    f"Dtype mismatch for '{fqn}' with --merge-save-dtype=same: "
                    f"expected {expected.dtype}, got {actual.dtype} in {checkpoint_dir}."
                )

    for fqn in first_keys:
        if _is_extra_state(_metadata_same_layout_path(fqn), None):
            continue
        for checkpoint_dir, tensor_metadata in zip(
            resolved_input_dirs, tensor_metadata_by_checkpoint
        ):
            dtype = _metadata_same_layout_tensor_dtype(fqn, tensor_metadata[fqn])
            if not torch.empty((), dtype=dtype).is_floating_point():
                raise WeightedMergeError(
                    f"Checkpoint {checkpoint_dir} key '{fqn}' has non-floating dtype "
                    f"{dtype}; weighted averaging is only supported for floating tensors."
                )

    return first_keys, first_byte_keys, first_layouts


def _metadata_same_layout_chunk_leaf(
    *,
    fqn: str,
    dtype: torch.dtype,
    global_shape: tuple[int, ...],
    global_offsets: tuple[int, ...],
    chunk_shape: tuple[int, ...],
) -> ShardedTensor:
    return ShardedTensor(
        key=fqn,
        data=torch.empty(chunk_shape, dtype=dtype, device="cpu"),
        dtype=dtype,
        local_shape=chunk_shape,
        global_shape=global_shape,
        global_offset=global_offsets,
        axis_fragmentations=None,
        replica_id=0,
    )


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _shape_numel(shape: Iterable[int]) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _estimate_merge_work_bytes(
    shape: Iterable[int],
    *,
    source_dtype: torch.dtype,
    target_dtype: torch.dtype,
    input_count: int,
) -> int:
    numel = _shape_numel(shape)
    source_bytes = numel * _dtype_nbytes(source_dtype) * input_count
    accumulator_bytes = numel * _dtype_nbytes(torch.float32)
    output_bytes = numel * _dtype_nbytes(target_dtype)
    return source_bytes + accumulator_bytes + output_bytes


def _rank_balanced_assignments(
    candidates: list[tuple[int, int, _DirectDcpWriteSpec]],
    *,
    world_size: int,
) -> tuple[list[list[_DirectDcpWriteSpec]], tuple[int, ...], tuple[int, ...]]:
    assigned_specs: list[list[_DirectDcpWriteSpec]] = [[] for _ in range(world_size)]
    rank_bytes = [0 for _ in range(world_size)]
    rank_chunks = [0 for _ in range(world_size)]
    for cost_bytes, _stable_index, spec in sorted(
        candidates,
        key=lambda item: (
            item[0],
            item[2].sharded_key,
            item[2].global_offsets,
            item[2].chunk_shape,
            item[1],
        ),
        reverse=True,
    ):
        target_rank = min(range(world_size), key=lambda idx: (rank_bytes[idx], rank_chunks[idx], idx))
        assigned_specs[target_rank].append(spec)
        rank_bytes[target_rank] += int(cost_bytes)
        rank_chunks[target_rank] += 1
    return assigned_specs, tuple(rank_bytes), tuple(rank_chunks)


def _build_metadata_same_layout_write_specs(
    *,
    selected_keys: tuple[str, ...],
    byte_extra_state_keys: tuple[str, ...],
    first_layouts: dict[str, _DcpMetadataTensorLayout],
    save_dtype: str,
    input_count: int,
    balance_rank_work: bool = False,
) -> _MetadataSameLayoutWorkPlan:
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = (
        dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    )
    target_dtype_override = SAVE_DTYPE_MAP.get(save_dtype)
    local_write_specs: list[_DirectDcpWriteSpec] = []
    byte_specs: list[_DcpMetadataByteSpec] = []
    candidate_specs: list[tuple[int, int, _DirectDcpWriteSpec]] = []
    rank_tensor_bytes = [0 for _ in range(world_size)]
    rank_tensor_chunks = [0 for _ in range(world_size)]
    merge_keys = 0
    extra_state_keys = 0
    stable_index = 0

    for fqn in selected_keys:
        layout = first_layouts[fqn]
        path = _metadata_same_layout_path(fqn)
        is_extra_state = _is_extra_state(path, None)
        if is_extra_state:
            extra_state_keys += 1
            target_dtype = layout.dtype
        else:
            merge_keys += 1
            target_dtype = (
                target_dtype_override if target_dtype_override is not None else layout.dtype
            )
        for chunk_index, (global_offsets, chunk_shape) in enumerate(layout.chunks):
            template_leaf = _metadata_same_layout_chunk_leaf(
                fqn=fqn,
                dtype=layout.dtype,
                global_shape=layout.global_shape,
                global_offsets=global_offsets,
                chunk_shape=chunk_shape,
            )
            spec = (
                _DirectDcpWriteSpec(
                    path=path,
                    load_path=path,
                    template_leaf=template_leaf,
                    sharded_key=fqn,
                    global_shape=layout.global_shape,
                    global_offsets=global_offsets,
                    chunk_shape=chunk_shape,
                    target_dtype=target_dtype,
                    chunk_dim=-1,
                    chunk_start=0,
                    chunk_length=1,
                    source_dtype=layout.dtype,
                    source_shape=layout.global_shape,
                    is_extra_state=is_extra_state,
                )
            )
            cost_bytes = _estimate_merge_work_bytes(
                chunk_shape,
                source_dtype=layout.dtype,
                target_dtype=target_dtype,
                input_count=input_count,
            )
            if balance_rank_work:
                candidate_specs.append((cost_bytes, stable_index, spec))
            elif chunk_index % world_size == rank:
                local_write_specs.append(spec)
                rank_tensor_bytes[rank] += cost_bytes
                rank_tensor_chunks[rank] += 1
            else:
                assigned_rank = chunk_index % world_size
                rank_tensor_bytes[assigned_rank] += cost_bytes
                rank_tensor_chunks[assigned_rank] += 1
            stable_index += 1

    if balance_rank_work:
        assigned_specs, assigned_bytes, assigned_chunks = _rank_balanced_assignments(
            candidate_specs,
            world_size=world_size,
        )
        local_write_specs = assigned_specs[rank]
        rank_tensor_bytes = list(assigned_bytes)
        rank_tensor_chunks = list(assigned_chunks)

    for byte_index, fqn in enumerate(byte_extra_state_keys):
        extra_state_keys += 1
        if byte_index % world_size != rank:
            continue
        template_leaf = ShardedObject.empty_from_unique_key(fqn)
        byte_specs.append(
            _DcpMetadataByteSpec(
                fqn=fqn,
                path=(fqn,),
                template_leaf=template_leaf,
            )
        )

    if merge_keys == 0:
        raise WeightedMergeError(
            "metadata-same-layout found no mergeable non-_extra_state model tensors."
        )
    return _MetadataSameLayoutWorkPlan(
        write_specs=local_write_specs,
        byte_specs=byte_specs,
        merge_keys=merge_keys,
        extra_state_keys=extra_state_keys,
        tensor_bytes_by_rank=tuple(rank_tensor_bytes),
        tensor_chunks_by_rank=tuple(rank_tensor_chunks),
    )


def merge_same_layout_dcp_metadata_checkpoints(
    input_paths: list[Union[str, Path]],
    weights: list[float],
    output_root: Union[str, Path],
    *,
    normalize: bool = False,
    save_dtype: str = "same",
    output_iteration: Optional[int] = None,
    write_latest: bool = True,
    extra_state_source_index: int = 0,
    byte_accounting: str = "rank0",
    overwrite_output: bool = False,
    atomic_output: bool = True,
    merge_style: Optional[str] = None,
    model_key_prefixes: tuple[str, ...] = METADATA_SAME_LAYOUT_MODEL_PREFIXES,
    balance_rank_work: bool = False,
) -> MergeResult:
    """Merge same-layout torch_dist checkpoints using public DCP metadata only.

    This path never builds a Megatron model or sharded-state template: it reads
    each source checkpoint's public DCP metadata (FQN, global shape, dtype, chunk
    layout, byte/object ``_extra_state``) and derives the merge structure from it.
    Because no model is constructed, it imports no CUDA-only kernels and runs
    CPU-only for every model family (GPT/Mamba/hybrid/TE).
    """

    total_start = time.perf_counter()
    discovery_start = time.perf_counter()
    ensure_process_group()

    if len(input_paths) != len(weights):
        raise WeightedMergeError(f"Got {len(input_paths)} input paths but {len(weights)} weights.")
    if not input_paths:
        raise WeightedMergeError("At least one input checkpoint is required.")
    if save_dtype != "same" and save_dtype not in SAVE_DTYPE_MAP:
        raise WeightedMergeError(
            f"Unsupported save dtype '{save_dtype}'. Use same, float32, float16, or bfloat16."
        )
    if byte_accounting not in BYTE_ACCOUNTING_MODES:
        raise WeightedMergeError(
            f"Unsupported byte accounting mode '{byte_accounting}'. "
            f"Use one of: {', '.join(BYTE_ACCOUNTING_MODES)}."
        )
    if extra_state_source_index < 0 or extra_state_source_index >= len(input_paths):
        raise WeightedMergeError(
            f"extra_state_source_index {extra_state_source_index} is out of range."
        )
    if not model_key_prefixes:
        raise WeightedMergeError("model_key_prefixes must contain at least one prefix.")
    if not atomic_output:
        raise WeightedMergeError(
            "Weighted merge requires atomic output publication; "
            "--no-atomic-merge-output is not supported because public "
            "DCP writes could otherwise expose a partial final checkpoint."
        )

    resolved_input_dirs = [resolve_checkpoint_dir(path) for path in input_paths]
    input_formats = [_checkpoint_format(path) for path in resolved_input_dirs]
    first_format = input_formats[0]
    for checkpoint_dir, checkpoint_format in zip(resolved_input_dirs[1:], input_formats[1:]):
        if checkpoint_format != first_format:
            raise WeightedMergeError(
                f"Checkpoint format mismatch: expected {first_format}, "
                f"got {checkpoint_format} in {checkpoint_dir}."
            )
    if first_format != "torch_dist":
        raise WeightedMergeError(
            "metadata-same-layout currently supports torch_dist checkpoints only; "
            f"got {first_format}."
        )

    weights = normalize_weights(weights) if normalize else validate_weights(weights)
    output_dir = output_checkpoint_dir(output_root, output_iteration)
    _reject_existing_atomic_overwrite(
        output_dir, overwrite_output=overwrite_output, atomic_output=atomic_output
    )

    metadata_snapshots = [
        _read_public_dcp_metadata(
            checkpoint_dir, model_key_prefixes=model_key_prefixes
        )
        for checkpoint_dir in resolved_input_dirs
    ]
    tensor_metadata_by_checkpoint = [
        snapshot.tensor_metadata for snapshot in metadata_snapshots
    ]
    byte_extra_state_keys_by_checkpoint = [
        snapshot.byte_extra_state_keys for snapshot in metadata_snapshots
    ]
    selected_keys, byte_extra_state_keys, first_layouts = _validate_metadata_same_layout(
        tensor_metadata_by_checkpoint=tensor_metadata_by_checkpoint,
        byte_extra_state_keys_by_checkpoint=byte_extra_state_keys_by_checkpoint,
        resolved_input_dirs=resolved_input_dirs,
        model_key_prefixes=model_key_prefixes,
        save_dtype=save_dtype,
        require_matching_chunks=True,
    )
    work_plan = _build_metadata_same_layout_write_specs(
        selected_keys=selected_keys,
        byte_extra_state_keys=byte_extra_state_keys,
        first_layouts=first_layouts,
        save_dtype=save_dtype,
        input_count=len(resolved_input_dirs),
        balance_rank_work=balance_rank_work,
    )
    memory_estimate = MergeMemoryEstimate(
        mergeable_tensors=work_plan.merge_keys,
        extra_state_entries=work_plan.extra_state_keys,
        template_devices=("cpu",),
    )
    discovery_time = time.perf_counter() - discovery_start

    print_rank_0(
        f"Merging {len(resolved_input_dirs)} same-layout metadata checkpoints into "
        f"{output_dir} with weights {weights}",
        flush=True,
    )
    if balance_rank_work:
        print_rank_0(
            "Rank-work balanced plan: "
            f"tensor_bytes_by_rank={list(work_plan.tensor_bytes_by_rank)}, "
            f"tensor_chunks_by_rank={list(work_plan.tensor_chunks_by_rank)}",
            flush=True,
        )

    temporary_output_dir = _prepare_temporary_output_dir(output_dir, overwrite_output)

    base_common_state = dist_checkpointing.load_common_state_dict(str(resolved_input_dirs[0]))
    common_state = _prepare_common_state(base_common_state, output_iteration)
    strict = StrictHandling.ASSUME_OK_UNEXPECTED
    _add_merge_provenance(
        common_state,
        input_dirs=resolved_input_dirs,
        weights=weights,
        normalize=normalize,
        save_dtype=save_dtype,
        output_iteration=output_iteration,
        extra_state_source_index=extra_state_source_index,
        strict=strict,
        merge_style=merge_style,
        execution_mode=METADATA_SAME_LAYOUT_MODE,
        balance_rank_work=balance_rank_work,
    )

    from megatron.core.dist_checkpointing.core import CheckpointingConfig, save_config
    from megatron.core.dist_checkpointing.strategies.common import save_common

    load_strategies = {
        checkpoint_dir: TorchDistLoadShardedStrategy(cache_metadata=True)
        for checkpoint_dir in resolved_input_dirs
    }
    planner = _WeightedMergeDirectOutputSavePlanner(
        write_specs=work_plan.write_specs,
        byte_specs=work_plan.byte_specs,
        resolved_input_dirs=resolved_input_dirs,
        weights=weights,
        load_strategies=load_strategies,
        extra_state_source_index=extra_state_source_index,
    )
    writer = FileSystemWriter(
        temporary_output_dir,
        single_file_per_rank=True,
        sync_files=True,
        thread_count=1,
        per_thread_copy_ahead=0,
    )

    bytes_read = sum(
        _directory_size_for_accounting(checkpoint_dir, byte_accounting)
        for checkpoint_dir in resolved_input_dirs
    )
    save_start = time.perf_counter()
    try:
        torch_dcp.save(
            {},
            storage_writer=writer,
            planner=planner,
            no_dist=_direct_dcp_save_uses_no_dist(),
        )
    except (Exception, CheckpointException) as exc:
        rank_suffix = (
            f" on rank {dist.get_rank()}"
            if dist.is_available() and dist.is_initialized()
            else ""
        )
        raise WeightedMergeError(
            f"Metadata same-layout DCP save failed{rank_suffix}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    def write_sidecars() -> None:
        save_common(common_state, str(temporary_output_dir))
        save_config(CheckpointingConfig("torch_dist", 1), str(temporary_output_dir))

    _run_rank0_filesystem_op("Writing metadata same-layout checkpoint sidecars", write_sidecars)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    _publish_temporary_output_dir(
        temporary_output_dir, output_dir, overwrite_output=overwrite_output
    )
    if dist.is_initialized():
        dist.barrier()
    if output_iteration is not None and write_latest:
        _run_rank0_filesystem_op(
            f"Writing {LATEST_CHECKPOINTED_ITERATION}",
            lambda: write_latest_checkpointed_iteration(output_dir, output_iteration),
        )
    if dist.is_initialized():
        dist.barrier()

    save_time = time.perf_counter() - save_start
    bytes_written = _directory_size_for_accounting(output_dir, byte_accounting)
    host_peak_bytes = _host_peak_memory_bytes()
    (
        rank,
        world_size,
        max_host_peak_rank,
        max_host_peak_bytes,
    ) = _distributed_memory_peaks(host_peak_bytes)
    timings = MergeTimings(
        discovery=discovery_time,
        load=planner.load_time,
        accumulation=planner.accumulation_time,
        save=save_time,
        total=time.perf_counter() - total_start,
    )
    return MergeResult(
        output_dir=output_dir,
        input_dirs=tuple(resolved_input_dirs),
        weights=tuple(weights),
        timings=timings,
        averaged_tensors=work_plan.merge_keys,
        copied_extra_states=work_plan.extra_state_keys,
        bytes_read=bytes_read,
        bytes_written=bytes_written,
        backend=first_format,
        host_peak_bytes=host_peak_bytes,
        max_host_peak_bytes=max_host_peak_bytes,
        max_host_peak_rank=max_host_peak_rank,
        world_size=world_size,
        rank=rank,
        implementation_mode=METADATA_SAME_LAYOUT_MODE,
        memory_estimate=memory_estimate,
        preflight_only=False,
        balance_rank_work=balance_rank_work,
        plan_tensor_bytes_by_rank=work_plan.tensor_bytes_by_rank,
        plan_tensor_chunks_by_rank=work_plan.tensor_chunks_by_rank,
    )


def add_merge_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="weighted checkpoint merge")
    group.add_argument(
        "--merge-inputs",
        nargs="+",
        required=True,
        help=(
            "Manual mode: PATH:WEIGHT entries. Window/range mode: one checkpoint root "
            "containing iter_* directories."
        ),
    )
    group.add_argument("--merge-output", required=True, help="Output checkpoint root.")
    group.add_argument("--normalize", action="store_true", help="Normalize manual weights.")
    group.add_argument("--start-checkpoint", type=int, help="Inclusive start iteration.")
    group.add_argument("--end-checkpoint", type=int, help="Inclusive target/end iteration.")
    group.add_argument(
        "--merge-window-btoks",
        type=int,
        help=(
            "Window size in billions of tokens. Requires --end-checkpoint and uses "
            "checkpoint seq_length/global_batch_size to derive the start iteration."
        ),
    )
    group.add_argument(
        "--merge-style",
        choices=get_valid_styles(),
        default="linear",
        help="Coefficient schedule for range/window mode.",
    )
    group.add_argument(
        "--coefficient-seed",
        type=int,
        default=0,
        help="Deterministic seed for '__scramble' coefficient styles.",
    )
    group.add_argument(
        "--min-iteration-interval",
        type=int,
        default=None,
        help="Only keep selected checkpoints separated by at least this many iterations.",
    )
    group.add_argument(
        "--min-checkpoints",
        type=int,
        default=None,
        help="Fail if fewer than this many input checkpoints are selected.",
    )
    group.add_argument(
        "--merge-save-dtype",
        choices=("same", "float32", "float16", "bfloat16"),
        default="same",
        help=(
            "Saved dtype for averaged tensors. 'same' preserves the loaded/template dtype "
            "and requires compatible input dtypes."
        ),
    )
    group.add_argument(
        "--merge-balance-rank-work",
        action="store_true",
        help=(
            "Greedily bin-pack source DCP chunks across distributed ranks to reduce "
            "per-rank merge work skew while preserving the source checkpoint chunk layout."
        ),
    )
    group.add_argument(
        "--output-iteration",
        type=int,
        default=None,
        help=(
            "If set, write --merge-output/iter_XXXXXXX and update "
            "latest_checkpointed_iteration.txt. Defaults to --end-checkpoint in range/window mode."
        ),
    )
    group.add_argument(
        "--merge-byte-accounting",
        choices=BYTE_ACCOUNTING_MODES,
        default="rank0",
        help=(
            "How to measure recursive checkpoint byte totals. 'rank0' avoids multiplying "
            "filesystem metadata traffic by distributed rank."
        ),
    )
    group.add_argument(
        "--merge-resource-log",
        choices=RESOURCE_LOG_MODES,
        default="none",
        help=(
            "Emit resource checkpoints around expensive setup phases. 'rank0' keeps logs "
            "compact; 'all' reports every distributed rank for memory diagnostics."
        ),
    )
    group.add_argument(
        "--overwrite-merge-output",
        action="store_true",
        help="Allow replacing an existing merged output checkpoint directory.",
    )
    group.add_argument(
        "--no-atomic-merge-output",
        action="store_true",
        help=(
            "Write directly to --merge-output instead of publishing from a temporary "
            "directory. Not supported: weighted merge always requires atomic output."
        ),
    )
    group.add_argument(
        "--extra-state-source-index",
        type=int,
        default=0,
        help="Input checkpoint index whose Transformer Engine _extra_state values are copied.",
    )
    group.add_argument(
        "--strict",
        choices=tuple(
            flag.value
            for flag in StrictHandling
            if not StrictHandling.requires_returning_mismatch_keys(flag)
        ),
        default=StrictHandling.RAISE_UNEXPECTED.value,
        help=(
            "Distributed-checkpoint strictness. The default requires requested model keys "
            "to exist while tolerating extra checkpoint shards such as optimizer state."
        ),
    )
    return parser


def _resolve_cli_inputs_and_weights(
    args: argparse.Namespace,
    *,
    seq_length: Optional[int] = None,
    global_batch_size: Optional[int] = None,
) -> tuple[list[Path], list[float], Optional[int], Optional[str]]:
    use_selection_mode = args.start_checkpoint is not None or args.merge_window_btoks is not None
    output_iteration = args.output_iteration

    if use_selection_mode:
        if len(args.merge_inputs) != 1:
            raise WeightedMergeError("Range/window mode expects exactly one --merge-inputs root.")
        if args.end_checkpoint is None:
            raise WeightedMergeError("--end-checkpoint is required for range/window mode.")

        checkpoint_root = Path(args.merge_inputs[0])
        selected_iterations = select_checkpoints_in_window(
            checkpoint_root,
            start_iteration=args.start_checkpoint,
            end_iteration=args.end_checkpoint,
            token_window_btok=args.merge_window_btoks,
            seq_length=seq_length,
            global_batch_size=global_batch_size,
            min_iteration_interval=args.min_iteration_interval,
        )
        coefficient_map = checkpoint_coefficients(
            selected_iterations, args.merge_style, seed=args.coefficient_seed
        )
        input_paths = checkpoint_paths_for_iterations(checkpoint_root, selected_iterations)
        weights = [coefficient_map[iteration] for iteration in selected_iterations]
        if output_iteration is None:
            output_iteration = args.end_checkpoint
        merge_style = args.merge_style
    else:
        input_paths, weights = parse_weighted_inputs(args.merge_inputs)
        merge_style = None
        warn_manual_weight_policy(weights, normalize=args.normalize)

    validate_min_checkpoints(len(input_paths), args.min_checkpoints)
    return input_paths, weights, output_iteration, merge_style


def _parse_metadata_same_layout_args(
    argv: Optional[list[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge strict same-layout torch_dist checkpoints using public DCP "
            "metadata without Megatron model/template initialization."
        )
    )
    add_merge_args(parser)
    parser.add_argument(
        "--ckpt-format",
        choices=("torch_dist",),
        default="torch_dist",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    _validate_metadata_same_layout_cli_args(args)
    return args


def _validate_metadata_same_layout_cli_args(args: argparse.Namespace) -> None:
    if args.merge_window_btoks is not None:
        raise WeightedMergeError(
            "--merge-window-btoks is not supported; the metadata-driven merge never "
            "builds a model and cannot read seq_length/global_batch_size to derive a "
            "token window. Use explicit PATH:WEIGHT inputs or start/end checkpoint "
            "selection."
        )
    if args.no_atomic_merge_output:
        raise WeightedMergeError(
            "Weighted merge requires atomic output publication; "
            "--no-atomic-merge-output is not supported."
        )

    unsupported_non_defaults = [
        (args.merge_resource_log != "none", "--merge-resource-log"),
        (args.strict != StrictHandling.RAISE_UNEXPECTED.value, "--strict"),
    ]
    unsupported = [flag for is_set, flag in unsupported_non_defaults if is_set]
    if unsupported:
        raise WeightedMergeError(
            "The metadata-driven merge does not use these options: "
            f"{', '.join(unsupported)}."
        )


def _run_metadata_same_layout_cli(args: argparse.Namespace) -> MergeResult:
    input_paths, weights, output_iteration, merge_style = _resolve_cli_inputs_and_weights(args)
    return merge_same_layout_dcp_metadata_checkpoints(
        input_paths,
        weights,
        args.merge_output,
        normalize=args.normalize,
        save_dtype=args.merge_save_dtype,
        output_iteration=output_iteration,
        extra_state_source_index=args.extra_state_source_index,
        byte_accounting=args.merge_byte_accounting,
        overwrite_output=args.overwrite_merge_output,
        atomic_output=not args.no_atomic_merge_output,
        merge_style=merge_style,
        balance_rank_work=args.merge_balance_rank_work,
    )


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def _format_bandwidth(num_bytes: int, seconds: float) -> str:
    if seconds <= 0:
        return "n/a"
    return f"{_format_bytes(int(num_bytes / seconds))}/s"


def _format_rank_bytes(values: tuple[int, ...]) -> str:
    return "[" + ", ".join(_format_bytes(value) for value in values) + "]"


def _print_merge_result(result: MergeResult) -> None:
    if result.preflight_only:
        print_rank_0(
            "Merge preflight complete: "
            f"averaged={result.averaged_tensors}, "
            f"copied_extra_state={result.copied_extra_states}, "
            f"backend={result.backend}, implementation_mode={result.implementation_mode}, "
            f"output_not_written={result.output_dir}",
            flush=True,
        )
    else:
        print_rank_0(
            "Merge complete: "
            f"averaged={result.averaged_tensors}, copied_extra_state={result.copied_extra_states}, "
            f"backend={result.backend}, implementation_mode={result.implementation_mode}, "
            f"verify_load={result.verified_load}, "
            f"output={result.output_dir}",
            flush=True,
        )
    print_rank_0(
        "Timing: "
        f"discovery={result.timings.discovery:.2f}s, "
        f"model_init={result.timings.model_init:.2f}s, "
        f"load={result.timings.load:.2f}s, "
        f"load_per_checkpoint={result.timings.load / len(result.input_dirs):.2f}s, "
        f"accumulation={result.timings.accumulation:.2f}s, "
        f"save={result.timings.save:.2f}s, "
        f"verification={result.timings.verification:.2f}s, "
        f"total={result.timings.total:.2f}s",
        flush=True,
    )
    print_rank_0(
        "I/O: "
        f"read={_format_bytes(result.bytes_read)} "
        f"({_format_bandwidth(result.bytes_read, result.timings.load)}), "
        f"wrote={_format_bytes(result.bytes_written)} "
        f"({_format_bandwidth(result.bytes_written, result.timings.save)})",
        flush=True,
    )
    if result.plan_tensor_bytes_by_rank:
        max_plan_bytes = max(result.plan_tensor_bytes_by_rank)
        min_plan_bytes = min(result.plan_tensor_bytes_by_rank)
        spread = max_plan_bytes / max(1, min_plan_bytes)
        print_rank_0(
            "Planned tensor work: "
            f"balance_rank_work={result.balance_rank_work}, "
            f"bytes_by_rank={_format_rank_bytes(result.plan_tensor_bytes_by_rank)}, "
            f"chunks_by_rank={list(result.plan_tensor_chunks_by_rank)}, "
            f"max_min_ratio={spread:.3f}",
            flush=True,
        )
    print(
        f"Memory rank={result.rank}: host_peak={_format_bytes(result.host_peak_bytes)}",
        flush=True,
    )
    print_rank_0(
        "Memory distributed max: "
        f"host_peak={_format_bytes(result.max_host_peak_bytes)} "
        f"(rank {result.max_host_peak_rank}/{result.world_size})",
        flush=True,
    )


def main() -> None:
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

    # The sole merge path derives the merge from public DCP metadata only. It
    # never bootstraps Megatron or its model-parallel state and constructs no
    # model, so it runs CPU-only for every model family (GPT/Mamba/hybrid/TE).
    args = _parse_metadata_same_layout_args()
    result = _run_metadata_same_layout_cli(args)
    _print_merge_result(result)


if __name__ == "__main__":
    main()
