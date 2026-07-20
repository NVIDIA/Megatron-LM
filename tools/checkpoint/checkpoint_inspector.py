# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Usage examples:
#   python checkpoint_inspector.py inspect /path/to/checkpoint
#
#   torchrun --nproc_per_node=8 --nnodes=1 checkpoint_inspector.py \
#       convert-torch-dist-to-fsdp-dtensor \
#       /path/to/input_checkpoint /path/to/output_checkpoint \
#       --swiglu-modules language_model --rename-mtp-keys
import gc
import io
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import click
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
    FileSystemReader,
    FileSystemWriter,
)
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, TensorStorageMetadata
from torch.distributed.checkpoint.state_dict_saver import _save_state_dict
from torch.distributed.tensor import DeviceMesh, Replicate, Shard

from megatron.core.dist_checkpointing.core import CheckpointingConfig, save_config
from megatron.core.dist_checkpointing.strategies.common import load_common, save_common
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
)
from megatron.core.dist_checkpointing.strategies.torch import TorchDistLoadShardedStrategy
from megatron.core.dist_checkpointing.validation import verify_checkpoint
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    redistribute_uneven_dtensor_to_replicated,
    split_dtensor,
)
from megatron.core.msc_utils import MultiStorageClientFeature


def rank0_echo(message):
    if torch.distributed.get_rank() == 0:
        click.echo(message)


def print_header(text, color="white"):
    click.echo(click.style(f"\n{'=' * 50}", fg=color))
    click.echo(click.style(f"=== {text.upper()} ===", fg=color, bold=True))
    click.echo(click.style(f"{'=' * 50}\n", fg=color))


@click.group()
@click.version_option("1.0")
def cli():
    """Megatron Core Distributed Checkpoint Inspector/Editor"""
    pass


@cli.command()
@click.argument("checkpoint_dir", type=click.Path(exists=True))
@click.option("--enable-msc", is_flag=True, help="Enable MultiStorageClient feature.")
@click.option("--not-ignore-param-to-group-meta", is_flag=True, help="Ignore parameter-to-group metadata.")
def inspect(checkpoint_dir, enable_msc, not_ignore_param_to_group_meta):
    """Inspect a Megatron Core Distributed Checkpoint"""
    ckpt_path = Path(checkpoint_dir)

    if not enable_msc:
        MultiStorageClientFeature.disable()

    # Metadata.json section
    metadata_json = ckpt_path / "metadata.json"
    if not metadata_json.exists():
        click.echo(
            click.style(
                "Metadata file not found in the checkpoint directory.",
                fg="red",
                bold=True,
            )
        )
    else:
        metadata_json = json.loads(metadata_json.read_text())
        print_header("checkpoint metadata", "blue")
        click.echo(
            click.style(json.dumps(metadata_json, indent=2), fg="bright_magenta")
        )

    try:
        # Strategies initialization
        sharded_strategy = TorchDistLoadShardedStrategy()
        sharded_strategy = FullyParallelLoadStrategyWrapper(sharded_strategy)
        verify_checkpoint(checkpoint_dir)
        assert isinstance(
            sharded_strategy.base_strategy, TorchDistLoadShardedStrategy
        ), click.style(
            f"Unsupported sharded strategy: {sharded_strategy}", fg="red", bold=True
        )

        # Common state section
        common_state = load_common(checkpoint_dir)
        print_header(f"common state ({len(common_state)} items)", "cyan")
        for key, value in common_state.items():
            bullet = click.style("•", fg="magenta")
            click.echo(
                f"  {bullet} {click.style(key, fg='green')}: {click.style(str(value), fg='white')}"
            )
    except:
        click.echo(
            click.style("Failed to load checkpoint strategies.", fg="red", bold=True)
        )

    # Tensor metadata section
    reader = FileSystemReader(ckpt_path)
    metadata = reader.read_metadata()
    total_tensors = len([
        v for v in metadata.state_dict_metadata.values()
        if isinstance(v, TensorStorageMetadata)
    ])
    total_elements = sum(
        v.size.numel()
        for v in metadata.state_dict_metadata.values()
        if isinstance(v, TensorStorageMetadata)
    )

    print_header("sharded tensors metadata", "yellow")
    stats = [
        click.style(
            f"Total Tensors: {total_tensors}", fg="bright_magenta"
        ),
        click.style(
            f"Total Elements: {total_elements / 1e9:.2f}B", fg="bright_magenta"
        ),
    ]
    click.echo(" | ".join(stats) + "\n")

    ignore_param_to_group_meta = not not_ignore_param_to_group_meta
    ignore_param_to_group_meta_count = 0
    for key, value in metadata.state_dict_metadata.items():
        bullet = click.style("►", fg="blue")
        key_styled = click.style(key, fg="green")

        if isinstance(value, TensorStorageMetadata):
            dtype = click.style(f"{value.properties.dtype}", fg="cyan")
            shape = click.style(f"{tuple(value.size)}", fg="magenta")
            click.echo(f"  {bullet} {key_styled} [{dtype}, shape={shape}]")
        elif isinstance(value, BytesStorageMetadata):
            if ignore_param_to_group_meta and key.startswith("optimizer.param_to_group_meta."):
                ignore_param_to_group_meta_count += 1
                continue
            click.echo(f"  {bullet} {key_styled} {click.style('[BYTES]', fg='yellow')}")
        else:
            click.echo(
                f"  {bullet} {key_styled} {click.style('[UNKNOWN TYPE]', fg='red')}"
            )
    if ignore_param_to_group_meta:
        click.echo(
            click.style(f"Ignored parameter-to-group metadata: {ignore_param_to_group_meta_count}", fg="yellow")
        )

    # MCore data section
    try:
        mcore_data = metadata.mcore_data
        print_header(f"mcore data ({len(mcore_data)} items)", "green")
        for key, value in mcore_data.items():
            bullet = click.style("▪", fg="yellow")
            click.echo(
                f"  {bullet} {click.style(key, fg='blue')}: {click.style(str(value), fg='white')}"
            )
    except:
        click.echo(
            click.style("No MCore data found in the checkpoint.", fg="red", bold=True)
        )
        pass


@cli.command()
@click.argument("checkpoint_dir", type=click.Path(exists=True))
@click.argument("key", type=str)
def print_tensor(checkpoint_dir, key):
    """Print tensor metadata from a Megatron Core Distributed Checkpoint"""
    ckpt_path = Path(checkpoint_dir)

    # Initialize reader
    reader = FileSystemReader(ckpt_path)
    metadata = reader.read_metadata()

    print_header("tensor metadata", "green")
    if key not in metadata.state_dict_metadata:
        click.echo(
            click.style(
                f"Key '{key}' not found in checkpoint metadata.", fg="red", bold=True
            )
        )
        return

    tensor_metadata = metadata.state_dict_metadata[key]
    if isinstance(tensor_metadata, TensorStorageMetadata):
        click.echo(click.style(f"Key: {key}", fg="blue"))
        click.echo(click.style(f"Shape: {tensor_metadata.size}", fg="cyan"))
        click.echo(
            click.style(f"Dtype: {tensor_metadata.properties.dtype}", fg="magenta")
        )
    elif isinstance(tensor_metadata, BytesStorageMetadata):
        click.echo(click.style(f"Key: {key} (Bytes Storage)", fg="blue"))
    else:
        click.echo(click.style(f"Key: {key} (Unknown Type)", fg="red"))

    # Initialize distributed process group
    dist.init_process_group(
        backend="nccl",
        rank=int(os.getenv("RANK", "0")),
        world_size=int(os.getenv("WORLD_SIZE", "1")),
    )
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))

    state_dict = {
        key: torch.distributed.tensor.empty(
            tensor_metadata.size,
            dtype=tensor_metadata.properties.dtype,
            device_mesh=DeviceMesh.from_group(
                group=dist.group.WORLD,
                device_type="cuda",
                mesh=torch.arange(dist.get_world_size()),
                mesh_dim_names=("world",),
            ),
            placements=[Shard(0)],
        )
    }
    torch.distributed.checkpoint.load(
        state_dict, storage_reader=reader, planner=DefaultLoadPlanner()
    )
    print(state_dict, state_dict[key].shape, state_dict[key]._local_tensor.shape)


def check_gpu_memory(threshold=0.9):
    """
    Check if the GPU memory is over the threshold.
    Args:
        threshold (float, optional): The threshold to check if the GPU memory is over.
            Defaults to 0.9.
    Returns:
        bool: True if the GPU memory is over the threshold.
    """
    if not torch.cuda.is_available():
        return False
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory

    allocated_ratio = allocated / total
    reserved_ratio = reserved / total

    near_full = allocated_ratio >= threshold or reserved_ratio >= threshold

    if near_full and torch.distributed.get_rank() == 0:
        print(
            f"GPU Memory: Allocated: {allocated_ratio:.2%}, Reserved: {reserved_ratio:.2%}"
        )
    return near_full


def flatten(obj, parent_key="", sep="."):
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            items.update(flatten(v, new_key, sep=sep))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(flatten(v, new_key, sep=sep))
    else:
        items[parent_key] = obj
    return items


def save_checkpoint_with_pickle_protocol(state_dict, output_dir, pickle_protocol=4):
    writer = FileSystemWriter(output_dir)
    planner = DefaultSavePlanner()

    def transform_object_override(write_item, obj):
        if isinstance(obj, (bytes, bytearray)):
            return io.BytesIO(obj)
        if isinstance(obj, io.BytesIO):
            return obj
        if isinstance(obj, torch.Tensor):
            return obj
        buffer = io.BytesIO()
        torch.save(obj, buffer, pickle_protocol=pickle_protocol)
        buffer.seek(0)
        return buffer

    planner.transform_object = transform_object_override

    _save_state_dict(
        state_dict=state_dict,
        storage_writer=writer,
        planner=planner,
        process_group=dist.group.WORLD,
    )


class VerboseLoadPlanner(DefaultLoadPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_up_planner(
        self, state_dict, metadata, is_coordinator: bool = False
    ) -> None:
        self.__total_items = len(state_dict)
        self.__resolve_items = {}
        super().set_up_planner(state_dict, metadata, is_coordinator)

    def resolve_tensor(self, read_item) -> torch.Tensor:
        self.__resolve_items[read_item.dest_index.fqn] = True
        rank0_echo(
            f"[{len(self.__resolve_items)}/{self.__total_items}] "
            f"Resolving '{read_item.dest_index.fqn}' "
            f"(Idx: {read_item.storage_index.index}, Off: {read_item.storage_index.offset})"
        )
        return super().resolve_tensor(read_item)


def convert_checkpoint(
    input_dir,
    output_dir,
    swiglu,
    process_group,
    optimizer_param_to_group_prefix="optimizer.param_to_group_meta.module.module.module",
    optimizer_state_prefix="optimizer.state.module.module.module",
    model_weight_prefix="model.module",
    param_to_param_group_map={},
    rename_mtp_keys=False,
    swiglu_modules=None,
):
    """Convert a Megatron Core Distributed Checkpoint from torch_dist to fsdp_dtensor format.

    \b
    Model-specific flags
    ====================
    Different model architectures require different conversion flags.
    Use the table below to determine which flags your model needs:

    \b
      Flag               When to use                              Auto-detected?
      ─────────────────  ───────────────────────────────────────  ──────────────
      --swiglu-modules   Model uses SWiGLU activation in MLP.     No (manual).
                         Check HuggingFace config for              Specify which
                         "hidden_act": "silu" + gate_proj.         modules use it.
      --swiglu           Same as above, but applies globally       No (manual).
                         to ALL modules. Use --swiglu-modules
                         if only some modules use SWiGLU.
      --rename-mtp-keys  Model uses Multi-Token Prediction          YES.
                         (MTP) with 'transformer_layer' naming
                         in the torch_dist checkpoint.

    \b
    Auto-detection: --rename-mtp-keys is auto-detected from checkpoint keys
    if not explicitly set. --swiglu / --swiglu-modules must always be specified
    manually because SWiGLU and GeLU MLP weights are indistinguishable in the
    MCore torch_dist format.

    \b
    Examples
    ========
    Qwen3.5-VL (SWiGLU in language_model only, has GDN + MTP):
      torchrun --nproc_per_node=8 checkpoint_inspector.py \\
        convert-torch-dist-to-fsdp-dtensor \\
        /path/to/torch_dist /path/to/fsdp_dtensor \\
        --swiglu-modules language_model

    Model with SWiGLU in all modules, no GDN/MTP:
      torchrun --nproc_per_node=8 checkpoint_inspector.py \\
        convert-torch-dist-to-fsdp-dtensor \\
        /path/to/torch_dist /path/to/fsdp_dtensor \\
        --swiglu
    """
    device_mesh = DeviceMesh.from_group(process_group, device_type="cuda")

    # 1. Initialize state_dict with proper metadata
    reader = FileSystemReader(input_dir)
    metadata = reader.read_metadata()
    state_dict = {}
    for key, md in metadata.state_dict_metadata.items():
        if isinstance(md, TensorStorageMetadata):
            # Initialize tensor storage
            assert len(md.size) > 0, (
                f"Expected size for key '{key}' to be non-empty, got {md.size}."
            )
            state_dict[key] = torch.distributed.tensor.empty(
                md.size,
                dtype=md.properties.dtype,
                device_mesh=device_mesh,
                placements=[Shard(0)],
            )
        elif isinstance(md, BytesStorageMetadata):
            # Initialize bytes storage
            state_dict[key] = io.BytesIO()
        else:
            raise NotImplementedError(f"Unsupported metadata type: {type(md)}")

    # Load original checkpoint with proper initialization
    start_time = time.time()
    dcp.load(state_dict, storage_reader=reader, planner=VerboseLoadPlanner())
    elapsed_time = time.time() - start_time
    rank0_echo(f"[Load] Finished loading state_dict in {elapsed_time:.2f}s.")

    # --- Resolve SWiGLU scope ---
    # --swiglu-modules (per-module) takes priority over --swiglu (global).
    if swiglu_modules is not None:
        _swiglu_prefixes = list(swiglu_modules)
        rank0_echo(f"[SWiGLU] Per-module splitting enabled for: {_swiglu_prefixes}")
    elif swiglu:
        _swiglu_prefixes = None  # None = match everything (backward compatible)
        rank0_echo("[SWiGLU] Global splitting enabled (all modules).")
    else:
        _swiglu_prefixes = []    # no SWiGLU splitting
        rank0_echo("[SWiGLU] Disabled (no --swiglu or --swiglu-modules specified).")

    # --- Auto-detect MTP from checkpoint keys ---
    # GDN sub-keys are kept un-merged; handle_gdn_in_state_dict() in
    # fsdp_dtensor_checkpoint.py wraps each sub-tensor with correct TP metadata
    # at runtime.
    all_keys = list(state_dict.keys())
    _detected = []

    if not rename_mtp_keys and any(
        ".mtp.layers." in k and ".transformer_layer." in k
        for k in all_keys
    ):
        rename_mtp_keys = True
        _detected.append("MTP (transformer_layer -> mtp_model_layer rename needed)")

    if _detected:
        rank0_echo(
            "[Auto-detect] Enabled transformations based on checkpoint keys:\n"
            + "\n".join(f"  - {d}" for d in _detected)
        )
    del all_keys

    # Handle optimizer state and module parameters
    fsdp_dtensor_state_dict = {}
    rerun_state_machine_state = None
    process_count = 0
    total_items = len(state_dict)
    _swiglu_split_count = 0

    def _free_up_some_gpu_memory():
        if check_gpu_memory(0.5):
            for key in fsdp_dtensor_state_dict:
                if isinstance(fsdp_dtensor_state_dict[key], torch.Tensor):
                    fsdp_dtensor_state_dict[key] = fsdp_dtensor_state_dict[key].cpu()
            gc.collect()
            torch.cuda.empty_cache()

    def split_layers(
        key: str,
        value: torch.Tensor,
        orig_shape: Optional[torch.Size] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Split layers into separate tensors.
        """
        _free_up_some_gpu_memory()
        layers = {}
        for i, v in enumerate(split_dtensor(value, 1, dim=0)):
            v = redistribute_uneven_dtensor_to_replicated(v).reshape(
                orig_shape[1:] if orig_shape else value.shape[1:]
            ).redistribute(placements=[Shard(0)])

            layer_key = key.replace(".layers.", f".layers.{i}.")
            layers[layer_key] = v

        return layers

    def split_expert_weights(
        key: str,
        value: torch.Tensor,
        orig_shape: Optional[torch.Size] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Split expert weights into separate tensors for each expert.
        """
        experts = {}
        layer_key = key.replace(".experts.experts.", ".experts.")
        expert_weights = split_dtensor(value, 1, dim=0)
        for expert_idx, expert_weight in enumerate(expert_weights):
            layer_key_parts = layer_key.split(".weight", 1)
            if len(layer_key_parts) == 1:
                expert_key = f"{layer_key}{expert_idx}"
            elif len(layer_key_parts) == 2:
                expert_key = f"{layer_key_parts[0]}.weight{expert_idx}{layer_key_parts[1]}"
            else:
                raise ValueError(f"Unexpected expert layer key: {layer_key}")

            expert_weight = redistribute_uneven_dtensor_to_replicated(expert_weight)
            expert_shape = orig_shape[1:] if orig_shape else value.shape[1:]
            # Handle optimizer states for expert linear_fc2 when ETP is enabled
            if (
                layer_key.startswith("optimizer.state.")
                and "linear_fc2" in layer_key
                and expert_weight.shape[-2] > 1
            ):
                tp_size = expert_weight.shape[-2]
                rows, cols = expert_shape
                # Reshape to split column dimension by tp_size
                expert_weight = expert_weight.reshape(
                    *expert_weight.shape[:-1], rows, cols // tp_size
                )
                dims = list(range(expert_weight.ndim))
                dims[-3], dims[-2] = dims[-2], dims[-3]
                expert_weight = (
                    expert_weight.permute(*dims)
                    .reshape(expert_shape)
                    .redistribute(placements=[Shard(0)])
                )
            else:
                expert_weight = expert_weight.reshape(expert_shape).redistribute(
                    placements=[Shard(0)]
                )
            experts[expert_key] = expert_weight
        return experts

    _SWIGLU_PATTERNS = [
        r"(.*)\.mlp\.linear_fc1\.weight",
        r"(.*)\.mlp\.linear_fc1\.bias",
        r"(.*)\.mlp\.experts\.linear_fc1\.weight(\d+)",
        r"(.*)\.mlp\.experts\.linear_fc1\.bias(\d+)",
        r"(.*)\.mlp\.experts\.local_experts\.(\d+)\.linear_fc1\.weight",
        r"(.*)\.mlp\.experts\.local_experts\.(\d+)\.linear_fc1\.bias",
        r"(.*)\.mlp\.shared_experts\.linear_fc1\.weight",
        r"(.*)\.mlp\.shared_experts\.linear_fc1\.bias",
    ]

    def is_swiglu_key(key):
        if not any(re.search(pat, key) for pat in _SWIGLU_PATTERNS):
            return False
        # _swiglu_prefixes=None means global (--swiglu), match all.
        # _swiglu_prefixes=[] means nothing detected, match none.
        # _swiglu_prefixes=["language_model", ...] means per-module match.
        if _swiglu_prefixes is None:
            return True
        return any(f".{mod}." in key or key.startswith(f"{mod}.") for mod in _swiglu_prefixes)

    def split_swiglu_weight(key: str, value: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Split SwiGLU weights/biases into separate _w and _v tensors.
        """
        value = redistribute_uneven_dtensor_to_replicated(value)
        swiglu_w_and_v = {}
        w, v = torch.chunk(value, 2, dim=0)
        w = w.redistribute(placements=[Shard(0)])
        v = v.redistribute(placements=[Shard(0)])
        w_key = re.sub(r'((?:weight|bias)\d*)(.*)', r'\1_w\2', key)
        v_key = re.sub(r'((?:weight|bias)\d*)(.*)', r'\1_v\2', key)
        swiglu_w_and_v[w_key] = w
        swiglu_w_and_v[v_key] = v
        return swiglu_w_and_v

    def has_layer_index(key: str) -> bool:
        return bool(re.search(r"layers\.(\d+)\.", key))

    while state_dict:
        key, value = state_dict.popitem()
        if torch.distributed.get_rank() == 0:
            # Print progress for the first rank
            click.echo(f"[Convert] [{process_count + 1}/{total_items}] Key: {key}")
        process_count += 1

        if "._extra_state" in key:
            # Skip extra states
            continue

        if isinstance(value, torch.Tensor):
            if key.startswith("optimizer.state."):
                # Special handling for optimizer state
                key_list = key.split(".")
                new_key = f"{optimizer_state_prefix}.{'.'.join(key_list[3:])}.{key_list[2]}"
                is_param = False
            else:
                # Special handling for module parameters
                new_key = f"{model_weight_prefix}.{key}"
                is_param = True

            # Handle dist-opt flatten tensors
            _has_mcore = hasattr(metadata, "mcore_data")
            if (
                _has_mcore
                and key in metadata.mcore_data
                and "nd_reformulated_orig_global_shape" in metadata.mcore_data[key]
            ):
                mcore_data = metadata.mcore_data[key]
                assert len(mcore_data) == 1, (
                    f"Expected exactly one reformulated shape for key '{key}'."
                )
                # Get the original global shape from mcore_data
                orig_shape = mcore_data["nd_reformulated_orig_global_shape"]
                metadata.mcore_data[key] = {}
            else:
                orig_shape = None

            # Handle multi-layer / experts tensors
            split_tensors = {}
            if ".layers." in new_key and not has_layer_index(new_key):
                split_tensors = split_layers(new_key, value, orig_shape)
            elif ".experts.experts." in new_key:
                split_tensors = split_expert_weights(new_key, value, orig_shape)
            else:
                if orig_shape:
                    value = redistribute_uneven_dtensor_to_replicated(value)
                    # Handle optimizer states with partition_dim=1 when TP is enabled
                    if (
                        new_key.startswith("optimizer.state.")
                        and value.ndim > 2
                        and value.shape[-2] > 1
                    ):
                        tp_size = value.shape[-2]
                        rows, cols = orig_shape
                        # Reshape to split column dimension by tp_size
                        value = value.reshape(*value.shape[:-1], rows, cols // tp_size)
                        dims = list(range(value.ndim))
                        dims[-3], dims[-2] = dims[-2], dims[-3]
                        value = (
                            value.permute(*dims)
                            .reshape(orig_shape)
                            .redistribute(placements=[Shard(0)])
                        )
                    else:
                        value = value.reshape(orig_shape).redistribute(placements=[Shard(0)])
                split_tensors = {new_key: value}

            # Handle SWiGLU weights (per-module: only for modules in _swiglu_prefixes)
            for key, value in list(split_tensors.items()):
                if is_swiglu_key(key):
                    swiglu_w_and_v = split_swiglu_weight(key, value)
                    split_tensors.update(swiglu_w_and_v)
                    del split_tensors[key]
                    _swiglu_split_count += 1

            fsdp_dtensor_state_dict.update(split_tensors)
            if is_param and key in param_to_param_group_map:
                for new_key in split_tensors.keys():
                    param_to_param_group_map[new_key] = param_to_param_group_map[key]
        elif key.startswith("rng_state"):
            # Skip RNG states
            continue
        elif key.startswith("rerun_state_machine_state"):
            if (
                rerun_state_machine_state is not None
                and torch.distributed.get_rank() == 0
            ):
                click.echo(
                    click.style(
                        "Warning: Multiple rerun_state_machine_state found, only the first one will be saved.",
                        fg="yellow",
                    )
                )
                continue
            # Flatten rerun_state_machine_state and add to fsdp_dtensor_state_dict
            fsdp_dtensor_state_dict.update(
                flatten(value[0], parent_key="rerun_state_machine.sharded", sep=".")
            )
            rerun_state_machine_state = value
        else:
            if torch.distributed.get_rank() == 0:
                click.echo(
                    click.style(
                        f"Warning: Key '{key}' is not a tensor and will be serialized as bytes.",
                        fg="yellow",
                    )
                )
            # Serialize non-tensor values as bytes
            serialized_data = io.BytesIO()
            torch.save(value, serialized_data)
            fsdp_dtensor_state_dict[key] = serialized_data

    if _swiglu_split_count > 0:
        rank0_echo(f"[SWiGLU] Split {_swiglu_split_count} fc1 keys into _w/_v pairs.")
    elif _swiglu_prefixes is not None and len(_swiglu_prefixes) > 0:
        rank0_echo("[SWiGLU] WARNING: modules specified but 0 keys were split — check module names.")

    # Rename MTP keys: torch_dist uses "transformer_layer" for MTP sub-modules,
    # but the FSDP model's state_dict() uses "mtp_model_layer".
    if rename_mtp_keys:
        _MTP_OLD = ".mtp.layers."
        _MTP_SRC = ".transformer_layer."
        _MTP_DST = ".mtp_model_layer."
        renamed_count = 0
        for k in list(fsdp_dtensor_state_dict.keys()):
            if _MTP_OLD in k and _MTP_SRC in k:
                new_k = k.replace(_MTP_SRC, _MTP_DST, 1)
                fsdp_dtensor_state_dict[new_k] = fsdp_dtensor_state_dict.pop(k)
                if k in param_to_param_group_map:
                    param_to_param_group_map[new_k] = param_to_param_group_map.pop(k)
                renamed_count += 1
        if renamed_count > 0:
            rank0_echo(f"[MTP rename] Renamed {renamed_count} keys: "
                       f"'transformer_layer' -> 'mtp_model_layer'.")

    # Move back to GPU if necessary
    for key in fsdp_dtensor_state_dict:
        if isinstance(fsdp_dtensor_state_dict[key], torch.Tensor):
            fsdp_dtensor_state_dict[key] = fsdp_dtensor_state_dict[key].cuda()

    # Check MCore data (may not exist for pretrained-only checkpoints)
    if hasattr(metadata, "mcore_data"):
        for key, value in list(metadata.mcore_data.items()):
            if len(value) == 0:
                del metadata.mcore_data[key]
        if len(metadata.mcore_data) != 0 and torch.distributed.get_rank() == 0:
            click.echo(
                click.style(
                    f"Warning: {metadata.mcore_data.keys()} MCore data items were not processed.",
                    fg="yellow",
                )
            )

    # Handle args, optimizer.param_groups and other shared objects.
    sharded_strategy = TorchDistLoadShardedStrategy()
    sharded_strategy = FullyParallelLoadStrategyWrapper(sharded_strategy)
    verify_checkpoint(str(input_dir))
    assert isinstance(sharded_strategy.base_strategy, TorchDistLoadShardedStrategy), (
        click.style(
            f"Unsupported sharded strategy: {sharded_strategy}", fg="red", bold=True
        )
    )
    common_state = load_common(input_dir)
    try:
        if "param_groups" in common_state["optimizer"]:
            ckpt_param_groups = common_state["optimizer"]["param_groups"]
        else:
            ckpt_param_groups = []
            for opt_state_dict in common_state["optimizer"].values():
                ckpt_param_groups.extend(opt_state_dict["optimizer"]["param_groups"])
    except:
        ckpt_param_groups = None
    common_state = flatten(common_state)
    for key, value in common_state.items():
        if key.startswith("optimizer.optimizer.param_groups."):
            key = key.replace(
                "optimizer.optimizer.param_groups.", "optimizer.param_groups."
            )
        assert key not in fsdp_dtensor_state_dict, (
            f"Key '{key}' already exists in fsdp_dtensor_state_dict."
        )
        fsdp_dtensor_state_dict[key] = value

    # set up per-parameter param_groups
    if param_to_param_group_map and ckpt_param_groups is not None:
        for name in list(fsdp_dtensor_state_dict.keys()):
            if not name.startswith(model_weight_prefix) or name.endswith(".expert_bias"):
                continue

            assert name in param_to_param_group_map, f"Missing param group for {name}"
            param_group_id = param_to_param_group_map[name]
            assert param_group_id < len(ckpt_param_groups), f"Invalid param group id {param_group_id} for {name}"
            name_without_prefix = name[len(model_weight_prefix):]
            fsdp_dtensor_state_dict[
                f"{optimizer_param_to_group_prefix}.{name_without_prefix}"
            ] = ckpt_param_groups[param_group_id]

    if "checkpoint_version" not in fsdp_dtensor_state_dict:
        fsdp_dtensor_state_dict["checkpoint_version"] = 3.0

    # Save modified checkpoint
    save_checkpoint_with_pickle_protocol(fsdp_dtensor_state_dict, output_dir)

    dist.barrier()              # Synchronize all ranks
    dist.destroy_process_group()


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--swiglu",
    is_flag=True,
    help="Split SWiGLU fc1 weights/biases into _w and _v for ALL modules. "
         "Use --swiglu-modules instead if only some modules use SWiGLU (e.g. VLMs). "
         "NOT auto-detected; must be specified manually.",
)
@click.option(
    "--swiglu-modules",
    type=str,
    default=None,
    help="Comma-separated module names that use SWiGLU (e.g. 'language_model'). "
         "Only these modules will have fc1 split into _w/_v. Overrides --swiglu. "
         "NOT auto-detected; must be specified manually.",
)
@click.option(
    "--oom-traceback", is_flag=True, help="Enable OOM traceback for debugging."
)
@click.option("--enable-msc", is_flag=True, help="Enable MultiStorageClient feature.")
@click.option(
    "--output-optimizer-state-prefix",
    default="optimizer.state.module.module.module",
    help="Prefix for optimizer state keys in the checkpoint.",
)
@click.option(
    "--output-model-weight-prefix",
    default="model.module",
    help="Prefix for model weight keys in the checkpoint.",
)
@click.option(
    "--param-to-param-group-map-json",
    type=str,
    default=None,
    help="Path to a JSON file mapping parameter names to optimizer param group ids. "
         "Required only if the source checkpoint has multiple optimizer param groups "
         "(e.g. different LR/weight-decay per group). Leave unset for single-group checkpoints."
)
@click.option(
    "--rename-mtp-keys",
    is_flag=True,
    help="Rename MTP layer keys from 'transformer_layer' to 'mtp_model_layer' "
         "to match the FSDP model's state_dict naming. "
         "Auto-detected if not set: enabled when '.mtp.layers.*.transformer_layer' "
         "keys are found in the checkpoint.",
)
def convert_torch_dist_to_fsdp_dtensor(
    input_dir,
    output_dir,
    swiglu,
    swiglu_modules,
    oom_traceback,
    enable_msc,
    output_optimizer_state_prefix,
    output_model_weight_prefix,
    param_to_param_group_map_json,
    rename_mtp_keys,
):
    """Convert a Megatron Core Distributed Checkpoint from torch_dist to fsdp_dtensor format.

    \b
    Model-specific flags
    ====================
    Different model architectures require different conversion flags.
    Use the guide below to determine which flags your model needs:

    \b
      Flag               When to use                            Auto-detected?
      ─────────────────  ────────────────────────────────────── ──────────────
      --swiglu-modules   Model uses SWiGLU activation in MLP.   No (manual).
                         Check HuggingFace config for            Specify which
                         "hidden_act": "silu" + gate_proj.       modules use it.
      --swiglu           Same as above, but applies globally     No (manual).
                         to ALL modules. Use --swiglu-modules
                         if only some modules use SWiGLU.
      --rename-mtp-keys  Model uses Multi-Token Prediction        YES.
                         (MTP) with 'transformer_layer' naming
                         in the torch_dist checkpoint.

    \b
    Auto-detection note:
      --rename-mtp-keys is auto-detected from checkpoint keys if not explicitly
      set. --swiglu / --swiglu-modules must always be specified manually.

    \b
    Examples
    ========
    Qwen3.5-VL (SWiGLU in language_model only, has GDN + MTP):

    \b
      torchrun --nproc_per_node=8 checkpoint_inspector.py \\
        convert-torch-dist-to-fsdp-dtensor \\
        /path/to/torch_dist /path/to/fsdp_dtensor \\
        --swiglu-modules language_model

    \b
    Model with SWiGLU in all modules, no GDN/MTP:

    \b
      torchrun --nproc_per_node=8 checkpoint_inspector.py \\
        convert-torch-dist-to-fsdp-dtensor \\
        /path/to/torch_dist /path/to/fsdp_dtensor \\
        --swiglu
    """
    if not enable_msc:
        MultiStorageClientFeature.disable()

    if oom_traceback:
        torch.cuda.memory._record_memory_history(
            True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,
            # record stack information for the trace events
            trace_alloc_record_context=True,
        )

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            click.echo(
                click.style(
                    f"OOM occurred on rank {torch.distributed.get_rank()} at device {device}.",
                    fg="red",
                    bold=True,
                )
            )
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump

            dump(
                snapshot,
                open(f"oom_rank-{torch.distributed.get_rank()}_snapshot.pickle", "wb"),
            )

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)


    # Initialize distributed process group
    init_process_group(f"convert_torch_dist_to_fsdp_dtensor from {input_dir} to {output_dir}")

    ckpt_path = Path(input_dir)
    output_dir = Path(output_dir)
    if param_to_param_group_map_json:
        with open(param_to_param_group_map_json, "r") as f:
            param_to_param_group_map = json.load(f)
    else:
        param_to_param_group_map = {}
    _swiglu_modules = (
        [m.strip() for m in swiglu_modules.split(",") if m.strip()]
        if swiglu_modules is not None else None
    )
    convert_checkpoint(
        ckpt_path, output_dir, swiglu, process_group=dist.group.WORLD,
        optimizer_state_prefix=output_optimizer_state_prefix,
        model_weight_prefix=output_model_weight_prefix,
        param_to_param_group_map=param_to_param_group_map,
        rename_mtp_keys=rename_mtp_keys,
        swiglu_modules=_swiglu_modules,
    )

    click.echo(
        click.style(
            f"Converted checkpoint saved to {output_dir}.", fg="green", bold=True
        )
    )


# ============================================================================
# Reverse conversion: fsdp_dtensor -> torch_dist
# ============================================================================
#
# This is the exact inverse of ``convert_checkpoint`` above. Whereas the forward
# path loads native torch_dist tensors, *splits* stacked layers/experts and
# SwiGLU fc1 weights, and re-prefixes keys into the Megatron-FSDP DTensor layout,
# the reverse path loads the fsdp_dtensor store, *merges* those pieces back, and
# rewrites native (bare) torch_dist keys.
#
# Key facts that make the reverse path simple and CPU-only:
#   * PyTorch DCP records each tensor's full ``global_shape`` in metadata, so a
#     plain single-rank ``dcp.load`` into ``torch.empty(global_shape)`` returns a
#     fully-gathered tensor regardless of the TP/PP/EP/FSDP sharding it was
#     written with (same trick as ``dist_checkpoint_io.load_dist_checkpoint_full``).
#   * A native torch_dist checkpoint keys model weights by the *bare* mcore FQN
#     (``ShardedTensor.key``) and optimizer state by
#     ``optimizer.state.<subkey>.<param>``; mcore's load validates key + global
#     shape, so the reverse must reproduce mcore's stacking (experts always
#     stacked; dense layers stacked) but must NOT emit ``mcore_data`` /
#     ``nd_reformulated_orig_global_shape`` (legacy/disabled).
#
# NOTE: resuming an N-D-parallel job *with optimizer state* from the produced
# checkpoint requires ``--dist-ckpt-optim-fully-reshardable`` — that is the only
# distributed-optimizer format whose on-disk layout is per-parameter and
# model-shaped (what the fsdp checkpoint holds and what this tool emits).

# mcore parameter FQNs never start with these; used to auto-detect prefixes.
_FSDP_MODEL_PREFIX_DEFAULT = "model.module"

_MTP_FSDP_INFIX = ".mtp_model_layer."
_MTP_TORCH_DIST_INFIX = ".transformer_layer."
# ``mtp.layers.`` at the start of a bare key or nested after a ``.``.
_MTP_LAYERS_RE = re.compile(r"(^|\.)mtp\.layers\.")

# First ``(weight|bias)[digits]`` immediately followed by the SwiGLU ``_w`` tag.
_SWIGLU_W_RE = re.compile(r"((?:weight|bias)\d*)_w(.*)")
# A grouped-expert parameter with a trailing expert index.
_EXPERT_RE = re.compile(r"^(.*\.mlp\.experts\.linear_fc[12]\.(?:weight|bias))(\d+)$")
# A non-grouped (SequentialMLP) routed-expert parameter with an explicit
# local-expert index. Even without ``--moe-grouped-gemm`` mcore's
# ``SequentialMLP.sharded_state_dict`` re-stacks every local expert into a single
# grouped ``experts.experts.linear_fcN`` tensor via a ShardedTensorFactory, so the
# reverse must stack these exactly like the grouped-gemm case. ``shared_experts``
# (``.mlp.shared_experts.``) never carry ``.experts.local_experts.`` and so are
# left untouched.
_LOCAL_EXPERT_RE = re.compile(
    r"^(.*\.mlp\.experts)\.local_experts\.(\d+)\.(linear_fc[12]\.(?:weight|bias))$"
)
# A per-layer parameter with an explicit layer index.
_LAYER_RE = re.compile(r"^(.*\.layers)\.(\d+)\.(.+)$")

# Key patterns the reverse converter cannot faithfully invert. Emitting a
# checkpoint for one of these would silently corrupt it, so fail loudly instead.
# A bare ``.layers.`` with no following index is an already-stacked multi-layer
# buffer whose per-layer structure cannot be recovered. (Mamba-2 and GatedDeltaNet
# fused projections are handled by :func:`_split_mamba_projections` /
# :func:`_split_gdn_projections`, not fenced.)
_UNSUPPORTED_SCOPE_RES = ((re.compile(r"\.layers\.(?!\d)"), "an un-indexed stacked-layer buffer"),)

# Persistent buffers that mcore deliberately keeps in fp32 regardless of the
# training compute dtype (the aux-loss-free load-balancing ``router.expert_bias``;
# see ``TopKRouter._maintain_float32_expert_bias``). Downcasting these to the
# model compute dtype corrupts routing on resume, so exclude them from the
# fp32 -> compute-dtype model downcast below.
_KEEP_FP32_KEY_RES = (re.compile(r"(^|\.)expert_bias$"),)


def _assert_supported_scope(tensors):
    """Raise ``NotImplementedError`` on any key the converter cannot invert.

    Runs on the classified (pre-transform) keys so an unsupported architecture
    fails loudly here instead of silently emitting a corrupt checkpoint.

    Args:
        tensors: mapping of classified model / ``optimizer.state.*`` keys.
    """
    for key in tensors:
        for pattern, what in _UNSUPPORTED_SCOPE_RES:
            if pattern.search(key):
                raise NotImplementedError(
                    f"{what} is not supported by the fsdp_dtensor -> torch_dist "
                    f"converter (offending key: {key!r})."
                )


def _is_keep_fp32_key(key):
    """Return ``True`` if ``key`` names a buffer that must stay fp32."""
    return any(pattern.search(key) for pattern in _KEEP_FP32_KEY_RES)


def _strip_fsdp_model_prefix(key, configured_prefix=_FSDP_MODEL_PREFIX_DEFAULT):
    """Return the bare mcore key for an fsdp model-weight key, else ``None``.

    Strips the configured prefix when it matches; otherwise strips a leading
    ``model.`` plus any run of ``module.`` wrappers (robust to wrapper depth).
    Real mcore parameter FQNs never start with ``model.``/``module.``.
    """
    if configured_prefix and key.startswith(configured_prefix + "."):
        rest = key[len(configured_prefix) + 1 :]
    elif key.startswith("model."):
        rest = key[len("model.") :]
    else:
        return None
    while rest.startswith("module."):
        rest = rest[len("module.") :]
    return rest


def _reverse_optimizer_state_key(key):
    """Invert the forward optimizer-key remap.

    fsdp:       ``optimizer.state.<module.>*<param_fqn>.<subkey>``
    torch_dist: ``optimizer.state.<subkey>.<param_fqn>``

    The trailing dotted token is the optimizer sub-state (``exp_avg`` /
    ``exp_avg_sq`` / ``step`` / ...); the ``module.`` wrapper run is stripped.
    """
    assert key.startswith("optimizer.state."), key
    rest = key[len("optimizer.state.") :]
    while rest.startswith("module."):
        rest = rest[len("module.") :]
    param_fqn, subkey = rest.rsplit(".", 1)
    return f"optimizer.state.{subkey}.{param_fqn}"


def _in_swiglu_modules(key, modules):
    """True if *key* is in one of the (optional) SwiGLU module scopes."""
    if not modules:
        return True
    return any(f".{m}." in key or key.startswith(f"{m}.") for m in modules)


def _reverse_mtp_keys(tensors):
    """Rename ``mtp_model_layer`` back to ``transformer_layer`` in MTP keys."""
    out = {}
    renamed = 0
    for key, value in tensors.items():
        if _MTP_LAYERS_RE.search(key) and _MTP_FSDP_INFIX in key:
            key = key.replace(_MTP_FSDP_INFIX, _MTP_TORCH_DIST_INFIX, 1)
            renamed += 1
        out[key] = value
    return out, renamed


def _merge_swiglu(tensors, swiglu_modules=None):
    """Concatenate SwiGLU ``_w``/``_v`` fc1 pairs back into a single tensor.

    Inverts ``split_swiglu_weight`` (``torch.chunk(..., 2, dim=0)``). Only fc1
    keys carry the ``_w``/``_v`` tags, so this is unambiguous; the optional
    ``swiglu_modules`` scope mirrors the forward ``--swiglu-modules`` filter.
    """
    out = dict(tensors)
    merged = 0
    for key in list(out.keys()):
        if key not in out or ".linear_fc1." not in key:
            continue
        if _SWIGLU_W_RE.search(key) is None:
            continue
        if not _in_swiglu_modules(key, swiglu_modules):
            continue
        base_key = _SWIGLU_W_RE.sub(r"\1\2", key, count=1)
        v_key = _SWIGLU_W_RE.sub(r"\1_v\2", key, count=1)
        if v_key not in out:
            continue  # unpaired _w — leave untouched
        w = out.pop(key)
        v = out.pop(v_key)
        out[base_key] = torch.cat([w, v], dim=0)
        merged += 1
    return out, merged


def _restack_experts(tensors):
    """Stack per-expert weights into one axis-0 ``(num_global_experts, *param)`` tensor.

    Inverts ``split_expert_weights``. Handles both expert storage layouts, which
    mcore's sharded_state_dict unifies into the same grouped
    ``...mlp.experts.experts.linear_fc{1,2}.<weight|bias>`` key:

    * grouped-gemm (TEGroupedMLP): ``...mlp.experts.linear_fc{1,2}.weight{idx}``
    * non-grouped (SequentialMLP): ``...mlp.experts.local_experts.{idx}.linear_fc{1,2}.weight``

    Applies uniformly to model and ``optimizer.state.*`` keys.
    """
    groups = {}  # grouped out_key -> {idx: tensor}
    out = {}
    for key, value in tensors.items():
        m = _EXPERT_RE.match(key)
        if m is not None:
            out_key = m.group(1).replace(
                ".mlp.experts.linear_fc", ".mlp.experts.experts.linear_fc", 1
            )
            groups.setdefault(out_key, {})[int(m.group(2))] = value
            continue
        m = _LOCAL_EXPERT_RE.match(key)
        if m is not None:
            out_key = f"{m.group(1)}.experts.{m.group(3)}"
            groups.setdefault(out_key, {})[int(m.group(2))] = value
            continue
        out[key] = value
    restacked = 0
    for out_key, by_idx in groups.items():
        n = len(by_idx)
        assert set(by_idx) == set(range(n)), (
            f"Expert indices for '{out_key}' are not contiguous 0..{n - 1}: "
            f"{sorted(by_idx)} — an incomplete or partially-gathered checkpoint?"
        )
        out[out_key] = torch.stack([by_idx[i] for i in range(n)], dim=0)
        restacked += 1
    return out, restacked


def _stack_layers(tensors):
    """Stack per-layer ``...layers.{i}.<param>`` into one ``...layers.<param>``.

    Inverts ``split_layers`` for dense/homogeneous blocks (global shape
    ``(num_layers, *param)``). Raises if the block is heterogeneous (a param is
    missing from some layers) so the caller can fall back to per-layer via
    ``--non-homogeneous-layers``.
    """
    groups = {}
    out = {}
    prefix_idxs = {}
    for key, value in tensors.items():
        m = _LAYER_RE.match(key)
        if m is None or _MTP_LAYERS_RE.search(key):
            # Non-layer key, or an MTP layer (a separate block that mcore always
            # stores per-layer) — leave untouched.
            out[key] = value
            continue
        prefix, idx, suffix = m.group(1), int(m.group(2)), m.group(3)
        groups.setdefault((prefix, suffix), {})[idx] = value
        prefix_idxs.setdefault(prefix, set()).add(idx)
    stacked_count = 0
    for (prefix, suffix), by_idx in groups.items():
        full = prefix_idxs[prefix]
        if set(by_idx) != full:
            raise ValueError(
                f"Heterogeneous layers: '{suffix}' under '{prefix}' appears in "
                f"layers {sorted(by_idx)} but the block spans {sorted(full)}. "
                f"Re-run with --non-homogeneous-layers to keep per-layer keys."
            )
        n = max(full) + 1
        assert sorted(full) == list(range(n)), (
            f"Non-contiguous layers under '{prefix}': {sorted(full)}"
        )
        out[f"{prefix}.{suffix}"] = torch.stack([by_idx[i] for i in range(n)], dim=0)
        stacked_count += 1
    return out, stacked_count


def _layers_are_homogeneous(keys):
    """Whether the transformer block's decoder layers are stored *stacked*.

    Mirrors ``TransformerBlock.sharded_state_dict`` (transformer_block.py:727-746):
    a block is stored **per-layer** (non-homogeneous) only when its layers differ
    in structure — interleaved MoE/dense (``moe_layer_freq > 1``), interleaved
    linear-attention/attention (``linear_attention_freq > 1``), or heterogeneous
    block specs. A **uniform** block — every layer identical, whether all-dense,
    all-MoE (``moe_layer_freq == 1``), or all-linear-attention — is stored
    **stacked** with a leading ``num_layers`` axis.

    Model-free proxy for that config rule: group model-weight keys by layer index
    and compare the index-stripped parameter-name sets. Identical across every
    layer ⇒ homogeneous ⇒ stack. MTP layers (``mtp.layers.``) are a separate
    block and never participate in the decoder-stacking decision.
    """
    per_prefix = {}  # layer-prefix -> {layer_idx: set(param-suffix)}
    for key in keys:
        if _MTP_LAYERS_RE.search(key):
            continue
        m = _LAYER_RE.match(key)
        if m is None:
            continue
        prefix, idx, suffix = m.group(1), int(m.group(2)), m.group(3)
        per_prefix.setdefault(prefix, {}).setdefault(idx, set()).add(suffix)
    if not per_prefix:
        return False  # nothing per-layer to stack (already stacked or no layers)
    for by_idx in per_prefix.values():
        suffix_sets = list(by_idx.values())
        if any(s != suffix_sets[0] for s in suffix_sets[1:]):
            return False  # layers differ in structure ⇒ non-homogeneous
    return True


# Gated-DeltaNet fuses q/k/v/gate/beta/alpha into one ``in_proj.weight`` and
# q/k/v into one ``conv1d.weight``. mcore's GatedDeltaNet.sharded_state_dict
# (ssm/gated_delta_net.py:671-700) stores these via a ``_split_tensor_factory``
# that splits dim-0 into named sub-tensors, so a classic torch_dist load expects
# the checkpoint to already carry the split ``<key>.<name>`` sub-keys. fsdp_dtensor
# stores the fused blob instead, so the reverse path must reproduce the split.
_GDN_INPROJ_SUFFIX = ".self_attention.in_proj.weight"
_GDN_CONV_SUFFIXES = (".self_attention.conv1d.weight", ".self_attention.conv1d.bias")

# Mamba-2 mixer fused projections. The classic ``MambaMixer.sharded_state_dict``
# stores these split (``in_proj`` 5-way, ``conv1d`` 3-way) with the underscore
# ``conv1d_weight``/``conv1d_bias`` nn.Parameters renamed to dotted ``conv1d.weight``/
# ``conv1d.bias`` on disk; the fsdp_dtensor source stores them fused.
_MAMBA_INPROJ_SUFFIX = ".mixer.in_proj.weight"
_MAMBA_CONV_RE = re.compile(r"^(.*\.mixer\.)(conv1d_weight|conv1d_bias)$")
_MAMBA_CONV_RENAME = {"conv1d_weight": "conv1d.weight", "conv1d_bias": "conv1d.bias"}


def _split_gdn_projections(tensors, args):
    """Split fused GatedDeltaNet ``in_proj``/``conv1d`` into named factory sub-keys.

    Mirrors ``GatedDeltaNet.sharded_state_dict`` at TP=1: ``in_proj`` splits dim-0
    into ``[qk, qk, v, v, nvh, nvh]`` named ``query,key,value,z,beta,alpha`` and
    ``conv1d`` into ``[qk, qk, v]`` named ``query,key,value`` (``qk =
    num_key_heads*key_head_dim``, ``v = num_value_heads*value_head_dim``, ``nvh =
    num_value_heads``). Applies to model weights and their ``optimizer.state.*``
    counterparts (the fully_reshardable optimizer mirrors the model factories).
    No-op unless the checkpoint's ``args`` selects the gated_delta_net variant.
    """
    if args is None or getattr(args, "experimental_attention_variant", None) != "gated_delta_net":
        return tensors, 0
    qk = args.linear_num_key_heads * args.linear_key_head_dim
    v = args.linear_num_value_heads * args.linear_value_head_dim
    nvh = args.linear_num_value_heads
    inproj = ([qk, qk, v, v, nvh, nvh], ["query", "key", "value", "z", "beta", "alpha"])
    conv = ([qk, qk, v], ["query", "key", "value"])

    def spec_for(key):
        if key.endswith(_GDN_INPROJ_SUFFIX):
            return inproj
        if any(key.endswith(s) for s in _GDN_CONV_SUFFIXES):
            return conv
        return None

    out = {}
    n_split = 0
    for key, value in tensors.items():
        spec = spec_for(key)
        if spec is None:
            out[key] = value
            continue
        sections, names = spec
        # Per-layer keys keep an explicit ``.layers.{idx}.`` (interleaved GDN) and
        # split dim 0; a stacked homogeneous block carries a leading num-layers axis
        # and splits dim 1.
        dim = 0 if re.search(r"\.layers\.\d+\.", key) else 1
        assert value.shape[dim] == sum(sections), (
            f"GDN split for '{key}': dim {dim} size {value.shape[dim]} != "
            f"sum({sections})={sum(sections)} — args/checkpoint mismatch?"
        )
        start = 0
        for size, name in zip(sections, names):
            out[f"{key}.{name}"] = value.narrow(dim, start, size).contiguous()
            start += size
        n_split += 1
    return out, n_split


def _split_mamba_projections(tensors, args):
    """Split fused Mamba-2 ``in_proj``/``conv1d`` into named factory sub-keys.

    Mirrors ``MambaMixer.sharded_state_dict`` (megatron/core/ssm/mamba_mixer.py):
    ``mixer.in_proj.weight`` splits dim-0 into ``[d_inner, d_inner, gds, gds, nheads]``
    named ``z,x,B,C,dt``; the fused ``mixer.conv1d_weight``/``conv1d_bias`` are renamed
    to the on-disk ``mixer.conv1d.weight``/``conv1d.bias`` and split into
    ``[d_inner, gds, gds]`` named ``x,B,C`` (``gds = mamba_num_groups *
    mamba_state_dim``). ``d_inner`` is recovered from each tensor's fused width, so
    the split needs only the group/state dims and the head dim from the checkpoint
    ``args`` — not the (optional) head count. ``A_log``/``D``/``dt_bias``/``norm``/
    ``out_proj`` pass through. Applies to model weights and their ``optimizer.state.*``
    counterparts. No-op unless the checkpoint carries Mamba mixer keys.
    """
    has_mamba = any(k.endswith(_MAMBA_INPROJ_SUFFIX) or _MAMBA_CONV_RE.match(k) for k in tensors)
    if not has_mamba:
        return tensors, 0
    if args is None or getattr(args, "mamba_state_dim", None) is None:
        raise NotImplementedError(
            "Mamba mixer keys found but the checkpoint args lack the Mamba config "
            "(mamba_state_dim / mamba_num_groups / mamba_head_dim) needed to split them."
        )
    gds = args.mamba_num_groups * args.mamba_state_dim
    headdim = args.mamba_head_dim

    out = {}
    n_split = 0
    for key, value in tensors.items():
        is_inproj = key.endswith(_MAMBA_INPROJ_SUFFIX)
        conv_m = _MAMBA_CONV_RE.match(key)
        if not is_inproj and conv_m is None:
            out[key] = value
            continue
        # Per-layer keys carry an explicit ``.layers.{idx}.`` (interleaved hybrid) and
        # split dim 0; a stacked homogeneous block carries a leading num-layers axis.
        dim = 0 if re.search(r"\.layers\.\d+\.", key) else 1
        width = value.shape[dim]
        if is_inproj:
            # width = 2*d_inner + 2*gds + nheads, with nheads = d_inner / headdim.
            d_inner = headdim * (width - 2 * gds) // (2 * headdim + 1)
            nheads = d_inner // headdim
            sections = [d_inner, d_inner, gds, gds, nheads]
            names = ["z", "x", "B", "C", "dt"]
            out_key = key
        else:
            d_inner = width - 2 * gds  # conv width = d_inner + 2*gds
            sections = [d_inner, gds, gds]
            names = ["x", "B", "C"]
            out_key = conv_m.group(1) + _MAMBA_CONV_RENAME[conv_m.group(2)]
        assert d_inner > 0 and sum(sections) == width, (
            f"Mamba split for '{key}': derived sections {sections} (sum {sum(sections)}) "
            f"!= dim {dim} size {width} — args/checkpoint mismatch?"
        )
        start = 0
        for size, name in zip(sections, names):
            out[f"{out_key}.{name}"] = value.narrow(dim, start, size).contiguous()
            start += size
        n_split += 1
    return out, n_split


def _unflatten(flat):
    """Inverse of ``flatten``: rebuild nested dicts, ints-keyed dicts -> lists."""

    def _lists_from_int_keys(obj):
        if not isinstance(obj, dict):
            return obj
        obj = {k: _lists_from_int_keys(v) for k, v in obj.items()}
        keys = list(obj.keys())
        if keys and all(isinstance(k, str) and k.isdigit() for k in keys):
            idxs = sorted(int(k) for k in keys)
            if idxs == list(range(len(idxs))):
                return [obj[str(i)] for i in range(len(idxs))]
        return obj

    root = {}
    for key, value in flat.items():
        parts = key.split(".")
        node = root
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value
    return _lists_from_int_keys(root)


# mcore matches checkpoint param_groups to the freshly-built optimizer's groups by
# this identifier tuple (optimizer.py:97; distrib_optimizer.py:894-912), so each
# group necessarily has a distinct tuple — deduping by it reproduces the groups.
_PARAM_GROUP_ID_KEYS = ("wd_mult", "lr_mult", "is_expert_parallel", "is_decoupled_lr")


def _rebuild_param_groups_from_meta(param_meta_flat, meta_prefix):
    """Reconstruct torch optimizer ``param_groups`` from Megatron-FSDP's map.

    Megatron-FSDP stores optimizer hyperparameters **per parameter**
    (``optimizer.param_to_group_meta.<param>.<attr>``) rather than as grouped
    ``param_groups``. A classic (non-FSDP) load expects standard ``param_groups``
    (distrib_optimizer.py:910) and matches checkpoint groups to the freshly-built
    optimizer's groups by :data:`_PARAM_GROUP_ID_KEYS`, discarding the checkpoint's
    ``params`` list. Group parameters by that identifier tuple and emit one group
    per distinct tuple, carrying its full hyperparameters (lr, betas, eps, step, …).
    """
    per_param = {}  # param_fqn -> {attr: value}
    for key, value in param_meta_flat.items():
        rest = key[len(meta_prefix) :]
        while rest.startswith("module."):
            rest = rest[len("module.") :]
        param_fqn, attr = rest.rsplit(".", 1)
        per_param.setdefault(param_fqn, {})[attr] = value

    groups = {}  # identifier tuple -> (representative meta, [param_fqns])
    order = []
    for param_fqn, meta in per_param.items():
        ident = tuple(meta.get(k) for k in _PARAM_GROUP_ID_KEYS)
        if ident not in groups:
            groups[ident] = (meta, [])
            order.append(ident)
        groups[ident][1].append(param_fqn)

    param_groups = []
    next_index = 0
    for ident in order:
        meta, params = groups[ident]
        group = dict(meta)
        # ``params`` is discarded on load (distrib_optimizer.py:918) — emit standard
        # contiguous integer indices purely for on-disk format fidelity.
        group["params"] = list(range(next_index, next_index + len(params)))
        next_index += len(params)
        param_groups.append(group)
    return param_groups


def _model_param_dtype(args):
    """Training compute dtype for model-section weights, read from saved ``args``.

    mcore stores model weights in the compute dtype (bf16/fp16) while the
    optimizer keeps fp32 masters. Returns ``None`` when ``args`` is unavailable
    (leave tensors as-is).
    """
    if args is None:
        return None
    if getattr(args, "bf16", False):
        return torch.bfloat16
    if getattr(args, "fp16", False):
        return torch.float16
    return torch.float32


def _output_group_id(fqn, do_stack):
    """Reduce a bare param FQN to the identity of the output tensor it feeds.

    Keys that must be transformed together (SwiGLU ``_w``/``_v`` halves, the
    experts of one layer, the layers of one param when stacking, and a param plus
    its ``optimizer.state.*`` subkeys) all reduce to the same group id, so sharding
    by group keeps every transform group whole on a single rank. Over-grouping is
    safe (only balance suffers); splitting a real group would corrupt the output.
    """
    g = re.sub(r"((?:weight|bias)\d*)_[wv](?=$|\.)", r"\1", fqn)  # SwiGLU _w/_v tag
    g = re.sub(r"(\.mlp\.experts\.linear_fc[12]\.(?:weight|bias))\d+$", r"\1", g)  # grouped expert idx
    g = re.sub(r"\.local_experts\.\d+\.", ".experts.", g)  # SequentialMLP expert idx
    if do_stack:
        g = re.sub(r"(\.layers)\.\d+(\.)", r"\1\2", g)  # homogeneous per-layer -> stacked
    return g


def _assign_tensor_keys_to_rank(
    md_items, input_prefix, include_optimizer, do_stack, rank, world_size
):
    """Return the set of fsdp tensor keys this rank owns for a sharded convert.

    Tensor items are partitioned by :func:`_output_group_id` and assigned to ranks
    round-robin over the sorted group list (deterministic across ranks). ``_extra_state``
    / RNG / rerun keys are skipped (never written). ``world_size == 1`` returns every key.
    """
    key_group = {}
    for key, md in md_items.items():
        if not isinstance(md, TensorStorageMetadata) or "_extra_state" in key:
            continue
        if key.startswith("rng_state") or key.startswith("rerun_state_machine"):
            continue
        bare = _strip_fsdp_model_prefix(key, input_prefix)
        if bare is not None:
            fqn = bare
        elif key.startswith("optimizer.state."):
            if not include_optimizer:
                continue
            rev = _reverse_optimizer_state_key(key)  # optimizer.state.<subkey>.<fqn>
            fqn = rev.split(".", 3)[3] if rev.count(".") >= 3 else rev
        else:
            fqn = "__misc__"
        key_group[key] = _output_group_id(fqn, do_stack)
    group_to_rank = {g: i % world_size for i, g in enumerate(sorted(set(key_group.values())))}
    return {key for key, group in key_group.items() if group_to_rank[group] == rank}


def _ensure_cpu_process_group():
    """Initialize a 1-rank gloo group if none is up (reverse path is CPU-only)."""
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(
        backend="gloo",
        rank=int(os.getenv("RANK", "0")),
        world_size=int(os.getenv("WORLD_SIZE", "1")),
    )


def reverse_convert_checkpoint(
    input_dir,
    output_dir,
    swiglu_modules=None,
    stack_layers=None,
    include_optimizer=True,
    input_model_weight_prefix=_FSDP_MODEL_PREFIX_DEFAULT,
    output_model_weight_prefix="",
):
    """Convert a Megatron-FSDP ``fsdp_dtensor`` checkpoint to ``torch_dist``.

    Args:
        input_dir: fsdp_dtensor checkpoint directory (a single DCP store).
        output_dir: destination torch_dist checkpoint directory.
        swiglu_modules: optional list of module scopes to restrict SwiGLU
            ``_w``/``_v`` merging to (default: merge every fc1 pair found).
        stack_layers: tri-state layer-stacking control. ``None`` auto-detects
            via :func:`_layers_are_homogeneous` (stack when every decoder layer is
            structurally identical — all-dense or all-MoE — keep per-layer when
            layers differ, e.g. interleaved MoE/dense or linear-attention/GDN);
            ``True`` forces stacking; ``False`` forces per-layer.
        include_optimizer: also convert ``optimizer.state.*`` tensors.
        input_model_weight_prefix: fsdp model-weight prefix to strip.
        output_model_weight_prefix: prefix to write for model weights (bare "").
    """
    _ensure_cpu_process_group()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # ---- 1. Load the fsdp store into fully-gathered CPU tensors ----------
    # Under a multi-rank launch the tensor items are sharded across ranks by
    # output-group (:func:`_assign_tensor_keys_to_rank`) so peak host RAM scales
    # ~1/world_size; the collective save below reassembles one torch_dist store.
    # Byte items (args / param_groups / ...) are small and loaded on every rank so
    # each has the ``args`` the transforms need. world_size == 1 loads everything.
    reader = FileSystemReader(input_dir)
    metadata = reader.read_metadata()
    md_items = metadata.state_dict_metadata

    # Layer stacking is a global property of the model keys; compute it from the
    # full key set so sharded ranks agree (never from a rank's partial subset).
    global_model_keys = []
    for k, md in md_items.items():
        if isinstance(md, TensorStorageMetadata) and "_extra_state" not in k:
            bare = _strip_fsdp_model_prefix(k, input_model_weight_prefix)
            if bare is not None:
                global_model_keys.append(bare)
    do_stack = (
        stack_layers if stack_layers is not None else _layers_are_homogeneous(global_model_keys)
    )

    owned = _assign_tensor_keys_to_rank(
        md_items, input_model_weight_prefix, include_optimizer, do_stack, rank, world_size
    )
    load_dict = {}
    for key, md in md_items.items():
        if isinstance(md, TensorStorageMetadata):
            if key in owned:
                load_dict[key] = torch.empty(md.size, dtype=md.properties.dtype, device="cpu")
        elif isinstance(md, BytesStorageMetadata):
            load_dict[key] = io.BytesIO()
        else:
            raise NotImplementedError(f"Unsupported metadata type: {type(md)}")

    start_time = time.time()
    dcp.load(load_dict, storage_reader=reader, planner=DefaultLoadPlanner())
    rank0_echo(
        f"[Load] Loaded fsdp_dtensor store in {time.time() - start_time:.2f}s "
        f"(rank {rank}/{world_size}, {len(load_dict)} items)."
    )

    # ---- 2. Classify keys into model / optimizer tensors and common state -
    # DCP restores tensor items as ``torch.Tensor`` and byte items as their
    # deserialized Python object (or a ``BytesIO``); classify on the value type.
    tensors = {}  # canonical (bare model | optimizer.state.<subkey>.<param>) -> tensor
    common_flat = {}
    param_meta_flat = {}  # optimizer.param_to_group_meta.* (fsdp-native)
    n_skipped_extra_state = 0
    for key, value in load_dict.items():
        if "_extra_state" in key:
            n_skipped_extra_state += 1
            continue
        if key.startswith("rng_state") or key.startswith("rerun_state_machine"):
            continue  # RNG is re-seedable; rerun state is optional on load.
        if isinstance(value, torch.Tensor):
            bare = _strip_fsdp_model_prefix(key, input_model_weight_prefix)
            if bare is not None:
                tensors[bare] = value
            elif key.startswith("optimizer.state."):
                if include_optimizer:
                    tensors[_reverse_optimizer_state_key(key)] = value
            else:
                rank0_echo(
                    click.style(f"[Warn] Unclassified tensor key dropped: {key}", fg="yellow")
                )
            continue
        # Non-tensor: common state (args, param_groups, checkpoint_version, ...).
        if isinstance(value, io.BytesIO):
            value.seek(0)
            value = torch.load(value, weights_only=False)
        if key.startswith("optimizer.param_to_group_meta."):
            # fsdp-native per-parameter optimizer config; reconstructed into
            # standard param_groups below (a classic load needs param_groups).
            if include_optimizer:
                param_meta_flat[key] = value
            continue
        common_flat[key] = value

    # ---- 2a. Fail loudly on architectures this tool cannot invert ----------
    # (Mamba fused conv1d, un-indexed stacked-layer buffers). Done before any
    # transform so an unsupported checkpoint raises rather than silently
    # producing a wrong one. GatedDeltaNet's dotted conv1d.weight is unaffected.
    _assert_supported_scope(tensors)

    # ---- 2b. Reconstruct fp32 optimizer masters + downcast the model section --
    # Megatron-FSDP stores fp32 model weights (they *are* the optimizer masters)
    # and no separate master copy. mcore's fully_reshardable torch_dist optimizer
    # instead expects, per trainable parameter, a fp32 master under
    # ``optimizer.state.param.<fqn>`` *plus* a model-section weight in the training
    # compute dtype (bf16/fp16). Synthesize the master from each fp32 model weight
    # that carries optimizer moments, then downcast the model section. Done here
    # (pre-transform, in the fsdp-split key space) so the master rides through the
    # SwiGLU/expert/layer transforms identically to ``exp_avg``. A checkpoint that
    # already carries masters (e.g. produced by the forward converter) is left
    # untouched. ``.contiguous()`` protects the later single-chunk write.
    n_masters = 0
    if include_optimizer and not any(k.startswith("optimizer.state.param.") for k in tensors):
        for key in list(tensors):
            if not key.startswith("optimizer.state.exp_avg."):
                continue
            fqn = key[len("optimizer.state.exp_avg.") :]
            master = tensors.get(fqn)
            if master is not None:
                tensors[f"optimizer.state.param.{fqn}"] = master.detach().clone()
                n_masters += 1
    model_dtype = _model_param_dtype(common_flat.get("args"))
    if model_dtype is not None and model_dtype != torch.float32:
        for key in list(tensors):
            value = tensors[key]
            if (
                not key.startswith("optimizer.")
                and isinstance(value, torch.Tensor)
                and value.dtype == torch.float32
                and not _is_keep_fp32_key(key)
            ):
                tensors[key] = value.to(model_dtype)

    # ---- 3. Apply the inverse transforms (order mirrors the forward split) -
    tensors, n_mtp = _reverse_mtp_keys(tensors)
    tensors, n_swiglu = _merge_swiglu(tensors, swiglu_modules)
    tensors, n_experts = _restack_experts(tensors)

    # ``do_stack`` was decided globally at load time (a per-rank subset must not
    # re-derive homogeneity from its partial key set).
    if do_stack:
        tensors, n_layers = _stack_layers(tensors)
    else:
        n_layers = 0

    # Gated-DeltaNet and Mamba-2 fused-projection splits (in_proj/conv1d -> named
    # sub-keys). Run after stacking so the per-layer/stacked dim is unambiguous.
    tensors, n_gdn = _split_gdn_projections(tensors, common_flat.get("args"))
    tensors, n_mamba = _split_mamba_projections(tensors, common_flat.get("args"))

    rank0_echo(
        f"[Convert] mtp={n_mtp} swiglu={n_swiglu} experts={n_experts} "
        f"layer-stacks={n_layers} gdn-splits={n_gdn} mamba-splits={n_mamba} "
        f"masters={n_masters} extra_state-dropped={n_skipped_extra_state} "
        f"layout={'stacked' if do_stack else 'per-layer'}"
    )

    # ---- 4. Rebuild common.pt (unflatten; drop rerun/rng; pin version) -----
    # Invert the forward converter's chained-optimizer flattening: mcore wraps
    # the DistributedOptimizer in a ChainedOptimizer, so its common state nests
    # ``optimizer.optimizer.param_groups`` (the forward collapsed the doubled
    # ``optimizer.`` to a single one).
    renamed_common = {}
    for key, value in common_flat.items():
        if key.startswith("optimizer.param_groups."):
            key = key.replace("optimizer.param_groups.", "optimizer.optimizer.param_groups.", 1)
        renamed_common[key] = value
    common_state = _unflatten(renamed_common)
    common_state.setdefault("checkpoint_version", 3.0)

    # mcore's load reads the distributed-optimizer sharding type from the
    # checkpoint's *content metadata* (``dist_checkpointing.load_content_metadata``,
    # stored under ``common['content_metadata']``). Without it, a classic
    # (non-FSDP) load defaults to the unsupported ``fully_sharded_model_space``
    # and raises "ShardedTensor.flattened_range is not supported.". Emit the
    # ``fully_reshardable`` content metadata (mirrors ``_build_sharded_state_dict_metadata``)
    # so the converted torch_dist checkpoint loads as fully_reshardable — the one
    # format whose on-disk layout matches what this tool writes.
    common_state["content_metadata"] = {
        "distrib_optim_sharding_type": "fully_reshardable",
        "distrib_optim_fully_reshardable_mem_efficient": False,
        "singleton_local_shards": False,
        "chained_optim_avoid_prefix": True,
    }

    # Real Megatron-FSDP checkpoints store a per-parameter ``param_to_group_meta``
    # map instead of grouped ``param_groups``; a classic load needs the latter.
    # Rebuild it into the ChainedOptimizer(DistributedOptimizer) nesting mcore
    # expects: ``common['optimizer']['optimizer']['param_groups']``
    # (optimizer.py:1382 -> distrib_optimizer.py:910), a sibling of the sharding
    # type. Forward-converted checkpoints already carry param_groups (handled by
    # the re-nesting above) and produce no ``param_meta_flat``.
    if include_optimizer and param_meta_flat:
        param_groups = _rebuild_param_groups_from_meta(
            param_meta_flat, "optimizer.param_to_group_meta."
        )
        common_state["optimizer"] = {
            "optimizer": {"param_groups": param_groups},
            "param_state_sharding_type": "fully_reshardable",
        }

    # ---- 5. Write the torch_dist checkpoint (bare keys, no mcore_data) ------
    if output_model_weight_prefix:
        tensors = {f"{output_model_weight_prefix}.{k}": v for k, v in tensors.items()}
    for key in tensors:
        if not tensors[key].is_contiguous():
            tensors[key] = tensors[key].contiguous()

    os.makedirs(output_dir, exist_ok=True)
    save_checkpoint_with_pickle_protocol(tensors, output_dir)
    if not dist.is_initialized() or dist.get_rank() == 0:
        if common_state:
            save_common(common_state, output_dir)
        save_config(CheckpointingConfig(sharded_backend="torch_dist"), output_dir)
    if dist.is_initialized():
        dist.barrier()


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--swiglu-modules",
    type=str,
    default=None,
    help="Comma-separated module scopes to restrict SwiGLU _w/_v merging to "
    "(e.g. 'language_model'). Default: merge every fc1 _w/_v pair found.",
)
@click.option(
    "--stack-layers/--non-homogeneous-layers",
    "stack_layers",
    default=None,
    help="Force a stacked or per-layer torch_dist layout. Default: auto-detect — "
    "stack when every decoder layer is structurally identical (all-dense or "
    "all-MoE), keep per-layer when layers differ (interleaved MoE/dense or GDN).",
)
@click.option(
    "--no-optimizer",
    is_flag=True,
    help="Convert model weights only; skip optimizer state.",
)
@click.option(
    "--input-model-weight-prefix",
    default=_FSDP_MODEL_PREFIX_DEFAULT,
    help="fsdp model-weight key prefix to strip (default: 'model.module').",
)
@click.option(
    "--output-model-weight-prefix",
    default="",
    help="Prefix to write for model-weight keys (default: '' — bare mcore keys, "
    "which is what a native torch_dist checkpoint uses).",
)
@click.option("--enable-msc", is_flag=True, help="Enable MultiStorageClient feature.")
def convert_fsdp_dtensor_to_torch_dist(
    input_dir,
    output_dir,
    swiglu_modules,
    stack_layers,
    no_optimizer,
    input_model_weight_prefix,
    output_model_weight_prefix,
    enable_msc,
):
    """Convert a Megatron-FSDP fsdp_dtensor checkpoint to torch_dist format.

    \b
    This is the inverse of ``convert-torch-dist-to-fsdp-dtensor``: it merges the
    SwiGLU ``_w``/``_v`` fc1 pairs, re-stacks per-expert (and, for dense models,
    per-layer) tensors into mcore's native stacked layout, renames MTP keys back
    to ``transformer_layer``, strips the FSDP wrapper prefixes, and rewrites
    native (bare) torch_dist keys plus ``common.pt`` / ``metadata.json``.

    \b
    IMPORTANT — optimizer resume
    ============================
    To resume an N-D-parallel (TP/PP/EP) job *with optimizer state* from the
    produced checkpoint, the resuming job must set:

    \b
      --dist-ckpt-optim-fully-reshardable

    That is the only distributed-optimizer checkpoint format whose on-disk
    layout is per-parameter and model-shaped — exactly what the fsdp_dtensor
    checkpoint holds and what this tool emits. FP8 ``_extra_state`` is not
    round-tripped (it was already discarded when the fsdp checkpoint was made),
    so this path targets bf16/fp32 checkpoints.

    \b
    Examples
    ========
    Qwen3.5-VL (SWiGLU in language_model only, has GDN + MTP):

    \b
      python checkpoint_inspector.py \\
        convert-fsdp-dtensor-to-torch-dist \\
        /path/to/fsdp_dtensor /path/to/torch_dist \\
        --swiglu-modules language_model

    \b
    Dense model (layers auto-stacked):

    \b
      python checkpoint_inspector.py \\
        convert-fsdp-dtensor-to-torch-dist \\
        /path/to/fsdp_dtensor /path/to/torch_dist
    """
    if not enable_msc:
        MultiStorageClientFeature.disable()

    _swiglu_modules = (
        [m.strip() for m in swiglu_modules.split(",") if m.strip()]
        if swiglu_modules is not None
        else None
    )

    reverse_convert_checkpoint(
        Path(input_dir),
        Path(output_dir),
        swiglu_modules=_swiglu_modules,
        stack_layers=stack_layers,
        include_optimizer=not no_optimizer,
        input_model_weight_prefix=input_model_weight_prefix,
        output_model_weight_prefix=output_model_weight_prefix,
    )

    click.echo(
        click.style(
            f"Converted checkpoint saved to {output_dir}.", fg="green", bold=True
        )
    )


def _modify_state_dict(input_dir, output_dir, ops, process_group, enable_msc=False):
    """Modify state dict items in a Megatron Core Distributed Checkpoint."""
    remove_items = []
    rename_items = []
    for op in ops:
        assert isinstance(op, str), f"Operation '{op}' must be a string."
        op_items = op.split()
        if op_items[0] == "remove":
            assert len(op_items) == 2, f"Remove operation requires exactly one argument: {op_items[1]}"
            remove_items.append(op_items[1])
        elif op_items[0] == "rename":
            assert len(op_items) == 3, f"Rename operation requires exactly two arguments: {op_items[1]} {op_items[2]}"
            rename_items.append((op_items[1], op_items[2]))
        else:
            raise NotImplementedError(f"Unsupported operation: {op} | {op_items}")
    combined_remove_items = "|".join(remove_items)

    reader = FileSystemReader(input_dir)
    metadata = reader.read_metadata()
    state_dict = {}
    for key, md in metadata.state_dict_metadata.items():
        if re.search(combined_remove_items, key):
            if torch.distributed.get_rank() == 0:
                click.echo(
                    click.style(f"Removing key '{key}' from state_dict.", fg="yellow")
                )
            if hasattr(metadata, "mcore_data") and key in metadata.mcore_data:
                del metadata.mcore_data[key]
            continue

        for old_key_pattern, new_key_pattern in rename_items:
            if re.search(old_key_pattern, key):
                new_key = re.sub(old_key_pattern, new_key_pattern, key)
                if torch.distributed.get_rank() == 0:
                    click.echo(
                        click.style(
                            f"Renaming key '{key}' to '{new_key}' in state_dict.", fg="green"
                        )
                    )
                if hasattr(metadata, "mcore_data") and key in metadata.mcore_data:
                    metadata.mcore_data[new_key] = metadata.mcore_data[key]
                    del metadata.mcore_data[key]
                key = new_key
                break

        if isinstance(md, TensorStorageMetadata):
            state_dict[key] = torch.distributed.tensor.empty(
                md.size,
                dtype=md.properties.dtype,
                device_mesh=DeviceMesh.from_group(
                    group=process_group,
                    device_type="cuda",
                ),
                placements=[Shard(0)],
            )
        elif isinstance(md, BytesStorageMetadata):
            state_dict[key] = io.BytesIO()
        else:
            raise NotImplementedError(f"Unsupported metadata type: {type(md)}")

    # Save the modified state dict
    click.echo(
        click.style(
            f"Saving modified state_dict to {output_dir}.", fg="green", bold=True
        )
    )
    save_checkpoint_with_pickle_protocol(
        state_dict,
        output_dir,
        pickle_protocol=4,  # Use protocol 4 for OOM issue
    )

    # Copy metadata.json, common.pt
    shutil.copy2(Path(input_dir) / "metadata.json", output_dir)
    shutil.copy2(Path(input_dir) / "common.pt", output_dir)


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option("--op", multiple=True, type=str)
@click.option("--enable-msc", is_flag=True, help="Enable MultiStorageClient feature.")
def modify_state_dict(input_dir, output_dir, op, enable_msc):
    """Modify state dict items in a Megatron Core Distributed Checkpoint."""
    # Initialize distributed process group
    init_process_group(f"modify_state_dict from {input_dir} to {output_dir}")

    if not enable_msc:
        MultiStorageClientFeature.disable()

    _modify_state_dict(
        Path(input_dir),
        Path(output_dir),
        op,
        process_group=dist.group.WORLD,
        enable_msc=enable_msc,
    )

    click.echo(
        click.style(
            f"State dict items modified and saved to {output_dir}.", fg="green", bold=True
        )
    )


def _compare_two_checkpoint(checkpoint_1, checkpoint_2):
    reader_1 = FileSystemReader(checkpoint_1)
    metadata_1 = reader_1.read_metadata()

    reader_2 = FileSystemReader(checkpoint_2)
    metadata_2 = reader_2.read_metadata()

    keys_1 = set(metadata_1.state_dict_metadata.keys())
    keys_2 = set(metadata_2.state_dict_metadata.keys())

    click.echo(click.style("Comparing checkpoints...", fg="blue"))

    # Compare keys
    missing_in_1 = keys_2 - keys_1
    missing_in_2 = keys_1 - keys_2
    common_keys = keys_1 & keys_2

    click.echo(click.style("Keys missing in checkpoint 1:", fg="red"))
    for key in missing_in_1:
        click.echo(click.style(f" - {key}", fg="red"))

    click.echo(click.style("Keys missing in checkpoint 2:", fg="red"))
    for key in missing_in_2:
        click.echo(click.style(f" - {key}", fg="red"))

    # Compare common keys
    click.echo(click.style("Common keys in both checkpoints:", fg="green"))
    for key in common_keys:
        meta_1 = metadata_1.state_dict_metadata[key]
        meta_2 = metadata_2.state_dict_metadata[key]

        if not isinstance(meta_1, TensorStorageMetadata):
            continue

        if meta_1.size != meta_2.size or meta_1.properties.dtype != meta_2.properties.dtype:
            click.echo(click.style(f" - {key} (metadata differ) meta_1: {meta_1}, meta_2: {meta_2}", fg="red"))
        else:
            value_1 = torch.empty(meta_1.size, dtype=meta_1.properties.dtype)
            value_2 = value_1.clone()

            dcp.load({key: value_1}, storage_reader=reader_1, planner=DefaultLoadPlanner())
            dcp.load({key: value_2}, storage_reader=reader_2, planner=DefaultLoadPlanner())

            if not torch.allclose(
                value_1, value_2, atol=1e-8, rtol=1e-5
            ):
                click.echo(click.style(f" - {key} (values differ) value_1: {value_1}, value_2: {value_2}", fg="red"))


@cli.command()
@click.argument("checkpoint_1", type=click.Path(exists=True))
@click.argument("checkpoint_2", type=click.Path(exists=True))
@click.option("--enable-msc", is_flag=True, help="Enable MultiStorageClient feature.")
def compare_two_checkpoint(checkpoint_1, checkpoint_2, enable_msc):
    """
    Compare two checkpoints.
    """
    init_process_group(f"compare_two_checkpoint from {checkpoint_1} to {checkpoint_2}")

    if not enable_msc:
        MultiStorageClientFeature.disable()

    _compare_two_checkpoint(
        Path(checkpoint_1),
        Path(checkpoint_2),
    )

    click.echo(
        click.style(
            f"Comparison between {checkpoint_1} and {checkpoint_2} completed.", fg="green", bold=True
        )
    )


@cli.command()
@click.argument("torch_dcp_dir", type=click.Path(exists=True))
def print_torch_dcp_in_json(torch_dcp_dir, model_weight_prefix="model.module"):
    # Use a temporary file context
    with tempfile.NamedTemporaryFile(suffix=".pth") as tmp_file:
        # Convert distributed checkpoint directory to a single-file checkpoint
        dcp_to_torch_save(torch_dcp_dir, tmp_file.name)

        # Load the state dict from the temporary file
        state_dict = torch.load(tmp_file.name, map_location="cpu")

        click.echo(f"torch dcp content: {json.dumps(state_dict)}")

        # Replace all "module.module." with model_weight_prefix in dict keys
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.module", model_weight_prefix)
            new_state_dict[new_key] = value
        
        # Convert state dict to JSON-serializable format
        serializable_dict = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in new_state_dict.items()}

        # Save to a JSON file
        json_file_path = os.path.join(torch_dcp_dir, "param_to_param_group_map.json")
        with open(json_file_path, "w") as json_file:
            json.dump(serializable_dict, json_file, indent=2)
        click.echo(f"Saved converted param_to_param_group_map to: {json_file_path}")


def init_process_group(message):
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    time.sleep(rank * 0.01)  # Ensure all ranks are synchronized before loading
    click.echo(f"[{rank}/{world_size}] [cuda:{local_rank}] {message}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )


if __name__ == "__main__":
    cli()
