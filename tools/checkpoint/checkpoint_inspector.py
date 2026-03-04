# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# python checkpoint_inspector.py inspect /path/to/checkpoint
# torchrun --nproc_per_node=8 --nnodes=1 checkpoint_inspector.py convert-torch-dist-to-fsdp-dtensor /path/to/input_checkpoint /path/to/output_checkpoint --swiglu
import gc
import io
import json
import os
from pathlib import Path
import time
import re
import shutil
from typing import Optional
import tempfile

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
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.state_dict_saver import _save_state_dict
from torch.distributed.tensor import DeviceMesh, Replicate, Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import split_dtensor, gather_uneven_dtensor_to_full_tensor

from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistLoadShardedStrategy,
)
from megatron.core.dist_checkpointing.validation import (
    verify_checkpoint_and_load_strategy,
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
        sharded_strategy = get_default_load_sharded_strategy(checkpoint_dir)
        sharded_strategy = FullyParallelLoadStrategyWrapper(sharded_strategy)
        sharded_strategy, common_strategy = verify_checkpoint_and_load_strategy(
            checkpoint_dir, sharded_strategy, common_strategy=None
        )
        assert isinstance(
            sharded_strategy.base_strategy, TorchDistLoadShardedStrategy
        ), click.style(
            f"Unsupported sharded strategy: {sharded_strategy}", fg="red", bold=True
        )

        # Common state section
        common_state = common_strategy.load_common(checkpoint_dir)
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
):
    """Convert a Megatron Core Distributed Checkpoint from torch_dist to standard fsdp_dtensor format."""
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

    # Handle optimizer state and module parameters
    fsdp_dtensor_state_dict = {}
    rerun_state_machine_state = None
    process_count = 0
    total_items = len(state_dict)

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
            v = gather_uneven_dtensor_to_full_tensor(v).reshape(
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

            expert_weight = gather_uneven_dtensor_to_full_tensor(expert_weight)
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

    def is_swiglu_key(key):
        return any(re.search(pat, key) for pat in [
            r"(.*)\.mlp\.linear_fc1\.weight",
            r"(.*)\.mlp\.linear_fc1\.bias",
            r"(.*)\.mlp\.experts\.linear_fc1\.weight(\d+)",
            r"(.*)\.mlp\.experts\.linear_fc1\.bias(\d+)",
            r"(.*)\.mlp\.experts\.local_experts\.(\d+)\.linear_fc1\.weight",
            r"(.*)\.mlp\.experts\.local_experts\.(\d+)\.linear_fc1\.bias",
            r"(.*)\.mlp\.shared_experts\.linear_fc1\.weight",
            r"(.*)\.mlp\.shared_experts\.linear_fc1\.bias",
        ])

    def split_swiglu_weight(key: str, value: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Split SwiGLU weights into separate tensors.
        """
        value = gather_uneven_dtensor_to_full_tensor(value)
        swiglu_w_and_v = {}
        w, v = torch.chunk(value, 2, dim=0)
        w = w.redistribute(placements=[Shard(0)])
        v = v.redistribute(placements=[Shard(0)])
        w_key = re.sub(r'(weight\d*)(.*)', r'\1_w\2', key)
        v_key = re.sub(r'(weight\d*)(.*)', r'\1_v\2', key)
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
            if (
                key in metadata.mcore_data
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
                    value = gather_uneven_dtensor_to_full_tensor(value)
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

            # Handle SWiGLU weights
            for key, value in list(split_tensors.items()):
                if swiglu and is_swiglu_key(key):
                    swiglu_w_and_v = split_swiglu_weight(key, value)
                    split_tensors.update(swiglu_w_and_v)
                    del split_tensors[key]

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

    # Move back to GPU if necessary
    for key in fsdp_dtensor_state_dict:
        if isinstance(fsdp_dtensor_state_dict[key], torch.Tensor):
            fsdp_dtensor_state_dict[key] = fsdp_dtensor_state_dict[key].cuda()

    # Check MCore data
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
    sharded_strategy = get_default_load_sharded_strategy(input_dir)
    sharded_strategy = FullyParallelLoadStrategyWrapper(sharded_strategy)
    sharded_strategy, common_strategy = verify_checkpoint_and_load_strategy(
        input_dir, sharded_strategy, common_strategy=None
    )
    assert isinstance(sharded_strategy.base_strategy, TorchDistLoadShardedStrategy), (
        click.style(
            f"Unsupported sharded strategy: {sharded_strategy}", fg="red", bold=True
        )
    )
    common_state = common_strategy.load_common(input_dir)
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
    help="SwiGLU is used in checkpoint, and MLP linear_fc1 is specially treated.",
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
    default="{}",
    help="JSON string representing the param to parameter group map."
)
def convert_torch_dist_to_fsdp_dtensor(
    input_dir,
    output_dir,
    swiglu,
    oom_traceback,
    enable_msc,
    output_optimizer_state_prefix,
    output_model_weight_prefix,
    param_to_param_group_map_json,
):
    """Convert a Megatron Core Distributed Checkpoint from torch_dist to fsdp_dtensor format."""
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
    with open(param_to_param_group_map_json, "r") as f:
        param_to_param_group_map = json.load(f)
    convert_checkpoint(
        ckpt_path, output_dir, swiglu, process_group=dist.group.WORLD,
        optimizer_state_prefix=output_optimizer_state_prefix,
        model_weight_prefix=output_model_weight_prefix,
        param_to_param_group_map=param_to_param_group_map,
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
