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
def inspect(checkpoint_dir, enable_msc):
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

    for key, value in metadata.state_dict_metadata.items():
        bullet = click.style("►", fg="blue")
        key_styled = click.style(key, fg="green")

        if isinstance(value, TensorStorageMetadata):
            dtype = click.style(f"{value.properties.dtype}", fg="cyan")
            shape = click.style(f"{tuple(value.size)}", fg="magenta")
            click.echo(f"  {bullet} {key_styled} [{dtype}, shape={shape}]")
        elif isinstance(value, BytesStorageMetadata):
            click.echo(f"  {bullet} {key_styled} {click.style('[BYTES]', fg='yellow')}")
        else:
            click.echo(
                f"  {bullet} {key_styled} {click.style('[UNKNOWN TYPE]', fg='red')}"
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
    optimizer_state_prefix="optimizer.state.module.module.module",
    model_weight_prefix="model.module",
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
            else:
                # Special handling for module parameters
                new_key = f"{model_weight_prefix}.{key}"

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

            # Handle multi-layer tensors
            if ".layers." in new_key:
                n_layer = value.shape[0]

                _free_up_some_gpu_memory()
                per_layer_values = [
                    gather_uneven_dtensor_to_full_tensor(v).redistribute(
                        placements=[Shard(len(v.shape) - 1)]
                    )
                    for v in split_dtensor(value, 1, dim=0)
                ]
                for i in range(n_layer):
                    if orig_shape is not None:
                        layer_shape = orig_shape[1:]
                    else:
                        layer_shape = value.shape[1:]

                    per_layer_values[i] = (
                        per_layer_values[i]
                        .reshape(layer_shape)
                        .redistribute(placements=[Shard(0)])
                    )
                for i in range(0, n_layer):
                    layer_key = new_key.replace(".layers.", f".layers.{i}.")
                    if swiglu and "mlp.linear_fc1.weight" in layer_key:
                        # Special case for SwiGLU
                        w, v = torch.chunk(per_layer_values[i], 2, dim=0)
                        w = w.redistribute(placements=[Shard(0)])
                        v = v.redistribute(placements=[Shard(0)])
                        w_key = layer_key.replace(
                            "mlp.linear_fc1.weight", "mlp.linear_fc1.weight_w"
                        )
                        v_key = layer_key.replace(
                            "mlp.linear_fc1.weight", "mlp.linear_fc1.weight_v"
                        )
                        # Store both w and v in the state_dict
                        fsdp_dtensor_state_dict[w_key] = w
                        fsdp_dtensor_state_dict[v_key] = v
                    elif (
                        "experts.experts.linear_fc1.weight" in layer_key
                        or "experts.experts.linear_fc2.weight" in layer_key
                    ):
                        # Special case for MoE
                        layer_key = layer_key.replace(".experts.experts.", ".experts.")
                        expert_weights = torch.split(per_layer_values[i], 1, dim=0)
                        for expert_idx, expert_weight in enumerate(expert_weights):
                            expert_key = f"{layer_key}{expert_idx}"
                            fsdp_dtensor_state_dict[expert_key] = expert_weight.squeeze(
                                0
                            )
                    else:
                        # General case
                        fsdp_dtensor_state_dict[layer_key] = per_layer_values[i]
            else:
                if orig_shape is not None:
                    _free_up_some_gpu_memory()
                    value = (
                        value.redistribute(placements=[Replicate()])
                        .reshape(orig_shape)
                        .redistribute(placements=[Shard(0)])
                    )
                fsdp_dtensor_state_dict[new_key] = value
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

    if "checkpoint_version" not in fsdp_dtensor_state_dict:
        fsdp_dtensor_state_dict["checkpoint_version"] = 3.0

    # Save modified checkpoint
    save_checkpoint_with_pickle_protocol(fsdp_dtensor_state_dict, output_dir)


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
    "--distributed-timeout-minutes",
    default=10,
    type=int,
    help="Timeout for distributed operations in minutes.",
)
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
def convert_torch_dist_to_fsdp_dtensor(
    input_dir,
    output_dir,
    swiglu,
    oom_traceback,
    enable_msc,
    distributed_timeout_minutes,
    output_optimizer_state_prefix,
    output_model_weight_prefix,
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
    convert_checkpoint(
        ckpt_path, output_dir, swiglu, process_group=dist.group.WORLD,
        optimizer_state_prefix=output_optimizer_state_prefix,
        model_weight_prefix=output_model_weight_prefix,
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
