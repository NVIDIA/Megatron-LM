# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Profile GDN saved-tensor lifetimes or compare activation offload with a baseline.

Run with torchrun, one process per GPU. Profiling and timed runs are separate:
saved-tensor hooks record host-side save/unpack order, not CUDA kernel timings.
"""

import argparse
import gc
import importlib.metadata
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_gated_delta_net_module_spec,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import PipelineOffloadManager
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig


def build_model(
    args: argparse.Namespace, offload: bool = False, fraction: float = 1.0
) -> torch.nn.ModuleList:
    """Build a stack of real GDN layers with FLA recurrence and TE projections."""
    config = TransformerConfig(
        num_layers=args.layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.key_heads,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=args.head_dim,
        linear_value_head_dim=args.head_dim,
        linear_num_key_heads=args.key_heads,
        linear_num_value_heads=args.value_heads,
        normalization="RMSNorm",
        activation_func=torch.nn.functional.silu,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        experimental_attention_variant="gdn",
        linear_attention_freq=[1] * args.layers,
        fine_grained_activation_offloading=offload,
        offload_modules=["gdn_core_attn"] if offload else [],
        min_offloaded_tensor_size=args.min_offloaded_tensor_size,
        activation_offload_fraction=fraction,
    )
    groups = ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(),
        cp=parallel_state.get_context_parallel_group(),
    )
    spec = get_gated_delta_net_module_spec(config)
    torch.manual_seed(args.seed)
    model_parallel_cuda_manual_seed(args.seed)
    return (
        torch.nn.ModuleList(
            [
                spec.module(config, spec.submodules, layer_number=i + 1, pg_collection=groups)
                for i in range(args.layers)
            ]
        )
        .cuda()
        .bfloat16()
    )


def forward(model: torch.nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
    """Run the GDN stack, preserving the residual between layers."""
    for layer in model:
        y, _ = layer(x, None)
        x = x + y
    return x


def profile(model: torch.nn.ModuleList, x: torch.Tensor) -> dict[str, Any]:
    """Record saves, shared storage, and first backward use without retaining extra tensors."""
    rows: list[dict[str, Any]] = []
    phase = "outside_recurrence"
    input_names: dict[int, str] = {}
    parameter_storages = {p.untyped_storage().data_ptr() for p in model.parameters()}
    original_kernels = [layer.gated_delta_rule for layer in model]

    def wrap(kernel: Callable[..., Any], layer_number: int) -> Callable[..., Any]:
        def call(**kwargs: Any) -> Any:
            nonlocal phase, input_names
            phase = f"gdn_core_attn.{layer_number}"
            input_names = {
                t.data_ptr(): name for name, t in kwargs.items() if isinstance(t, torch.Tensor)
            }
            try:
                return kernel(**kwargs)
            finally:
                phase = "outside_recurrence"
                input_names = {}

        return call

    def pack(tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
        index = len(rows)
        storage = tensor.untyped_storage()
        rows.append(
            {
                "index": index,
                "phase": phase,
                "name": input_names.get(tensor.data_ptr(), "internal"),
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype),
                "tensor_bytes": tensor.numel() * tensor.element_size(),
                "storage": storage.data_ptr(),
                "storage_bytes": storage.nbytes(),
                "parameter_alias": storage.data_ptr() in parameter_storages,
                "save_host_seconds": time.perf_counter(),
                "first_unpack_host_seconds": None,
                "last_unpack_host_seconds": None,
                "unpack_count": 0,
            }
        )
        return tensor.detach(), index

    def unpack(saved: tuple[torch.Tensor, int]) -> torch.Tensor:
        tensor, index = saved
        unpack_time = time.perf_counter()
        if rows[index]["first_unpack_host_seconds"] is None:
            rows[index]["first_unpack_host_seconds"] = unpack_time
        rows[index]["last_unpack_host_seconds"] = unpack_time
        rows[index]["unpack_count"] += 1
        return tensor

    try:
        for i, layer in enumerate(model):
            layer.gated_delta_rule = wrap(original_kernels[i], i + 1)
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            y = forward(model, x)
            torch.cuda.synchronize()
            forward_done = time.perf_counter()
            y.float().square().mean().backward()
            torch.cuda.synchronize()
    finally:
        for layer, kernel in zip(model, original_kernels):
            layer.gated_delta_rule = kernel

    storages: dict[int, dict[str, Any]] = {}
    for row in rows:
        storage = storages.setdefault(
            row["storage"], {"bytes": row["storage_bytes"], "save_indices": [], "phases": []}
        )
        storage["save_indices"].append(row["index"])
        if row["phase"] not in storage["phases"]:
            storage["phases"].append(row["phase"])
    recurrence_only_bytes = sum(
        s["bytes"]
        for s in storages.values()
        if all(p.startswith("gdn_core_attn.") for p in s["phases"])
    )
    return {
        "forward_done_host_seconds": forward_done,
        "saved_tensor_bytes_with_aliases": sum(r["tensor_bytes"] for r in rows),
        "unique_saved_storage_bytes": sum(s["bytes"] for s in storages.values()),
        "recurrence_only_storage_bytes": recurrence_only_bytes,
        "rows": rows,
        "storages": storages,
    }


def snapshot_step(
    model: torch.nn.ModuleList, x: torch.Tensor, y: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Copy outputs and gradients after timing, without retaining the autograd graph."""
    if x.grad is None:
        raise RuntimeError("Input gradient is missing.")
    return {
        "output": y.detach().cpu(),
        "input_grad": x.grad.detach().cpu(),
        **{
            f"grad.{name}": p.grad.detach().cpu()
            for name, p in model.named_parameters()
            if p.grad is not None
        },
    }


def benchmark_arm(
    args: argparse.Namespace,
    inputs_cpu: list[torch.Tensor],
    offload: bool,
    fraction: float,
    reference: list[dict[str, torch.Tensor]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, torch.Tensor]]]:
    """Measure complete steps and compare every replay outside timing."""
    off_interface.reset_instance()
    gc.collect()
    torch.cuda.empty_cache()
    model = build_model(args, offload, fraction)
    times = []
    peak_allocated = []
    end_forward_allocated = []
    peak_reserved = []
    snapshots = []
    max_pool_in_use = 0
    for step, input_cpu in enumerate(inputs_cpu):
        model.zero_grad(set_to_none=True)
        x = input_cpu.cuda().requires_grad_()
        if offload:
            off_interface.reset(process_group=torch.distributed.group.WORLD)
            off_interface.init_chunk_handler(
                0, None, None, args.min_offloaded_tensor_size, 0, fraction
            )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        y = forward(model, x)
        # The default does not join transfer streams before backward. The opt-in
        # phase synchronization supports diagnostics, but changes runtime overlap.
        if args.sync_forward:
            torch.cuda.synchronize()
        forward_bytes = torch.cuda.memory_allocated()
        y.float().square().mean().backward()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        if step >= args.warmup:
            times.append(elapsed)
            peak_allocated.append(torch.cuda.max_memory_allocated())
            peak_reserved.append(torch.cuda.max_memory_reserved())
            end_forward_allocated.append(forward_bytes)
        # Every warmup/measured step uses distinct inputs and is checked outside
        # timing. Identical repeated inputs could conceal stale cached activations.
        snapshot = snapshot_step(model, x, y)
        if reference is None:
            snapshots.append(snapshot)
        else:
            expected = reference[step]
            if snapshot.keys() != expected.keys():
                raise RuntimeError(f"Gradient coverage differs at step {step}.")
            mismatches = [
                name for name in expected if not torch.equal(expected[name], snapshot[name])
            ]
            if mismatches:
                raise RuntimeError(
                    f"Offload changed outputs/gradients at step {step}: {mismatches}"
                )
        if offload:
            pool = PipelineOffloadManager.get_instance().cpu_tensor_pool
            in_use = pool.get_pool_status()["global_stats"]["current_in_use"]
            max_pool_in_use = max(max_pool_in_use, in_use)
            if in_use:
                raise RuntimeError(f"Pinned buffers remain in use after step {step}: {in_use}")
        del snapshot
        del x, y
    manager = PipelineOffloadManager.get_instance() if offload else None
    result = {
        "offload": offload,
        "fraction": fraction,
        "step_ms": times,
        "median_step_ms": statistics.median(times),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "end_forward_allocated_bytes": end_forward_allocated,
        "selected_offload_bytes": manager.offload_summary_bytes if manager else {},
        "steady_offloaded_groups": (
            sum(
                g.offload and g.total_offload_bytes > 0
                for c in manager._cached_chunks_forward
                for g in c.offload_groups
            )
            if manager
            else 0
        ),
    }
    if reference is not None:
        result.update(
            bitwise_equal=True,
            checked_steps=len(inputs_cpu),
            mismatches=[],
            max_pinned_buffers_in_use_after_backward=max_pool_in_use,
        )
    del model
    off_interface.reset_instance()
    return result, snapshots


def main() -> None:
    """Write machine-readable profile or comparison evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["profile", "benchmark"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--key-heads", type=int, default=16)
    parser.add_argument("--value-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--min-offloaded-tensor-size", type=int, default=1048576)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--sync-forward",
        action="store_true",
        help="Synchronize after forward in both arms for phase diagnostics (changes overlap).",
    )
    args = parser.parse_args()
    if args.mode == "benchmark" and (args.warmup < 2 or args.iterations < 1 or args.layers < 2):
        parser.error("Use at least two layers, two warmup steps, and one measured step.")
    if (
        min(
            args.layers,
            args.seq_length,
            args.batch_size,
            args.hidden_size,
            args.key_heads,
            args.value_heads,
            args.head_dim,
        )
        < 1
    ):
        parser.error("Model dimensions must be positive.")
    if args.min_offloaded_tensor_size < 0 or any(not 0 <= f <= 1 for f in args.fractions):
        parser.error("Use a non-negative tensor threshold and fractions in [0, 1].")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.distributed.init_process_group("nccl")
    parallel_state.initialize_model_parallel()
    try:
        generator = torch.Generator().manual_seed(args.seed)
        inputs_cpu = [
            torch.randn(
                args.seq_length,
                args.batch_size,
                args.hidden_size,
                dtype=torch.bfloat16,
                generator=generator,
            )
            for _ in range(1 if args.mode == "profile" else args.warmup + args.iterations)
        ]
        evidence: dict[str, Any] = {
            "config": {**vars(args), "output": str(args.output)},
            "input_policy": "seeded_distinct_input_per_step",
            "environment": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "fla": importlib.metadata.version("flash-linear-attention"),
                "transformer_engine": importlib.metadata.version("transformer-engine"),
                "torch_compile_disable": os.environ.get("TORCH_COMPILE_DISABLE", "0"),
            },
        }
        if args.mode == "profile":
            model = build_model(args)
            evidence["profile"] = profile(model, inputs_cpu[0].cuda().requires_grad_())
        else:
            baseline, reference = benchmark_arm(args, inputs_cpu, False, 1.0)
            arms = [baseline]
            for fraction in args.fractions:
                arm, _ = benchmark_arm(args, inputs_cpu, True, fraction, reference)
                arms.append(arm)
            evidence["arms"] = arms
        output = args.output
        if torch.distributed.get_world_size() > 1:
            output = output.with_name(f"{output.stem}.rank{torch.distributed.get_rank()}.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2) + "\n")
        print(f"Wrote {output}", flush=True)  # pylint: disable=bad-builtin
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
