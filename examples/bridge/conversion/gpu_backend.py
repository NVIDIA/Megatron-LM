# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/08035723e7a86f94fc54a7f4473bd0519f2ac941/scripts/conversion/gpu_backend.py
# Source license: Apache-2.0.
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Distributed GPU checkpoint conversion backend."""

import datetime
import os
from collections.abc import Iterable
from pathlib import Path

import torch
import yaml
from rich.console import Console
from utils import (
    _configure_distributed_env,
    _configure_model_config,
    _configure_model_provider,
    _hf_tokenizer_kwargs,
    _maybe_generate_pipeline_layout,
    _uses_model_builder,
    parse_dtype,
    prepare_output_directory,
    resolve_hf_model_revision,
    validate_output_path,
)

from megatron.bridge import AutoBridge
from megatron.bridge.models.decorators import torchrun_main
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.utils.common_utils import print_rank_0

_IGNORE_PRECISION_PARAMS = (
    "e_score_correction_bias",
    "A_log",
    "linear_attn.norm.weight",
    "dt_bias",
    "expert_bias",
    "q_norm.weight",
    "k_norm.weight",
    "block_sparse_moe.gate.weight",
    "mlp.gate.weight",
    "moe.gate.weight",
)
_FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}
_ROUNDTRIP_ATOL = 1e-1
_ROUNDTRIP_RTOL = 1e-5
_CONSOLE = Console()


def _ensure_distributed_initialized(timeout_minutes: int | None, *, use_cpu: bool = False) -> None:
    """Initialize a distributed process group from torchrun or Slurm task state."""
    if torch.distributed.is_initialized():
        return
    _configure_distributed_env()
    if os.environ.get("WORLD_SIZE") is None:
        raise RuntimeError(
            "Distributed conversion must be launched through NeMo Run's local or Slurm executor."
        )
    if not use_cpu:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
    kwargs: dict[str, object] = {"backend": "gloo" if use_cpu else "nccl"}
    if timeout_minutes is not None:
        kwargs["timeout"] = datetime.timedelta(minutes=timeout_minutes)
    torch.distributed.init_process_group(**kwargs)


def _prepare_distributed_output(
    path: str, *, overwrite: bool, source_paths: Iterable[str] = ()
) -> None:
    """Prepare an output directory once and synchronize all ranks."""
    source_paths = tuple(source_paths)
    validate_output_path(path, source_paths=source_paths)
    if torch.distributed.get_rank() == 0:
        prepare_output_directory(path, overwrite=overwrite, source_paths=source_paths)
    # Without device_ids, torch guesses the barrier's device as
    # rank % local_gpu_count. Slurm assigns SLURM_PROCID cyclically across
    # nodes (RANK = num_nodes * local_rank + node_id, not block-wise), so
    # that guess collides across ranks on the same node once more than one
    # node is used, and NCCL aborts with "Multiple ranks detected using the
    # same GPU on this node." Passing the real device (already set via
    # torch.cuda.set_device in _ensure_distributed_initialized) avoids the
    # guess, matching the pattern used by training/initialize.py's own
    # first post-init barrier.
    if torch.distributed.get_backend() == "gloo":
        torch.distributed.barrier()
    else:
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])


def _rebalance_pipeline_layout(saved_layout: list[list[str]], pp: int) -> list[list[str]]:
    """Redistribute a saved flexible pipeline layout across a new PP size."""
    layers = [layer for stage in saved_layout for layer in stage]
    base, remainder = divmod(len(layers), pp)
    rebalanced_layout = []
    offset = 0
    for stage_index in range(pp):
        stage_size = base + (1 if stage_index < remainder else 0)
        rebalanced_layout.append(layers[offset : offset + stage_size])
        offset += stage_size
    return rebalanced_layout


def _maybe_restore_pipeline_layout(
    bridge: AutoBridge, model_provider: GPTModelProvider, megatron_path: str, pp: int
) -> None:
    """Restore a serialized pipeline layout or regenerate it for export.

    The provider adopts the checkpoint's MTP layer count because export builds the model from the
    checkpoint config, so a restored or regenerated layout must place the MTP layers that model has.
    """
    checkpoint_path = Path(megatron_path)
    iteration_paths = [path for path in checkpoint_path.glob("iter_*") if path.is_dir()]

    def _iteration_number(path: Path) -> int:
        try:
            return int(path.name.removeprefix("iter_"))
        except ValueError:
            return -1

    config_root = (
        max(iteration_paths, key=_iteration_number) if iteration_paths else checkpoint_path
    )
    config_path = config_root / "run_config.yaml"
    root_config_path = checkpoint_path / "run_config.yaml"
    if not config_path.exists() and root_config_path.exists():
        config_path = root_config_path
    if config_path.exists():
        with config_path.open() as config_file:
            config = yaml.safe_load(config_file) or {}
        model_config = config.get("model", {})
        # Builder-backed model configs serialize the Megatron-Core fields under ``transformer``.
        model_config = {**(model_config.get("transformer") or {}), **model_config}
        if "mtp_num_layers" in model_config:
            model_provider.mtp_num_layers = model_config["mtp_num_layers"]
        saved_layout = model_config.get("pipeline_model_parallel_layout")
        saved_pp = model_config.get("pipeline_model_parallel_size")
        is_valid_layout = isinstance(saved_layout, list) and all(
            isinstance(stage, list) for stage in saved_layout
        )
        is_same_topology = saved_pp == pp or (
            saved_pp is None and len(saved_layout or []) % pp == 0
        )
        if is_valid_layout and is_same_topology and len(saved_layout) % pp == 0:
            model_provider.pipeline_model_parallel_layout = saved_layout
            # Hybrid models take their stage boundaries from the '|' segments saved with this layout.
            saved_pattern = model_config.get("hybrid_layer_pattern")
            if isinstance(saved_pattern, str) and "|" in saved_pattern:
                model_provider.hybrid_layer_pattern = saved_pattern
            print_rank_0(
                f"Read pipeline layout from checkpoint ({len(saved_layout)} PP/VPP stages)"
            )
            return
        if is_valid_layout:
            print_rank_0(
                f"Rebalancing checkpoint pipeline layout from PP={saved_pp or 'unknown'} to requested PP={pp}"
            )
            if not _maybe_generate_pipeline_layout(bridge, model_provider, pp):
                model_provider.pipeline_model_parallel_layout = _rebalance_pipeline_layout(
                    saved_layout, pp
                )
            return
    _maybe_generate_pipeline_layout(bridge, model_provider, pp)


def _roundtrip_weights_match(
    name: str, exported: torch.Tensor, original: torch.Tensor
) -> tuple[bool, bool]:
    """Compare one exported parameter with its original Hugging Face value.

    Returns:
        A pair containing whether the values match and whether comparison was
        skipped because either value uses a lossy FP8 dtype.
    """
    if original.dtype in _FP8_DTYPES or exported.dtype in _FP8_DTYPES:
        return True, True
    if exported.dtype != original.dtype or any(part in name for part in _IGNORE_PRECISION_PARAMS):
        exported = exported.float()
        original = original.float()
    return (
        torch.allclose(
            exported, original.to(exported.device), atol=_ROUNDTRIP_ATOL, rtol=_ROUNDTRIP_RTOL
        ),
        False,
    )


def _verify_roundtrip_weights(bridge: AutoBridge, megatron_model: list[torch.nn.Module]) -> None:
    """Exhaustively verify exported Megatron weights against the original Hugging Face state.

    Every rank participates in the collective Megatron-to-Hugging-Face export, but only rank 0 lazily reads each
    original Hugging Face tensor and compares it serially. This does not materialize a complete Hugging Face model
    on every rank. For very large checkpoints, the rank-0 work can create prolonged rank skew before the result
    broadcast, exceed the process-group collective timeout, and incur substantial transient tensor memory and
    storage I/O.
    """
    is_rank_0 = torch.distributed.get_rank() == 0
    all_match = True
    verified_count = 0
    mismatch_count = 0
    mismatch_samples: list[str] = []
    fp8_skip_count = 0
    fp8_skip_samples: list[str] = []

    for name, exported in bridge.export_hf_weights(megatron_model, show_progress=False):
        if not is_rank_0:
            continue
        original = bridge.hf_pretrained.state[name]
        match, skipped_fp8 = _roundtrip_weights_match(name, exported, original)
        if skipped_fp8:
            fp8_skip_count += 1
            if len(fp8_skip_samples) < 20:
                fp8_skip_samples.append(
                    f"{name}: exported {exported.dtype} vs original {original.dtype}"
                )
        all_match = all_match and match
        verified_count += 1
        if not match:
            mismatch_count += 1
            if len(mismatch_samples) < 20:
                mismatch_samples.append(name)

    mismatch = torch.tensor(not all_match, dtype=torch.int32, device=torch.cuda.current_device())
    torch.distributed.broadcast(mismatch, src=0)

    if is_rank_0:
        compared_count = verified_count - fp8_skip_count
        matched_count = compared_count - mismatch_count
        color = "green" if mismatch_count == 0 else "red"
        _CONSOLE.print(
            f"[{color}]Weight verification: {matched_count}/{compared_count} compared weights matched[/]"
        )
        if mismatch_count:
            _CONSOLE.print(f"[red]{mismatch_count} weight mismatches (showing up to 20):[/red]")
            for entry in mismatch_samples:
                _CONSOLE.print(f"  [red]{entry}[/red]")
        if fp8_skip_count:
            _CONSOLE.print(
                f"[yellow]WARNING: {fp8_skip_count} FP8 params skipped allclose (dequantisation is lossy):[/yellow]"
            )
            for entry in fp8_skip_samples:
                _CONSOLE.print(f"  [yellow]{entry}[/yellow]")
            if fp8_skip_count > len(fp8_skip_samples):
                _CONSOLE.print(
                    f"  [yellow]... and {fp8_skip_count - len(fp8_skip_samples)} more[/yellow]"
                )
    if mismatch.item():
        raise ValueError("Weight mismatch detected")


@torchrun_main
def import_checkpoint(
    *,
    hf_model: str,
    hf_revision: str | None,
    megatron_path: str,
    tp: int,
    pp: int,
    ep: int,
    etp: int,
    torch_dtype: str,
    trust_remote_code: bool,
    low_memory_save: bool,
    distributed_timeout_minutes: int | None,
    overwrite: bool,
    text_only: bool = False,
) -> None:
    """Import a Hugging Face model into a distributed Megatron checkpoint.

    Args:
        hf_model: Hugging Face model ID or local path.
        hf_revision: Hugging Face Hub revision to load.
        megatron_path: Destination Megatron checkpoint path.
        tp: Tensor parallelism size.
        pp: Pipeline parallelism size.
        ep: Expert parallelism size.
        etp: Expert tensor parallelism size.
        torch_dtype: Weight dtype name.
        trust_remote_code: Allow custom Hugging Face repository code.
        low_memory_save: Reduce peak GPU memory while saving the imported checkpoint.
        distributed_timeout_minutes: Process-group timeout in minutes.
        overwrite: Delete a non-empty destination before conversion.
        text_only: Convert only the supported model's language component.
    """
    _ensure_distributed_initialized(distributed_timeout_minutes)
    _prepare_distributed_output(megatron_path, overwrite=overwrite, source_paths=[hf_model])
    dtype = parse_dtype(torch_dtype)

    print_rank_0(f"GPU import: {hf_model} -> {megatron_path}")
    print_rank_0(f"Parallelism: TP={tp} PP={pp} EP={ep} ETP={etp}; dtype={torch_dtype}")
    revision_kwargs = {"revision": hf_revision} if hf_revision is not None else {}
    if text_only:
        revision_kwargs["text_only"] = True
    bridge = AutoBridge.from_hf_pretrained(
        hf_model,
        trust_remote_code=is_safe_repo(trust_remote_code=trust_remote_code, hf_path=hf_model),
        torch_dtype=dtype,
        **revision_kwargs,
    )
    if _uses_model_builder(bridge):
        model_config = bridge.get_model_config()
        _configure_model_config(model_config, tp=tp, pp=pp, ep=ep, etp=etp, dtype=dtype)
        _maybe_generate_pipeline_layout(bridge, model_config, pp)
        megatron_model = bridge.get_model(
            model_config, wrap_with_ddp=False, mixed_precision_wrapper=None
        )
    else:
        model_provider = bridge.to_megatron_provider(load_weights=True)
        _configure_model_provider(model_provider, tp=tp, pp=pp, ep=ep, etp=etp, dtype=dtype)
        _maybe_generate_pipeline_layout(bridge, model_provider, pp)
        model_provider.finalize()
        model_provider.initialize_model_parallel(seed=0, create_gloo_process_groups=False)
        megatron_model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    bridge.save_megatron_model(
        megatron_model,
        megatron_path,
        hf_tokenizer_path=hf_model,
        hf_tokenizer_kwargs=_hf_tokenizer_kwargs(bridge, trust_remote_code=trust_remote_code),
        low_memory_save=low_memory_save,
    )
    print_rank_0(f"GPU import complete: {megatron_path}")


@torchrun_main
def export_checkpoint(
    *,
    hf_model: str,
    hf_revision: str | None,
    megatron_path: str,
    hf_path: str,
    tp: int,
    pp: int,
    ep: int,
    etp: int,
    torch_dtype: str,
    trust_remote_code: bool,
    strict: bool,
    show_progress: bool,
    distributed_save: bool,
    save_every_n_ranks: int,
    text_only: bool = False,
    distributed_timeout_minutes: int | None,
    export_weight_dtype: str | None,
    overwrite: bool,
    use_cpu: bool = False,
) -> None:
    """Export a distributed Megatron checkpoint to Hugging Face format.

    Args:
        hf_model: Hugging Face model ID or local config reference.
        hf_revision: Immutable Hugging Face Hub revision to load.
        megatron_path: Source Megatron checkpoint path.
        hf_path: Destination Hugging Face checkpoint path.
        tp: Tensor parallelism size.
        pp: Pipeline parallelism size.
        ep: Expert parallelism size.
        etp: Expert tensor parallelism size.
        torch_dtype: Model dtype name.
        trust_remote_code: Allow custom Hugging Face repository code.
        strict: Require source and destination parameter keys to match.
        show_progress: Display export progress.
        distributed_save: Let ranks save assigned Hugging Face shards independently.
        save_every_n_ranks: Write files from every Nth rank.
        distributed_timeout_minutes: Process-group timeout in minutes.
        export_weight_dtype: Optional dtype for exported weights.
        overwrite: Delete a non-empty destination before conversion.
        use_cpu: Use Gloo and CPU model initialization instead of NCCL/CUDA.
    """
    if use_cpu:
        _ensure_distributed_initialized(distributed_timeout_minutes, use_cpu=True)
    else:
        _ensure_distributed_initialized(distributed_timeout_minutes)
    if not Path(megatron_path).exists():
        raise FileNotFoundError(f"Megatron checkpoint does not exist: {megatron_path}")
    _prepare_distributed_output(
        hf_path, overwrite=overwrite, source_paths=[megatron_path, hf_model]
    )
    dtype = parse_dtype(torch_dtype)

    device_label = "CPU" if use_cpu else "GPU"
    print_rank_0(f"Distributed {device_label} export: {megatron_path} -> {hf_path}")
    print_rank_0(f"Parallelism: TP={tp} PP={pp} EP={ep} ETP={etp}; dtype={torch_dtype}")
    trusted = is_safe_repo(trust_remote_code=trust_remote_code, hf_path=hf_model)
    revision_kwargs = {"revision": hf_revision} if hf_revision is not None else {}
    if text_only:
        revision_kwargs["text_only"] = True
    bridge = AutoBridge.from_hf_pretrained(
        hf_model, trust_remote_code=trusted, torch_dtype=dtype, **revision_kwargs
    )
    reference_model = (
        resolve_hf_model_revision(hf_model, hf_revision, config_only=True)
        if text_only
        else resolve_hf_model_revision(hf_model, hf_revision)
    )
    checkpoint_config_bridge = AutoBridge.from_auto_config(
        megatron_path, reference_model, trust_remote_code=trusted
    )
    # Preserve the reference wrapper's streaming state source and shard map while
    # exporting the checkpoint-derived architecture and vocabulary configuration.
    bridge.hf_pretrained.config = checkpoint_config_bridge.hf_pretrained
    if _uses_model_builder(bridge):
        model_config = bridge.get_model_config()
        _configure_model_config(
            model_config, tp=tp, pp=pp, ep=ep, etp=etp, dtype=dtype, use_cpu=use_cpu
        )
        _maybe_restore_pipeline_layout(bridge, model_config, megatron_path, pp)
        resolved_pipeline_layout = model_config.pipeline_model_parallel_layout
        model_config.finalize()
        resolved_hybrid_layer_pattern = getattr(model_config, "hybrid_layer_pattern", None)
        bridge._get_or_initialize_pg_collection(model_config.transformer)
    else:
        model_provider = bridge.to_megatron_provider(load_weights=False)
        _configure_model_provider(
            model_provider, tp=tp, pp=pp, ep=ep, etp=etp, dtype=dtype, use_cpu=use_cpu
        )
        _maybe_restore_pipeline_layout(bridge, model_provider, megatron_path, pp)
        resolved_pipeline_layout = model_provider.pipeline_model_parallel_layout
        model_provider.finalize()
        resolved_hybrid_layer_pattern = getattr(model_provider, "hybrid_layer_pattern", None)
        model_provider.initialize_model_parallel(seed=0, create_gloo_process_groups=False)

    model_parallel_overrides: dict[str, object] = {
        "tensor_model_parallel_size": tp,
        "pipeline_model_parallel_size": pp,
        "expert_model_parallel_size": ep,
        "expert_tensor_parallel_size": etp,
        "pipeline_dtype": dtype,
        "params_dtype": dtype,
    }
    if isinstance(resolved_pipeline_layout, list):
        model_parallel_overrides["pipeline_model_parallel_layout"] = resolved_pipeline_layout
        # load_megatron_model drops the '|' stage boundaries from the checkpoint's hybrid layer pattern,
        # so pass the segments that match this layout; HybridModel allocates pipeline stages from them.
        if isinstance(resolved_hybrid_layer_pattern, str) and "|" in resolved_hybrid_layer_pattern:
            model_parallel_overrides["hybrid_layer_pattern"] = resolved_hybrid_layer_pattern

    megatron_model = bridge.load_megatron_model(
        megatron_path, mp_overrides=model_parallel_overrides, wrap_with_ddp=False
    )
    bridge.save_hf_pretrained(
        megatron_model,
        hf_path,
        show_progress=show_progress,
        strict=strict,
        distributed_save=distributed_save,
        save_every_n_ranks=save_every_n_ranks,
        weight_dtype=parse_dtype(export_weight_dtype) if export_weight_dtype else None,
    )
    print_rank_0(f"Distributed {device_label} export complete: {hf_path}")


@torchrun_main
def roundtrip_checkpoint(
    *,
    hf_model: str,
    tp: int,
    pp: int,
    ep: int,
    etp: int,
    trust_remote_code: bool,
    distributed_timeout_minutes: int | None,
) -> None:
    """Validate a Hugging Face to Megatron to Hugging Face round trip.

    This workflow performs exhaustive equality validation and is not intended as the scalable conversion path for
    very large checkpoints. When the goal is conversion rather than exhaustive validation, prefer separate
    ``convert.sh import`` and ``convert.sh export --distributed-save`` workflows. If a full round trip is required,
    provision sufficient time and memory and choose an appropriate ``--distributed-timeout-minutes`` value. A
    longer timeout can accommodate expected rank skew, but does not reduce serial verification cost or memory and
    I/O pressure.

    Args:
        hf_model: Hugging Face model ID or local path.
        tp: Tensor parallelism size.
        pp: Pipeline parallelism size.
        ep: Expert parallelism size.
        etp: Expert tensor parallelism size.
        trust_remote_code: Allow custom Hugging Face repository code.
        distributed_timeout_minutes: Process-group timeout in minutes.
    """
    _ensure_distributed_initialized(distributed_timeout_minutes)
    dtype = torch.bfloat16
    print_rank_0(f"GPU round trip: {hf_model}")
    print_rank_0(f"Parallelism: TP={tp} PP={pp} EP={ep} ETP={etp}; dtype=bfloat16")
    bridge = AutoBridge.from_hf_pretrained(
        hf_model,
        trust_remote_code=is_safe_repo(trust_remote_code=trust_remote_code, hf_path=hf_model),
        torch_dtype=dtype,
    )

    if _uses_model_builder(bridge):
        model_config = bridge.get_model_config()
        _configure_model_config(model_config, tp=tp, pp=pp, ep=ep, etp=etp, dtype=dtype)
        _maybe_generate_pipeline_layout(bridge, model_config, pp)
        megatron_model = bridge.get_model(
            model_config, wrap_with_ddp=False, mixed_precision_wrapper=None
        )
    else:
        model_provider = bridge.to_megatron_provider(load_weights=True)
        _configure_model_provider(model_provider, tp=tp, pp=pp, ep=ep, etp=etp, dtype=dtype)
        _maybe_generate_pipeline_layout(bridge, model_provider, pp)
        model_provider.finalize()
        model_provider.initialize_model_parallel(seed=0, create_gloo_process_groups=False)
        megatron_model = model_provider.provide_distributed_model(wrap_with_ddp=False)
    _verify_roundtrip_weights(bridge, megatron_model)
    print_rank_0("GPU round-trip validation complete")
