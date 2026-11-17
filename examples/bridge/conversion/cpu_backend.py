# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/08035723e7a86f94fc54a7f4473bd0519f2ac941/scripts/conversion/cpu_backend.py
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
"""CPU checkpoint conversion with single-process or distributed import."""

import datetime
import logging
import os
import shutil
from pathlib import Path

import torch
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
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo

logger = logging.getLogger(__name__)


def _ensure_distributed_initialized(timeout_minutes: int | None) -> None:
    """Initialize Gloo from torchrun or Slurm without selecting a CUDA device."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_backend() != "gloo":
            raise RuntimeError("Distributed CPU import requires a Gloo process group.")
        return
    _configure_distributed_env()
    if os.environ.get("WORLD_SIZE") is None:
        raise RuntimeError("Distributed CPU import must be launched through torchrun or Slurm.")
    kwargs: dict[str, object] = {"backend": "gloo"}
    if timeout_minutes is not None:
        kwargs["timeout"] = datetime.timedelta(minutes=timeout_minutes)
    torch.distributed.init_process_group(**kwargs)


def _find_run_config(checkpoint_path: Path) -> Path:
    """Find the run config used to synthesize an exported HF config."""
    config_files = list(checkpoint_path.glob("**/run_config.yaml"))
    if config_files:
        return config_files[0]

    iteration_dirs = [
        path
        for path in checkpoint_path.iterdir()
        if path.is_dir() and path.name.startswith("iter_")
    ]
    if iteration_dirs:
        latest_iteration = max(
            iteration_dirs, key=lambda path: int(path.name.removeprefix("iter_"))
        )
        config_path = latest_iteration / "run_config.yaml"
        if config_path.exists():
            return config_path
    raise FileNotFoundError(
        f"Could not find run_config.yaml in {checkpoint_path}. Ensure this is a valid Megatron checkpoint."
    )


def import_checkpoint(
    *,
    hf_model: str,
    hf_revision: str | None,
    megatron_path: str,
    torch_dtype: str,
    trust_remote_code: bool,
    overwrite: bool,
    text_only: bool = False,
    use_distributed: bool = False,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    etp: int = 1,
    distributed_timeout_minutes: int | None = None,
) -> None:
    """Import a Hugging Face model into a CPU-initialized Megatron checkpoint.

    Args:
        hf_model: Hugging Face model ID or local path.
        hf_revision: Hugging Face Hub revision to load.
        megatron_path: Destination Megatron checkpoint path.
        torch_dtype: Weight dtype name.
        trust_remote_code: Allow custom Hugging Face repository code.
        overwrite: Delete a non-empty destination before conversion.
        text_only: Convert only the supported model's language component.
        use_distributed: Import CPU model shards across an existing launcher world.
        tp: Tensor parallelism size for distributed import.
        pp: Pipeline parallelism size for distributed import.
        ep: Expert parallelism size for distributed import.
        etp: Expert tensor parallelism size for distributed import.
        distributed_timeout_minutes: Optional Gloo process-group timeout.
    """
    if use_distributed:
        # Keep elastic error handling and process-group cleanup specific to the
        # distributed path; single-process callers retain ordinary exceptions.
        @torchrun_main
        def _import_distributed() -> None:
            _ensure_distributed_initialized(distributed_timeout_minutes)
            validate_output_path(megatron_path, source_paths=[hf_model])
            rank = torch.distributed.get_rank()
            if rank == 0:
                prepare_output_directory(
                    megatron_path, overwrite=overwrite, source_paths=[hf_model]
                )
                logger.info("Distributed CPU import: %s -> %s", hf_model, megatron_path)
                logger.info(
                    "Parallelism: TP=%s PP=%s EP=%s ETP=%s; dtype=%s", tp, pp, ep, etp, torch_dtype
                )
            torch.distributed.barrier()
            dtype = parse_dtype(torch_dtype)
            revision_kwargs = {"revision": hf_revision} if hf_revision is not None else {}
            if text_only:
                revision_kwargs["text_only"] = True
            bridge = AutoBridge.from_hf_pretrained(
                hf_model,
                trust_remote_code=is_safe_repo(
                    trust_remote_code=trust_remote_code, hf_path=hf_model
                ),
                torch_dtype=dtype,
                **revision_kwargs,
            )
            if _uses_model_builder(bridge):
                model_config = bridge.get_model_config()
                _configure_model_config(
                    model_config, tp=tp, pp=pp, ep=ep, etp=etp, dtype=dtype, use_cpu=True
                )
                _maybe_generate_pipeline_layout(bridge, model_config, pp)
                megatron_model = bridge.get_model(
                    model_config, wrap_with_ddp=False, mixed_precision_wrapper=None
                )
            else:
                model_provider = bridge.to_megatron_provider(load_weights=True)
                _configure_model_provider(
                    model_provider, tp=tp, pp=pp, ep=ep, etp=etp, dtype=dtype, use_cpu=True
                )
                _maybe_generate_pipeline_layout(bridge, model_provider, pp)
                model_provider.finalize()
                model_provider.initialize_model_parallel(seed=0, create_gloo_process_groups=False)
                megatron_model = model_provider.provide_distributed_model(wrap_with_ddp=False)

            bridge.save_megatron_model(
                megatron_model,
                megatron_path,
                hf_tokenizer_path=hf_model,
                hf_tokenizer_kwargs=_hf_tokenizer_kwargs(
                    bridge, trust_remote_code=trust_remote_code
                ),
                low_memory_save=False,
            )
            if rank == 0:
                logger.info("Distributed CPU import complete: %s", megatron_path)

        _import_distributed()
        return
    if any(size != 1 for size in (tp, pp, ep, etp)):
        raise ValueError("CPU model parallelism requires use_distributed=True.")
    prepare_output_directory(megatron_path, overwrite=overwrite, source_paths=[hf_model])
    trusted = is_safe_repo(trust_remote_code=trust_remote_code, hf_path=hf_model)
    logger.info("CPU import: %s -> %s", hf_model, megatron_path)
    revision_kwargs = {"revision": hf_revision} if hf_revision is not None else {}
    if text_only:
        revision_kwargs["text_only"] = True
    AutoBridge.import_ckpt(
        hf_model_id=hf_model,
        megatron_path=megatron_path,
        torch_dtype=parse_dtype(torch_dtype),
        device_map="cpu",
        trust_remote_code=trusted,
        **revision_kwargs,
    )
    logger.info("CPU import complete: %s", megatron_path)


def export_checkpoint(
    *,
    hf_model: str,
    hf_revision: str | None,
    megatron_path: str,
    hf_path: str,
    show_progress: bool,
    strict: bool,
    trust_remote_code: bool,
    overwrite: bool,
    text_only: bool = False,
) -> None:
    """Export a Megatron checkpoint to Hugging Face format on CPU.

    Args:
        hf_model: Hugging Face model ID or local config reference.
        hf_revision: Immutable Hugging Face Hub revision to load.
        megatron_path: Source Megatron checkpoint path.
        hf_path: Destination Hugging Face checkpoint path.
        show_progress: Display conversion progress.
        strict: Require source and destination parameter keys to match.
        trust_remote_code: Allow custom Hugging Face repository code.
        overwrite: Delete a non-empty destination before conversion.
    """
    checkpoint_path = Path(megatron_path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Megatron checkpoint does not exist: {checkpoint_path}")
    config_path = _find_run_config(checkpoint_path)
    prepare_output_directory(hf_path, overwrite=overwrite, source_paths=[megatron_path, hf_model])

    trusted = is_safe_repo(trust_remote_code=trust_remote_code, hf_path=hf_model)
    logger.info("CPU export: %s -> %s", megatron_path, hf_path)
    logger.info("Using Megatron run config: %s", config_path)
    revision_kwargs = {"revision": hf_revision} if hf_revision is not None else {}
    if text_only:
        revision_kwargs["text_only"] = True
    bridge = AutoBridge.from_hf_pretrained(hf_model, trust_remote_code=trusted, **revision_kwargs)
    reference_model = (
        resolve_hf_model_revision(hf_model, hf_revision, config_only=True)
        if text_only
        else resolve_hf_model_revision(hf_model, hf_revision)
    )
    checkpoint_config_bridge = AutoBridge.from_auto_config(
        megatron_path, reference_model, trust_remote_code=trusted
    )
    # Preserve the reference wrapper's state source and shard map so model
    # families with packed HF weights export in their canonical representation.
    bridge.hf_pretrained.config = checkpoint_config_bridge.hf_pretrained
    try:
        bridge.export_ckpt(
            megatron_path=megatron_path, hf_path=hf_path, show_progress=show_progress, strict=strict
        )
    except Exception as error:
        if strict:
            shutil.rmtree(Path(hf_path), ignore_errors=True)
            raise RuntimeError(
                f"Strict Megatron-to-HF export failed: {error}. Partial output at {hf_path} was deleted. "
                "Re-run with --not-strict only when unmatched keys are expected."
            ) from error
        raise
    logger.info("CPU export complete: %s", hf_path)
