# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/08035723e7a86f94fc54a7f4473bd0519f2ac941/scripts/conversion/utils.py
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
"""Utilities shared by CPU and distributed GPU conversion backends."""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.gpt_provider import GPTModelProvider


DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _configure_distributed_env() -> None:
    """Derive the PyTorch distributed environment for direct Slurm launches."""
    if os.environ.get("WORLD_SIZE") is not None or os.environ.get("SLURM_NTASKS") is None:
        return

    from megatron.bridge.utils.slurm_utils import (
        resolve_slurm_master_addr,
        resolve_slurm_master_port,
    )

    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    master_addr = resolve_slurm_master_addr()
    master_port = resolve_slurm_master_port()
    if master_addr is not None:
        os.environ["MASTER_ADDR"] = master_addr
    if master_port is not None:
        os.environ["MASTER_PORT"] = str(master_port)


def _validate_hf_revision_target(hf_model: str, hf_revision: str | None) -> None:
    """Reject revision pinning for local model paths."""
    if hf_revision is not None and Path(hf_model).expanduser().exists():
        raise ValueError(
            "--hf-revision applies only to Hugging Face Hub model IDs, not local paths."
        )


def resolve_hf_commit_revision(hf_model: str, hf_revision: str | None) -> str | None:
    """Resolve a Hub branch, tag, or commit to one immutable commit SHA.

    Args:
        hf_model: Hugging Face model ID or local path.
        hf_revision: Hub branch, tag, or commit to resolve.

    Returns:
        The immutable Hub commit SHA, or ``None`` when no revision was supplied.

    Raises:
        ValueError: If a revision is paired with an existing local path.
        RuntimeError: If the Hub response does not contain a commit SHA.
    """
    _validate_hf_revision_target(hf_model, hf_revision)
    if hf_revision is None:
        return None
    if re.fullmatch(r"[0-9a-f]{40}", hf_revision):
        return hf_revision

    from huggingface_hub import HfApi

    resolved_revision = HfApi().model_info(repo_id=hf_model, revision=hf_revision).sha
    if not resolved_revision:
        raise RuntimeError(
            f"Hugging Face Hub did not return a commit SHA for {hf_model}@{hf_revision}."
        )
    return resolved_revision


def resolve_hf_model_revision(
    hf_model: str, hf_revision: str | None, *, config_only: bool = False
) -> str:
    """Resolve a remote Hugging Face model revision to an immutable local snapshot.

    Args:
        hf_model: Hugging Face model ID or local path.
        hf_revision: Hub branch, tag, or commit to resolve. ``None`` preserves
            the original model reference.
        config_only: Download only config and custom Python dependencies, not weights.

    Returns:
        The original model reference when no revision is provided, otherwise
        the local path of the resolved Hub snapshot.

    Raises:
        ValueError: If a revision is paired with an existing local path.
    """
    _validate_hf_revision_target(hf_model, hf_revision)
    if hf_revision is None:
        return hf_model

    from huggingface_hub import snapshot_download

    kwargs = {"allow_patterns": ["config.json", "*.py"]} if config_only else {}
    return snapshot_download(repo_id=hf_model, revision=hf_revision, **kwargs)


def parse_dtype(name: str) -> torch.dtype:
    """Resolve a CLI dtype name.

    Args:
        name: User-facing dtype name.

    Returns:
        Matching PyTorch dtype.

    Raises:
        ValueError: If the dtype name is unsupported.
    """
    try:
        return DTYPE_MAP[name]
    except KeyError:
        raise ValueError(f"Unsupported dtype '{name}'. Choose from {sorted(DTYPE_MAP)}.") from None


def validate_output_path(path: str, *, source_paths: Iterable[str]) -> Path:
    """Reject a conversion destination that overlaps an existing local source.

    Args:
        path: Destination directory.
        source_paths: Local or remote source references. Nonexistent paths are
            treated as remote identifiers and skipped.

    Returns:
        Destination as a ``Path``.

    Raises:
        ValueError: If the destination equals, contains, or is contained by a
            local source path.
    """
    output_path = Path(path).expanduser()
    resolved_output = output_path.resolve()
    for source in source_paths:
        source_path = Path(source).expanduser()
        if not source_path.exists():
            continue
        resolved_source = source_path.resolve()
        if (
            resolved_output == resolved_source
            or resolved_output in resolved_source.parents
            or resolved_source in resolved_output.parents
        ):
            raise ValueError(f"Destination {output_path} overlaps conversion source {source_path}.")
    return output_path


def prepare_output_directory(
    path: str, *, overwrite: bool, source_paths: Iterable[str] = ()
) -> Path:
    """Validate and optionally clear a conversion destination.

    Args:
        path: Destination directory.
        overwrite: Delete a non-empty destination when true.
        source_paths: Local or remote source references that must not overlap
            the destination.

    Returns:
        Destination as a ``Path``.

    Raises:
        FileExistsError: If the destination is non-empty and overwrite is false.
        ValueError: If the destination overlaps a local source or overwrite
            targets the filesystem root.
    """
    output_path = validate_output_path(path, source_paths=source_paths)
    if not output_path.exists() or not any(output_path.iterdir()):
        return output_path
    if not overwrite:
        raise FileExistsError(
            f"Destination is not empty: {output_path}. Pass --overwrite to replace it."
        )
    if output_path.resolve() == Path("/"):
        raise ValueError("Refusing to overwrite the filesystem root.")
    shutil.rmtree(output_path)
    return output_path


def _uses_model_builder(bridge: AutoBridge) -> bool:
    """Return whether the selected bridge supports native builder construction."""
    return getattr(bridge._model_bridge, "USE_MODEL_CONFIG_FOR_CONVERSION", False)


def _configure_model_provider(
    model_provider: GPTModelProvider,
    *,
    tp: int,
    pp: int,
    ep: int,
    etp: int,
    dtype: torch.dtype,
    use_cpu: bool = False,
) -> None:
    """Apply distributed parallelism and dtype settings to a model provider."""
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    model_provider.expert_tensor_parallel_size = etp
    model_provider.pipeline_dtype = dtype
    model_provider.params_dtype = dtype
    if use_cpu:
        model_provider.use_cpu_initialization = True


def _configure_model_config(
    model_config, *, tp: int, pp: int, ep: int, etp: int, dtype: torch.dtype, use_cpu: bool = False
) -> None:
    """Apply distributed parallelism and dtype settings to a builder config."""
    _configure_model_provider(
        model_config.transformer, tp=tp, pp=pp, ep=ep, etp=etp, dtype=dtype, use_cpu=use_cpu
    )


def _maybe_generate_pipeline_layout(
    bridge: AutoBridge, model_provider: GPTModelProvider, pp: int
) -> bool:
    """Generate a bridge-specific pipeline layout when the model requires one.

    A bridge returns ``None`` when the default pipeline split already applies. A bridge whose
    stage boundaries also live in other fields, such as the ``|`` segments of a hybrid layer
    pattern, implements ``configure_pipeline_layout`` to update the model config in place instead.
    """
    if pp <= 1:
        return False
    if hasattr(bridge._model_bridge, "configure_pipeline_layout"):
        bridge._model_bridge.configure_pipeline_layout(model_provider)
        from megatron.bridge.utils.common_utils import print_rank_0

        stage_sizes = [
            stage.count("decoder") for stage in model_provider.pipeline_model_parallel_layout or []
        ]
        print_rank_0(
            f"Configured pipeline layout for PP={pp} ({stage_sizes} decoder layers per stage)"
        )
        return True
    if not hasattr(bridge._model_bridge, "generate_pipeline_layout"):
        return False
    num_layers = bridge.hf_pretrained.config.num_hidden_layers
    # The layout must match the model being built, which may omit the checkpoint's MTP layers.
    model_config = getattr(model_provider, "transformer", model_provider)
    mtp_layers = getattr(model_config, "mtp_num_layers", None) or 0
    layout = bridge._model_bridge.generate_pipeline_layout(num_layers, pp, mtp_layers)
    if layout is None:
        return False
    model_provider.pipeline_model_parallel_layout = layout
    from megatron.bridge.utils.common_utils import print_rank_0

    print_rank_0(
        f"Auto-generated pipeline layout for PP={pp} ({num_layers} layers, {mtp_layers} MTP)"
    )
    return True


def _hf_tokenizer_kwargs(bridge: AutoBridge, *, trust_remote_code: bool) -> dict[str, object]:
    """Build tokenizer metadata for a saved Megatron checkpoint."""
    tokenizer_kwargs: dict[str, object] = {}
    if hasattr(bridge._model_bridge, "get_hf_tokenizer_kwargs"):
        tokenizer_kwargs = bridge._model_bridge.get_hf_tokenizer_kwargs() or {}
    if trust_remote_code:
        tokenizer_kwargs["trust_remote_code"] = True
    if bridge.hf_model_revision is not None:
        tokenizer_kwargs["revision"] = bridge.hf_model_revision
    return tokenizer_kwargs
