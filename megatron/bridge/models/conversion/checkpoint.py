# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Adapted from Megatron Bridge's weights-only model_load_save/checkpointing flow:
# https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/training
# Source license: Apache-2.0.
"""Experimental weights-only conversion I/O using MCore and MLM config metadata.

This intentionally does not import Bridge's training loop or install global MLM
arguments. It supports builder-backed torch_dist checkpoints only. Optimizer,
training progress, ModelOpt transforms, adapters and destructive low-memory saves
remain outside this initial port.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch.distributed as dist

from megatron.core import dist_checkpointing, parallel_state
from megatron.core.utils import unwrap_model
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename
from megatron.training.config import (
    CheckpointConfig,
    InferenceConfigContainer,
    InferenceSetupConfig,
)
from megatron.training.models.base import ModelConfig
from megatron.training.utils.checkpoint_utils import (
    get_checkpoint_run_config_filename,
    read_run_config,
)
from megatron.training.vocab_utils import model_vocab_size


@contextmanager
def temporary_distributed_context(backend: str = "gloo") -> Iterator[None]:
    """Own a single-rank process group only when the caller did not supply one."""
    if dist.is_initialized():
        yield
        return
    dist.init_process_group(backend=backend, store=dist.HashStore(), world_size=1, rank=0)
    try:
        # AutoBridge initializes model-parallel groups from the supplied model config.
        yield
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Resolve MLM's tracker file, or accept an explicit distributed iteration dir."""
    path = Path(path)
    tracker = Path(get_checkpoint_tracker_filename(str(path)))
    if tracker.is_file():
        # MLM's training reader reduces a CUDA tensor. Conversion also supports
        # Gloo/CPU, so read the same selector format without that training collective.
        selector = tracker.read_text().strip()
        release = selector == "release"
        iteration = 0 if release else int(selector)
        if iteration < 0:
            raise ValueError(f"Invalid checkpoint iteration in {tracker}: {selector}")
        path = Path(get_checkpoint_name(str(path), iteration, release, return_base_dir=True))
    if not dist_checkpointing.check_is_distributed_checkpoint(str(path)):
        raise ValueError(f"Not a torch_dist checkpoint directory: {path}")
    return path


def load_model_config(path: str | Path) -> tuple[ModelConfig, None]:
    """Read native builder metadata; legacy args/provider reconstruction is not ported."""
    from megatron.training.config.instantiate_utils import _resolve_target

    checkpoint = resolve_checkpoint_path(path)
    metadata = read_run_config(get_checkpoint_run_config_filename(str(checkpoint)))
    model_dict = metadata["model"]

    def check_targets(value: Any, key: str) -> None:
        # from_dict reconstructs nested dataclasses directly. Preserve the same
        # target allowlist as instantiate before allowing those imports.
        if isinstance(value, dict):
            if "_target_" in value:
                _resolve_target(value["_target_"], full_key=f"{key}._target_", check_callable=False)
            for name, item in value.items():
                check_targets(item, f"{key}.{name}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                check_targets(item, f"{key}.{index}")

    check_targets(model_dict, "model")
    model_config_cls = _resolve_target(model_dict["_target_"], full_key="model._target_")
    if not isinstance(model_config_cls, type) or not issubclass(model_config_cls, ModelConfig):
        raise TypeError("This conversion port requires a builder-backed model config")
    # Match Bridge's loader: model-specific from_dict restores activation callables
    # and other serialized fields before reverse HF config translation consumes them.
    config = model_config_cls.from_dict(model_dict)
    return config, None


def get_hf_source_from_model_config(config: ModelConfig) -> tuple[str | None, str | None]:
    """Read the source id and revision that AutoBridge records on native configs."""
    metadata = config.extra_checkpoint_metadata or {}
    return metadata.get("hf_model_id"), metadata.get("hf_model_revision")


def get_hf_model_id_from_checkpoint(path: str | Path) -> str | None:
    """Return the saved HF source id without constructing a model."""
    config, _ = load_model_config(path)
    return get_hf_source_from_model_config(config)[0]


def _model_state_dict(models: list, metadata: dict | None = None) -> dict:
    """Use the same model/modelN envelope as both MLM and Bridge checkpointing."""
    metadata = dict(metadata or {})
    metadata["dp_cp_group"] = parallel_state.get_data_parallel_group(with_context_parallel=True)
    return {
        "model" if len(models) == 1 else f"model{i}": model.sharded_state_dict(metadata=metadata)
        for i, model in enumerate(unwrap_model(models))
    }


def _rank_zero_action(action) -> None:
    """Propagate filesystem/tokenizer failures before peers enter the next collective."""
    errors = [None]
    if dist.get_rank() == 0:
        try:
            action()
        except Exception as error:
            errors[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(errors, src=0)
    if errors[0] is not None:
        raise RuntimeError(errors[0])


def save_megatron_model(
    model: list,
    path: str | Path,
    hf_tokenizer_path: str | Path | None = None,
    low_memory_save: bool = False,
    hf_tokenizer_kwargs: dict | None = None,
) -> None:
    """Save weights and native run_config.yaml, publishing the tracker last."""
    if low_memory_save:
        raise NotImplementedError("Destructive low-memory saves are not included in this port")
    config = getattr(model[0], "model_config", None)
    if not isinstance(config, ModelConfig):
        raise TypeError("Expected models built by AutoBridge.get_model with native model_config")
    if config.restore_modelopt_state:
        raise NotImplementedError("ModelOpt checkpoint conversion is not included in this port")
    if not dist.is_initialized() or not parallel_state.is_initialized():
        raise RuntimeError("Save requires the process groups used to construct the model")
    root = Path(path)
    iteration_dir = Path(get_checkpoint_name(str(root), 0, return_base_dir=True))
    cfg = InferenceConfigContainer(
        model=config,
        inference=InferenceSetupConfig(),
        checkpoint=CheckpointConfig(save=str(root), save_optim=False, save_rng=False),
    )

    def prepare() -> None:
        # Refuse to overwrite a previous export, including an unfinished export.
        iteration_dir.mkdir(parents=True, exist_ok=False)
        if hf_tokenizer_path is not None:
            from transformers import AutoTokenizer

            tokenizer_dir = iteration_dir / "tokenizer"
            tokenizer = AutoTokenizer.from_pretrained(
                hf_tokenizer_path, **(hf_tokenizer_kwargs or {})
            )
            tokenizer.save_pretrained(tokenizer_dir)
            cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
            cfg.tokenizer.tokenizer_model = str(tokenizer_dir)
            cfg.tokenizer.padded_vocab_size = model_vocab_size(
                config.vocab_size,
                should_pad_vocab=config.should_pad_vocab,
                make_vocab_size_divisible_by=config.make_vocab_size_divisible_by,
                tensor_model_parallel_size=config.transformer.tensor_model_parallel_size,
            )
        cfg.to_yaml(get_checkpoint_run_config_filename(str(iteration_dir)))

    _rank_zero_action(prepare)
    state = _model_state_dict(model)
    state.update(checkpoint_version=3.1, iteration=0)
    dist_checkpointing.save(state, str(iteration_dir))
    _rank_zero_action(lambda: Path(get_checkpoint_tracker_filename(str(root))).write_text("0"))
    dist.barrier()


def load_megatron_model(
    bridge: Any, path: str | Path, *, mp_overrides: dict | None = None, **kwargs: Any
) -> list:
    """Reconstruct a native model and load only its weights through MCore.

    Unlike the full Bridge loader, this prototype preserves saved topology unless
    explicit overrides are supplied. No silent topology or precision conversion.
    """
    checkpoint = resolve_checkpoint_path(path)
    config, _ = load_model_config(checkpoint)
    if config.restore_modelopt_state:
        raise NotImplementedError("ModelOpt checkpoint conversion is not included in this port")
    from megatron.bridge.models.conversion.runtime import (
        _get_checkpoint_weight_topology,
        _validate_checkpoint_weight_topology,
    )

    saved_topology = _get_checkpoint_weight_topology(config)
    saved_pp = config.transformer.pipeline_model_parallel_size
    overrides = mp_overrides or {}
    for name, value in overrides.items():
        if not hasattr(config, name):
            raise ValueError(f"Unknown model override: {name}")
        setattr(config, name, value)
    pp_changed = config.pipeline_model_parallel_size != saved_pp
    clear_layout = (pp_changed and "pipeline_model_parallel_layout" not in overrides) or (
        config.pipeline_model_parallel_size == 1
        and config.virtual_pipeline_model_parallel_size is None
    )
    if clear_layout:
        config.pipeline_model_parallel_layout = None
    if (pp_changed or clear_layout) and "hybrid_layer_pattern" not in overrides:
        pattern = getattr(config, "hybrid_layer_pattern", None)
        if isinstance(pattern, str):
            config.hybrid_layer_pattern = pattern.replace("|", "")
    _validate_checkpoint_weight_topology(
        saved=saved_topology, requested=_get_checkpoint_weight_topology(config)
    )
    config.transformer.perform_initialization = False
    config.transformer.use_cpu_initialization = kwargs.pop(
        "use_cpu_initialization", dist.is_initialized() and dist.get_backend() == "gloo"
    )
    kwargs.setdefault("wrap_with_ddp", False)
    models = bridge.get_model(config, load_weights=False, **kwargs)
    common = dist_checkpointing.load_common_state_dict(str(checkpoint))
    metadata = dist_checkpointing.load_content_metadata(preloaded_state_dict=common)
    template = _model_state_dict(models, metadata)
    loaded = dist_checkpointing.load(template, str(checkpoint))
    for i, model in enumerate(unwrap_model(models)):
        model.load_state_dict(loaded["model" if len(models) == 1 else f"model{i}"], strict=True)
    return models
