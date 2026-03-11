# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""NVIDIA DLFw Inspect integration for tensor inspection and statistics collection."""

from typing import Any, List, Optional

from megatron.training.utils import print_rank_0


MISSING_NVINSPECT_MSG = (
    "nvdlfw_inspect is not available. Please install it with `pip install nvdlfw-inspect`."
)

try:
    import nvdlfw_inspect.api as nvinspect_api
    from nvdlfw_inspect.logging import BaseLogger, MetricLogger, wrap_tensorboard_writer

    HAVE_NVINSPECT = True
except (ImportError, ModuleNotFoundError):
    HAVE_NVINSPECT = False
    nvinspect_api = None
    BaseLogger = None
    MetricLogger = None

    def wrap_tensorboard_writer(x):
        return x


def _get_default_feature_dirs() -> List[str]:
    """Get default feature directories from installed packages."""
    feature_dirs = []
    try:
        import importlib
        from pathlib import Path

        te_features_mod = importlib.import_module("transformer_engine.debug.features")
        te_features_dir = Path(te_features_mod.__file__).parent
        if te_features_dir.exists():
            feature_dirs.append(str(te_features_dir))
    except Exception:
        pass

    return feature_dirs


def _clean_metric_name(name: str) -> str:
    """Strip model wrapper prefixes from metric names for cleaner logging."""
    prefixes = ["model.module.module.", "model.module.", "model."]
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _maybe_attach_metric_loggers(tensorboard_logger: Any, wandb_logger: Any) -> None:
    """Attach TensorBoard and W&B loggers to nvdlfw_inspect."""
    if not HAVE_NVINSPECT:
        return

    try:
        if tensorboard_logger is not None:
            tb_logger = wrap_tensorboard_writer(tensorboard_logger)
            MetricLogger.add_logger(tb_logger)

        if wandb_logger is not None and hasattr(wandb_logger, "log"):
            if BaseLogger is None:
                return

            class _WandbModuleLogger(BaseLogger):
                def __init__(self, wandb_module):
                    super().__init__()
                    self._wandb = wandb_module

                def log_scalar(self, name: str, value: float, iteration: int, **kwargs):
                    clean_name = _clean_metric_name(name)
                    self._wandb.log({clean_name: value}, step=iteration)

            MetricLogger.add_logger(_WandbModuleLogger(wandb_logger))

    except Exception as e:
        print_rank_0(f"Warning: Failed to attach metric loggers to tensor inspection: {e}")


def initialize_tensor_inspect_pre_model(
    enabled: bool,
    config_file: Optional[str] = None,
    feature_dirs: Optional[List[str]] = None,
    log_dir: Optional[str] = None,
    init_training_step: int = 0,
) -> None:
    """Initialize NVIDIA-DL-Framework-Inspect before model construction."""
    if not enabled:
        return

    if not HAVE_NVINSPECT:
        raise ImportError(MISSING_NVINSPECT_MSG)

    if feature_dirs is None:
        feature_dirs = _get_default_feature_dirs()

    nvinspect_api.initialize(
        config_file=config_file or "",
        feature_dirs=feature_dirs,
        log_dir=log_dir or ".",
        statistics_logger=None,
        init_training_step=init_training_step,
        default_logging_enabled=True,
    )
    print_rank_0("Initialized NVIDIA DLFw Inspect.")


def finalize_tensor_inspect_post_model(
    enabled: bool,
    model: List[Any],
    tensorboard_logger: Any = None,
    wandb_logger: Any = None,
    current_training_step: Optional[int] = None,
    include_context_parallel: bool = True,
) -> None:
    """Finalize tensor inspection setup after model creation."""
    if not enabled:
        return

    if not HAVE_NVINSPECT:
        raise ImportError(MISSING_NVINSPECT_MSG)

    from megatron.core.parallel_state import get_tensor_and_data_parallel_group

    _maybe_attach_metric_loggers(tensorboard_logger, wandb_logger)

    if current_training_step is not None:
        nvinspect_api.initialize_training_step(int(current_training_step))

    nvinspect_api.infer_and_assign_layer_names(model)
    nvinspect_api.set_tensor_reduction_group(
        get_tensor_and_data_parallel_group(with_context_parallel=include_context_parallel)
    )
    print_rank_0("Finalized NVIDIA DLFw Inspect.")


def tensor_inspect_step(enabled: bool) -> None:
    """Advance the tensor inspection step counter."""
    if not enabled:
        return

    if not HAVE_NVINSPECT:
        raise ImportError(MISSING_NVINSPECT_MSG)

    nvinspect_api.step()


def tensor_inspect_end(enabled: bool) -> None:
    """Shutdown tensor inspection."""
    if not enabled or not HAVE_NVINSPECT:
        return

    nvinspect_api.end_debug()
