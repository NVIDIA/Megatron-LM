# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optional experiment backends for the training loop's scalar writer interface.

Implements Bridge's MLflow/Comet behavior against MLM's runtime services.
Reference: NVIDIA-NeMo/Megatron-Bridge, revision
6bfaf4b007fe0db2b3b2267815fa402b0eb9ffd1, src/megatron/bridge/training/state.py
and utils/{mlflow,comet}_utils.py (Apache-2.0). Independently implemented here.
"""

import atexit
import logging
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _flatten_config(config: dict, prefix: str = "") -> dict:
    """Flatten serialized config into tracking parameters, omitting credentials."""
    result = {}
    for key, value in config.items():
        key = str(key)
        if key == "mlflow_tracking_uri" or any(
            part in key.lower() for part in ("password", "secret", "api_key", "access_token")
        ):
            continue
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten_config(value, name))
        else:
            # Tracking parameters are a summary; the checkpoint retains the full config.
            result[name] = str(value)[:500]
    return result


class ExperimentWriter:
    """Send scalar metrics to optional backends and the existing W&B writer.

    Only the reporting rank constructs this object. The ``log`` signature is
    shared with W&B so existing MoE, MTP and other scalar helpers can reuse it.
    SDK objects are runtime services, never part of a serialized config.
    """

    def __init__(self, cfg, wandb_writer=None) -> None:
        self.wandb = wandb_writer
        self.mlflow = None
        self.comet = None
        self._owns_mlflow_run = False
        self._closed = False
        self._previous_excepthook = None
        self._failure_hook = None
        self.log_artifacts = cfg.logger.mlflow_log_artifacts
        settings = cfg.logger
        # Do not serialize runtime objects or the legacy args namespace.
        params = _flatten_config(cfg.to_dict())
        try:
            if settings.mlflow_experiment:
                try:
                    import mlflow
                except ImportError as exc:
                    raise ImportError("MLflow logging requires 'pip install mlflow'.") from exc
                self.mlflow = mlflow
                if settings.mlflow_tracking_uri:
                    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
                active_run = mlflow.active_run()
                if active_run is None:
                    mlflow.set_experiment(settings.mlflow_experiment)
                    active_run = mlflow.start_run(
                        run_id=settings.mlflow_run_id,
                        run_name=settings.mlflow_run_name,
                        tags=settings.mlflow_tags,
                        description=settings.mlflow_description,
                    )
                    self._owns_mlflow_run = True
                elif settings.mlflow_run_id and active_run.info.run_id != settings.mlflow_run_id:
                    raise ValueError("The active MLflow run differs from logger.mlflow_run_id.")
                elif settings.mlflow_tags:
                    mlflow.set_tags(settings.mlflow_tags)
                # Parameters are immutable across resumes. Keep the original values and
                # record only newly introduced keys when attaching to an existing run.
                existing_params = mlflow.get_run(active_run.info.run_id).data.params
                mlflow_params = {
                    re.sub(r"[^\w. /:\-]", "_", key)[:250]: value for key, value in params.items()
                }
                items = [
                    (key, value)
                    for key, value in mlflow_params.items()
                    if key not in existing_params
                ]
                # log_params has a per-request limit; a model config can contain many fields.
                for start in range(0, len(items), 100):
                    mlflow.log_params(dict(items[start : start + 100]))
            if settings.comet_project:
                try:
                    import comet_ml
                except ImportError as exc:
                    raise ImportError("Comet logging requires 'pip install comet-ml'.") from exc
                self.comet = comet_ml.Experiment(
                    project_name=settings.comet_project,
                    workspace=settings.comet_workspace,
                    auto_metric_logging=False,
                    auto_param_logging=False,
                )
                self.comet.set_name(settings.comet_experiment_name)
                if settings.comet_tags:
                    self.comet.add_tags(settings.comet_tags)
                self.comet.log_parameters(params)
        except BaseException:
            self.finish("FAILED")
            raise

        previous = sys.excepthook

        def on_exception(exc_type, exc_value, traceback):
            self.finish("FAILED")
            previous(exc_type, exc_value, traceback)

        self._previous_excepthook = previous
        self._failure_hook = on_exception
        sys.excepthook = on_exception
        atexit.register(self.finish)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Log scalars at the same step as W&B, without changing its metric names."""
        if self._closed:
            return
        if self.wandb is not None:
            self.wandb.log(metrics, step)
        scalars = {name: float(value) for name, value in metrics.items()}
        if self.mlflow is not None:
            # MLflow rejects characters used by some model-specific metric names.
            sanitized = {
                re.sub(r"[^\w. /:\-]", "_", name): value for name, value in scalars.items()
            }
            self.mlflow.log_metrics(sanitized, step=step)
        if self.comet is not None:
            self.comet.log_metrics(scalars, step=step)

    def checkpoint_saved(self, path: str, iteration: int) -> None:
        """Report a durable global checkpoint; called only after save finalization."""
        if self._closed:
            return
        path = str(path)
        if self.mlflow is not None and self.log_artifacts:
            try:
                # MSC URIs are already remotely stored and are not local artifact directories.
                if "://" not in path:
                    self.mlflow.log_artifacts(path, artifact_path=f"checkpoints/{Path(path).name}")
            except Exception:
                logger.warning("Failed to upload checkpoint artifacts to MLflow", exc_info=True)
        if self.comet is not None:
            try:
                self.comet.log_other("last_saved_checkpoint", path)
                self.comet.log_other("last_saved_iteration", iteration)
            except Exception:
                logger.warning("Failed to record checkpoint in Comet", exc_info=True)

    def checkpoint_loaded(self, path: str, load_dir: str) -> None:
        """Record the checkpoint selected by the loader, without fetching artifacts."""
        if self._closed:
            return
        tags = {"last_loaded_checkpoint": str(path), "checkpoint_base_dir": str(load_dir)}
        if self.mlflow is not None:
            try:
                self.mlflow.set_tags(tags)
            except Exception:
                logger.warning("Failed to record loaded checkpoint in MLflow", exc_info=True)
        if self.comet is not None:
            try:
                for name, value in tags.items():
                    self.comet.log_other(name, value)
            except Exception:
                logger.warning("Failed to record loaded checkpoint in Comet", exc_info=True)

    def finish(self, status: str = "FINISHED") -> None:
        """Close owned runs once; leave externally managed MLflow runs open."""
        if self._closed:
            return
        self._closed = True
        if self._failure_hook is not None and sys.excepthook is self._failure_hook:
            sys.excepthook = self._previous_excepthook
        atexit.unregister(self.finish)
        if self._owns_mlflow_run:
            try:
                self.mlflow.end_run(status=status)
            except Exception:
                logger.warning("Failed to finish MLflow run", exc_info=True)
        if self.comet is not None:
            try:
                self.comet.log_other("training_status", status)
            except Exception:
                logger.warning("Failed to record Comet run status", exc_info=True)
            try:
                self.comet.end()
            except Exception:
                logger.warning("Failed to finish Comet experiment", exc_info=True)
