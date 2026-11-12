# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optional experiment backends for the training loop's scalar writer interface.

Implements Bridge's MLflow/Comet behavior against MLM's runtime services.
Reference: NVIDIA-NeMo/Megatron-Bridge, revision
6bfaf4b007fe0db2b3b2267815fa402b0eb9ffd1, src/megatron/bridge/training/state.py
and utils/{mlflow,comet}_utils.py (Apache-2.0). Independently implemented here.
"""

import atexit
import logging
import os
import re
import sys
import time
from pathlib import Path
from threading import Lock
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
    """Third-party experiment writer for W&B, Comet, and MLflow.

    Only the reporting rank constructs this object. Scalar helpers use ``log``
    to report metrics to every enabled backend at the supplied step.
    SDK objects are runtime services, never part of a serialized config.
    """

    def __init__(self, cfg, wandb_writer=None) -> None:
        self.wandb = wandb_writer
        self.mlflow = None
        self.comet = None
        self._owns_mlflow_run = False
        self._mlflow_run_id = None
        self._finish_lock = Lock()
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
                # Address every operation by run ID. Fluent active runs are thread-local,
                # but in-process restart finalization can run on a different thread.
                if settings.mlflow_tracking_uri:
                    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
                self.mlflow = mlflow.tracking.MlflowClient()
                self._mlflow_entities = mlflow.entities
                active_run = mlflow.active_run()
                run_id = settings.mlflow_run_id or os.environ.get("MLFLOW_RUN_ID")
                if active_run is not None:
                    if run_id and active_run.info.run_id != run_id:
                        raise ValueError("The active MLflow run differs from logger.mlflow_run_id.")
                    self._mlflow_run_id = active_run.info.run_id
                elif run_id:
                    self.mlflow.update_run(run_id, status="RUNNING")
                    self._mlflow_run_id = run_id
                    self._owns_mlflow_run = True
                else:
                    experiment = mlflow.set_experiment(settings.mlflow_experiment)
                    tags = dict(settings.mlflow_tags or {})
                    if settings.mlflow_description:
                        tags["mlflow.note.content"] = settings.mlflow_description
                    run = self.mlflow.create_run(
                        experiment.experiment_id, tags=tags, run_name=settings.mlflow_run_name
                    )
                    self._mlflow_run_id = run.info.run_id
                    self._owns_mlflow_run = True
                if self._owns_mlflow_run and settings.mlflow_description:
                    self.mlflow.set_tag(
                        self._mlflow_run_id, "mlflow.note.content", settings.mlflow_description
                    )
                for name, value in (settings.mlflow_tags or {}).items():
                    self.mlflow.set_tag(self._mlflow_run_id, name, value)
                # Parameters are immutable across resumes. Keep original values and
                # record only newly introduced keys when attaching to an existing run.
                existing_params = self.mlflow.get_run(self._mlflow_run_id).data.params
                items = [
                    mlflow.entities.Param(name, value)
                    for key, value in params.items()
                    if (name := re.sub(r"[^\w. /:\-]", "_", key)[:250]) not in existing_params
                ]
                for start in range(0, len(items), 100):
                    self.mlflow.log_batch(
                        self._mlflow_run_id, params=items[start : start + 100], synchronous=True
                    )
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
        # Only the training lifecycle can confirm successful completion.
        atexit.register(self.finish, "KILLED")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Log scalars to every enabled backend at the supplied training step."""
        self._log_scalars(metrics, step)

    def _log_scalars(
        self, metrics: dict[str, Any], step: int, *, independent_step: bool = False
    ) -> None:
        """Dispatch scalars, adapting independent axes to each backend's step API."""
        if self._closed:
            return
        scalars = {name: float(value) for name, value in metrics.items()}
        if self.wandb is not None:
            if independent_step:
                # Custom axes carry their step in the metrics. Keep W&B's internal
                # sequence monotonic when inference or older-checkpoint eval runs.
                self.wandb.log(scalars, commit=True)
            else:
                self.wandb.log(scalars, step)
        if self.mlflow is not None:
            # MLflow rejects characters used by some model-specific metric names.
            sanitized = {
                re.sub(r"[^\w. /:\-]", "_", name): value for name, value in scalars.items()
            }
            timestamp = int(time.time() * 1000)
            self.mlflow.log_batch(
                self._mlflow_run_id,
                metrics=[
                    self._mlflow_entities.Metric(name, value, timestamp, step)
                    for name, value in sanitized.items()
                ],
                synchronous=True,
            )
        if self.comet is not None:
            self.comet.log_metrics(scalars, step=step)

    @property
    def supports_media(self) -> bool:
        """Whether tables and plots can be logged (currently supported by W&B)."""
        return not self._closed and self.wandb is not None

    def table(self, *, columns: list[str], data: list) -> Any:
        """Build a table for media logging, or return None when unsupported."""
        if self.supports_media:
            return self.wandb.Table(columns=columns, data=data)
        return None

    def histogram(self, table: Any, column: str, title: str) -> Any:
        """Build a histogram for media logging, or return None when unsupported."""
        if self.supports_media and table is not None:
            return self.wandb.plot.histogram(table, column, title)
        return None

    def log_media(self, metrics: dict[str, Any], step: int) -> None:
        """Log tables and plots to W&B; other backends currently receive scalars only."""
        if self.supports_media and metrics:
            self.wandb.log(metrics, step=step)

    def start_inference_logging(self) -> int:
        """Configure the inference axis and return its last recorded step on resume."""
        offset = 0
        if self.wandb is not None:
            self.wandb.define_metric('inference/*', step_metric='inference/inference_step')
            run = self.wandb.run
            if run is not None:
                history = self.wandb.Api().run(f'{run.entity}/{run.project}/{run.id}')
                for row in history.scan_history(keys=['inference/inference_step']):
                    value = row.get('inference/inference_step')
                    if isinstance(value, (int, float)):
                        offset = max(offset, int(value))
        if self.mlflow is not None:
            for metric in self.mlflow.get_metric_history(
                self._mlflow_run_id, 'inference/inference_step'
            ):
                offset = max(offset, int(metric.value))
        return offset

    def log_inference(self, metrics: dict[str, Any]) -> None:
        """Use an independent inference step without advancing the training axis."""
        if self._closed:
            return
        self._log_scalars(metrics, int(metrics['inference/inference_step']), independent_step=True)

    def log_evaluation(self, metrics: dict[str, Any], iteration: int, standalone=False) -> None:
        """Log evaluation scalars, including evaluations of older checkpoints."""
        if not standalone:
            self.log(metrics, iteration)
            return
        if self._closed:
            return
        values = {
            'eval_checkpoint_iter': iteration,
            **{f'eval_only/{k}': v for k, v in metrics.items()},
        }
        if self.wandb is not None:
            self.wandb.define_metric('eval_only/*', step_metric='eval_checkpoint_iter')
        self._log_scalars(values, iteration, independent_step=True)

    def log_metadata(self, values: dict[str, Any]) -> None:
        """Record mutable run metadata, including the job name after a restart."""
        if self._closed:
            return
        if self.wandb is not None:
            self.wandb.config.update(values, allow_val_change=True)
        if self.mlflow is not None:
            for name, value in values.items():
                self.mlflow.set_tag(self._mlflow_run_id, name, value)
        if self.comet is not None:
            for name, value in values.items():
                self.comet.log_other(name, value)

    def checkpoint_saved(self, path: str, iteration: int) -> None:
        """Report a durable global checkpoint; called only after save finalization."""
        if self._closed:
            return
        path = str(path)
        if self.wandb is not None:
            try:
                self.wandb.config.update(
                    {"last_saved_checkpoint": path, "last_saved_iteration": iteration},
                    allow_val_change=True,
                )
            except Exception:
                logger.warning("Failed to record checkpoint in W&B", exc_info=True)
        if self.mlflow is not None:
            try:
                self.mlflow.set_tag(self._mlflow_run_id, "last_saved_checkpoint", path)
                self.mlflow.set_tag(self._mlflow_run_id, "last_saved_iteration", iteration)
            except Exception:
                logger.warning("Failed to record checkpoint in MLflow", exc_info=True)
        if self.mlflow is not None and self.log_artifacts:
            try:
                # MSC URIs are already remotely stored and are not local artifact directories.
                if "://" not in path:
                    self.mlflow.log_artifacts(
                        self._mlflow_run_id, path, artifact_path=f"checkpoints/{Path(path).name}"
                    )
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
        if self.wandb is not None:
            try:
                self.wandb.config.update(tags, allow_val_change=True)
            except Exception:
                logger.warning("Failed to record loaded checkpoint in W&B", exc_info=True)
        if self.mlflow is not None:
            try:
                for name, value in tags.items():
                    self.mlflow.set_tag(self._mlflow_run_id, name, value)
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
        with self._finish_lock:
            if self._closed:
                return
            self._closed = True
        if self._failure_hook is not None and sys.excepthook is self._failure_hook:
            sys.excepthook = self._previous_excepthook
        atexit.unregister(self.finish)
        if self.wandb is not None:
            try:
                self.wandb.finish(exit_code=0 if status == "FINISHED" else 1)
            except Exception:
                logger.warning("Failed to finish W&B run", exc_info=True)
        if self._owns_mlflow_run:
            try:
                self.mlflow.set_terminated(self._mlflow_run_id, status=status)
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
