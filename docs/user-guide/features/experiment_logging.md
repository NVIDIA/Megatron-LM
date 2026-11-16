# Experiment logging

MLflow and Comet are optional experiment backends alongside W&B and TensorBoard.
Only the last global rank creates an experiment and reports metrics, following
W&B's convention. W&B, Comet, and MLflow share one experiment writer and do not
require TensorBoard to be enabled. TensorBoard uses its own writer.
Install the selected SDK (`pip install mlflow` or `pip install comet-ml`) in the
training environment. Disabled backends do not import their SDKs.

## Configuration

CLI example (append to your normal training command):

```bash
--mlflow-experiment pretraining --mlflow-run-name run-1 \
--mlflow-tracking-uri https://your-tracking-server \
--mlflow-tags '{"team": "training"}' \
--comet-project pretraining --comet-experiment-name run-1 \
--comet-workspace your-workspace --comet-tags pretraining experiment
```

The equivalent recipe sets fields on `cfg.logger` (`LoggerConfig`):

```python
cfg.logger.mlflow_experiment = "pretraining"
cfg.logger.mlflow_run_name = "run-1"
cfg.logger.mlflow_tags = {"team": "training"}
cfg.logger.comet_project = "pretraining"
cfg.logger.comet_experiment_name = "run-1"
```

Use the SDK's environment variables for authentication, for example
`COMET_API_KEY` or MLflow's tracking credentials. Do not put secrets in config.
The tracking URI itself is omitted from logged parameters; the serialized run
config still contains it, so do not embed credentials in the URI.

### Environment defaults and precedence

MLflow and W&B can also be enabled without logging CLI flags:

```bash
export MLFLOW_EXPERIMENT_NAME=pretraining
export MLFLOW_RUN_NAME=run-1
# Or select an existing experiment by ID instead of name:
# unset MLFLOW_EXPERIMENT_NAME
# export MLFLOW_EXPERIMENT_ID=42

export WANDB_PROJECT=pretraining
export WANDB_NAME=run-1
```

| Environment variable | Logger config / CLI option |
|---|---|
| `MLFLOW_EXPERIMENT_NAME` | `mlflow_experiment` / `--mlflow-experiment` |
| `MLFLOW_EXPERIMENT_ID` | `mlflow_experiment_id` / `--mlflow-experiment-id` |
| `MLFLOW_RUN_NAME` | `mlflow_run_name` / `--mlflow-run-name` |
| `WANDB_PROJECT` | `wandb_project` / `--wandb-project` |
| `WANDB_NAME` | `wandb_exp_name` / `--wandb-exp-name` |
| `WANDB_ENTITY` | `wandb_entity` / `--wandb-entity` |
| `WANDB_RUN_ID` | `wandb_run_id` / `--wandb-run-id` |
| `WANDB_DIR` | `wandb_save_dir` / `--wandb-save-dir` |
| `WANDB_RESUME` | `wandb_resume` / `--wandb-resume` |

Explicit CLI or recipe settings take precedence over environment defaults.
Environment defaults take precedence over generated defaults. Only unset (`None`)
config fields inherit environment values; empty environment values are ignored.
For example, `WANDB_PROJECT=launcher --wandb-project=recipe` selects `recipe`.
Overriding the project does not discard an environment-provided run name.

MLflow experiment name and ID are two ways to select the same group of runs:

- An explicit `mlflow_experiment` **or** `mlflow_experiment_id` overrides **both**
  environment selectors. An unrelated environment ID cannot override a CLI name.
- With no explicit selection, either environment selector enables MLflow.
- When both selected name and ID are present, they must identify the same existing
  experiment. A mismatch raises an error before creating a run; neither silently
  wins. This also applies when both selectors are explicitly configured.
- A name alone may create a new experiment; an ID must already exist.
- A run name is still required, from explicit config or `MLFLOW_RUN_NAME`.
  As before, resuming a run or attaching to an active run uses that run's existing
  experiment rather than selecting a new one.

For W&B, an environment run ID wins over the ID otherwise derived from the display
name, and `WANDB_DIR` wins over the checkpoint-directory fallback. Resume policy
defaults to `allow` unless config or `WANDB_RESUME` selects `allow`, `must`, `never`,
or `auto`. `WANDB_RUN_ID` identifies one run, **not** an MLflow-style experiment group.

Validation checks these effective settings without modifying the recipe.
Container finalization stores the resolved values in `cfg.logger` before logger
enablement and config logging, so CLI and native recipes follow the same rules.
With no project/experiment selection in config or environment, the backend stays
disabled. SDK-managed credentials and connection settings, including `WANDB_API_KEY`,
`WANDB_BASE_URL`, `WANDB_MODE`, and `MLFLOW_TRACKING_URI`, retain their SDK behavior;
an explicit `mlflow_tracking_uri` overrides its environment default.

Comet is unchanged: `comet_project` and `comet_experiment_name` must be set in the
CLI or recipe. SDK environment defaults such as `COMET_API_KEY` and
`COMET_WORKSPACE` still apply; explicit workspace config wins.

## Metrics and checkpoints

All three experiment backends receive the same training/evaluation scalar metrics,
including loss, learning rate, gradient norm, throughput and timer metrics.
Existing logging intervals and metric-selection flags apply, including
`--tensorboard-log-interval` and `--log-timers-to-tensorboard`. The latter also
controls experiment-backend timers despite its historical name. MLflow metric
names replace unsupported characters with `_`; W&B/Comet names stay unchanged.
Training tensor-metric observers, RL scalar metrics, and inference metrics also
use these backends. The existing `--inference-wandb-logging` flag enables
inference reporting to any configured experiment backend. Basic W&B training
and evaluation scalars now log without TensorBoard, including when evaluation
sets `write_to_tensorboard=False`; TensorBoard output remains unchanged.
Standalone evaluation (`--skip-train` or no training data) logs `eval_only/*`
against `eval_checkpoint_iter`, so evaluating an older checkpoint does not send
an out-of-order training step to W&B. In-training evaluation keeps its existing
metric names and training-step axis.
Custom callbacks use `get_experiment_writer()`, which returns the shared writer
even when only W&B is configured. It provides `log(metrics, step)` and
`log_metadata(values)`; normal shutdown closes all enabled backends together.
Use `on_experiment_log` for callback metrics at `--tensorboard-log-interval`.
It runs on the rank owning the experiment writer regardless of TensorBoard;
`on_tensorboard_log` remains restricted to ranks owning a TensorBoard writer.
Tables and plots use its `table`, `histogram`, and `log_media` methods and currently
require W&B. Check `supports_media` before preparing media. The separate
`get_wandb_writer()` accessor is removed; migrate existing callbacks to the shared writer.

Direct dynamic-engine callers supply an `InferenceMetricsWriter` through
`InferenceConfig.metrics_writer`. This interface provides
`start_inference_logging()` and `log_inference(metrics)`. The shared
`ExperimentWriter` implements it for all three backends, including W&B.

All backends log the structured `cfg.to_dict()` configuration rather than the
legacy command-line args namespace. W&B and Comet receive nested JSON-compatible
config; MLflow receives flattened parameter strings (values capped at 500
characters). TensorBoard records the config as JSON under its `config` text tag.
Non-JSON values are converted to text and credential fields are omitted. Legacy
args-only fields and the separate kitchen-file contents snapshot are not logged.
Full config remains in the checkpoint's `run_config.yaml`. Runtime SDK objects
are never stored in the config or checkpoint.

After a persistent checkpoint finishes, all three backends record its path and
iteration. Checkpoint files are not uploaded to experiment backends. Reporting
adds no distributed barrier; metadata SDK calls still run during finalization.
All three backends record the checkpoint path selected on successful load.
Reporting failures warn without invalidating a successfully saved checkpoint.

## Resume and shutdown

Use `--mlflow-run-id` to resume an existing run; MLflow's `MLFLOW_RUN_ID`
environment variable is also supported. Names alone do not identify a run to
resume. Existing MLflow parameters are retained because the SDK treats them as
immutable; metrics continue using the restored training iteration. An already
active MLflow run is reused and left open for its owner to close. Internally owned
MLflow runs use explicit run IDs, so cleanup from a restart thread closes the
correct run without leaving a thread-local active run behind. As in Bridge,
Comet creates a new experiment for each invocation; restoring a checkpoint does
not reattach to an earlier Comet experiment.

Owned runs finish after pending asynchronous checkpoints complete. Early training
exit marks them killed (failed for a nonzero exit code), and uncaught Python
exceptions mark them failed. Explicit nonzero exits during initialization or
checkpoint loading also mark them failed. If shutdown bypasses the training
lifecycle, the process-exit fallback records an interrupted run, never a
successful completion. W&B represents unsuccessful endings through a nonzero
exit code. Abrupt termination such as SIGKILL cannot flush an SDK.
