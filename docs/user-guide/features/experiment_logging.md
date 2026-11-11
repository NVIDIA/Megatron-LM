# MLflow and Comet logging

MLflow and Comet are optional experiment backends alongside W&B and TensorBoard.
Only the last global rank creates an experiment and reports metrics, following
W&B's convention. Neither backend requires TensorBoard or W&B to be enabled.
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

## Metrics and checkpoints

Both backends receive the training/evaluation scalar metrics sent to W&B,
including loss, learning rate, gradient norm, throughput and timer metrics.
Existing logging intervals and metric-selection flags apply, including
`--tensorboard-log-interval` and `--log-timers-to-tensorboard`. The latter also
controls experiment-backend timers despite its historical name. MLflow metric
names replace unsupported characters with `_`; W&B/Comet names stay unchanged.
Training tensor-metric observers also use these backends. W&B-specific media
and custom callbacks that access `get_wandb_writer()` remain W&B-only; custom
scalar callbacks can instead use `get_experiment_writer()`.

Config parameters are flattened and summarized (values capped at 500 characters).
Full config remains in the checkpoint's `run_config.yaml`. Runtime SDK objects
are never stored in the config or checkpoint.

After a persistent checkpoint finishes, MLflow uploads its directory and Comet
records its path and iteration. MLflow uploads can be expensive for large
models: set `--no-mlflow-log-artifacts` or
`cfg.logger.mlflow_log_artifacts = False` to disable them. Non-persistent/local
checkpoints are not uploaded; MSC remote paths are not downloaded for upload.
Both backends record the checkpoint path selected on successful load.
Artifact/reporting failures warn without invalidating a successfully saved checkpoint.

## Resume and shutdown

Use `--mlflow-run-id` to resume an existing run; MLflow's `MLFLOW_RUN_ID`
environment variable is also supported. Names alone do not identify a run to
resume. Existing MLflow parameters are retained because the SDK treats them as
immutable; metrics continue using the restored training iteration. An already
active MLflow run is reused and left open for its owner to close. As in Bridge,
Comet creates a new experiment for each invocation; restoring a checkpoint does
not reattach to an earlier Comet experiment.

Owned runs finish after pending asynchronous checkpoints complete. Early training
exit marks them killed (failed for a nonzero exit code), and uncaught Python
exceptions mark them failed. Abrupt termination such as SIGKILL cannot flush an SDK.
