# Shared configuration settings for training drivers

Megatron Bridge can consume canonical Megatron config classes while retaining
extensions for behavior that has not moved upstream. Core owns optimizer and
DDP configuration; training-driver settings live in `megatron.training.config`.
Neither layer imports Bridge.

## Native training-loop controls

`TrainingConfig.check_optimizer_step_success` defaults to `True`. The loop
performs a logical AND across model-parallel ranks before deciding whether to
advance the learning-rate scheduler. `--no-check-optimizer-step-success` uses the
local optimizer result instead. Disable it only when results are consistent
across model-parallel ranks; otherwise scheduler state can diverge.

`TrainingConfig.skip_sync_grad_norm_across_mp` defaults to `False`.
`--skip-sync-grad-norm-across-mp` skips the reporting-only MAX reduction after the
optimizer step. It does not change gradient clipping or the optimizer's own norm
calculation. Ranks without trainable parameters may then report no norm. Zero
counts still synchronize when their logging option is enabled. Bridge uses the
same shared reduction helpers with explicit model-parallel groups.

## Timing reports

`--timing-log-level -1` disables timer reports and optional level-1/2 operator
timers. Level-0 measurements remain active because training progress, throughput,
evaluation and checkpoint bookkeeping consume them. This does not suppress
ordinary progress metrics or eliminate required timer synchronization overhead.
Levels 0, 1, and 2 retain their existing behavior; the default remains 0.

## Native schedule and initialization controls

`--max-steps` / `SchedulerConfig.max_steps` sets the full iteration horizon for
learning-rate decay and weight-decay growth while `train_iters` still determines
when the run stops. An explicit horizon overrides `lr_decay_iters`; fractional
warmup follows that horizon. With WSD, an unspecified decay duration also uses
the horizon. The value must be positive and at least `train_iters`; sample-based
training is unsupported. The default `None` preserves all existing scheduler
resolution, including an explicitly supplied LR decay length.

`--external-gpu-device-mapping` / `DistributedInitConfig.external_gpu_device_mapping`
selects CUDA device 0 when a launcher exposes one GPU per process, even if the
process has a nonzero local rank. NCCL restart initialization and GPU telemetry
use the same device index. The default `False` retains local-rank device selection.
Bridge alone provides the compatibility property `lazy_init`, backed by the
existing shared `lazy_mpu_init` field.

`--spiky-loss-factor` / `RerunStateMachineConfig.spiky_loss_factor` controls the
maximum-loss multiplier used by GPT, hybrid and elastification loss checks when
`check_for_spiky_loss` is enabled. The default 10.0 matches their previous fixed
threshold. Gradient checks and RL's separate percentage-based detector retain
their existing behavior.

## Default behavior

Leaving the new fields unset preserves native behavior: scheduler horizon `None`,
external GPU mapping `False`, spiky-loss factor `10.0`, optimizer-success checking
`True`, norm-sync skipping `False`, and timing level `0`. Regression tests exercise
both defaults and explicit overrides at the runtime consumers.

## Deferred migrations

- TODO: move `num_epochs` and its private derived state with epoch scheduling.
- Evaluation iteration counts, intervals, and `skip_train` belong to
  `ValidationConfig`, never `TrainingConfig`.
- Checkpoint configuration extensions are deferred to a separate PR.
- MLflow/Comet fields and artifact policy belong to the dedicated logger migration.
  The Bridge-only training-metrics gate is not added here.
- Torch FSDP2 uses `TorchFullyShardedDataParallelConfig`; GTP FP32 accumulation
  uses the existing DDP field. Bridge retains experimental feature selection,
  explicit process-group routing, evaluation CP rebinding and its legacy GTP
  NCCL-buffer switch. The unused positive straggler-startup field is removed;
  `disable_straggler_on_startup` remains the runtime control.
