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

## Shared driver settings

`SchedulerConfig.max_steps` carries Bridge's full schedule horizon for shortened
runs. `DistributedInitConfig.external_gpu_device_mapping` and its `lazy_init`
alias preserve driver construction. `RerunStateMachineConfig.spiky_loss_factor`
sets Bridge's spiky-loss threshold. These fields do not introduce new native CLI
options or implementations of their Bridge-specific behavior.

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
