# Shared configuration schemas for training drivers

Megatron Bridge and Megatron-LM can construct the same dataclass types from
`megatron.training.config`. Megatron Core continues to own optimizer and DDP
configuration. Importing these classes does not import Bridge or install any
optional logging, checkpoint-export, or distributed-runtime package.

Some schema fields are consumed by Bridge's Python training driver. Adding them
to a shared schema preserves typed construction and saved-config reload; it does
not add their runtime implementation to the native Megatron-LM training loop.
They are excluded from Megatron-LM's generated CLI until that loop supports them.

| Section | Bridge-consumed additions |
| --- | --- |
| Training | Fractional `num_epochs`, derived epoch state, optimizer-step checking, gradient-norm synchronization policy; legacy nested evaluation fields |
| Scheduler | `max_steps` full schedule horizon for shortened runs |
| Checkpoint | CPU optimizer-state staging, per-rank writer count, write-result multiprocessing mode, per-DP RNG state, custom checkpoint manager, additional Hugging Face export settings |
| Distributed initialization | External device mapping |
| Logging | Training-metrics gate, legacy Comet credential, MLflow artifact-upload policy; `timing_log_level=-1` for Bridge's disabled timer logging |
| Rerun | `spiky_loss_factor` used by Bridge's loss checks |

Torch FSDP2 is selected by `TorchFullyShardedDataParallelConfig`, and GTP FP32
reduce-scatter accumulation belongs to DDP configuration. Bridge retains its
experimental feature switch, process-group routing, evaluation CP rebinding,
and legacy GTP NCCL-buffer switch locally. The unused positive straggler-startup
field is removed; `disable_straggler_on_startup` remains the runtime control.

The native CLI retains its existing timing levels, validation flags, defaults,
and runtime behavior. The separate MLflow/Comet integration supplies the shared
logger backend fields. This schema change does not implement those integrations.
Prefer `COMET_API_KEY` over the legacy programmatic credential field.

Validation and derived values must run after recipe overrides, through the
owning driver's runtime initialization. Schema construction alone does not
validate a complete training run. Bridge keeps its filesystem, optional-package,
precision, and training-loop checks in its runtime setup. Core's existing explicit
optimizer and DDP lifecycle methods remain unchanged.
