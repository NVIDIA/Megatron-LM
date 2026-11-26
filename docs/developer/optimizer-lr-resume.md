# Override learning rates while resuming the distributed optimizer

`DistributedOptimizer.load_state_dict(state, override_lr=True)` retains the current
groups' `lr`, `max_lr`, `min_lr` and LR logging classification while restoring the
checkpoint's remaining optimizer metadata. MLM passes this option for the standard
distributed optimizer when `SchedulerConfig.override_opt_param_scheduler` is enabled,
then evaluates the scheduler at the restored sample count. Custom group bounds are
not replaced with the base LR after loading.

The checkpoint format, tensor-loading routines and ordinary resume behavior are
unchanged. This option does not add group IDs or a checkpoint version. Without the
option, matching still uses the full existing identifier and checkpoint settings.

During an explicit override, exact identifiers are preferred when their configuration
metadata remains compatible. Otherwise the lookup ignores LR bounds and uses the
current group's other settings (including muP's epsilon) to select compatible saved
groups. Step counters and scheduled weight decay are restored from the checkpoint,
not compared with freshly initialized values. If several candidates remain, all
retained metadata, including checkpoint-only fields, must agree. Ambiguous cases
raise rather than selecting by group position.

Separate exact and LR-independent lookups are built once. The existing pre-DSA
split and NeMo aliases remain supported; returned weight-decay aliases are normalized
so the scheduler sees the intended multiplier. Conflicts in unused saved groups do
not prevent loading. Current LR logging classification survives state allocation.

Moments and master parameters continue to load through the distributed optimizer's
parameter/buffer mappings. Tests cover changing, crossing, merging and splitting LR
groups with `dp_reshardable` and `fully_reshardable` disk checkpoints on two DP ranks,
checking the tensors and subsequent updates against a reference.

This remains an LR-only operation under the existing model/layout compatibility
requirements. It does not validate arbitrary membership changes or add resharding
capabilities. FSDP, layer-wise and positional non-distributed optimizer loaders do not
support the new argument being true; their default loading behavior is preserved.
Chained and MIMO wrappers forward the option; MIMO retains its existing group-count
and checkpoint-layout restrictions.

Other training frameworks using Core must explicitly pass the option when their
runtime scheduler policy requests it. Load the scheduler afterwards using that same
policy and preserve the per-group bounds retained by Core. Updating a Core dependency
alone does not change a framework's training-layer resume policy.
