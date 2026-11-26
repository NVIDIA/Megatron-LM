# LSQ scale optimizer policy

Learned quantization scales can set their peak LR, minimum LR and weight-decay
multiplier independently through `ModelTransformConfig`:

```python
model.transformations.lsq_scale_lr = 1e-3
model.transformations.lsq_scale_min_lr = 1e-5
model.transformations.lsq_scale_wd_mult = 0.0
```

The ModelOpt entry points also accept `--lsq-scale-lr`, `--lsq-scale-min-lr`
and `--lsq-scale-wd-mult`. Unset fields retain the existing base policy.
Only differing fields create overrides, so specifying the base values does
not create an indistinguishable extra group during checkpoint loading.

These settings apply to `_amax_pre` and `_amax_post` parameters and follow the
existing scheduler. Zero weight-decay multiplier disables decay. Negative or
nonfinite settings and inverted LR bounds are rejected. Core checks conflicts
with other rules, including standard shape-based weight-decay exclusions.

Policy is recorded on each constructed student model and composed through the
shared optimizer mapping interface. It does not change other runs or the frozen
distillation teacher. Existing peak-LR-only callers retain their behavior.

This change configures optimizer policy; it does not enable scale gradients in
an unsupported ModelOpt build or add checkpoint layout conversion. Runtime LR
changes on resume additionally require the explicit LR-override loader support.

## Shared API and Bridge compatibility

The configuration fields retain the `lsq_scale_` prefix. Internally, the shared
`megatron.post_training.optimizer.get_lsq_config_overrides` API keeps its existing
`scale_min_lr` and `scale_wd_mult` keyword arguments, including the default decay
multiplier of `1.0`. The MLM setup maps an unset configuration multiplier to that
default, preserving standard weight-decay rules.

Bridge's LSQ callable and legacy provider already delegate to this helper. They
continue to use the same API and receive the additional input validation; no
second Bridge policy implementation or separate adoption is required here. Bridge
retains its ModelOpt gradient-support check.

Comparisons involving inherited LR bounds run after model and recipe contributions
are composed. A scale maximum below the global minimum remains valid when the
recipe supplies a lower scale minimum. Explicit negative or nonfinite scale fields
are rejected immediately; inverted final bounds are rejected before optimizer construction.
