# Configure optimizer parameter groups in training recipes

`PretrainConfigContainer.optimizer_config_overrides` accepts Core's existing
`dict[ParamKey, ParamGroupOverride]` or a callable returning that mapping. The callable
runs after model construction, with keyword arguments `optimizer_config` and `model`.
It needs no provider superclass. For example:

```python
from megatron.core.optimizer.optimizer_config import ParamKey

cfg.optimizer_config_overrides = {
    ParamKey(name="*.linear_out.weight"): {"max_lr": 0.001, "min_lr": 0.0001}
}
```

For type annotations, import `OptimizerConfigOverrides` and
`OptimizerConfigOverrideHook` from `megatron.training.config.optimizer_config`.
This is the shared public import path for MLM and Bridge recipes.

`optimizer_config_overrides_mode="extend"` is the default: the contribution is combined
with standard weight-decay and decoupled-LR rules. An empty contribution preserves
those rules. Set `"replace"` explicitly to supply their replacement, including an
empty mapping to disable standard rules. Enabled model features such as muP and LSQ
are composed in either mode; replacing standard rules does not disable an algorithm.

Core's `merge_config_overrides` merges disjoint fields and accepts equal assignments
on identical selectors. Conflicts raise instead of letting dictionary order determine
precedence. Core also checks conflicts between different selectors when they match
the same parameter. If a custom policy needs a different standard WD rule, use replace
mode and supply the intended complete standard/custom mapping. For muP conflicts,
resolve the feature configuration or selectors explicitly.

Use a configured dataclass callable or an importable function for model-dependent
rules. Dataclass callables and static mappings round-trip through the existing config
serializer/target allowlist; static selectors are encoded as entries because dataclass
objects cannot be YAML mapping keys. This is configuration serialization, not a new
optimizer checkpoint format. Opaque lambdas are unsuitable for serialized recipes.

`megatron.post_training.optimizer.get_lsq_config_overrides` builds shared LSQ
contributions. In addition to the peak LR, callers can supply `scale_min_lr`,
`base_min_lr`, and `scale_wd_mult`. Unset bounds inherit the base schedule; the
default multiplier of 1.0 preserves the existing weight-decay policy. Only fields
that differ from the base settings are emitted, keeping equal parameter groups
from acquiring duplicate checkpoint identities. Standard rules are composed by
the training setup, not by this helper.

Bridge's `LSQOptimizerConfigOverrides` callable uses this helper and retains its
ModelOpt gradient-support check. New Bridge recipes assign that callable to
`optimizer_config_overrides`; the old LSQ provider remains a compatibility wrapper
for existing configurations. Bridge retains the provider API for downstream integrations.

LSQ's model hook records the actual scale LR on the constructed model; setup consumes
it per run. Flextron's router LR now uses the recipe mapping. Neither operation patches
a process-global training function. Standard rules remain Core-owned.

The ordinary training hook does not configure MIMO's separate per-module optimizer
factories. Explicit custom/replacement mappings are rejected for that entry point
instead of being silently ignored.

Bridge can reuse `merge_config_overrides` and adapt its providers at the training
boundary. Existing providers that return the complete standard/custom mapping should
be adapted as replacement mappings. Providers returning only additions use extend
mode. No import-only provider-base-class migration is assumed. This branch changes
MLM and Core only; it does not modify the Bridge repository.

## Construct an optimizer and scheduler outside the training loop

`megatron.training.optim.setup_optimizer` accepts explicit optimizer and scheduler
configs, model chunks, a resolved parameter-group policy and process groups. It does
not read the training loop's global arguments. MLM's training setup calls this helper,
and Bridge's existing setup function delegates construction to it.

Validation and finalization remain with the caller: run the framework's normal
configuration preparation before constructing the model. This includes converting
iteration horizons into scheduler sample counts. The helper does not finalize configs
again, so custom `finalize()` implementations run at their existing boundary.

```python
from megatron.training.config.optimizer_config import build_optimizer_config_overrides
from megatron.training.optim import setup_optimizer

# cfg has already been validated/finalized; model and its process groups exist.
overrides = build_optimizer_config_overrides(
    cfg.optimizer,
    model,
    cfg.optimizer_config_overrides,
    mode=cfg.optimizer_config_overrides_mode,
)
optimizer, scheduler = setup_optimizer(
    cfg.optimizer,
    cfg.scheduler,
    model,
    config_overrides=overrides,
    pg_collection=pg_collection,
    use_gloo_process_groups=cfg.dist.use_gloo_process_groups,
)
```

The construction helper takes an already resolved `config_overrides` mapping. It
preserves `None` versus an empty mapping and does not normalize an existing complete
policy again. New recipe contributions should first use `build_optimizer_config_overrides`.
Explicit process groups are forwarded unchanged. With no groups, the standard Core
factory retains its legacy MPU fallback. MLM training uses the model's process groups
when no explicit collection is supplied.

For custom construction, pass `optimizer_factory` with the optimizer configuration
already bound. For example, `functools.partial(get_megatron_optimizer, config=cfg.optimizer)`
uses the standard factory, while `cfg.optimizer.provide` preserves an existing custom
config factory. The callable receives exactly `model_chunks`, `config_overrides`,
`use_gloo_process_groups`, and `pg_collection` as keyword arguments. A custom factory
receives the original single model or model list and returns a Megatron-compatible
optimizer with parameter groups, update methods, and checkpoint state. The default
factory normalizes a single model into a list.

`build_optimizer_param_scheduler` exposes the same explicit-config scheduler builder
for an optimizer that already exists. The old
`megatron.training.training.get_optimizer_param_scheduler(optimizer)` entry point is
retained as a wrapper. Scheduler state and optimizer checkpoint formats are unchanged.
