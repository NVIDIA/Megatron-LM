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
