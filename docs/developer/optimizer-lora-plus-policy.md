# LoRA+ optimizer policy

With adapters already constructed and the intended base parameters frozen, use
the shared recipe mapping hook:

```python
from megatron.training.config.optimizer_config import LoRAPlusOptimizerOverrides

cfg.optimizer_config_overrides = LoRAPlusOptimizerOverrides(ratio=16.0)
cfg.scheduler.lr_warmup_init = 0.0
```

Adapter A (`*.linear_in.weight`) uses the base optimizer LR. Adapter B
(`*.linear_out.weight`) uses `ratio` times both the peak and minimum LR. With
zero initial warmup LR, this preserves the ratio throughout warmup and decay.
The default extend mode keeps standard WD exclusions and decoupled-LR rules.
Conflicting contributions are rejected by the existing Core mapping checks.
Using the convenience callable together with muP is rejected: that combination
requires an explicit policy for both adapter groups and their interaction with muP.

The callable contributes only B's differing bounds. Ratio one returns an empty
mapping, and A stays in the base group. This avoids creating another group with
the same checkpoint identity as other trainable base weights.

MLM setup rejects unequal A/B bounds selected by exact, name-only standard adapter
globs with a nonzero initial warmup LR. This includes hand-written mappings using
those exact selectors. General globs and selectors combined with attributes or
predicates require validation by their policy; they do not imply LoRA intent.
The check does not depend on rank-local adapter presence. The callable serializes through the existing recipe target mechanism. Its ratio
can be changed through recipe CLI/YAML overrides. Bridge accepts this same
callable through its mapping/callable interface.

This configures optimizer groups only; adapter construction, freezing and adapter
checkpoint filtering remain the responsibility of the PEFT integration. Runtime
LR overrides on resume additionally require the corresponding Core/MLM resume fixes.

## Existing Bridge callers and checkpoints

`get_lora_plus_lr_overrides(config, ratio)` exposes the shared LR calculation as
explicit A and B entries. It does not add standard WD/decoupled-LR rules; callers
compose those with `merge_config_overrides`. An unset `min_lr` is interpreted as
zero, preserving the existing Bridge helper's behavior.

Bridge retains `get_lora_plus_config_overrides` as a thin wrapper around that
helper and the standard rules. This preserves its existing parameter groups,
including the explicit A and B entries at ratio one, so existing checkpoints
keep their group membership and parameter order. Bridge also delegates the
warmup check to `validate_lora_plus_warmup`.

Use the original policy when resuming an existing run. Switching from the legacy
helper to the new callable can change group structure even when the effective
learning rates are identical. This change does not convert checkpoint groups or
introduce a checkpoint format version.
