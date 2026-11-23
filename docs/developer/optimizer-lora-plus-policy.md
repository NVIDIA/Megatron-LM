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

MLM setup rejects unequal A/B bounds selected by the standard adapter name globs
with a nonzero initial warmup LR, including hand-written mappings using those
globs. Arbitrary layer-specific selectors require validation by their provider.
The check does not depend on rank-local adapter presence. The callable serializes through the existing recipe target mechanism
and can also be supplied to Bridge's shared mapping/callable interface.

This configures optimizer groups only; adapter construction, freezing and adapter
checkpoint filtering remain the responsibility of the PEFT integration. Runtime
LR overrides on resume additionally require the corresponding Core/MLM resume fixes.
