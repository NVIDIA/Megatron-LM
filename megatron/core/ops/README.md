# Operation backends

Model code asks for an implementation once, while it builds a spec, and calls the result
directly from then on:

```python
provider = get_backend_spec_provider(config)
norm_cls = provider.layer_norm(rms_norm=True)   # returns the class; nothing else happens
self.input_layernorm = norm_cls(config=config, hidden_size=..., eps=...)
```

There is no registry lookup, no resolver, and no dispatch in the forward path.

## The pieces

| File | Holds |
| --- | --- |
| `operations.py` | `Operation`: the set of methods a backend can own. Each value *is* the method name. |
| `spec_provider.py` | `BackendSpecProvider`, the protocol, and `compose()`, the only way backends combine. |
| `options.py` | `BackendOptions`: every setting that can change which implementation you get. |
| `resolve.py` | The backend name table and the single construction path. |
| `<operation>/<backend>.py` | The implementations for one operation, plus the backend that owns it. |
| `<operation>/__init__.py` | The contract that family's implementations have to meet. |
| `providers/` | Backends that implement the whole protocol and can serve as a base. |

## How selection works

Exactly three steps, in this order:

1. `--transformer-impl` picks the base backend, which answers every operation.
2. Settings that predate `--op-backend` (today, the cross entropy fusion flags) take over the
   operations they name.
3. `--op-backend` takes over last.

Two settings claiming the same operation is an error, not a silent win for one of them.

## Combining backends

A backend is any object implementing one or more provider methods. A backend that implements
all of them can be a base; one that implements a subset can take over just those operations:

```bash
# Transformer Engine everywhere, except normalization, which comes from Apex
--transformer-impl transformer_engine --op-backend layer_norm=apex
```

or from a file, which is the same thing:

```yaml
# backends.yaml, passed as --op-backend-config backends.yaml
layer_norm: apex
vocab_parallel_cross_entropy: te_cross_entropy
```

`compose()` attaches the owning backend's bound method to the provider, so combining backends
costs one attribute lookup while the model is built and nothing at all afterwards.

## Adding a backend

Say you want Liger's RMSNorm.

1. Write `megatron/core/ops/norm/liger.py`. Meet the contract in `norm/__init__.py`, keep the
   import lazy, and let `_availability.require` produce the error when it is missing:

   ```python
   from megatron.core.ops import _availability

   class LigerNormBackend:
       """Owns ``layer_norm`` using Liger's fused RMSNorm."""

       def __init__(self) -> None:
           _availability.require("liger_kernel", backend="liger")

       def layer_norm(self, rms_norm=False, for_qk=False, has_residual=False):
           if not rms_norm:
               raise ValueError("The liger backend implements RMSNorm only")
           from liger_kernel.ops.rms_norm import LigerRMSNorm

           return LigerRMSNorm
   ```

2. Add one entry to `_BACKENDS` in `resolve.py`:

   ```python
   "liger": _liger_norm,
   ```

3. Add its targets to the family's equivalence test in
   `tests/unit_tests/ops/test_backend_targets.py`.

That is the whole change. No existing branch moves, no call site changes, and
`--op-backend layer_norm=liger` works immediately. A backend pointed at an operation it does
not implement is rejected while the model is built, with the operation and the backend named.

## Adding an operation

Add an `Operation` member whose value is the new provider method name, declare the method on
`BackendSpecProvider`, and give it a default there if most backends should share one. Add a
member only when an existing class, callable, or builder already owns that boundary at
construction time — an implementation that has to branch on shape, phase, or communication
keeps its current owner and exposes a target through `ops` instead.
