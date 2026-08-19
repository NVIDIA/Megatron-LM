# Operation backends

Model code asks for an implementation once, while it builds a spec, and calls the result
directly from then on:

```python
provider = get_backend_spec_provider(config)
norm_cls = provider.layer_norm(rms_norm=True)   # returns the class; nothing else happens
self.input_layernorm = norm_cls(config=config, hidden_size=..., eps=...)
```

There is no registry lookup, no resolver, and no dispatch in the forward path.

## Two layers

**Central, and it stops changing.** `operations.py` defines what an operation *is*.
`spec_provider.py` defines `BackendSpecProvider`, the one type model code ever holds.
`options.py` lists every setting that can change which implementation you get. `resolve.py`
builds a provider and names no individual backend.

**Per family, and it grows.** Each `megatron/core/ops/<family>/` declares its own operations,
its contract, its backend table, and its default. Adding a backend touches one family;
adding a family is a milestone.

```
megatron/core/ops/norm/
  contract.py               # the contract, the Operation constants, and the slot methods
  __init__.py               # BACKENDS, DEFAULT
  reference.py apex.py megatron.py transformer_engine.py
```

Every backend module exposes one class named for its family — `reference.Norm`,
`transformer_engine.Norm`, `megatron.Linear`, `inference.Moe` — so a table reads as
"this name, that vendor's implementation" and nothing collides with the targets themselves.

A family can split into sub-folders when its operations have disjoint backend sets, which is
how `attention.dsa` will work: `FAMILY = "attention.dsa"` is the module path, so the resolver
finds it with no extra machinery, and DSA can offer `tilelang` without offering it for
`core_attention`.

## How selection works

Three steps, in this order:

1. `--transformer-impl` names a preset. Each family uses its entry for that name, or its own
   `DEFAULT` if it has none.
2. Settings that predate `--op-backend` take over the operations they name.
3. `--op-backend` takes over last.

Two settings claiming the same operation is an error, not a silent win for one of them.

Backend names are scoped to the operation, so the same word means "that vendor's
implementation of whatever you named":

```bash
--op-backend layer_norm=transformer_engine core_attention=transformer_engine
--op-backend layer_norm=apex                       # Apex norm on a Transformer Engine model
--op-backend-config backends.yaml                  # same thing from a file
```

`BackendSpecProvider` binds the owning backend's method onto itself while the model is being
built, so combining backends costs one attribute lookup then and nothing afterwards. A slot no
backend owns raises and says how to select one.

## Adding a backend

Say you want Liger's RMSNorm.

1. Write `megatron/core/ops/norm/liger.py`. Meet the contract in `norm/contract.py`, keep the
   import lazy, and let `_availability.require` produce the error when it is missing:

   ```python
   class Norm:
       """Owns ``layer_norm`` using Liger's fused RMSNorm."""

       REQUIRES = "liger_kernel"

       def layer_norm(self, rms_norm=False, for_qk=False, has_residual=False):
           if not rms_norm:
               raise ValueError("The liger backend implements RMSNorm only")
           from liger_kernel.ops.rms_norm import LigerRMSNorm

           return LigerRMSNorm
   ```

   `REQUIRES` is the whole dependency story. The resolver checks it once, in one place, while
   arguments are parsed — so `--op-backend layer_norm=liger` on a machine without Liger fails
   before anything is built. A backend that needs more than "is the package importable", such
   as a version check, does that in its own constructor.

2. Add one entry to `BACKENDS` in `norm/__init__.py`:

   ```python
   from megatron.core.ops.norm import liger
   ...
   "liger": liger.Norm,
   ```

That is the whole change. Nothing central moves, no call site changes, and
`--op-backend layer_norm=liger` works immediately. `tests/unit_tests/ops/test_conformance.py`
is parametrized off `BACKENDS`, so the new backend is covered the moment it is registered — if
it misses a slot its family declares, that test fails by name.

Backends that need a setting bound up front expose `from_options(options)` instead of a bare
constructor; the loss family's `transformer_engine.Loss` does this for CUDA-graph capture.

## Adding a family

Create `megatron/core/ops/<family>/contract.py` with the contract, the `Operation` constants,
and a `*Slots` class whose methods raise `unowned(...)`. Add `BACKENDS` and `DEFAULT` to the
family's `__init__.py`, list the family in `resolve.py:_FAMILIES`, and add its `*Slots` to the
bases of `BackendSpecProvider`.

Declare an operation only when an existing class, callable, or builder already owns that
boundary at construction time. An implementation that has to branch on shape, phase, or
communication keeps its current owner and exposes a target through `ops` instead.
