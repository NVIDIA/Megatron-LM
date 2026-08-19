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
  __init__.py               # the contract, the operations, the slots, BACKENDS, DEFAULT
  backends.py               # every implementation, side by side
```

Two files per family, split by who reads them: `__init__.py` is what a *caller* needs — what
this family promises and which backends exist — and `backends.py` is what a *maintainer* of an
implementation needs. A family becomes a package with sub-packages when its operations have
disjoint backend sets, which is how `attention/dsa/` will work. Each backend class is named for the family and the key that
selects it — `NormApex`, `NormTE`, `LinearLocal`, `MoeTE` — so `--op-backend layer_norm=apex`
and `NormApex` are the same word, and nothing collides with the target classes themselves
(`TENorm` and `TELinear` are real modules in `extensions/transformer_engine.py`).

Backends whose implementations belong to another subsystem live there and are referenced by
name: the inference-optimized ones are in `megatron/core/inference/ops/backends.py`.

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

## Where a choice is made

One selection, end to end, so the hops are written down rather than discovered:

```
--transformer-impl transformer_engine --op-backend layer_norm=apex
  |
  options.py      BackendOptions          every selector, each with its valid values
  resolve.py      _preset_owners          preset -> each family's entry, or its DEFAULT
  resolve.py      _override_owners        legacy flags, then --op-backend, last word
  norm/__init__   BACKENDS["apex"]        -> NormApex
  norm/backends   NormApex.layer_norm     -> megatron.core.fusions.fused_layer_norm

and at runtime, repr(provider) names the backend that won every slot:

  BackendSpecProvider(activation_func=MoeTE, column_parallel_linear=LinearTE,
                      layer_norm=NormApex, ...)
```

Two rules make that traceable:

- **`resolve.py` names no family and no backend.** It asks each family for its table, its
  default, and its `legacy_backends`. Grepping a backend name never lands there.
- **A family's `__init__.py` is the whole story for its operations** — which backends exist,
  which is the default, and how any older flag maps onto them. The cross entropy flags
  (`--cross-entropy-fusion-impl native`) turn into backend names in
  `loss/__init__.py:legacy_backends`, next to the `BACKENDS` table they refer to.

## Adding a backend

Say you want Liger's RMSNorm.

1. Add a class to `megatron/core/ops/norm/backends.py`. Meet the contract in
   `norm/contract.py`, and keep the target import inside the method that returns it — nothing
   in that file may import an optional package at module scope:

   ```python
   class NormLiger:
       """Liger's fused RMSNorm."""

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
   "liger": NormLiger,
   ```

That is the whole change. Nothing central moves, no call site changes, and
`--op-backend layer_norm=liger` works immediately. `tests/unit_tests/ops/test_conformance.py`
is parametrized off `BACKENDS`, so the new backend is covered the moment it is registered — if
it misses a slot its family declares, that test fails by name.

Backends that need a setting bound up front expose `from_options(options)` instead of a bare
constructor; the loss family's `LossTEFused` does this for CUDA-graph capture.

## Adding a family

Create `megatron/core/ops/<family>/` with an `__init__.py` carrying the contract, the
`Operation` constants, a `*Slots` class whose methods raise `unowned(...)`, `BACKENDS` and
`DEFAULT`, plus a `backends.py` with the implementations. Then list the family in
`resolve.py:_FAMILIES` and add its `*Slots` to the bases of `BackendSpecProvider`. If any
older setting selects one of its operations, give it a `legacy_backends(options)` so that
mapping lives beside the table it refers to.

Declare an operation only when an existing class, callable, or builder already owns that
boundary at construction time. An implementation that has to branch on shape, phase, or
communication keeps its current owner and exposes a target through `ops` instead.
