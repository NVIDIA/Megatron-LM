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

## Fusing across operations

A backend may fuse whatever it likes. What it may not do is fuse silently: it has to say which
operations it reaches across, and what it needs from each.

`MlpTEOpFuser` is the worked example. It fills `mlp_module`, but folds the linears it is handed
into Transformer Engine operations, and `TEFusedMLP` raises on anything that is not a TE
linear. So it declares:

```python
FUSES = {
    COLUMN_PARALLEL_LAYER_NORM_LINEAR: "transformer_engine",
    ROW_PARALLEL_LINEAR: "transformer_engine",
}
```

`resolve.py` then refuses a configuration the fusion cannot serve — including
`--transformer-impl local --use-transformer-engine-op-fuser`, and
`--op-backend column_parallel_layer_norm_linear=local` alongside the fuser. Both used to build
a spec that died with a type error partway through model construction.

A backend that *replaces* a slot rather than consuming it names itself as the backend for it.

`FUSES` is for a backend that stays inside one family and is picky about its neighbours. A
kernel that swallows operations from *several* families is the next section.

## Megakernels: `ops/fusions/`

A megakernel cannot be a backend in any single family, because it does not answer any single
family's question. It gets a family of its own, one file per fusion — so two vendors adding two
kernels never touch the same file, and neither has to reorganise anything that already exists.

`fusions/attn_moe.py` is a complete worked example, `ReferenceFusedAttentionMoELayer`. It runs
Megatron Core's ordinary attention and experts rather than a kernel, so the mechanism is
testable before any kernel exists and a vendor can see exactly what their own file owes:

```python
class AttnMoeReference:
    REQUIRES = None            # a vendor: "vendor_kernels>=0.4", checked while args are parsed
    DETERMINISM = "unknown"    # a vendor: "nondeterministic", refused by --deterministic-mode
    SPANS = (CORE_ATTENTION, GROUPED_MLP_MODULES)

    def fused_moe_layer(self):
        return ReferenceFusedAttentionMoELayer
```

Selected the same way as anything else, with no new flag:

```
--op-backend fused_moe_layer=attn_moe_reference
```

Three things make an arbitrary span safe to accept:

- **`SPANS` is the whole rule.** Fuse across whatever you like, but list the operations your
  kernel performs itself. `resolve.py:_check_spans` then refuses `--op-backend
  core_attention=...` alongside the fusion, because that backend would never be built — the run
  would look configured and be nothing of the sort. Everything the fusion did *not* swallow
  stays selectable.
- **The target is a `TransformerLayer` subclass**, handed the same `TransformerLayerSubmodules`
  the ordinary layer would have got. Those are specs, not modules, so the parts the kernel
  performs itself are simply never built. Its state dict still has to load a checkpoint the
  ordinary layer wrote, or it is a different model rather than a faster one.
- **`REQUIRES`, `DETERMINISM` and `FUSES` mean what they always mean.** A fusion is a backend
  in a table; it gets the version check, the `--deterministic-mode` refusal, and the
  conformance tests for free.

### One slot per handover point, not per kernel

The slots are named for **where the layer hands over control**, not for what any one kernel
swallows. There are two, matching the two layer specs `get_gpt_decoder_layer_specs` already
builds:

```python
def fused_dense_layer(self) -> Optional[type]   # a dense layer, as one kernel
def fused_moe_layer(self) -> Optional[type]     # an MoE layer, as one kernel
```

That split is what keeps the family from growing per contributor:

- **A kernel that reaches wider needs no new slot.** One that also eats the input norm still
  fills `fused_moe_layer` and just lists `LAYER_NORM` in `SPANS`. The layer hands it the full
  `TransformerLayerSubmodules`; the norm it ignores is never built.
- **An attention-plus-dense-MLP kernel is a new *file*, not a new slot.** It fills
  `fused_dense_layer`, declares `SPANS = (CORE_ATTENTION, MLP_MODULE)`, and can be selected
  *alongside* an attention-plus-MoE kernel from someone else, because the two are separate
  operations. One generic `fusion(*ops)` slot could not do that — `--op-backend` binds one
  backend per operation, so only one fusion could ever be enabled.
- **What does force a new slot** is a kernel needing control somewhere the layer does not hand
  it over — spanning the residual into the *next* layer, say, which is a block-level handover.
  That needs a call site beside these two, which is a milestone rather than a drive-by.

Which is the trade on offer: the kernel is yours to write, the handover point is ours to name.

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
