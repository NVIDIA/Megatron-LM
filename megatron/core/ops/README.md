# `megatron/core/ops`

Where the implementation of every operation lives, organized by operation.

This is an implementation home, **not** a selection layer. There is no registry and no
resolver here, and nothing in this package decides which backend a run gets. That decision
belongs to the `BackendSpecProvider` implementations:

| provider | lives in |
| --- | --- |
| `LocalSpecProvider`, `InferenceSpecProvider` | `megatron/core/models/backends.py` |
| `TESpecProvider`, `KitchenSpecProvider` | `megatron/core/extensions/transformer_engine_spec_provider.py` |

## Layout

Two files per family:

```
<family>/__init__.py    the contract every backend for this family must meet
<family>/backends.py    the backends themselves, side by side
```

Backends are named for the family and the implementation — `NormTE`, `NormApex`, `NormTorch`,
`LinearTE`, `MoeLocal` — so the class name alone says what it is.

They sit side by side in one file on purpose: a family's backends answer the same question and
are read by comparison. Splitting them one-per-file would also mean importing one imports its
optional package, which is the thing this package exists to avoid.

## The one rule

**Every optional-package import goes inside the method that returns its target**, never at
module scope:

```python
class NormTE:
    REQUIRES = "transformer_engine"
    DETERMINISM = "unknown"

    def layer_norm(self, rms_norm=False, for_qk=False, has_residual=False):
        from megatron.core.extensions.transformer_engine import TENorm   # here, not above
        return TENorm
```

So importing `megatron.core.ops` pulls in no optional dependency and no backend nobody
selected. `tests/unit_tests/ops/test_import_hygiene.py` pins it.

That is what lets a provider name a backend without the *call site* having to guard on whether
it is installed — which is what the scattered `HAVE_TE` / `HAVE_APEX` flags were doing.

## Tracing which implementation a run gets

Open the provider. `TESpecProvider.layer_norm` names `NormTE`; `NormTE.layer_norm` names
`TENorm`. Two hops, no globals, no dependence on import order or on what happens to be
installed.

## Adding a backend

Say you want Liger's RMSNorm.

1. Add a class to `norm/backends.py`, meeting the contract in `norm/__init__.py`:

   ```python
   class NormLiger:
       """Liger's fused RMSNorm."""

       REQUIRES = "liger_kernel"
       DETERMINISM = "unknown"

       def layer_norm(self, rms_norm=False, for_qk=False, has_residual=False):
           if not rms_norm:
               raise ValueError("The liger backend implements RMSNorm only")
           from liger_kernel.ops.rms_norm import LigerRMSNorm

           return LigerRMSNorm
   ```

2. Have a provider return it — either a new provider, or an existing one under a flag.

`REQUIRES` and `DETERMINISM` are declarations a provider can check at construction:
`REQUIRES` names the optional package, optionally with a version (`"transformer_engine>=1.13"`);
`DETERMINISM` is `deterministic`, `nondeterministic`, or `unknown`, so that "nobody audited
this" cannot read as "safe" under `--deterministic-mode`.
