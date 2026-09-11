# SSM and Sparse-Attention Kernels

This package owns kernel implementations and their optional dependency adapters.
It does not own model parameters, inference state, checkpoints, context-parallel
communication, or index-sharing state. Those remain with the model/SSM callers.
Importing a family namespace does not import its optional kernel libraries.

## Families

| Location | Contents |
| --- | --- |
| `ssm/common` | Causal convolution, determinism and intermediate extraction |
| `ssm/mamba2` | Mamba SSD training, varlen and decode kernels |
| `ssm/gated_delta` | GDN/GDN2 reference recurrences and FLA adapters |
| `ssm/gdp` | GDP chunk, recurrent, decode and backend adapters |
| `attention/dsa` | DSA layout/masking, indexer loss, reference and fused adapters |
| `attention/dsa/kernels` | TileLang/Triton kernel implementations |
| `attention/csa` | CSA indexing, masks, compressor pooling and sparse attention |

The family docstrings and callable signatures describe tensor layouts, masks,
state ownership and distributed inputs. Moving a kernel does not certify its
determinism or change its supported dtypes, layouts or numerical tolerances.
Existing determinism guards and backend-specific restrictions still apply.

## Selection

The existing `BackendSpecProvider` has optional `dsa_kernels`, `gated_delta_rule`
and `gated_delta_product` slots. Local and TE providers preserve the same existing
family defaults. `backend_slot` supplies those defaults for older providers.
There is no additional registry or new CLI option.

DSA/GDN spec builders preserve an explicit provider through the `kernel_backend`
constructor argument. Direct construction resolves the provider from config when
that argument is omitted; custom GDP specs can supply it too.

DSAttention captures concrete hooks in an immutable `DSAKernels` object at
construction. Its forward does not resolve a backend or consult the legacy
module-global selection cache. A hook can still return `None` for unsupported
runtime inputs; reference fallback remains the caller's responsibility. Changing
the backend setting after construction requires rebuilding the module.

GDN/GDN2 and GDP likewise bind the existing selected recurrence once. Mamba's
phase-specific entry points remain direct calls; a provider is not needed for
every helper. No extra callable wrapper is inserted into these kernel calls.

## Compatibility

The former `ssm/ops`, `ssm/triton_cache_manager` and sparse-attention helper
module paths remain importable. Leaf modules alias the canonical module object,
so caches and Triton module state are not duplicated. References extracted from
model files are explicitly re-exported under their old names. New production
imports should use this package. Tests patching an implementation dependency
must patch its canonical owner, not a re-exported name in a model module.

Vendored kernel files retain their original licenses and internal file structure.
Future changes should keep low-level computation here and stateful model or
distributed orchestration in its existing owner.
