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

## Metadata and Initialization Checks

Each family exports an immutable `KERNELS` inventory from its lightweight
`kernel_metadata.py`. It describes selectable implementations and public
phase-specific entry points, not every private JIT helper. The inventory is for
inspection and conformance tests, not registration or forward dispatch.

Every declaration uses the same template:

- `name`: operation and implementation identifier.
- `requires`: direct optional dependencies, their actual Python import paths,
  required exports, and version bounds where known. Core project dependencies
  such as PyTorch remain in `pyproject.toml`; this is not another lockfile.
- `determinism`: an explicit tri-state assessment and its scope/reason.
- `contract`: the family docstring describing layouts, modes and ownership.
- `determinism_check`: an optional construction-time environment assessment.

After choosing the implementation, the provider or model owner calls
`validate_kernel`, or `validate_kernels` for a group of selected entry points:

```python
from megatron.core.ops.kernel_metadata import DeterminismPolicy, validate_kernels
from megatron.core.ops.ssm.gated_delta.kernel_metadata import GDN_TORCH

validate_kernels(
    (GDN_TORCH,),
    features=("qk_l2norm",),
    determinism=DeterminismPolicy.WARN,
)
```

Do not pass the entire family inventory: unselected backends must not become
requirements. Checks import the declared module and verify its required exports,
not just package presence. Versioned dependencies also require installed
distribution metadata. Import and native-library loading failures retain their
cause and identify the kernel. Checks do not install packages, compile kernels,
allocate model state or create process groups.

DSA/GDN/GDP selectors check their chosen recurrence/hooks. SSM constructors check
their selected auxiliary kernels. `MambaInferenceStateConfig.from_model` checks
additional prefill/decode targets when dynamic inference is initialized, so a
training-only GDN does not need recurrent-inference exports. Older/custom mixers
without `get_inference_kernel_metadata` retain their existing initialization;
their dependencies are not certified by these checks. Custom providers likewise
own validation for their custom targets.

Ordinary execution uses `IGNORE`. Existing deterministic-mode paths use `WARN`:
unknown implementations warn, known nondeterministic ones fail, and existing
stricter model guards still apply. `ERROR` is an explicit strict helper policy
that also rejects unknown implementations; it does not silently change the
meaning of the global configuration flag. No implementation is certified merely
because it is written in Torch or passes metadata conformance tests.

Input-dependent constraints stay at execution: shape/dtype support, packed
layouts, index validity, device-specific fallbacks, and conditional features not
known at construction. For example, direct GDN reference calls still check for
the FLA normalization helper if Q/K normalization is requested later. Numerical,
gradient, graph-capture and scoped repeatability tests remain separate from
metadata validation.

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

The `megatron/core/ssm` and `transformer/experimental_attention_variant` folders
remain model owners, not obsolete kernel folders. Removing compatibility leaf
modules is a separate API-deprecation decision; deleting their parent folders
would also delete live model and distributed code.
