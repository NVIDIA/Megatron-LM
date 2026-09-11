# SSM and Sparse-Attention Operations

This is an operation-implementation package, not just a kernel directory. It
owns concrete operation modules, kernels, backend adapters, operation-local
parameters and checkpoint mappings, state updates, and operation-specific
communication. Model assembly and global runtime management stay outside.

Import family namespaces for contracts and metadata without loading optional
kernel libraries. Import concrete implementation modules explicitly when
constructing an operation. Moving an implementation does not change its
parameter names, registered submodules, checkpoint layout or numerical behavior.

## Families

| Location | Contents |
| --- | --- |
| `ssm/common` | Convolution, packing, checkpoint helpers and per-operation inference execution |
| `ssm/mamba2` | `mixer.py`, context-parallel transforms, SSD training and inference kernels |
| `ssm/gated_delta` | GDN/GDN2 modules, reference recurrences and FLA adapters |
| `ssm/gdp` | `mixer.py`, context-parallel transforms, training adapters and inference kernels |
| `ssm/context_parallel` | Chunkwise SSM communication and GDP backend implementations |
| `attention/dsa` | `modules.py`, layout/masking, indexer loss, reference and fused adapters |
| `attention/dsa/kernels` | TileLang/Triton kernel implementations |
| `attention/csa` | `modules.py` with compressor/indexer/attention modules and reference kernels |
| `attention/mla.py` | Absorbed MLA operation and its projection/layout helpers |
| `attention/dsv4.py` | DeepSeek-v4 hybrid attention operation |

## Ownership Boundary

An implementation may own state when it belongs to that operation: parameters,
local buffers, checkpoint sharding, or communication using supplied process
groups. Merely subclassing `nn.Module` does not put an implementation outside
`ops`. Existing operation boundaries are preserved; relocation adds no wrapper.

The following remain outside:

- Model/layer assembly, hybrid allocation, configuration and spec builders.
- Inference contexts, global cache allocation, request scheduling and stack-wide
  recurrent-state configuration (`ssm/ssm_inference.py`).
- Training-wide indexer-loss tracking and gradient-scale management
  (`transformer/dsa_loss.py`).

Operations may use shared infrastructure such as embeddings, `MegatronModule`,
checkpoint utilities, inference contexts and explicit process groups. They must
not import concrete model assembly or legacy SSM/attention compatibility paths.
The existing provider API is used only at construction; no new reverse
dependency on model spec builders is introduced.

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

`KERNELS` describes kernel entry points, not a blanket dependency or determinism
certificate for the operation modules that compose them. An operation validates
its selected targets; adding a module to this package does not certify all its
training, communication or inference behavior.

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

Former kernel and operation-module paths remain importable. Leaf modules alias
the canonical module object, so classes, caches, monkeypatches and Triton state
are not duplicated. Package exports and the two split ownership modules keep
explicit identity-preserving re-exports. Historical pickle globals still resolve
through those paths; new pickles record the canonical implementation path.
State-dict keys do not depend on the source directory and remain unchanged.
Production and numerical tests use canonical paths; dedicated compatibility
tests exercise legacy imports separately.

Vendored kernel files retain their original licenses and internal file structure.
Future changes should keep cohesive operation implementations here and model
assembly or global runtime lifecycle in its existing subsystem.

The `megatron/core/ssm` and `transformer/experimental_attention_variant` folders
now retain assembly/configuration code and compatibility imports, rather than
the migrated operation implementations. Removing compatibility paths is a
separate API-deprecation decision, not part of the implementation relocation.
