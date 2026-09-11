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
  recurrent-state configuration (`inference/ssm_config.py`).
- Training-wide indexer-loss tracking and gradient-scale management
  (`transformer/dsa_loss.py`).

Operations may use shared infrastructure such as embeddings, `MegatronModule`,
checkpoint utilities, inference contexts and explicit process groups. They must
not import concrete model assembly or retired SSM/attention module paths.
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
  required exports (including dotted lazy-namespace attributes), and version bounds
  where known. Core project dependencies such as PyTorch remain in `pyproject.toml`;
  this is not another lockfile.
- `determinism`: an explicit tri-state assessment and its scope/reason.
- `contract`: the family docstring describing layouts, modes and ownership.
- `determinism_check`: an optional construction-time environment assessment.

Keep each declaration self-contained: write its `Dependency` entries, determinism
assessment and contract directly, using named fields. Do not assemble requirements
from shared tuples, another kernel's `.requires`, or `dataclasses.replace`.
Repeating a dependency is intentional: changing one implementation's requirements
must not silently change another's. Share validation logic, not declaration data.

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

For example, DSA validates concrete TileLang entry points and cuDNN's lazy
`DSA` wrapper exports, not just importable `tilelang` or `cudnn` packages. A
missing implementation is an initialization error; runtime shape/layout refusal
remains the documented reference-fallback case.

DSA/GDN/GDP selectors check their chosen recurrence/hooks before importing the
concrete target. Mamba binds its scan targets and their supported arguments once
at construction. GDP validates its chunkwise-CP adapter only when that CP path is
selected. SSM constructors check convolution, normalization, and other selected
auxiliary kernels before allocating parameters. Optional normalization must not
require its dependency when disabled, and a custom recurrence must not be gated
by the default recurrence's dependency or determinism declarations.

`MambaInferenceStateConfig.from_model` checks
additional prefill/decode targets when dynamic inference is initialized, so a
training-only GDN does not need recurrent-inference exports. Older/custom mixers
without `get_inference_kernel_metadata` retain their existing initialization;
their dependencies are not certified by these checks. Custom providers likewise
own validation for their custom targets. The `gated_delta_product` provider slot
accepts `deterministic` as well as `use_cutedsl`, so the selector applies the
requested policy to its own target.

Do not duplicate these checks with module-level `HAVE_*` flags, placeholder
implementations, or constructor assertions. Concrete optional implementations,
including the Mamba/GDP normalization modules, are imported after validation.
Mamba's ordinary scan preserves its historical Torch-convolution fallback when
`causal-conv1d` is absent. A broken installation or missing export fails instead;
the memory-efficient path and GDP require the external convolution. Training
does not require the separate CUDA decode-update export. Convolution determinism
uses the existing reduction guard through metadata rather than a second copy of
the environment and version checks in each mixer.
Low-level JIT import scaffolding and TE/GTP class checks are distinct from
operation backend selection; runtime-only requirements, such as TE's packed
THD partition helper, are checked when the packed input is first known. See the
[contribution rules](../../../docs/developer/contribute.md#kernel-backend-selection).

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
bound phase-specific entry points remain direct calls; a provider is not needed for
every helper. No extra callable wrapper is inserted into these kernel calls.

## Import Migration

The former `megatron.core.ssm` and
`megatron.core.transformer.experimental_attention_variant` packages are removed.
Import implementations from their canonical owners; there are no compatibility
files, `sys.modules` aliases or replacement import hooks for the retired paths.
This is an intentional breaking change to Python imports and serialized objects
that record those paths. Historical pickles referring to the removed modules
are not supported. Ordinary state-dict keys and checkpoint tensor mappings do
not depend on the source directory and remain unchanged.

Paths below are relative to `megatron.core`:

| Former owner | Canonical owner |
| --- | --- |
| `ssm.mamba_mixer`, `ssm.gated_delta_product`, `ssm.gated_delta_net` | `ops.ssm.mamba2.mixer`, `ops.ssm.gdp.mixer`, `ops.ssm.gated_delta.modules` |
| `ssm.ops.{common,mamba2,gdp}` | `ops.ssm.{common,mamba2,gdp}` |
| SSM CP, packing and checkpoint helpers | `ops.ssm` operation families and `ops.ssm.common` |
| Experimental DSA/CSA, absorbed MLA and DeepSeek-v4 attention | `ops.attention.{dsa,csa}.modules`, `ops.attention.mla`, `ops.attention.dsv4` |
| Experimental DSA kernel adapters and helpers | `ops.attention.dsa` and `ops.attention.dsa.kernels` |
| `ssm.mamba_layer`, `ssm.mlp_layer` and their layer-config classes | `transformer.mamba_layer`, `transformer.mlp_layer` and `transformer.*_layer_config` |
| Experimental `dsa_layer_config` | `transformer.dsa_layer_config` |
| Experimental `deepseek_v4_hybrid_attention_module_specs` | `models.gpt.deepseek_v4_hybrid_attention_module_specs` |
| `ssm.ssm_inference.SSMChunking` and `ssm_chunking` | `inference.ssm_config` |
| `ssm.ssm_inference.SSMDynamicInferenceMixin` | `ops.ssm.common.inference` |
| `ssm.mamba_block`, `ssm.mamba_hybrid_layer_allocation` | `models.hybrid.hybrid_block`, `models.hybrid.hybrid_layer_allocation` |

Update launch-script module strings as well as Python imports. In particular,
the cache-manager setting is now:

```bash
export TRITON_CACHE_MANAGER=megatron.core.ops.ssm.triton_cache_manager:ParallelFileCacheManager
```

Tests cover canonical module/class ownership and pickle round trips, construction
import order, absence of retired source files and stale runtime references.

Vendored kernel files retain their original licenses and internal file structure.
Future changes should keep cohesive operation implementations here and model
assembly or global runtime lifecycle in its existing subsystem.
