# Megatron Core Operations (`megatron/core/ops`)

`megatron/core/ops` is the implementation home for Megatron Core's operation
families — the modules, kernels and backend adapters behind SSM mixers, sparse
attention and, over time, the other fused operations. It is organized by
operation, not by vendor or library.

This is an operation-implementation package, not just a kernel directory. It
owns concrete operation modules, kernels, backend adapters, operation-local
parameters and checkpoint mappings, state updates, and operation-specific
communication. Model assembly and global runtime management stay outside.


Three rules hold everywhere in this package:

1. **Choose once, call directly.** An operation binds its kernels in `__init__`
   and calls them directly in `forward`. No registry, no dispatch wrapper, no
   availability flag or optional import in the forward path.
2. **A named backend is supported or fails early.** Missing optional libraries
   are reported at construction, naming the operation, with the original error
   chained. Nothing falls back silently to another implementation.
3. **Selection is configured once, from the existing settings.** Operation modules
   ask the `BackendSpecProvider` for their kernels; the provider was configured
   from `TransformerConfig` when it was built. No new CLI flags for choosing
   kernels.

## Layout

```
megatron/core/ops/
├── __init__.py            package docstring only; imports nothing optional
├── _backends.py           require(): construction-time optional-dependency checks
├── _compat.py             deprecated_module(): forwarders for moved import paths
├── ssm/
│   ├── common/            causal_conv1d_cp, packing, checkpointing, inference mixin
│   ├── mamba2/            mixer.py, context_parallel.py, SSD kernels
│   ├── gated_delta/       common.py, gdn.py, gdn2.py
│   ├── gdp/               mixer.py, context_parallel.py, kernels
│   └── context_parallel/  chunkwise CP protocol and the FLA / CuTeDSL GDP adapters
└── attention/
    ├── dsa/               modules.py, layout/masking, kernels/, TileLang & cuDNN adapters
    ├── csa/               modules.py
    ├── mla.py             absorbed MLA
    └── dsv4.py            DeepSeek-v4 hybrid attention
```

A family gains `backends.py` and `reference.py` when it adopts the selector pattern
below; until then its module carries the availability checks it had before the move.

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

Within a family the file roles are fixed:

- `__init__.py` — the **contract**: a docstring describing tensor layouts, masks,
  state ownership and distributed inputs, plus any typing `Protocol` for the
  kernel interface. Importing it must not load optional libraries.
- `backends.py` — the **selectors**: `select_<op>(...)` functions that turn the
  existing config values into concrete callables, using `require()`. This is the
  only place in the family that decides between implementations.
- `modules.py` / `mixer.py` — the **operation**: the `nn.Module` that owns
  parameters and state, binds kernels in `__init__`, and calls them in `forward`.
- `kernels/`, `*_kernels.py` — **implementations and adapters**: in-tree Triton /
  TileLang kernels and thin adapters over external libraries. An adapter imports
  its library at module top and fails with a clear `ImportError` if it is absent.
- `reference.py` — eager PyTorch reference paths used for parity tests and
  deterministic mode.

## Ownership boundary

An implementation may own state when it belongs to that operation: parameters,
local buffers, checkpoint sharding, or communication using supplied process
groups. Merely subclassing `nn.Module` does not put an implementation outside
`ops`. Existing operation boundaries are preserved; relocation adds no wrapper.

The following remain outside:

- Model/layer assembly, hybrid allocation, configuration and spec builders
  (`transformer/`, `models/`).
- Inference contexts, global cache allocation and request scheduling (`inference/`).

This move retains `SSMChunking` and `ssm_chunking` in `ops.ssm.common.inference`,
and indexer-loss tracking in `ops.attention.dsa.modules`. Extracting that stack-wide
state into inference/transformer infrastructure is separate work.

Operations may use shared infrastructure such as embeddings, `MegatronModule`,
checkpoint utilities, inference contexts and explicit process groups. They must
not import concrete model assembly or the deprecated SSM/attention module paths.
The provider API is used only at construction; no reverse dependency on model
spec builders is introduced.

Moving a kernel does not certify its determinism or change its supported dtypes,
layouts or numerical tolerances. Existing determinism guards and backend-specific
restrictions still apply.

## Adding a new operation

Say you are adding a gated linear recurrence called `foo`.

1. **Create the family directory**, under the closest existing family or a new
   one: `megatron/core/ops/ssm/foo/`.

2. **Write the contract first**, in `foo/__init__.py`: what the operation
   computes, tensor layouts (`[B, T, D]`, THD packing, …), which masks and
   `PackedSeqParams` fields it honors, what state it owns across decode steps,
   and which process groups it uses. If backends must share a call signature,
   define it as a `typing.Protocol` here. Keep this module free of `torch`
   kernels and optional imports so tooling can import it cheaply.

3. **Add the kernels or adapters.** In-tree Triton/TileLang kernels go in
   `foo/kernels/`. An adapter over an external library (`foo_fla_kernels.py`)
   imports that library at the top of the file:

   ```python
   # megatron/core/ops/ssm/foo/foo_fla_kernels.py
   from fla.ops.foo import chunk_foo  # raises ImportError if FLA is missing

   def run_chunk_foo(q, k, v, *, cu_seqlens=None):
       return chunk_foo(q, k, v, cu_seqlens=cu_seqlens)
   ```

4. **Add the selector** in `foo/backends.py`. It takes the *values* of existing
   config fields, not the config object, so a provider can be configured once:

   ```python
   # megatron/core/ops/ssm/foo/backends.py
   from megatron.core.ops._backends import require

   def select_foo_recurrence(deterministic: bool = False):
       """Return the callable the mixer will bind; never falls back silently."""
       if deterministic:
           from megatron.core.ops.ssm.foo.reference import torch_chunk_foo
           return torch_chunk_foo
       return require("fla.ops.foo", "chunk_foo", needed_by="Foo recurrence").chunk_foo
   ```

5. **Write the operation module** in `foo/mixer.py`. Bind in `__init__`, before
   any parameter is allocated, through the provider; call directly in `forward`:

   ```python
   from megatron.core.models.backends import backend_slot, resolve_kernel_backend
   from megatron.core.ops._backends import require
   from megatron.core.ops.ssm.foo.backends import select_foo_recurrence

   class FooMixer(MegatronModule):
       def __init__(self, config, submodules, d_model, *, pg_collection=None,
                    kernel_backend=None):
           super().__init__(config)
           provider = resolve_kernel_backend(kernel_backend, config)
           self.recurrence = backend_slot(
               backend=provider, name="foo_recurrence",
               default=lambda: select_foo_recurrence(config.deterministic_mode),
           )
           # auxiliary kernels the operation owns, independent of the recurrence
           self.causal_conv1d = require(
               "causal_conv1d", "causal_conv1d_fn", needed_by="Foo convolution"
           ).causal_conv1d_fn
           ...  # parameters are created after the checks above

       def forward(self, hidden_states, ...):
           x = self.causal_conv1d(...)
           return self.recurrence(q, k, v, ...)   # no selection, no checks here
   ```

   `backend_slot` asks the provider for `foo_recurrence()` and, if the provider
   predates the slot, falls back to `default()`. `resolve_kernel_backend` uses a
   spec-injected provider when there is one and otherwise derives one from
   `config`. Where a family has not adopted the provider slots yet, its module calls its
   selector directly.

6. **Add the provider slot** only if a different provider could reasonably answer
   it (an existing callable or builder already owns that boundary). Add a typed,
   argument-free method to `BackendSpecProvider` and implement it in
   `KernelSelectionMixin` in `megatron/core/models/backends.py`, reading its
   settings from `KernelSelection`. If only one implementation will ever exist,
   skip the slot and call the selector directly.

7. **Reuse existing settings.** Deterministic mode, memory-efficient paths and
   backend names are already `TransformerConfig` fields; read them through
   `KernelSelection.from_config`. A new field or CLI flag needs a design review.

8. **Wire the spec.** The module spec builder passes the provider into the module
   (`params={"kernel_backend": backend}`) so a user-supplied provider is honored.
   Module-level specs assembled without a config leave it out; the module then
   derives the provider from `config`.

9. **Tests** (`tests/unit_tests/ops/` and the family's test directory):
   - the selector imports only the selected library (monkeypatch
     `megatron.core.ops._backends.import_module` and assert what was asked for);
   - a missing library fails at construction with a message naming the operation,
     before any parameter is allocated;
   - a custom provider's answer is bound without the default being imported;
   - numerical parity between the fused kernel and `reference.py`, and the
     determinism guard if the operation has one;
   - the canonical-ownership test (`test_deprecated_imports.py`) passes.

What does **not** go in `ops`: the layer that wraps the mixer (`transformer/`),
the layer-config dataclass, hybrid allocation, module specs (`models/`), and
inference cache allocation (`inference/`).

## Adding a new backend kernel to an existing operation

Say DSA gains a third fused backend, `flashinfer`, next to `tilelang` and `cudnn`.

1. **Write the adapter** as a sibling of the existing ones,
   `attention/dsa/dsa_flashinfer_kernels.py`, implementing the same hook
   signatures the operation already calls (`run_fused_qk_topk`,
   `run_fused_dsa_attention`, …). Import the library at module top. A hook may
   return `None` for inputs it does not support; the caller then runs the
   reference path — that is the only permitted fallback, and it is per call, not
   per installation.

2. **Extend the selector**, not the module. Add the backend name to the existing
   config value's accepted set and to the selector in `attention/dsa/backends.py`:

   ```python
   _NATIVE_REQUIREMENTS["flashinfer"] = (("flashinfer", ("sparse_attention",)),)

   def select_dsa_kernels(backend: str, *, fused: bool = True) -> DSAKernels:
       ...
       for module, symbols in _NATIVE_REQUIREMENTS[backend]:
           require(module, *symbols, needed_by=f"DSA kernel backend {backend!r}")
       adapter = require(backend_module_name(backend), needed_by=...)
       return DSAKernels(backend=backend, run_fused_qk_topk=getattr(adapter, ...), ...)
   ```

   The module (`DSAttention`) does not change: it already binds whatever
   `dsa_kernels()` returns. If the backend needs a minimum version, say so with
   `require(..., min_version="x.y")`; `require` reads `__version__` first so
   source checkouts work.

3. **Do not** add a `HAVE_FLASHINFER` flag, a `try/except ImportError` in the
   module, an `if backend == ...` branch in `forward`, or a new CLI option when an
   existing one (`--dsa-kernel-backend`) already names the choice. "auto" never
   turns a new backend on by itself; the user names it.

4. **Kernel-only additions** (a faster implementation of one hook) go into the
   existing adapter module or `kernels/`; the selector picks them by the existing
   backend name, and the operation still binds once.

5. **Tests:** extend the selector tests (`select_dsa_kernels("flashinfer")` binds
   the hook set once; a missing library is an `ImportError` naming the backend;
   `"none"`/`unfused` never import it), add numerical parity against
   `reference.py` for the new hooks, and run the existing DSA suites with the new
   backend name.

## Dependency checks

Optional kernel libraries are checked once, at construction, with
`megatron.core.ops._backends.require`:

```python
from megatron.core.ops._backends import require

ssd = require("mamba_ssm.ops.triton.ssd_combined", "mamba_chunk_scan_combined", needed_by="Mamba2")
self.scan = ssd.mamba_chunk_scan_combined
```

`require(module, *symbols, min_version=None, dist=None, needed_by=...)` imports
the module, checks that each named export exists and is not `None` (dotted names
reach into lazy namespaces such as `cudnn.DSA`), optionally checks a minimum
version (`module.__version__` first, then distribution metadata), and returns
the module. Every failure is an `ImportError` naming the operation that asked —
including a native extension that is installed but fails to load — with the
original error chained. `ModuleNotFoundError` keeps its type and `name`.

Rules:

- `require` is construction-time only. A capability that depends on
  execution-time input — packed sequences under CP, say — is decided once
  (`is_available`, `has_min_version`, `packed_cp_conv_supported`) and a bool is
  checked per call.
- Selectors import only what was selected.
- Operation constructors `require` the auxiliary kernels they own (convolution,
  normalization, fused RoPE) separately from the provider-owned kernel, before
  parameters are allocated.
- Inference-only kernels are bound by `bind_dynamic_inference_kernels`, which
  dynamic-inference setup calls on every pipeline-local mixer, so a missing
  library fails there rather than in the first decode step.
- When a module's own imports already fail clearly (the GDP chunkwise-CP adapters
  do), `require(module, needed_by=...)` is the whole check.
- Determinism is not declared per kernel. The existing guards
  (`assert_causal_conv1d_deterministic`, the Torch reference recurrences selected
  by `deterministic_mode`, `CSA_OPERATION_DETERMINISM`) stay with their owners;
  `docs/developer/determinism` describes what has been audited.

## Selection

`BackendSpecProvider` is the only construction API. A provider is configured once,
when it is built, from the existing config fields collected in
`megatron.core.models.backends.KernelSelection` (`deterministic_mode`,
`use_mamba_mem_eff_path`, `gdp_cutedsl_kernel`, `gdp_num_chunk_states_to_recompute`,
`dsa_kernel_backend`, `attention_backend`). The kernel slots take no
implementation-selection arguments:

| Slot | Returns | Bound by |
| --- | --- | --- |
| `mamba_kernels()` | `MambaKernels` (scan, optional fused conv+scan, conv) | `MambaMixer` |
| `gated_delta_rule(variant)` | GDN or GDN2 recurrence; `variant` names the operation, not the backend | `GatedDeltaNet`, `GatedDeltaNet2` |
| `gated_delta_product()` | FLA or CuTeDSL chunked gated delta product | `GatedDeltaProductMixer` |
| `gated_delta_product_cp_backend()` | chunkwise-CP adapter matching the GDP kernel | `GatedDeltaProductMixer` when CP > 1 |
| `dsa_kernels()` | immutable `DSAKernels` hook set (or none) | `DSAttention` |

Local and TE providers share one implementation of these slots
(`KernelSelectionMixin`): TE has no SSM or sparse-attention kernels of its own,
and sharing the body keeps every slot overridable by a partial provider that does.
`backend_slot` supplies the family default for providers written before a slot
existed. There is no registry and no new CLI option.

Every operation module accepts a `kernel_backend` provider from its module spec
(`params={"kernel_backend": provider}`) and resolves it with
`resolve_kernel_backend(kernel_backend, config)`:

- A provider built with a selection (`get_backend_from_config`, or
  `kernels=KernelSelection(...)`) is used as is; the explicit selection wins even
  where it disagrees with `config`.
- A bare provider (`TESpecProvider()`) is configured from `config` at bind time, on
  a shallow copy, so the existing per-operation settings still decide the kernels
  and a provider shared across a spec is never mutated. Asking a bare provider for
  a kernel slot directly is an error, never a silent default.
- A wrapper or custom provider without the mixin is used untouched; `backend_slot`
  supplies the family default for slots it does not implement. Build wrappers
  through `get_backend`/`get_backend_from_config` so the fallback they delegate to
  is configured; a wrapper around a bare fallback fails loudly on a kernel slot.
- Specs assembled without a config — the module-level hybrid stack specs — cannot
  inject a provider, so those modules derive one from `config` through the same
  `get_backend_from_config` path the spec builders use; both routes select
  identically.

Kernels are bound once, in `__init__`, and called directly from `forward`. No
selection, availability check or optional import happens in the forward path.
DSAttention's hooks may still return `None` for unsupported runtime inputs, in
which case the caller runs the reference implementation. GDN dynamic inference uses
FLA's fused decode/prefill family (which takes `A_log`/`dt_bias` and fuses the
gates, so it is not interchangeable with the training recurrence); it is bound once
through `bind_dynamic_inference_kernels`.

## Deprecated import paths

The former `megatron.core.ssm` and
`megatron.core.transformer.experimental_attention_variant` module paths are
deprecated, not removed. Every pre-move module still exists as a two-line
forwarder built on `megatron.core.ops._compat.deprecated_module`:

- Importing an old path emits one `DeprecationWarning` naming the replacement.
- Attributes resolve lazily through PEP 562 module `__getattr__`, so importing the
  old path does not import the implementation or its optional kernel libraries.
- `from old import *`, private names and pickles that recorded the old
  `__module__` keep working, and every object is the canonical one.
- The forwarders are scheduled for removal in the version recorded by
  `_compat.REMOVAL_VERSION`. In-tree code must use canonical paths; a unit test
  enforces this.

These are read-through import aliases. Assigning attributes on a deprecated module
(including monkeypatching a kernel function) does not update the canonical module's
globals. Patches must target the canonical path. Module `__file__` also names the
forwarder; use the canonical module when inspecting implementation source.

Ordinary state-dict keys and checkpoint tensor mappings do not depend on the
source directory and are unchanged. The full old-to-new table is
`tests/unit_tests/ops/deprecated_paths.py`. The main entries, relative to
`megatron.core`:

| Former owner | Canonical owner |
| --- | --- |
| `ssm.mamba_mixer`, `ssm.gated_delta_product`, `ssm.gated_delta_net` | `ops.ssm.mamba2.mixer`, `ops.ssm.gdp.mixer`, `ops.ssm.gated_delta` |
| `ssm.ops.{common,mamba2,gdp}` | `ops.ssm.{common,mamba2,gdp}` |
| SSM CP, packing and checkpoint helpers | `ops.ssm` operation families and `ops.ssm.common` |
| Experimental DSA/CSA, absorbed MLA and DeepSeek-v4 attention | `ops.attention.{dsa,csa}.modules`, `ops.attention.mla`, `ops.attention.dsv4` |
| Experimental DSA kernel adapters and helpers | `ops.attention.dsa` and `ops.attention.dsa.kernels` |
| `ssm.mamba_layer`, `ssm.mlp_layer` and their layer-config classes | `transformer.mamba_layer`, `transformer.mlp_layer` and `transformer.*_layer_config` |
| Experimental `dsa_layer_config` | `transformer.dsa_layer_config` |
| Experimental `deepseek_v4_hybrid_attention_module_specs` | `models.gpt.deepseek_v4_hybrid_attention_module_specs` |
| `ssm.ssm_inference.SSMChunking` and `ssm_chunking` | `ops.ssm.common.inference` |
| `ssm.ssm_inference.SSMDynamicInferenceMixin` | `ops.ssm.common.inference` |
| `ssm.mamba_block`, `ssm.mamba_hybrid_layer_allocation` | `models.hybrid.hybrid_block`, `models.hybrid.hybrid_layer_allocation` |

## Tests

`tests/unit_tests/ops/` holds the package-level tests. In this PR:
`test_deprecated_imports.py` (canonical module/class ownership and pickle round
trips, import-path validation, the deprecated-path forwarders, and the absence
of deprecated imports in the tree). Selector, provider and `require` tests join it as families adopt the
selection pattern. Family behaviour tests live with their family
under `tests/unit_tests/ssm/` and `tests/unit_tests/transformer/`.
