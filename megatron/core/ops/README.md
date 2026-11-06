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
3. **Selection uses the existing settings.** Operation constructors pass the
   relevant `TransformerConfig` values to a family-local selector and store the
   returned callables. Reuse an existing backend choice when one is available.

## Layout

The shared helpers live at the package root. The family paths below describe
where to place implementations as operations are added or moved:

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

When relocating an operation, preserve its existing state ownership and
checkpoint behavior. Extracting shared state into inference or transformer
infrastructure is a separate change from moving the implementation.

Operations may use shared infrastructure such as embeddings, `MegatronModule`,
checkpoint utilities, inference contexts and explicit process groups. They must
not import concrete model assembly or paths that have been replaced by
deprecated forwarders. Model spec builders may select operation classes; the
operation implementation must not depend on those builders.

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

   Add the matching PyTorch implementation, `torch_chunk_foo`, in
   `foo/reference.py`. Use the same tensor contract and verify its determinism
   before selecting it for deterministic mode.

4. **Add the selector** in `foo/backends.py`. It takes the *values* of existing
   config fields, so its choices can be tested independently of model assembly:

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
   any parameter is allocated; call directly in `forward`:

   ```python
   from megatron.core.ops.ssm.foo.backends import select_foo_recurrence
   from megatron.core.transformer.module import MegatronModule

   class FooMixer(MegatronModule):
       def __init__(self, config):
           super().__init__(config)
           self.recurrence = select_foo_recurrence(
               deterministic=config.deterministic_mode,
           )
           # Allocate any operation parameters after dependency checks.

       def forward(self, q, k, v):
           return self.recurrence(q, k, v)
   ```

6. **Reuse existing settings.** Read relevant fields, such as
   `config.deterministic_mode`, in the operation constructor. Pass their values
   to the selector. A new config field or CLI flag needs a design review.

7. **Wire the spec** in the model's spec builder, using the existing `ModuleSpec`
   interface. The caller supplies `config` when it builds the module:

   ```python
   from megatron.core.ops.ssm.foo.mixer import FooMixer
   from megatron.core.transformer.spec_utils import ModuleSpec

   foo_spec = ModuleSpec(module=FooMixer)
   ```

   The `foo` files and external `fla.ops.foo` in these examples are illustrative:
   implement them with the operation's real tensor contract before wiring it
   into a model. No extension of `BackendSpecProvider` is needed for this recipe.

8. **Tests** (`tests/unit_tests/ops/` and the family's test directory):
   - the selector imports only the selected library (monkeypatch
     `megatron.core.ops._backends.import_module` and assert what was asked for);
   - a missing library fails at construction with a message naming the operation,
     before any parameter is allocated;
   - deterministic and fused choices bind the expected callable once;
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

2. **Extend the family selector.** Add the backend name to the existing config
   value's accepted set. If the operation still selects kernels inline, extract
   that construction-time logic into `backends.py` while preserving its behavior.
   Check the selected adapter with `require`, then return its callables using the
   same interface as the existing backends. Bind that result in the constructor,
   as in the `FooMixer` example above. If the backend needs a minimum version,
   use `require(..., min_version="x.y")`; `require` reads `__version__` first so
   source checkouts work. The family selector and its return type belong to the
   operation being implemented; the shared package does not supply them.

3. **Do not** add a `HAVE_FLASHINFER` flag, a `try/except ImportError` in the
   module, an `if backend == ...` branch in `forward`, or a new CLI option when an
   existing one (`--dsa-kernel-backend`) already names the choice. "auto" never
   turns a new backend on by itself; the user names it.

4. **Kernel-only additions** (a faster implementation of one hook) go into the
   existing adapter module or `kernels/`; the selector picks them by the existing
   backend name, and the operation still binds once.

5. **Tests:** extend the selector tests (the new backend binds
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
  (`is_available`, `has_min_version`) and a bool is
  checked per call.
- Selectors import only what was selected.
- Operation constructors `require` the auxiliary kernels they own (convolution,
  normalization, fused RoPE) separately from the selected recurrence, before
  parameters are allocated.
- Bind inference-only kernels during inference setup, before the first decode
  step, using the operation's existing initialization lifecycle.
- When a module's own imports already fail clearly (the GDP chunkwise-CP adapters
  do), `require(module, needed_by=...)` is the whole check.
- Determinism is not declared per kernel. The existing guards
  (`assert_causal_conv1d_deterministic`, the Torch reference recurrences selected
  by `deterministic_mode`) stay with their owners;
  `docs/developer/determinism` describes what has been audited.

## Selection

An operation constructor calls its family selector with the relevant config
values and stores the returned callables. `forward` calls those functions
directly. This is the pattern used by the new-operation recipe above.

`BackendSpecProvider` in `megatron/core/models/backends.py` supplies the existing
model-spec components, such as parallel linear layers, normalization, attention,
and cross entropy. It does not currently provide SSM or sparse-attention kernel
slots. Keep operation-local kernel selection in the family until an extension
to that provider interface and its callers is implemented together.

Runtime dispatch for input-dependent behavior remains inside the operation. For
example, an adapter hook may return `None` for an unsupported tensor layout if
the operation explicitly supports a reference path for that case. Missing
optional libraries are construction-time errors, not a reason to choose a
reference path silently.

## Deprecated import paths

When moving a module, keep its old path as a forwarder built on
`megatron.core.ops._compat.deprecated_module` and register the mapping in
`tests/unit_tests/ops/deprecated_paths.py`. Paths absent from that table retain
their existing ownership; the table is empty until modules are moved.

A forwarder provides the following compatibility behavior:

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
source directory. Preserve them when moving an operation, and use the
old-to-new table to drive the migration-specific regression tests.

## Tests

`tests/unit_tests/ops/test_deprecated_imports.py` tests the compatibility helper
with synthetic modules, independently of any migration entries. It covers
warnings, lazy imports, private and wildcard exports, multiple targets, and
package child imports. The same file checks registered deprecated paths,
canonical class identity and pickle round trips, import-path resolution, and
in-tree use of canonical paths as operations move.

Add selector and dependency-check tests alongside the family that uses them.
Family behavior tests live under `tests/unit_tests/ssm/` and
`tests/unit_tests/transformer/`.
