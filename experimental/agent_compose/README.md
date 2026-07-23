# Agent Compose (experimental)

Agent Compose is the upstream home for incrementally reviewed Megatron Lite
components. It makes Megatron-LM development agentic-native by composing
Megatron Core primitives with coding agents, rather than introducing a new
standalone product or training stack.

`experimental/agent_compose` is the project and review location. The Python
package keeps the public namespace `megatron.lite`; `experimental.agent_compose`
is not an import path.

## Main And Dev

The complete work-in-progress implementation remains on the `dev` branch under
[`experimental/lite`](https://github.com/NVIDIA/Megatron-LM/tree/dev/experimental/lite).
Code is promoted from that preview one independently validated primitive at a
time.

| Surface | `main` | `dev` |
| --- | --- | --- |
| Project tree | `experimental/agent_compose` | `experimental/lite` |
| Role | Reviewed upstream subset | Work-in-progress superset |
| Python namespace | `megatron.lite` | `megatron.lite` |

The upstream package has no runtime dependency on the preview tree. Do not add
both source roots to the same `PYTHONPATH`; select the tree from the branch being
tested.

## Architecture

The initial package establishes three layers:

- `primitive`: replaceable lower-level components built from Megatron Core.
- `model`: model declarations and composition from validated primitives.
- `runtime`: lifecycle and training orchestration through model protocols.

Dependencies flow from runtime to model to primitive. Model and primitive code
may use an explicitly stable runtime contract, but they must not import runtime
backends. Primitive code must remain model-agnostic.

```text
experimental/agent_compose/
  README.md
  docs/
    architecture.md
    model.md
  megatron/
    lite/
      primitive/
      model/
      runtime/
  skills/
    basic/
    primitive/
    model/
    runtime/
  tests/
    unit/
```

For local source-tree use:

```bash
export PYTHONPATH=/path/to/Megatron-LM/experimental/agent_compose:$PYTHONPATH
```

The skeleton exposes the stable runtime interface and shared runtime contracts,
but contains no built-in runtime backend, model, or primitive implementation.
Those implementations will be added in separate reviewable PRs.

```python
from megatron.lite.runtime import Runtime, RuntimeConfig, create_runtime, register_runtime
```

Backends subclass `Runtime` and register a module-level factory before
`create_runtime` is called. The skeleton intentionally registers no built-in
backend.

## Documentation

- [Three-layer architecture](docs/architecture.md)
- [Runtime interface](docs/runtime.md)
- [Model layer and protocol](docs/model.md)

## Skills

`skills/` contains agent-agnostic operational contracts. The initial skills set
the global constraints and the minimum contract for each architecture layer.
Each primitive PR should add or update the corresponding leaf skill together
with its reference and validation path.

## Principles

- **Compose, don't fork.** Reuse Megatron Core wherever appropriate. Document
  why a separate implementation is necessary when reuse is not possible.
- **Reviewable by construction.** Keep runtime, model, and primitive contracts
  small enough to review independently.
- **Reference before implementation.** Every implementation needs a checkable
  Megatron, Hugging Face, Torch, or first-principles reference.
- **Core performance.** Validate accepted code against Megatron Core for both
  correctness and speed where applicable.

## Status

The package and skill boundaries are established here. Implementations will
land incrementally; use the `dev` preview for surfaces not yet present on
`main`.
