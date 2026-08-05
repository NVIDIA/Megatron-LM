# Agent Compose (experimental)

Agent Compose is an experimental incubation surface for incrementally reviewed
Megatron capabilities. It makes Megatron-LM development agentic-native by
combining Megatron Core primitives with coding agents, rather than maintaining
a fork or introducing a standalone training stack.

`experimental/agent_compose` is the project and review location. The Python
package uses the public namespace `megatron.experimental.agent_compose`;
`experimental.agent_compose` is not an import path.

## Main And Dev

The complete work-in-progress implementation remains in `experimental/lite` on
the `dev` branch, while `experimental/agent_compose` on `main` is its
independently reviewed upstream incubation surface. Code is promoted from the
development preview one vertically complete and independently validated slice
at a time.

| Surface | `main` | `dev` |
| --- | --- | --- |
| Project tree | `experimental/agent_compose` | Development preview |
| Role | Reviewed upstream subset | Work-in-progress superset |
| Python namespace | `megatron.experimental.agent_compose` | Preview-local |

The upstream package has no runtime dependency on the preview tree. Do not add
both source roots to the same `PYTHONPATH`; select the tree from the branch being
tested.

## Incubation

Incomplete prototypes remain on `dev`. Changes promoted to `main` must be
functionally complete and validated for their declared scope.

Successful, reusable capabilities should graduate to their long-term owners:

- reusable primitives and backend-neutral interfaces to Megatron Core;
- training orchestration to Megatron Bridge or Automodel;
- integration-specific behavior to the owning integration project.

Agent Compose first switches to the graduated implementation and validates
parity. Duplicate incubating code is removed only after that transition, so the
Agent Compose path remains coherent and runnable.

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
    experimental/
      agent_compose/
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

The skeleton exposes the initial runtime interface and shared runtime contracts,
but contains no built-in runtime backend, model, or primitive implementation.
Those implementations will be added in separate reviewable PRs.

```python
from megatron.experimental.agent_compose.runtime import Runtime, RuntimeConfig, create_runtime, register_runtime
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
