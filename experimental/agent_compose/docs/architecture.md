# Three-Layer Architecture

Agent Compose provides three reviewable layers under the public
`megatron.experimental.agent_compose` namespace.

## Layers

### Primitive

`megatron.experimental.agent_compose.primitive` owns reusable lower-level
components: parallel operations and state, modules, checkpoint conversion,
optimizer integration, and focused math or kernel shims. A primitive must be
independently selectable and validated. It may build on Megatron Core, but it
must not know model family names or runtime backend implementations.

### Model

`megatron.experimental.agent_compose.model` owns model-family configuration and
the protocol that composes validated primitives into model chunks. It also owns
model-specific checkpoint mappings and forward adaptation. It does not own the
training loop or distributed runtime lifecycle.

### Runtime

`megatron.experimental.agent_compose.runtime` owns the backend-neutral
lifecycle: model construction, mode changes, forward/backward microbatch
orchestration, checkpoint dispatch, optimizer and scheduler steps, weight
export, and optional device offload. Concrete backends implement the public
`Runtime` interface.

## Dependency Direction

```text
runtime orchestration -> model protocol -> primitive -> Megatron Core
          |                    |
          +---- runtime contracts <----+
```

`runtime.contracts` is the shared boundary surface, not a runtime backend.
Model and primitive code may import these stable data types. They must not
import `runtime.backends` or backend implementation modules.

The static layering test enforces import direction:

- primitive does not import model;
- primitive and model do not import runtime implementation code;
- reviewed code never imports development-preview source at runtime.

The runtime skill and human review additionally require runtime code to remain
model-family agnostic; that semantic rule cannot be fully expressed as an
import-prefix check before model families exist in this tree.

## Composition Flow

1. A runtime resolves a model protocol without importing model-family details
   into the runtime layer.
2. The model protocol selects validated primitives and constructs model chunks.
3. The protocol returns backend-consumable model state through shared contracts.
4. The runtime drives training without reaching into model internals.

## Incubation Lifecycle

1. Prototype and iterate on `dev`.
2. Promote a vertically complete, referenced, and validated slice into Agent
   Compose.
3. Validate correctness, composition, end-to-end behavior, and performance for
   the declared scope.
4. Select the long-term owner: Megatron Core for reusable primitives and
   backend-neutral interfaces, Megatron Bridge or Automodel for training
   orchestration, or the owning project for integration-specific behavior.
5. Switch Agent Compose to consume the graduated implementation and demonstrate
   parity.
6. Remove the duplicate incubating implementation only after the Agent Compose
   path is complete and runnable with the graduated capability.

The current skeleton exposes the runtime interface and shared contracts. Model,
primitive, and backend implementations will land in separate PRs.
