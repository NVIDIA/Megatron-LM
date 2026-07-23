# Three-Layer Architecture

Agent Compose upstreams Megatron Lite as three reviewable layers under the
public `megatron.lite` namespace.

## Layers

### Primitive

`megatron.lite.primitive` owns reusable lower-level components: parallel
operations and state, modules, checkpoint conversion, optimizer integration,
and focused math or kernel shims. A primitive must be independently selectable
and validated. It may build on Megatron Core, but it must not know model family
names or runtime backend implementations.

### Model

`megatron.lite.model` owns model-family configuration and the protocol that
composes validated primitives into model chunks. It also owns model-specific
checkpoint mappings and forward adaptation. It does not own the training loop
or distributed runtime lifecycle.

### Runtime

`megatron.lite.runtime` owns the backend-neutral lifecycle: model construction,
mode changes, forward/backward microbatch orchestration, checkpoint dispatch,
optimizer and scheduler steps, weight export, and optional device offload.
Concrete backends implement the public `Runtime` interface.

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
- reviewed code never imports the `dev/experimental/lite` preview at runtime.

The runtime skill and human review additionally require runtime code to remain
model-family agnostic; that semantic rule cannot be fully expressed as an
import-prefix check before model families exist in this tree.

## Composition Flow

1. A runtime resolves a model protocol without importing model-family details
   into the runtime layer.
2. The model protocol selects validated primitives and constructs model chunks.
3. The protocol returns backend-consumable model state through shared contracts.
4. The runtime drives training without reaching into model internals.

The current skeleton exposes the runtime interface and shared contracts. Model,
primitive, and backend implementations will land in separate PRs.
