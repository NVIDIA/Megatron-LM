# Megatron Lite Architecture

Megatron Lite is organized around three narrow contracts:

1. Runtime backends own execution.
2. Model implementations own model construction.
3. Primitive packages expose reusable building blocks through explicit
   dataclasses and protocols.

The first PR keeps those layers intentionally small. The only executable backend
is the local `mlite` backend, and the only model is `toy_dense`.

## Package Layers

`megatron.lite.runtime` provides `RuntimeConfig`, `create_runtime`, and
`register_runtime`. A backend implements the `Runtime` abstract base class and
returns `ModelHandle` objects from `build_model`.

`megatron.lite.model.registry` maps a public model name and implementation name
to a protocol module. Runtime backends use the registry instead of importing
model packages directly.

`megatron.lite.primitive` contains only interface-level objects in this slice:
`ModelBundle`, primitive config dataclasses, and protocol type definitions.
Concrete distributed primitives are deliberately out of scope.

## Review Boundary

The package lives under `experimental/lite` and is imported by adding that
directory to `PYTHONPATH`. This keeps the experimental API isolated from the
main Megatron package until the interface is stable enough to promote.
