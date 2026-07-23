# Model Layer

Model code lives under `megatron.lite.model`. The layer turns a model-family
declaration into backend-consumable model state by composing validated
primitives; it is not a second runtime.

## Responsibilities

The model layer owns:

- typed architecture configuration;
- construction of model chunks from primitives;
- model-specific forward input/output adaptation;
- recompute and offload placement choices;
- Hugging Face load and export mappings;
- model-specific optimizer wiring when it cannot be expressed generically.

It does not own distributed initialization, the training-step lifecycle,
checkpoint scheduling, or runtime backend selection.

## Protocol Direction

The Lite preview currently uses module-level model protocols. A model protocol
provides these required operations:

```text
ImplConfig
build_model_config(source, **overrides)
build_model(model_cfg, *, impl_cfg)
```

Optional operations include Hugging Face load/export helpers and vocabulary
metadata. This document records the intended boundary; no concrete model or
registry is included in the skeleton PR. The first model PR should upstream the
smallest protocol surface justified by its selected primitives rather than
freezing every preview-only hook here.

## Rules For Adding A Model

1. Declare required features before choosing primitives.
2. Use only primitives with a checkable reference and validation path.
3. Keep model-family imports out of runtime code.
4. Keep heavyweight optional imports inside protocol operations where possible.
5. Add a composition test and an end-to-end runtime validation before delivery.
