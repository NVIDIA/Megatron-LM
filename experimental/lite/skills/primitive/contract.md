# Primitive Contract

Primitive work should start with a typed contract before adding optimized code.

For each primitive, document:

- Inputs and outputs.
- Shape and dtype expectations.
- Distributed assumptions.
- Fallback behavior.
- Validation against a simple PyTorch reference.

This PR includes only the shared contract layer; concrete primitives are
follow-up work.
