# Megatron Lite Tests

The PR1 validation suite is intentionally local and CPU-only. It verifies that
the package imports, the registries resolve the toy model, and the local runtime
can run one dense-model training step.

Future primitive parity tests should be added here as real primitives land.
