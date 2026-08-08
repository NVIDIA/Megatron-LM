# Primitive Design

Primitive implementations should be independent units that can be tested without
bringing up a full model.

Design rules:

- Keep primitive configuration explicit.
- Prefer pure PyTorch references for first validation.
- Add optimized kernels only after the reference behavior is covered.
- Avoid hidden dependencies on global distributed state.

If a primitive needs process groups, pass them through config or bundle metadata
instead of importing runtime internals.
