<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Extending Instrumentation

Megatron call sites use the local telemetry facade:

```python
from megatron.core.telemetry import telemetry as _otel
```

`megatron/core/telemetry/telemetry.py` is the only Megatron module that imports
Lens instrumentation helpers or registers Megatron groups. It provides no-op
helpers when Lens is absent. An installed but incompatible Lens fails visibly
instead of silently disabling instrumentation.

## Group-gated operations

Choose an existing group from [Span Groups](span-groups.md). Keep the operation
itself outside telemetry enablement checks so disabled telemetry cannot skip
application work.

```python
with _otel.managed_span(_otel.DETAIL, "example.operation") as span:
    result = do_work()
    _otel.set_attributes(span, {"example.result": result})
```

The example names are placeholders, not new Megatron signals. Use the canonical
schema for production names and keep shared names in Lens semantic-convention
constants. Megatron-owned names and helpers belong in its telemetry facade.

`set_attributes` tolerates absent/nonrecording spans and contains attribute
assignment failures. Do not call raw `set_attribute` or `set_attributes` at
instrumented application sites. Time-varying numeric signals, immutable
Resource identity, and span attributes have distinct roles; do not copy a
process Resource map onto every span.

Argument expressions are evaluated before a context manager is entered.
Place expensive telemetry-only work behind `_otel.is_enabled(group)`;
never add tensor reads, synchronization, or collectives only to populate
attributes. Expensive or distributed application work must remain unchanged
when telemetry is off.

## Adding a group

Add the namespaced constant, `GROUPS` member, and any justified preset
membership in `telemetry.py`. Update group documentation and tests together.
Do not define a local `all` preset: Lens supplies the process-wide wildcard.
Adding a group to `default` expands production instrumentation and requires
explicit review.

## Resource ownership

Megatron maps trainer arguments and composes launcher identity in
`resource_attrs.py`, using Lens's shared parsing, composition, conversion,
and detector helpers. Preserve the documented [precedence](configuration.md#resource-attributes).
