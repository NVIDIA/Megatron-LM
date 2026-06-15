# Extending Instrumentation

To add new spans or metrics to Megatron code, use the instrumentation primitives from `nemo.lens`. The primitives themselves are documented in
[lens: instrumentation](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/user-guide/instrumentation.md).
This page covers Megatron conventions.

## Adding a custom span

### Simple block — `span_cm`

```python
from megatron.training.global_vars import get_telemetry
from nemo.lens.helpers import span_cm

telemetry = get_telemetry()
if telemetry is not None:
    with span_cm("megatron.my_custom_op", tracer=telemetry.tracer, param_count=1e9):
        ...  # your code
```

`span_cm` always creates a span when telemetry is active — good for cold paths.

### Group-gated block — `managed_span`

For hot paths where you want minimal cost when the group is disabled:

```python
from megatron.core.telemetry.span_groups import MegatronSpanGroup
from nemo.lens.helpers import managed_span

with managed_span(MegatronSpanGroup.STEP, "megatron.my_custom_step",
                  iteration=iteration) as span:
    result = do_work()
    if span is not None:
        span.set_attribute("megatron.my_custom.result", result)
```

`managed_span` yields `None` when the group is disabled; the body still runs. Check `if span is not None` before setting attributes.

### Fallback pattern

Every import of lens in Megatron code must use the try/except fallback idiom so the code runs when lens isn't installed:

```python
try:
    from nemo.lens.helpers import managed_span as _otel_managed_span
    from nemo.lens.state import is_span_group_enabled as _otel_sg_enabled
except ImportError:
    from megatron.core.telemetry._fallbacks import managed_span as _otel_managed_span
    from megatron.core.telemetry._fallbacks import is_span_group_enabled as _otel_sg_enabled
```

`megatron/core/telemetry/_fallbacks.py` re-exports from `nemo.lens.fallbacks` when lens is installed, otherwise provides inline no-ops.

See [lens: optional dependency](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/design/optional-dependency.md).

## Naming conventions

| Kind | Convention | Example |
|---|---|---|
| Span name | `megatron.<subsystem>.<op>` | `megatron.train_step`, `megatron.microbatch.forward` |
| Span attribute | `megatron.<attr>` for Megatron-specific, `dl.<attr>` for shared | `megatron.iteration`, `dl.rank` |
| Resource attribute | `megatron.<attr>` for Megatron-specific, `dl.<attr>` for shared, standard for host/SLURM/K8s | `megatron.num_layers`, `dl.tensor_parallel.size`, `host.name` |
| Metric name | `megatron.<subsystem>.<metric>` | `megatron.training.loss` |

Use the constants in `nemo.lens.semconv` when a name is shared across consumers (`DL_RANK`, `NEMO_RUN_ID`, etc.). For Megatron-specific names, hard-coded strings are fine — they're short and grep-able.

## Choosing a span group

When adding a new span, decide which group it belongs to:

- **Always want it in production?** → `job` (very rare outside setup spans)
- **Once per iteration?** → `step`
- **Inside the forward/backward?** → `forward_backward` or `microbatch`
- **Inside the optimizer?** → `optimizer`
- **Related to checkpointing?** → `checkpoint`
- **Related to evaluation?** → `evaluate`
- **Cross-rank communication?** → `communication`
- **Inference request path?** → `inference`

Don't invent a new group unless no existing group fits — new groups add to `MegatronSpanGroup` and require preset updates.

## Adding a new span group

If you do need a new group:

1. Edit `megatron/core/telemetry/span_groups.py`:

    ```python
    class MegatronSpanGroup(SpanGroup):
        # ... existing groups ...
        MY_NEW_GROUP = "my_new_group"

        ALL_GROUPS = frozenset([*SpanGroup.ALL_GROUPS, ..., MY_NEW_GROUP])

        _PRESETS = {
            "default": frozenset([SpanGroup.JOB, SpanGroup.CHECKPOINT, SpanGroup.EVALUATE,
                                  INFERENCE]),   # typically don't add new groups to default
            "per_step": frozenset([...]),         # add to per_step if it's per-iteration
            "all": ALL_GROUPS,                    # always in all
        }
    ```

2. Document it in [Span Groups](span-groups.md) with the spans it controls and typical frequency.

3. Update dashboard queries if the group introduces new metric labels.

## Adding a metric

For domain-specific metrics, add a module under `megatron/core/instruments/` following the `nemo.lens.instruments.inference` pattern:

```python
# megatron/core/instruments/training.py
import weakref
from opentelemetry import metrics

_INSTRUMENTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

def _get_instruments(meter: metrics.Meter) -> dict:
    instruments = _INSTRUMENTS.get(meter)
    if instruments is None:
        instruments = {
            "my_new_metric": meter.create_histogram(
                name="megatron.training.my_new_metric_ms",
                unit="ms",
                description="...",
            ),
        }
        _INSTRUMENTS[meter] = instruments
    return instruments

def record_training_metrics(meter, *, my_new_value_ms=None, ...):
    i = _get_instruments(meter)
    if my_new_value_ms is not None:
        i["my_new_metric"].record(my_new_value_ms)
```

Call `record_training_metrics(meter=handle.meter, my_new_value_ms=42.0)` only on the export rank (check `handle.is_exporting`).

See [lens: metrics](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/user-guide/metrics.md) for the pattern rationale.

## Testing new instrumentation

Megatron's telemetry tests live at `tests/unit_tests/telemetry/` and use the fixture pattern from lens's `conftest.py` (global OTel state reset before/after each test).

When adding a span:

1. Add a test in `tests/unit_tests/telemetry/` that asserts the span is emitted when its group is enabled and absent when disabled.
2. Use `InMemorySpanExporter` (from lens's `conftest.py`, shared via `sys.path` or a test utility) to capture spans.
3. Assert on span name, attributes, and parent relationships.

See [lens: testing](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/developer/testing.md) for fixture patterns.

## When not to add instrumentation

- **Inside a tight inner loop** (per-token, per-parameter). Even `managed_span`'s frozenset lookup adds up over trillions of invocations.
- **On code that runs on all ranks with unbounded cardinality**. If the span attribute includes something like a tensor shape with high variance, you get cardinality explosion at the backend.
- **As a replacement for logging**. Structured logs belong in logs (and can be correlated via the [log bridge](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/user-guide/logging-bridge.md)). Spans describe bounded operations, not every interesting event.

When in doubt, start with a coarse span at the boundary of the subsystem, not a fine-grained one at every internal call.
