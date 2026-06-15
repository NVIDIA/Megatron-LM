# Pipeline-Parallel Trace Correlation

When pipeline parallel size > 1, each rank normally creates an independent trace per iteration — making cross-stage causality invisible in Jaeger. Megatron solves this by broadcasting rank 0's trace context to all ranks and linking each stage's receive operation to that context.

The generic primitives (`broadcast_trace_context`, `create_linked_span`) live in lens; see
[lens: distributed tracing](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/user-guide/distributed-tracing.md)
for how they work.

This page covers Megatron's specific integration.

## The problem

Without correlation:

```
Rank 0 trace (trace_id: AAA):
  megatron.train_step
    megatron.microbatch.forward [microbatch 0]
    megatron.p2p.send_forward

Rank 1 trace (trace_id: BBB):    # different trace!
  megatron.p2p.recv_forward
  megatron.microbatch.forward [microbatch 0]
```

Rank 0 sends a tensor to rank 1, but there's nothing in the trace that shows this relationship. Jaeger treats them as unrelated activities.

## The solution

At the start of every `megatron.train_step`, rank 0's trace context is broadcast to all ranks via `torch.distributed`. Each non-first pipeline stage then emits a `megatron.pp.recv_forward.linked` span with an **OTel Link** (not parent-child) to the broadcast context.

### After integration

```
Rank 0 trace (trace_id: ABC):
  megatron.train_step
    megatron.microbatch.forward [microbatch 0]
    megatron.p2p.send_forward

Rank 1 trace (same trace_id: ABC):
  megatron.pp.recv_forward.linked [LINK to rank 0's train_step context]
  megatron.p2p.recv_forward
  megatron.microbatch.forward [microbatch 0]

Rank 2 trace (same trace_id: ABC):
  megatron.pp.recv_forward.linked [LINK to rank 0's train_step context]
  ...
```

All PP stages of a given step share a single `trace_id`. Each stage's `megatron.pp.recv_forward.linked` shows a link back to the broadcast context — visible in Jaeger's span detail panel.

## Where the integration lives

### Broadcast — `megatron/training/training.py`

Inside the main training loop, just before the `megatron.train_step` managed_span opens:

```python
# Broadcast rank 0's trace context to all PP ranks.
# Collective — all ranks must participate.
_pp_carrier = None
try:
    from megatron.core import parallel_state
    if (torch.distributed.is_initialized()
            and parallel_state.get_pipeline_model_parallel_world_size() > 1
            and get_telemetry() is not None):
        from nemo.lens.distributed import broadcast_trace_context
        from nemo.lens.state import set_pp_trace_carrier
        _pp_carrier = broadcast_trace_context(
            rank=torch.distributed.get_rank(), src_rank=0
        )
        set_pp_trace_carrier(_pp_carrier)
except Exception:
    _pp_carrier = None
```

### Linked spans — `megatron/core/pipeline_parallel/schedules.py`

Inside `forward_backward_pipelining_without_interleaving`, wrapping the warmup and pre-1F1B `recv_forward` calls:

```python
# OTel: create linked span to sender's trace context on recv_forward.
_recv_link_span = None
if (not is_pp_first_stage(p2p_communicator.pp_group)
        and _otel_sg_enabled('communication')
        and _otel_create_linked_span is not None):
    _carrier = _otel_get_pp_carrier()
    if _carrier is not None:
        from opentelemetry import trace as _otel_trace_mod
        _recv_link_span = _otel_create_linked_span(
            _otel_trace_mod.get_tracer('nemo.lens'),
            'megatron.pp.recv_forward.linked',
            remote_carrier=_carrier,
            **{'dl.pipeline_parallel.rank': rank, 'dl.microbatch_id': i},
        )
input_tensor = p2p_communicator.recv_forward(...)
if _recv_link_span is not None:
    _recv_link_span.end()
```

## What's instrumented vs what isn't

| recv site | Instrumented? |
|---|---|
| Warmup `recv_forward` (lines ~2166 in `schedules.py`) | Yes |
| Pre-1F1B `recv_forward` (line ~2196) | Yes |
| Steady-state 1F1B `send_forward_recv_backward` | No — high frequency; instrumenting every hop would dominate signal |
| Cooldown `recv_backward` | No — redundant with warmup coverage |

The warmup and pre-1F1B receives establish the pipeline structure for Jaeger viewing. Steady-state and cooldown receives are skipped to keep the per-step span count bounded — the link from rank N's warmup recv to rank 0's context is enough to visualise the pipeline wave.

## Cost

- `broadcast_trace_context`: one `torch.distributed.broadcast` of a small carrier payload (length int64 + ~200 bytes), runs once per step (not per microbatch).
- `megatron.pp.recv_forward.linked` spans: only created when the `communication` span group is enabled AND PP > 1 AND telemetry was initialised.

Does not scale with number of microbatches — one broadcast per step, a handful of linked spans per step.

## Collective correctness

`broadcast_trace_context` is a **collective operation** — every rank must call it or the job deadlocks. The gate uses `get_telemetry() is not None`, which is uniformly true or false across ranks (telemetry was initialised in `_set_telemetry` on all ranks, regardless of whether they export).

Do not gate on `handle.is_exporting` — that differs per rank and would cause some ranks to call broadcast while others skip it.

## Disabling

Set `MEGATRON_OTEL_SPAN_GROUPS=default` (which excludes `communication`) to skip the linked spans. The broadcast still happens (cheap), but the spans aren't emitted.

Alternatively, run with PP size 1 — the broadcast is gated on PP > 1 and won't fire.

To disable both: `MEGATRON_OTEL_ENABLED=0` — `get_telemetry()` returns `None`, the broadcast is skipped entirely.

## Viewing in Jaeger

1. Open the Jaeger UI (`:16686` in the local stack).
2. Search for traces with `service.name=megatron-lm` and `nemo.run.id=<your-run-id>`.
3. Click a `megatron.train_step` trace.
4. In the waterfall, find a `megatron.pp.recv_forward.linked` span (only present on non-first PP ranks).
5. Click the span → the detail panel shows a "Links" section with a clickable reference to the upstream span context.

Jaeger renders the link as a visible reference rather than a parent-child edge. This is the correct representation: the stages are concurrent, not hierarchical.

## Why links instead of parent-child

Parent-child implies sequential dependency: "the parent was running, spawned this child, then continued." Pipeline-parallel stages run concurrently — stage 1 doing forward on microbatch N doesn't "spawn" stage 2's forward on microbatch N-1; they happen at the same time with a tensor exchange between them.

Links model "these are related" without implying temporal ordering. This is the correct shape for concurrent distributed work.

See [lens: distributed tracing](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/user-guide/distributed-tracing.md) for a deeper discussion.
