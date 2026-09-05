<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Rank-local determinism tracing

Rank-local determinism traces help locate the first semantic boundary at which
two otherwise identical distributed training runs diverge. The tracer writes
one append-only JSONL stream per global rank and never introduces a distributed
collective, so a diagnostic cannot change collective ordering.

This tool complements [deterministic training](deterministic-training.md):
deterministic mode constrains execution, while determinism tracing explains
where two executions first stopped matching.

## Start with low-cost evidence

Enable lifecycle and loss tracing on every rank:

```bash
python pretrain_gpt.py \
  --determinism-trace-dir /traces/run-a \
  --determinism-trace-mode metadata \
  --determinism-trace-ranks all \
  --determinism-trace-iterations 20-40 \
  <other args ...>
```

The training loop records iteration start/result events, reported losses,
gradient norm, and zero-gradient count. Iteration selectors are 1-indexed,
matching user-visible training iteration numbers. Rank and iteration selectors
accept comma-separated values and inclusive ranges, such as
`0,4-7` and `20,24-28`.

Each rank creates `rank_XXXXXX.jsonl`. Existing files cause the run to fail
rather than overwrite evidence. Use `--determinism-trace-append` only to
continue an intentionally preserved stream; the tracer validates existing
records and continues sequence and occurrence counters.

## Evidence modes

The configured mode applies to tensor records:

| Mode | Evidence | Interpretation |
|---|---|---|
| `metadata` | shape, stride, dtype, layout, device type | Structure only; tensor values are not read |
| `summary` | metadata, finite status, min, max, mean, L2 norm | Low-volume value diagnostics; not an equality certificate |
| `sampled` | metadata and SHA-256 over deterministic evenly spaced elements | Narrows large tensors at bounded copy volume; unsampled values can differ |
| `full` | metadata and SHA-256 over every tensor byte | Exact tensor equality certificate for the captured boundary |

`full` hashes all bytes but does not store the tensor payload. It is intended
to establish equality or locate divergence without duplicating model-sized
data. A matching sampled or summary record is diagnostic evidence, not proof
that the full tensors are equal.

Value-bearing modes snapshot the selected data when the record is created and
move it to the host when the trace is flushed. This adds device work and a
synchronization at flush time. Start with metadata, narrow the rank/iteration
window, then increase evidence strength around the first mismatch. Do not call
summary, sampled, or full capture inside CUDA graph capture; metadata mode is
safe because it does not read tensor values.

## Instrument an internal boundary

The global trace is available to model code after training initialization:

```python
from megatron.training.determinism_trace import get_determinism_trace


def forward(self, hidden_states):
    trace = get_determinism_trace()
    if trace is not None:
        trace.record_tensor(
            "decoder.layer0.input",
            hidden_states,
            iteration=self.current_iteration,
            phase="forward",
            metadata={"layer": 0},
        )
    ...
```

Use a stable, semantic name rather than an object id or memory address. Include
`microbatch` when the same boundary executes for multiple microbatches. Repeated
records with the same semantic key receive an occurrence counter, making loops
and recomputation distinguishable without relying on timestamps.

For a localized investigation, the tracer can also be constructed directly:

```python
from megatron.training.determinism_trace import RankLocalTrace, TraceConfig

trace = RankLocalTrace(
    TraceConfig(
        output_dir="/traces/run-a",
        rank=global_rank,
        mode="sampled",
        sample_count=512,
        append=False,
        rank_spec="0,4",
        iteration_spec="24-26",
    )
)
```

Call `flush()` at a safe boundary and `close()` during shutdown. The context
manager form closes automatically.

## Compare two runs

Compare rank streams offline:

```bash
python tools/determinism/compare_traces.py \
  /traces/run-a \
  /traces/run-b \
  --output /traces/comparison.json
```

Exit codes are:

- `0`: all semantic events and evidence match
- `1`: a missing event, content mismatch, or event-order mismatch was found
- `2`: either trace is malformed or violates the schema

The report identifies the first divergence by rank, iteration, microbatch,
phase, event name, and occurrence. Sequence numbers are validated for strict
monotonicity but compared separately from semantic identity, so one missing
record does not make every later record appear to be a content mismatch.

`match_strength` describes what a successful comparison establishes:

- `structure_only`: all tensor records are metadata-only
- `diagnostic_tensor_match`: at least one summary or sampled record is present
- `full_tensor_certificate`: every tensor record hashes every byte
- `event_match`: the traces contain semantic events but no tensor records

## Operational guidance

- Store different runs in different output directories.
- Preserve the training command, checkpoint identity, source revision, and
  environment alongside each trace. Matching traces do not compensate for
  different inputs.
- Compare the same selected rank and iteration set on both sides.
- Escalate capture strength only after a lower-cost pass has narrowed the
  boundary.
- Treat abrupt process termination as an incomplete trace. Explicit
  iteration-boundary flushing limits the incomplete region to pending work.
- Do not use trace files as checkpoints or tensor payload archives.
