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
collective. It can still perturb execution timing, allocation, and stream
overlap; absence of new collectives is not a guarantee of a faithful observer.

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
records using the same schema as the offline loader and continues sequence and
occurrence counters. A complete final JSON object without a trailing newline is
preserved and separated from the next record. A truncated object is rejected.
Use one writer per rank file; append is continuation, not concurrent writing.

## Evidence modes

The configured mode applies to `record_tensor` calls without a mode override:

| Mode | Evidence | Interpretation |
|---|---|---|
| `metadata` | shape, stride, dtype, layout, device type | Structure only; tensor values are not read |
| `summary` | metadata, finite status, min, max, mean, L2 norm | Low-volume value diagnostics; not an equality certificate |
| `sampled` | metadata and SHA-256 over deterministic evenly spaced elements | Gathers selected logical elements, including from non-contiguous tensors; unsampled values can differ |
| `full` | metadata and SHA-256 over every tensor byte | Exact tensor equality certificate for the captured boundary |

`full` hashes all bytes but does not store the tensor payload. It is intended
to establish equality or locate divergence without duplicating model-sized
data. A matching sampled or summary record is diagnostic evidence, not proof
that the full tensors are equal.

`record_scalar` is an explicit exception: scalar tensors always use `full`,
including the training loop's scalar losses. Therefore a training run configured
with `--determinism-trace-mode metadata` still reads scalar tensor values and
can synchronize at flush. Python scalar metrics are recorded as JSON events.
Use `record_tensor(..., mode="metadata")` when a boundary must be structure-only.

Value-bearing modes snapshot the selected data when the record is created and
move it to the host when the trace is flushed. `summary` scans the entire tensor,
performs several reductions, and may allocate a full float64 conversion.
`sampled` allocates indices and gathered values; its device work depends on the
sample count and tensor dimensionality. `full` retains a complete tensor copy
until flush and performs CPU SHA-256 hashing. Multiple buffered full captures
can consume substantial device memory. Small output files do not imply low
observer cost. Set a narrow rank/iteration window and an appropriate
`--determinism-trace-flush-every` limit; frequent flushing adds synchronization
and file I/O, whereas delayed flushing retains more snapshots.

Capture runs on the caller's current stream for the tensor's device. The caller
must order earlier input writes before capture, and subsequent mutations after
capture, using the normal PyTorch stream dependency rules. The recorder records
a CUDA completion event for each value-bearing snapshot. Flush waits for that
event before host materialization, so it may execute on another stream. It does
not synchronize the whole device or infer dependencies for the input tensor.

Do not call summary, sampled, full, or tensor `record_scalar` inside CUDA graph
capture, or flush pending CUDA values during capture. Metadata-only tensor
capture does not read values, but its Python record operation executes only
during capture, not on every graph replay. Instrument replay lifecycle outside
the graph; capture-time metadata cannot certify per-replay coverage.

## Instrument an internal boundary

The reusable API lives in `megatron.core.determinism.trace`; Core boundary sites
do not import the training layer. The process-local trace is available after
the training adapter initializes it at the start of `train()`:

```python
from megatron.core.determinism.trace import get_determinism_trace


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
from megatron.core.determinism.trace import RankLocalTrace, TraceConfig

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
- `1`: a missing event, content mismatch, differing boundary, or event-order mismatch was found
- `2`: either trace is malformed or violates the schema

Report version 2 returns `rank_results`, keyed by decimal global rank. Each
entry contains `match`, `first_divergence`, and
`compared_records_before_divergence` (the length of that rank's equal execution
prefix). It examines events in increasing per-rank sequence order, considering
order and missing-event differences before comparing later content. Each
candidate includes `left_sequence` and `right_sequence`; a null sequence means
that side's stream has ended. Semantic identity still uses iteration,
microbatch, phase, event name/type, rank, and occurrence. Different numeric
sequence offsets alone do not constitute divergence.

For example, if rank 0 first differs at sequence 90 and rank 1 at sequence 3,
both candidates are returned. Neither is declared globally earlier. When both
runs reach different new events at the same boundary, `event_mismatch` reports
both keys. When an event exists only on one side, `missing_event` reports the
missing side. The report intentionally removes the old top-level
`first_divergence` and prefix count; consumers must inspect `rank_results`.
Trace record schema version remains 1.

Directory inputs require full filename matches of `rank_<digits>.jsonl` and
reject rank aliases, malformed matching filenames, and every empty rank file.
Explicit file inputs may use arbitrary filenames but must contain exactly one
rank. Empty selected windows are not equality evidence. Missing non-empty ranks
are reported as divergences, and each side lists its observed ranks. Invalid
JSON, duplicate fields, invalid UTF-8, inconsistent tensor evidence, duplicate
semantic keys, and non-increasing sequences produce exit code 2.

`match_strength` describes what a successful comparison establishes:

- `structure_only`: all tensor records are metadata-only
- `diagnostic_tensor_match`: summary/sampled evidence or mixed full/metadata records
- `full_tensor_certificate`: every tensor record hashes every byte
- `event_match`: the traces contain semantic events but no tensor records

A full certificate is conditional on valid recorder evidence and covers only
captured tensor records. The comparator checks evidence consistency (including
shape, element counts, mode flags, and sample indices); it cannot recompute a
digest without the original tensor or establish equality of uncaptured state.

## Observer validity and GPU qualification

Before interpreting a trace causally, compare paired runs with tracing disabled
and enabled under the same model, data, seed, checkpoint, software, and topology.
The original equality or divergence verdict must be preserved. A trace that
makes the original divergence disappear has failed this validity gate, even if
its own captures match. Changing evidence mode, selected boundaries, flush
frequency, or rank/iteration coverage requires requalification.

The CUDA unit tests exercise non-default and mixed producer streams, flush on a
different stream, mutation after snapshot, graph metadata coverage, and a small
observer validity check with an injected divergence. In the Megatron CI container:

```bash
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/determinism/test_determinism_trace.py \
  tests/unit_tests/determinism/test_trace_comparison.py \
  tests/unit_tests/determinism/test_determinism_trace_cuda.py
```

These tests do not qualify application-scale performance. Measure paired
tracing-off/tracing-on runs in both orders, excluding initialization,
compilation, graph capture, and warmup. Record steady-state median and p95 step
times, peak device memory, evidence mode, selectors, boundaries, sample count,
and flush frequency. Include the tracing flush in timing. Apply the roadmap's
overhead decision gate to the target workload: below 1% is acceptable, above 7%
is a blocker, and 1–7% requires an explicit product decision. Report the
observer-validity verdict alongside performance, not just file size or kernel
timing. See [the determinism roadmap](https://github.com/NVIDIA/Megatron-LM/issues/5785).

A deferred device-digest backend and FSDP/collective/TE/optimizer-specific
boundaries remain follow-ups. The current value-bearing implementation is a
progressive diagnostic tool, not a scale-safe backend qualification.

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
