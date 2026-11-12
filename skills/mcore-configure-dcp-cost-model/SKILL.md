---
name: mcore-configure-dcp-cost-model
description: Configure the runtime DCP scheduler selected with --dynamic-cp-runtime-config. Use when choosing between the dcp_benchmark and dynamic_usp scheduler types, checking that a model and launch are covered, measuring the non-attention times, writing the runtime JSON, or reading the start-up result and trace.
---

# Configure a DCP scheduler

Work relative to the Megatron-LM checkout. Two documents hold the facts this
procedure uses:

- `megatron/core/datasets/dcp_runtime_scheduler/README.md`: the keys of the
  runtime file, the start-up checks, CP group sizes, what each cost model
  covers and the trace records.
- [references/model-options.md](references/model-options.md): a runtime file
  and launch arguments for every cost model, how to measure the non-attention
  times, and what to change for each start-up message.

## 1. Choose the scheduler type

The runtime file is passed with `--dynamic-cp-runtime-config
/path/to/runtime.json`. Its `type` selects the cost model and the search
algorithm together:

- `dcp_benchmark`: packaged attention predictions plus the user's
  non-attention times, planned by the lightweight planner. The training
  configuration decides between ordinary GQA and full Qwen3.5-35B-A3B.
- `dynamic_usp`: a DynamicUSP cost profile planned by the DynamicUSP search.
  `{"type": "dynamic_usp"}` selects the original profile, which does not
  read the model configuration. `"hardware": "h100"` selects the calibration
  measured for full Qwen3-30B-A3B with classic P2P.

There are no other types and no separate solver, model or profile key.

## 2. Check that the configuration is covered

Read the actual training configuration: layer types and their order, attention
heads, query groups and `kv_channels`, hardware, DPxCP, the tensor, pipeline
and expert parallel sizes, precision, recomputation, `--cp-comm-type`,
`--max-seqlen-per-dp-cp-rank`, padding and tail policy. A model name alone
does not establish coverage. Compare it with the README:

- Ordinary GQA: the coverage tables (hardware, attention geometry, DPxCP, CP
  group sizes per `max_seqlen_per_dp_cp_rank`) and the required training
  settings.
- Full Qwen3.5-35B-A3B and the H100 calibration: the table of required
  settings. Each admits one configuration.
- Original DynamicUSP profile: DPxCP and the minimum CP size are 8, 16 or 32.

Do not change a model or hardware label to make another configuration fit.
Report a configuration that no cost model covers.

`hardware` also fixes the NVLink topology the cost model assumes; no key or
argument sets it. Check the rank placement of the launch against the README
section "NVLink topology": on `h100` every node runs 8 consecutive ranks, on
`gb200` all ranks belong to one 72-GPU NVLink domain. Start-up cannot check
the placement.

Decide `max_cp_size` from the README section "CP group sizes": it is required
when the largest CP group the run can form is not the full DPxCP group, and it
must be one of the sizes the run can form.

## 3. Obtain the non-attention times

Only `dcp_benchmark` takes them. They are the user's measurements; do not
infer them, reuse numbers from another configuration or enter zero in their
place. Follow "Measuring non-attention times" in the configuration reference:
the part of a layer the cost model predicts differs between ordinary GQA (core
attention only) and full Qwen3.5 (the whole attention module), and the values
are milliseconds per layer, per microbatch and per rank. Collect new timings
only when the user asks for it.

## 4. Write the runtime file and launch

Start from the file and the launch block of the cost model in the
configuration reference. Add `max_cp_size`, `async` or `trace_dir` when
needed; any other key is rejected with the list of keys the type accepts. The
file must be the same on every node.

## 5. Read the start-up result

Training validates the file on every rank before the first step. When it
passes, rank 0 prints one line that starts with `Runtime DCP scheduler:` and
names the scheduler type, the cost model, the hardware with its NVLink domain
(`hardware=h100 (NVLink domain: 8 consecutive ranks)`), the CP group sizes,
the sequence cap, `async` and the file. Check that the cost model and the CP
group sizes are the intended ones and that the launch places the ranks as
the NVLink domain says.

When it fails, the message names what must change; the table "When start-up
rejects the configuration" in the configuration reference lists the messages.

## 6. Inspect the plans

Set `trace_dir` to see what the scheduler does. Every rank writes
`rank<NNNNN>.jsonl`; the README section "Trace records" explains the fields.
Useful checks: `lookahead` shows whether batches are planned one step ahead,
`statistics.incumbent_rejection` whether the default layout was discarded,
`rows` and `predicted_ms` the plan and its predicted time, and
`model_metadata` what the cost model loaded.

A batch that cannot be scheduled stops every rank with a
`RuntimeScheduleError` that names the sample and the limits.
