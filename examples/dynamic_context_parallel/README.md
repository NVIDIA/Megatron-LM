# Dynamic Context Parallel Benchmark

## Bounded CPU runtime planner (PP1/VPP1)

Cost models and solvers are bundled in MCore; no external `dcp-scheduler` package
is required. Model selection is independent of solver selection. For example,
add these flags to a compatible native-DCP training configuration:

```bash
--dynamic-cp-runtime-model calibrated_attention \
--dynamic-cp-runtime-model-options '{"hardware":"gb200","query_heads":32,"kv_heads":4,"head_dim":128,"layer_count":48,"allow_provisional":true}' \
--dynamic-cp-runtime-trace-dir /path/to/telemetry
```

The geometry above is illustrative and must match the trained model. The
`calibrated_attention` model covers per-layer FA core costs, not projections or
the rest of an arbitrary model; transfer to eager/native is provisional. The
alternative `qwen35_gb200` includes the frozen W32/EP4/local16K five-layer
proxy's (3 GDN + 2 FA)
paired FA remainder/GDN/non-attention parameters, not a universal Qwen model.
Neither is a substitute for GPU validation. Numba acceleration is optional.

With no solver flag, the lightweight arbitrary-width planner uses a cooperative
0.8-second budget and scores at most four candidates; mandatory work may overrun.
To select the communication-agnostic DynamicUSP Stage 1/count-search policy:

```bash
--dynamic-cp-runtime-solver dynamic_usp \
--dynamic-cp-runtime-solver-options '{"candidate_count":4,"outlier_threshold":1.0}'
```

This policy retains the upstream power-of-two candidate widths. It uses the
selected cost model, not the upstream USP calibration, and does not choose an
attention implementation. Both solvers retain the configured powers incumbent
when their search does not predict an improvement. Model/solver options accept a
JSON object or file path; custom `module:factory` hooks are still supported.
See the [runtime contracts and model limits](../../megatron/core/datasets/dcp_runtime_scheduler/README.md).

The existing DataLoader keeps each DP rank's sample stream. `wrap_data_iterator`
buffers one future global batch on the host and gathers its logical/physical lengths
on the training thread, using the existing DP collective. A persistent CPU executable
on the DPxCP leader solves it while the current batch trains. Only lengths, options
and incumbent rows enter the worker; CUDA visibility is disabled. It does not import
the training entry point or call CUDA/NCCL. An I/O-only thread handles the worker
protocol; the training thread broadcasts the completed rows when consumed. No new
process group is created, and the rerun iterator retains its original consumption order.

`--no-dynamic-cp-runtime-async` is a same-solver synchronous control;
`--dynamic-cp-runtime-force-powers` selects the configured powers packing. To compare
against a particular historical revision, set `--dynamic-cp-runtime-baseline
module:factory`; this factory takes the geometry dictionary and returns a
`padded_pairs -> rank_rows` callable. Without it the baseline is the current
checkout's non-arbitrary scheduler, **not** an authenticated historical dev SHA. Legacy
transport and configurations without a runtime model keep their existing path.
The runtime requires PP1/no VPP and sequence-parallel size one. The injected cost
model must match the model's layouts, including contiguous or zigzag partitioning. Training
lookahead is enabled for the in-tree Varlen path or worker-backed loaders; external
iterators and evaluation do not read ahead. Stochastic transforms must run in
DataLoader workers so speculative reads do not change training-process RNG state.

Per-rank JSONL includes submit/solve-start/solve-end/consume monotonic and wall
timestamps, global logical/physical length digest, rows, residual CPU wait,
worker CUDA/distributed initialization status, and queue depth (at most one). Global
IDs are step-local in DP-rank order, **not dataset indices**. Compare leader monotonic
timestamps on the same host and use GPU traces to establish compute overlap. The
first step waits for initialization; subsequent steps wait only for unfinished work.

This example compares regular DP-balanced packed-sequence training against
Dynamic Context Parallelism (DCP) on the same variable-length mock workload.

The script reuses the normal Megatron-LM training stack:

- `pretrain_gpt.py` builds and trains the GPT model.
- `MockVarlenDataset` creates THD-format variable-length samples.
- `DefaultDynamicCPScheduler` is enabled only for the DCP run.

No model class or custom dataset class is introduced by this example.

## Run

From the Megatron-LM repository root:

```bash
GPUS_PER_NODE=8 bash examples/dynamic_context_parallel/benchmark_dcp.sh
```

The default topology is `TP=1, CP=4, PP=1`, so the default run expects eight
GPUs and uses data parallel size two. By default the script sets
`NUM_MICROBATCHES=8` and
`GLOBAL_BATCH_SIZE=MICRO_BATCH_SIZE * DP_SIZE * NUM_MICROBATCHES`. Keeping more
than one microbatch per global batch gives the DCP scheduler enough
variable-length samples to assign smaller local CP groups instead of expanding
every sample back to the full fixed CP size. This mock THD workload also keeps
`MICRO_BATCH_SIZE=1`; increase effective batch size with `NUM_MICROBATCHES` and
data-parallel ranks.

The script runs two jobs:

1. Baseline packed sequence training:

   ```text
   --sequence-packing-scheduler dp_balanced
   --context-parallel-size 4
   ```

2. Dynamic CP training:

   ```text
   --dynamic-context-parallel
   --sequence-packing-scheduler default_dynamic_cp
   --context-parallel-size 4
   ```

Both runs use the same model, batch size, sequence distribution, and
`--max-seqlen-per-dp-cp-rank`. The script also sets
`--moe-token-dispatcher-type alltoall`, which Megatron-Core currently requires
when sequence packing is enabled. It sets `--num-workers 0` by default to keep
mock collation in the main process for this variable-length THD workload.

## What Makes DCP Useful Here

The mock dataset draws sequence lengths from a lognormal distribution:

```json
{"mode":"distribution","type":"lognormal","format":"thd","min_seq_len":128,"max_seq_len":8192,"mean_seq_len":1024,"lognormal_sigma":1.5}
```

With the default `--max-seqlen-per-dp-cp-rank 2048`, DCP can assign different
local CP sizes:

- Short samples up to 2048 tokens can use one rank.
- Medium samples up to 4096 tokens can use two ranks.
- Long samples up to 8192 tokens can use four ranks.

The baseline keeps the full fixed CP size for the packed workload. DCP can
spread short samples over the DPxCP domain instead of making every sample occupy
the full CP group.

## Automatic native transport

For the default GPT builder with TE full attention, eager FP16/BF16 and P2P CP,
Megatron automatically uses native transport when the installed TE extension and
the DP×CP parent communicator support it. No additional transport flag is needed.
TE must be built with `NVTE_WITH_NCCL_DEVICE_CP=1` and matching NCCL headers/runtime.
All ranks agree before dynamic CP groups are created. This enables arbitrary
integer CP sizes and routes attention, MoE auxiliary reductions and MTP through
one parent communicator. The log reports `dynamic CP transport: native (automatic)`.

Old TE builds, custom/hybrid attention, unsupported precision or CUDA Graphs
retain the legacy ProcessGroup path, with the reason logged at startup. Failures
after native initialization begins are errors, not runtime transport fallbacks.

## NVLink-aware native DCP scheduling

For a Cartesian rank layout with each consecutive eight global GPU ranks in
one NVLink domain, add these arguments to a native DCP training command:

```text
--dynamic-context-parallel
--dynamic-cp-nvlink-domain-size 8
--dynamic-cp-communication-cost 1024 2304
```

The domain size is an explicit placement contract, **not** GPUs per Slurm host
or automatic topology discovery. Use 72 only when consecutive 72-rank blocks
really describe NVL72 domains. The scheduler maps global ranks to DP×CP
positions and takes the common boundaries across TP/PP planes so all stages
choose the same schedule. Custom non-Cartesian process-group layouts are not
supported by this option. Omitting the domain size preserves compute-only
scheduling; legacy multi-ProcessGroup DCP is unchanged.
Explicit topology-aware scheduling requires native transport; initialization
reports an error if automatic selection cannot provide it.

The lightweight score retains the padded compute proxy `Q = sum(L_i²) / CP`.
For a group crossing a domain boundary, it adds only the extra exposed ring
communication, using `CP - 1` overlapping exchanges:

```text
q = Q / CP
u_local, u_cross = communication_cost * payload_tokens_per_rank
score = Q + (CP - 1) * (max(q, u_cross) - max(q, u_local))
```

Both terms use the same token-squared units; the communication coefficients
are equivalent-token weights, not milliseconds or bandwidths. They apply to
the complete pack, not separately to each sequence. Fully hidden traffic is
only a tie-break. Payload size follows the existing `pad_packed_seq_alignment`
setting (including full-capacity `max` padding), not just the valid tokens.
The two default weights are a rounded H100/Qwen3 attention
reference, not native-transport calibration; override them for a different
model, payload or interconnect. Equal weights or a single NVLink domain preserve
the existing packing/filling policy, including its full-parent packing fallback.

All integer CP sizes remain eligible, including groups that must cross a
domain. Spare ranks are distributed across packs by a bounded dynamic program,
instead of all being assigned to one pack. It minimizes the slowest group, then
total rank-work and cross-domain traffic, exactly for each fixed pack order.
Only two orders are tried for rank allocation (original and topology-greedy),
so this is not a globally optimal packing solver. Each order costs O(groups × ranks²); VPP
splits use the same allocator before committing an expansion.
The training path loads no predictor, uses no external solver, and creates no
process groups or topology-probing collectives. This heuristic does not guarantee an
E2E improvement: EP imbalance, communication overlap and native timings still
need offline prediction or measurement for the workload of interest.

## Output

At the end, the script prints:

```text
=== Dynamic CP benchmark summary ===
Iteration-time statistics exclude the first 10 logged iterations.
Baseline dp_balanced average:        ...
Dynamic CP average:                  ...
Average speedup:                     ...
Baseline dp_balanced median:         ...
Dynamic CP median:                   ...
Median speedup:                      ...
Baseline dp_balanced 10% trimmed avg: ...
Dynamic CP 10% trimmed avg:           ...
10% trimmed mean speedup:             ...
```

It parses Megatron-LM's regular training log line:

```text
elapsed time per iteration (ms): ...
```

Logs and TensorBoard output are written under `dcp_benchmark_output/` by
default.

## Slurm Benchmark Results

The following single-run measurements were collected on June 9, 2026 on a
Slurm cluster using four GPUs per node and
`/home/tolong/nvidian+nemo+26.02.rc5.sqsh`.
The run used the default benchmark shape (`TP=1`, `CP=4`, `PP=1`,
`TRAIN_ITERS=30`, `WARMUP_ITERS=10`, `NUM_MICROBATCHES=8`,
`MAX_SEQLEN_PER_DP_CP_RANK=2048`) and the default lognormal mock VarlenDataset
distribution shown above. Statistics exclude the first 10 logged iterations.

The 10% trimmed mean is the primary comparison because short DCP runs can have
large first-use spikes when dynamic groups are exercised. Arithmetic means are
included to show that variance.

| Nodes | GPUs | Slurm job | Baseline trimmed mean (ms) | DCP trimmed mean (ms) | Trimmed speedup | Baseline avg (ms) | DCP avg (ms) |
| ----- | ---- | --------- | -------------------------- | --------------------- | --------------- | ----------------- | ------------ |
| 1 | 4 | 3239751 | 195.269 | 151.906 | 1.285x | 216.125 | 155.150 |
| 2 | 8 | 3239752 | 220.119 | 155.850 | 1.412x | 226.515 | 157.905 |
| 4 | 16 | 3239753 | 226.775 | 187.125 | 1.212x | 231.620 | 189.470 |
| 8 | 32 | 3239754 | 286.206 | 208.088 | 1.375x | 287.780 | 225.875 |
| 16 | 64 | 3239955 | 271.988 | 181.912 | 1.495x | 281.480 | 217.620 |

## Useful Overrides

Use a larger model:

```bash
NUM_LAYERS=32 HIDDEN_SIZE=4096 FFN_HIDDEN_SIZE=16384 NUM_ATTENTION_HEADS=32 \
GPUS_PER_NODE=8 bash examples/dynamic_context_parallel/benchmark_dcp.sh
```

Use a more skewed long/short distribution:

```bash
VARLEN_DATASET_JSON='{"mode":"distribution","type":"lognormal","format":"thd","min_seq_len":128,"max_seq_len":8192,"mean_seq_len":768,"lognormal_sigma":1.8}' \
GPUS_PER_NODE=8 bash examples/dynamic_context_parallel/benchmark_dcp.sh
```

Use exact sequence lengths from a CSV:

```bash
VARLEN_DATASET_JSON='{"mode":"file","format":"thd","path":"/path/to/lengths.csv"}' \
GPUS_PER_NODE=8 bash examples/dynamic_context_parallel/benchmark_dcp.sh
```

Reduce the runtime:

```bash
TRAIN_ITERS=12 WARMUP_ITERS=3 GPUS_PER_NODE=8 \
bash examples/dynamic_context_parallel/benchmark_dcp.sh
```

Use a specific Python interpreter:

```bash
PYTHON=/opt/venv/bin/python GPUS_PER_NODE=8 \
bash examples/dynamic_context_parallel/benchmark_dcp.sh
```

For stable numbers, keep the same GPU allocation, run more iterations, and
avoid checkpointing or evaluation during the measured window.
