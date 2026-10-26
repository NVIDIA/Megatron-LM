# Dynamic Context Parallel Benchmark

This example compares regular DP-balanced packed-sequence training against
Dynamic Context Parallelism (DCP) on the same variable-length mock workload.

The script reuses the normal Megatron-LM training stack:

- `pretrain_gpt.py` builds and trains the GPT model.
- `MockVarlenDataset` creates THD-format variable-length samples.
- `DefaultDynamicCPScheduler` is enabled only for the DCP run.

No model class or custom dataset class is introduced by this example.

## Runtime cost models and async planning

The existing classic CP ProcessGroups and TE attention path can use either
in-tree CPU solver. Add a model to enable the runtime planner:

```text
--dynamic-cp-runtime-model calibrated_attention
--dynamic-cp-runtime-model-options '{"hardware":"gb200","query_heads":32,"kv_heads":4,"head_dim":128,"layer_count":48,"allow_provisional":true}'
```

The default lightweight solver searches legal power-of-two CP widths and
size-aligned rank intervals. Select the independent DynamicUSP allocator with
`--dynamic-cp-runtime-solver dynamic_usp`; model and solver selection are
independent. No custom native transport or external scheduler package is needed.
The frozen models have limited calibration domains; see the
[model and solver documentation](../../megatron/core/datasets/dcp_runtime_scheduler/README.md)
before using their predictions.

A persistent CPU worker on the DPxCP leader prepares one future schedule while
the current batch trains. Use `--no-dynamic-cp-runtime-async` as a same-solver
synchronous control and `--dynamic-cp-runtime-trace-dir PATH` for timing, input
digests, executed rows and worker telemetry. PP1/VPP1 and sequence-parallel size
one are currently required. Mock varlen data or DataLoader workers enable the
lookahead training hook; evaluation and external iterators do not read ahead.
Stochastic transforms must run in DataLoader workers to preserve training RNG.
`--dynamic-cp-runtime-force-powers` is a baseline-only diagnostic and bypasses
the solver; it is disabled by default.

## Topology-aware scheduling

For a Cartesian rank mapping, `--dynamic-cp-nvlink-domain-size 8` declares that
each consecutive eight global ranks share an NVLink domain. The heuristic
maps those domains into the DPxCP groups and scores exposed communication with
`--dynamic-cp-communication-cost 1024 2304`. These equivalent-token weights are
configurable reference coefficients, not a latency calibration. Classic CP
keeps power-of-two widths and size-aligned starts during placement and spare-rank
allocation. Omitting the domain size preserves compute-only scheduling.

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
