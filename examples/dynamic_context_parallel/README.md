# Dynamic Context Parallel Benchmark

This example compares regular DP-balanced packed-sequence training against
Dynamic Context Parallelism (DCP) on the same variable-length workload. It can
use either the built-in mock dataset or an SFT dataset such as
`nvidia/Nemotron-SFT-Math-v3`.

The script reuses the normal Megatron-LM training stack:

- `pretrain_gpt.py` builds and trains the GPT model.
- `VarlenDataset` tokenizes SFT conversations into variable-length THD samples.
- `MockVarlenDataset` provides a synthetic alternative for quick benchmarks.
- `DefaultDynamicCPScheduler` is enabled only for the DCP run.

No model class or custom dataset class is introduced by this example.

## Prerequisites and Limitations

- Run from a CUDA-enabled Megatron-LM environment with NCCL, FlashAttention,
  and Transformer Engine 2.9 or newer.
- The benchmark currently supports one node only. `NUM_NODES` must be `1` and
  `NODE_RANK` must be `0`; this avoids unsafe cross-node log and TensorBoard
  writes until rank-aware collection is added.
- Packed-sequence DCP currently supports dense models and the `single`
  dataloader. Megatron FSDP and CUDA graphs are not supported. The script
  explicitly passes `--cuda-graph-impl none` to both cases.
- SFT mode also requires the Hugging Face `datasets`, `transformers`, and
  `pandas` packages and access to the selected dataset and tokenizer.
- Transformer Engine nondeterministic algorithms default to disabled so the
  loss comparison is meaningful. Override
  `NVTE_ALLOW_NONDETERMINISTIC_ALGO=1` only when measuring a production setting,
  and do not compare that run with results collected under the deterministic
  default.

## Run With Mock Data

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

Both runs use the same model, batch size, sequence distribution,
`--max-seqlen-per-dp-cp-rank`, and `--cuda-graph-impl none`. They also start
from the same seed. The script sets `--num-workers 0` by default to keep mock
collation in the main process for this variable-length THD workload, and
disables dataloader attention-mask creation because packed THD attention uses
sequence boundaries instead.

## Run With Nemotron-SFT-Math-v3

Set `DATASET_PATH` to switch from the mock tokenizer and dataset to
`VarlenDataset` and `SFTTokenizer`:

```bash
HF_DATASETS_CACHE=/path/to/huggingface/cache \
DATASET_PATH=nvidia/Nemotron-SFT-Math-v3 \
TOKENIZER_MODEL=Qwen/Qwen3-30B-A3B \
GPUS_PER_NODE=8 \
bash examples/dynamic_context_parallel/benchmark_dcp.sh
```

The Hugging Face cache must have enough space for the dataset. A local Parquet,
JSON, or JSONL benchmark snapshot can be passed through `DATASET_PATH` instead.
Hub repositories can change, so use an immutable local snapshot and record its
external checksum before publishing a comparison.

The tokenizer must have a chat template that supports the dataset's tool and
reasoning metadata. The default is `Qwen/Qwen3-30B-A3B`. Both benchmark cases
use `--sft-tokenizer-prompt-format default`, which delegates conversation
formatting to that Hugging Face chat template.

The `default` prompt format uses full-sequence causal loss: system, user,
assistant, and tool tokens are all training targets. This benchmark therefore
checks DCP against the fixed-CP baseline under that loss definition; it is not
an assistant-only masking recipe.

Without a checkpoint, both cases start from the same seeded random
initialization. To run actual fine-tuning from a pretrained model, pass a
matching Megatron distributed checkpoint:

```bash
LOAD_PATH=/path/to/megatron/checkpoint \
DATASET_PATH=nvidia/Nemotron-SFT-Math-v3 \
TOKENIZER_MODEL=Qwen/Qwen3-30B-A3B \
GPUS_PER_NODE=8 \
bash examples/dynamic_context_parallel/benchmark_dcp.sh
```

`LOAD_PATH` adds `--load --finetune --exit-on-missing-checkpoint` to both runs.
The model-shape overrides supplied to this script must match the checkpoint.

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

At the end, the script prints throughput statistics and compares the logged
loss trajectories:

```text
=== Dynamic CP throughput summary ===
Iteration-time statistics exclude the first 10 logged iterations.
Baseline dp_balanced average:        ...
Dynamic CP average:                  ...
Baseline average throughput:         ... samples/s
Dynamic CP average throughput:       ... samples/s
Average sample-throughput speedup:   ...
Baseline dp_balanced median:         ...
Dynamic CP median:                   ...
Median sample-throughput speedup:    ...
Baseline dp_balanced 10% trimmed avg: ...
Dynamic CP 10% trimmed avg:           ...
10% trimmed throughput speedup:       ...

=== Dynamic CP loss comparison ===
Loss records compared:                 ...
Iteration/sample alignment:            PASS
Maximum absolute loss difference:      ...
Maximum relative loss difference:      ...
Loss parity (atol=0.001, rtol=0.001):  PASS
```

Because the model, global batch size, and consumed samples are identical,
iteration-time speedup is also sample-throughput speedup. Timing statistics
exclude warmup iterations and require exactly `TRAIN_ITERS` positive, finite
records. Samples per iteration are derived from constant positive deltas in the
logged consumed-sample counts and must match between the two cases. Loss
comparison covers every logged iteration. The script parses Megatron-LM's
regular training log fields:

```text
iteration ... | consumed samples: ... |
elapsed time per iteration (ms): ...
lm loss: ...
```

By default, mismatched iteration/sample records or losses outside
`LOSS_ATOL=1e-3` and `LOSS_RTOL=1e-3` fail the benchmark. Set
`CHECK_LOSS_PARITY=0` to report out-of-tolerance finite losses as a warning
instead. Incomplete or duplicate timing/loss records, skipped or NaN
iterations, and sample-misaligned runs always fail. Set `ANALYZE_ONLY=1` to
recompute the report from existing `baseline.log` and `dcp.log` files without
launching training again. Repeat the `TRAIN_ITERS` and `WARMUP_ITERS` values
that produced those logs when using analysis-only mode. Throughput is derived
from the logs, so a current-shell `GLOBAL_BATCH_SIZE` value cannot change an
analysis-only result.

Logs and TensorBoard output are written under `dcp_benchmark_output/` by
default. Each new run also writes `reproducibility_manifest.txt` with the git
SHA and dirty state, selected arguments, GPU and software versions, dataset or
mock configuration, and checkpoint identity. It deliberately does not dump the
process environment or authentication tokens. `ANALYZE_ONLY=1` reuses the
existing manifest instead of replacing the identity of the run being analyzed.

## Historical Mock-Only Results

An earlier dev-stack version of this example reported mock-only, multi-node
measurements. Those numbers predate this main-stack port, did not use
Nemotron-SFT-Math-v3, and are not validation for the current implementation, so
they are intentionally omitted. This example does not claim Math-v3 loss parity
or throughput until a complete run is recorded from the rebuilt stack with its
manifest and an immutable dataset snapshot.

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

Adjust the loss tolerances:

```bash
LOSS_ATOL=5e-4 LOSS_RTOL=5e-4 GPUS_PER_NODE=8 \
bash examples/dynamic_context_parallel/benchmark_dcp.sh
```

Reverse the case order to check for cache or thermal bias:

```bash
CASE_ORDER=dcp_first OUTPUT_DIR=dcp_benchmark_output_reversed \
GPUS_PER_NODE=8 bash examples/dynamic_context_parallel/benchmark_dcp.sh
```

For stable numbers, keep the same GPU allocation, run more iterations, and
avoid checkpointing or evaluation during the measured window. Run both case
orders before publishing a throughput claim; each invocation reports one
ordered comparison.
