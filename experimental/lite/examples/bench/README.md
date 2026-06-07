# MLite Bench Example

This example runs the same small pretrain-style benchmark through either
`backend=mlite` or `backend=bridge`.

`bridge` is a runtime backend backed by Megatron-Bridge. It requires an
environment where `import megatron.bridge` works. Dry-run mode does not import
Megatron-Bridge and is safe for config validation.

## Dry-Run

```bash
export PYTHONPATH=/path/to/Megatron-LM/experimental/lite:$PYTHONPATH

python experimental/lite/examples/bench/bench.py \
  --backend mlite \
  --hf-path /models/Qwen3.5-35B-A3B \
  --model-name qwen3_5 \
  --truncate-layers 2 \
  --disable-mtp \
  --dry-run

python experimental/lite/examples/bench/bench.py \
  --backend bridge \
  --hf-path /models/Qwen3.5-35B-A3B \
  --model-name qwen3_5 \
  --truncate-layers 2 \
  --disable-mtp \
  --dry-run
```

The dry-run output is JSON containing the resolved `RuntimeConfig` and session
settings. Use it to confirm the two backend runs differ only on the intended
axis.

## Pair Script

```bash
HF_PATH=/models/Qwen3.5-35B-A3B \
DRY_RUN=1 \
bash experimental/lite/examples/bench/scripts/run_qwen35_pair.sh
```

Set `DRY_RUN=0` to run the benchmark under `torchrun`. Results are written to
`experimental/lite/examples/bench/outputs/`.

## Validated Run

The following paired run completed on 2026-06-07 with 8x NVIDIA H100 80GB GPUs:

```bash
HF_PATH=/models/Qwen3.5-35B-A3B \
OUTPUT_DIR=experimental/lite/examples/bench/outputs/qwen35_pair \
DRY_RUN=0 \
NPROC=8 \
MASTER_PORT=31841 \
MASTER_PORT_BRIDGE=31842 \
STEPS=15 \
WARMUP=5 \
SEQ_LEN=1024 \
NUM_MICROBATCHES=4 \
TRUNCATE_LAYERS=8 \
KEEP_EXPERTS=8 \
SAME_DATA_ACROSS_DP=1 \
bash experimental/lite/examples/bench/scripts/run_qwen35_pair.sh
```

Slurm job `12624917` completed with exit code `0:0`. The run used
`torch==2.10.0+cu129`; `transformer_engine`, `einops`, and `megatron.bridge` were
available in the runtime environment.

| Runtime | Impl | Optimizer backend | Measured steps | Avg step ms | Tokens/s | Tokens/s/GPU | Peak memory GB | TFLOPs/GPU |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlite` | `lite` | `mc_full` | 10 | 309.433 | 105896.935 | 13237.117 | 14.324 | 80.444 |
| `bridge` | `bridge` | `mc` | 10 | 332.201 | 98639.089 | 12329.886 | 17.987 | 74.931 |

The two runs used the same synthetic input stream. Loss matched within
`atol=0.05, rtol=0.005` across 10 measured samples
(`max_abs_diff=0.000500`). Grad norm did not match that tolerance
(`max_abs_diff=2.491823`), so treat the recorded correctness result as
loss-consistent rather than full optimizer-metric parity.

## Real Run

```bash
torchrun --nproc_per_node 1 experimental/lite/examples/bench/bench.py \
  --backend mlite \
  --hf-path /models/Qwen3.5-35B-A3B \
  --model-name qwen3_5 \
  --steps 5 \
  --warmup 1 \
  --seq-len 2048 \
  --num-microbatches 1 \
  --truncate-layers 2 \
  --disable-mtp \
  --output-json /tmp/qwen35_mlite_bench.json

torchrun --nproc_per_node 1 experimental/lite/examples/bench/bench.py \
  --backend bridge \
  --hf-path /models/Qwen3.5-35B-A3B \
  --model-name qwen3_5 \
  --steps 5 \
  --warmup 1 \
  --seq-len 2048 \
  --num-microbatches 1 \
  --truncate-layers 2 \
  --disable-mtp \
  --output-json /tmp/qwen35_bridge_bench.json
```

Compare `loss`, `grad_norm`, `avg_step_ms`, `tok_per_s`, peak memory, and
`tflops_per_gpu` in the two JSON outputs. Benchmarks are performance evidence;
they are not a replacement for precision tests.

## Deterministic Correctness

Use `correctness.py` for strict deterministic parity. It emits exact scalar
fingerprints for `loss` and `grad_norm`, SHA256 fingerprints for logits and
post-step exported weights, and a strict comparison artifact.

```bash
export MEGATRON_LITE_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

torchrun --nproc_per_node 1 experimental/lite/examples/bench/correctness.py run \
  --backend mlite \
  --hf-path /models/Qwen3.5-35B-A3B \
  --model-name qwen3_5 \
  --steps 2 \
  --seq-len 128 \
  --num-microbatches 1 \
  --truncate-layers 2 \
  --disable-mtp \
  --same-data-across-dp \
  --output-json /tmp/qwen35_mlite_correctness.json

torchrun --nproc_per_node 1 experimental/lite/examples/bench/correctness.py run \
  --backend bridge \
  --hf-path /models/Qwen3.5-35B-A3B \
  --model-name qwen3_5 \
  --steps 2 \
  --seq-len 128 \
  --num-microbatches 1 \
  --truncate-layers 2 \
  --disable-mtp \
  --same-data-across-dp \
  --output-json /tmp/qwen35_bridge_correctness.json

python experimental/lite/examples/bench/correctness.py compare \
  /tmp/qwen35_mlite_correctness.json \
  /tmp/qwen35_bridge_correctness.json \
  --fail-on-mismatch
```
