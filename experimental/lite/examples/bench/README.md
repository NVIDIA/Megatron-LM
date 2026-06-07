# MLite Bench Example

This example runs the same small pretrain-style benchmark through either
`backend=mlite` or `backend=bridge`.

`bridge` is a runtime backend backed by Megatron-Bridge. It requires an
environment where `import mbridge` works. Dry-run mode does not import
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
