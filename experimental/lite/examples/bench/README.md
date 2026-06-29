# MLite Bench Example

This example runs the same small pretrain-style benchmark through `backend=mlite`
and a reference backend.

The validated reference backend in this PR is `backend=mbridge`, backed by the
legacy `mbridge` package. `backend=bridge` is reserved for the real
Megatron-Bridge package and requires an environment where `import
megatron.bridge` works. Dry-run mode does not import either reference package and
is safe for config validation.

The `backend=mlite` path remains native MLite. For deterministic Qwen3.5 MoE
runs it mounts the Qwen3.5 vision module from the local Hugging Face model code
through `transformers`; it does not import `mbridge` or Megatron-Bridge to build
the MLite model.

This benchmark separates two validation lines:

- `mbridge`: the validated reference for MLite vs Megatron-Core distributed
  optimizer (`distopt`) parity.
- `bridge`: a real Megatron-Bridge environment check when `import
  megatron.bridge` works.

Use the `mbridge` line for Core/distopt precision and speed claims.

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
  --backend mbridge \
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
REFERENCE_BACKEND=mbridge \
DRY_RUN=1 \
bash experimental/lite/examples/bench/scripts/run_qwen35_pair.sh
```

Set `DRY_RUN=0` to run the benchmark under `torchrun`. Results are written to
`experimental/lite/examples/bench/outputs/`. Set `REFERENCE_BACKEND=bridge` only
when Megatron-Bridge is installed and `import megatron.bridge` succeeds.

## Validated Run

The following paired run completed on 2026-06-07 with 8x NVIDIA H100 80GB GPUs:

```bash
HF_PATH=/models/Qwen3.5-35B-A3B \
OUTPUT_DIR=experimental/lite/examples/bench/outputs/qwen35_pair \
REFERENCE_BACKEND=mbridge \
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
`torch==2.10.0+cu129`; `transformer_engine`, `einops`, and `mbridge` were
available in the runtime environment.

| Runtime | Impl | Optimizer backend | Measured steps | Avg step ms | Tokens/s | Tokens/s/GPU | Peak memory GB | TFLOPs/GPU |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlite` | `lite` | `distopt` | 10 | 309.433 | 105896.935 | 13237.117 | 14.324 | 80.444 |
| `mbridge` | `bridge` | `distopt` | 10 | 332.201 | 98639.089 | 12329.886 | 17.987 | 74.931 |
| `bridge` | `bridge` | `distopt` | 10 | 334.936 | 97833.496 | 12229.187 | 16.403 | 74.319 |

The two runs used the same synthetic input stream. Loss matched within
`atol=0.05, rtol=0.005` across 10 measured samples
(`max_abs_diff=0.000500`). This long benchmark is performance evidence; the
strict optimizer-metric evidence is the deterministic run below.

## Deterministic mbridge Correctness

Slurm job `12630675` completed a strict deterministic MLite vs `mbridge` run
with 1x GPU, `seed=42`, Qwen3.5 MoE, `seq_len=8`, `truncate_layers=1`,
`keep_experts=2`, `optimizer_backend=distopt`, and `steps=2`.

The comparison passed with bitwise scalar parity and no mismatches:

- `samples=2`
- `max_loss_abs=0.0`
- `max_grad_norm_abs=0.0`
- `mismatches=[]`
- step 0: `loss=13.027458190917969`,
  `grad_norm=120.75512734973202`, post-step weight SHA256
  `1e3176a8cb18d68c5da9bfa5f31a507fa8d51a3a5ddc10fbcd821260a4c6c980`
- step 1: `loss=14.698704719543457`,
  `grad_norm=96.15656991334498`, post-step weight SHA256
  `e6d034f7e05ee5ee6a42ceddda8970874c6742baf59cade38f53550abd7aec29`
- eval logits canonical bf16 SHA256
  `2f805802633927852c5ec87455b1afa3a68597e1e411b979f2678eaedfb1c710`

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
  --backend mbridge \
  --hf-path /models/Qwen3.5-35B-A3B \
  --model-name qwen3_5 \
  --steps 5 \
  --warmup 1 \
  --seq-len 2048 \
  --num-microbatches 1 \
  --truncate-layers 2 \
  --disable-mtp \
  --output-json /tmp/qwen35_mbridge_bench.json
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
  --backend mbridge \
  --hf-path /models/Qwen3.5-35B-A3B \
  --model-name qwen3_5 \
  --steps 2 \
  --seq-len 128 \
  --num-microbatches 1 \
  --truncate-layers 2 \
  --disable-mtp \
  --same-data-across-dp \
  --output-json /tmp/qwen35_mbridge_correctness.json

python experimental/lite/examples/bench/correctness.py compare \
  /tmp/qwen35_mlite_correctness.json \
  /tmp/qwen35_mbridge_correctness.json \
  --fail-on-mismatch
```

For the PR signoff pair script:

```bash
export MEGATRON_LITE_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

HF_PATH=/models/Qwen3.5-35B-A3B \
REFERENCE_BACKEND=mbridge \
DRY_RUN=0 \
bash experimental/lite/examples/bench/scripts/run_qwen35_correctness_pair.sh
```
