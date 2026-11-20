<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Post-Training Quantization and Distillation

Post-training quantization lowers the numerical precision of a model's weights
and activations after training has finished. The result is a smaller checkpoint
that runs faster on hardware with low-precision support, at the cost of some
accuracy.

Megatron-LM performs quantization through the
[NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) library
(`nvidia-modelopt`). The scripts live in
[`examples/post_training/modelopt`](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/post_training/modelopt)
and the supporting code lives in `megatron/post_training`.

This page covers two workflows:

- **Post-training quantization (PTQ)** converts a trained checkpoint to a lower
  precision in a single calibration pass. No training data or gradient steps are
  required beyond a small calibration set.
- **Quantization-aware distillation (QAD)** continues to train the quantized
  model afterwards, using the original full-precision model as a teacher. This
  recovers accuracy that PTQ alone loses, at the cost of a training run.

The examples use `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`. The same commands
apply to the other models listed in the support matrix in the
[ModelOpt examples README](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/post_training/modelopt/README.md),
including the Super and Ultra sizes of Nemotron 3.

## Prerequisites

### Environment

Run the examples in an NVIDIA GPU container. The
[NGC NeMo container catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/nemo/-/tags)
lists the current release; it ships both Megatron-Bridge and a compatible Model
Optimizer. Mount your Megatron-LM checkout and a workspace directory that
outlives the container:

```bash
docker run --rm -it --gpus all --shm-size=24g \
    -e HF_TOKEN \
    -v "/path/to/Megatron-LM:/workspace/Megatron-LM" \
    -v "/path/to/modelopt-workspace:/workspace/modelopt-workspace" \
    -w /workspace/Megatron-LM/examples/post_training/modelopt \
    nvcr.io/nvidia/nemo:26.08 \
    bash
```

Inside the container, set `MLM_SKIP_INSTALL=1` when running the scripts so they
use the container's Model Optimizer. Without it, every invocation reinstalls
from `requirements.txt`, which replaces an installation you set up yourself.

### Checkpoint

Every script in this directory operates on a **Megatron-Core distributed
checkpoint**, not on Hugging Face weights. Import a Hugging Face model first
with the
[Megatron-Bridge checkpoint conversion CLI](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/conversion#2-stable-checkpoint-conversion-cli),
then pass the imported checkpoint as `MLM_MODEL_CKPT`. To quantize a Hugging
Face model directly instead, use the
[Megatron-Bridge quantization examples](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/quantization).

### Configuration

Each script takes a configuration name as its first positional argument. That
name selects a file under `examples/post_training/modelopt/conf/`, which
supplies the model architecture arguments, and the model you run must be
covered by one of those files. For Nemotron 3 Nano the name is
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.

Paths and parallelism are passed as environment variables rather than flags:

| Variable | Meaning |
| --- | --- |
| `MLM_MODEL_CKPT` | Megatron-Core checkpoint to load. |
| `MLM_MODEL_SAVE` | Where to write the resulting Megatron-LM checkpoint. |
| `EXPORT_DIR` | Where to write the exported deployment checkpoint. |
| `MLM_SKIP_INSTALL` | Set to `1` to keep the environment's Model Optimizer. |
| `TP`, `PP`, `EP`, `CP`, `DP` | Tensor, pipeline, expert, context, and data parallel sizes. Each defaults to `1`. |
| `ETP` | Expert-tensor parallel size. Defaults to whatever `TP` is set to. |
| `MLM_EXTRA_ARGS` | Any additional Megatron-LM arguments. |

For the full list, see
[Advanced Topics](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/post_training/modelopt/ADVANCED.md).

On a single node the scripts launch through `torchrun` and size themselves as
`ETP × EP × PP × CP × DP` processes, so those variables decide how many GPUs a
command needs. Because `ETP` follows `TP`, raising `TP` and `EP` together
multiplies the requirement: `TP=8 EP=8` asks for 64 processes, not 8. The
examples below use `TP=1 EP=8`, which is eight processes on one 8-GPU node.
Multi-node runs under Slurm size differently; see
[Running on Multiple Nodes](#running-on-multiple-nodes).

## Post-Training Quantization

Run `quantize.sh` with the configuration name and a quantization format. The
script loads the checkpoint, runs a calibration pass, and saves a Megatron-LM
distributed checkpoint that can be resumed for inference, further training, or
export.

```bash
cd examples/post_training/modelopt

TP=1 \
EP=8 \
MLM_SKIP_INSTALL=1 \
MLM_MODEL_CKPT=<megatron_core_checkpoint> \
MLM_MODEL_SAVE=/tmp/nemotron3-nano-nvfp4 \
./quantize.sh nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 NVFP4_DEFAULT_CFG
```

The second positional argument is the quantization format. Pass the full config
name in capitals, such as `NVFP4_DEFAULT_CFG` or `FP8_DEFAULT_CFG`. The available
formats come from Model Optimizer; see its
[quantization configs](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/quantization/config.py)
for the current list. If you omit the argument, the script defaults to
`FP8_DEFAULT_CFG` and prints a warning.

By default the scripts simulate low-precision arithmetic rather than using
low-precision kernels, so calibration runs on any GPU of compute capability 8.0
or higher. Storing genuinely low-precision parameters and running low-precision
compute depends on the GPU.

### Quantizing the Key-Value Cache

During generation a model stores the key and value tensors it has already
computed, so it does not recompute them for every new token. That cache is
created at serving time and differs for every request, so calibration cannot
quantize its contents. What it does instead is fit scale factors for the key and
value projections and record them in the checkpoint; the serving runtime then
uses those scales to store each request's cache in the lower precision.

This is worth doing because at long sequence lengths the cache can occupy more
memory than the weights, and every generated token reads all of it.

```bash
MLM_EXTRA_ARGS="--export-kv-cache-quant fp8"
```

Accepted values are `fp8`, `fp8_affine`, `nvfp4`, `nvfp4_affine`, and
`nvfp4_rotate`; the default is `none`. Cache precision is set independently of
the weight format. The two need not match, and an FP8 cache with NVFP4 weights
is a common combination.

When exporting a checkpoint that was trained after quantization, add
`--no-clamp-kv-scales` to preserve the learned scales instead of clamping them.

### Using a Prepared Recipe

Model Optimizer ships recipes that bundle a format, an algorithm, and a
key-value cache setting, some of them tuned for a specific model. Pass a recipe
name in place of the format and the script forwards it as `--recipe`:

```bash
TP=1 \
EP=8 \
MLM_SKIP_INSTALL=1 \
MLM_MODEL_CKPT=<megatron_core_checkpoint> \
MLM_MODEL_SAVE=/tmp/nemotron3-super-nvfp4 \
./quantize.sh nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 models/Nemotron-3-Super-120B-A12B/super-nvfp4
```

The script recognizes a recipe by the `/` in the name, or by a `.yaml` or `.yml`
suffix if you pass a path to your own file. A recipe takes precedence over
`--export-quant-cfg` and `--export-kv-cache-quant`, which are then ignored.
Size the parallelism for the model you are running; the 120B checkpoint above
needs more than one node.

### Letting the Search Choose Per-Layer Formats

Some layers lose more accuracy than others when quantized. Auto Quantize searches
for a per-layer assignment of formats subject to a target average bit width,
which is usually more accurate than applying one format everywhere.

Pass `auto` as the format and give the target width through `MLM_EXTRA_ARGS`:

```bash
TP=1 \
EP=8 \
MLM_SKIP_INSTALL=1 \
MLM_MODEL_CKPT=<megatron_core_checkpoint> \
MLM_MODEL_SAVE=/tmp/nemotron3-nano-auto \
MLM_EXTRA_ARGS="--auto-quantize-bits 4.0" \
./quantize.sh nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 auto
```

| Argument | Default | Description |
| --- | --- | --- |
| `--auto-quantize-bits` | *(required)* | Target effective bits per weight, for example `4.0` or `4.8`. |
| `--auto-quantize-formats` | `NVFP4_DEFAULT_CFG FP8_DEFAULT_CFG` | Formats to search over. |
| `--auto-quantize-method` | `gradient` | How layer sensitivity is scored, either `gradient` or `kl_div`. |
| `--auto-quantize-score-size` | `128` | Number of samples used for scoring. |
| `--auto-quantize-checkpoint` | `None` | Path to save and restore search state between runs. |

Auto Quantize requires `PP=1` and Model Optimizer 0.46 or newer.

### Checking the Result

Evaluate the quantized checkpoint before exporting it, so that any accuracy loss
shows up while the checkpoint is still easy to change:

```bash
TP=1 \
EP=8 \
MLM_SKIP_INSTALL=1 \
MLM_MODEL_CKPT=/tmp/nemotron3-nano-nvfp4 \
./mmlu.sh nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

`generate.sh` produces sample text from the same checkpoint if you want to read
the output directly.

## Quantization-Aware Distillation

PTQ is a single calibration pass, so at aggressive precisions such as NVFP4 the
quantized model can land noticeably below the original. QAD trains the quantized
model further while the original full-precision model supervises it: the teacher
runs alongside the student and the student learns to match the teacher's output
distribution rather than only the training labels.

QAD is the quantization case of the knowledge distillation support described in
[Knowledge Distillation](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/post_training/modelopt/distillation.md).
The student is the quantized checkpoint produced by `quantize.sh`, and the
teacher is the checkpoint you quantized. Running `finetune.sh` without a teacher
performs quantization-aware training instead, fitting the quantized model to the
data with the ordinary loss.

```bash
TP=1 \
EP=8 \
MLM_SKIP_INSTALL=1 \
MLM_MODEL_CKPT=/tmp/nemotron3-nano-nvfp4 \
MLM_MODEL_SAVE=/tmp/nemotron3-nano-nvfp4-qad \
MLM_EXTRA_ARGS="--export-kd-teacher-load <path_to_bf16_checkpoint> \
    --export-kd-teacher-model-config <path_to_bf16_checkpoint>/model_config.yaml" \
./finetune.sh nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

The teacher checkpoint should carry a `model_config.yaml`, or be given one
through `--export-kd-teacher-model-config`. Without either, the run continues
with a warning and assumes the teacher's architecture matches the student's,
which holds for QAD but not for distillation between different models.

`finetune.sh` supplies a complete default training recipe when you do not
override it, including a public instruction dataset, a fixed sample budget, and
a learning-rate schedule. Set `DATASET`, `MLM_DATA_ARGS`, `MLM_OPTIM_ARGS`, and
`MLM_TRAIN_ARGS` to train on your own data and schedule rather than accepting
those defaults.

To change the loss weighting or distill from intermediate layers as well as the
final logits, pass a distillation config through `--export-kd-cfg`. The fields
are documented in
[Knowledge Distillation](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/post_training/modelopt/distillation.md).

Interleaved pipeline parallelism is not supported during distillation.

### Models with Multi-Token Prediction Heads

Models that carry multi-token prediction heads have two parts that can be
trained: the base model and the heads themselves. Training both at once while
the base model is also adapting to quantization makes results hard to attribute,
so `--qad-train-target` selects which side receives gradients:

| Value | Effect |
| --- | --- |
| `mtp` | Train the prediction heads, freeze the base model. |
| `base` | Train the base model, freeze the prediction heads. |
| `both` | Train the base model and the heads together. |

The production recipe runs in two phases: distill the quantized base model
first with `--qad-train-target base`, then add the prediction heads and train
them against the fixed base with `--qad-train-target mtp`.

```bash
MLM_EXTRA_ARGS="--export-kd-teacher-load <path_to_bf16_checkpoint> --qad-train-target base"
```

This option only means something for a model configured with multi-token
prediction layers. None of the files under `conf/` enable them, so a run that
adds `--qad-train-target` to a stock configuration has no prediction-head
parameters to select between. In that situation `mtp` leaves nothing trainable
and the run proceeds without training anything, rather than reporting an error.

On a mixture-of-experts model, freezing a side also pins that side's router bias
terms. Those terms are updated from token counts during training rather than
from gradients, so they would otherwise keep drifting after the rest of the side
had been frozen.

`--freeze-base-for-mtp` is a deprecated spelling of `--qad-train-target mtp`.
Use `--qad-train-target`.

## Exporting for Deployment

Export converts a Megatron-LM checkpoint into a Hugging Face-style checkpoint
that TensorRT-LLM, vLLM, and SGLang can load:

```bash
PP=8 \
MLM_SKIP_INSTALL=1 \
MLM_MODEL_CKPT=/tmp/nemotron3-nano-nvfp4-qad \
EXPORT_DIR=/tmp/nemotron3-nano-export \
./export.sh nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

Export runs the model with tensor parallelism of 1 and spreads it across GPUs
with pipeline parallelism instead. The script forces `--tensor-model-parallel-size 1`
and warns if `TP` was set to anything else, but it does so after the launcher
has already sized itself, so `TP` and `EP` still determine how many processes
start. Set them to `1` for export, and use `PP` alone, or the command will
request more GPUs than it uses.

## Running on Multiple Nodes

Models that do not fit on one node run through Slurm with the wrapper in
`examples/post_training/modelopt/slurm/sbatch.sh`. Under Slurm the variables set
on the command line are not forwarded to the job, so they must be collected in a
setup script and passed through `SANDBOX_ENV_SETUP`:

```bash
export USER_FSW=<path_to_scratch_space>
export CONTAINER_IMAGE=<path_to_container_image>
export SANDBOX_ENV_SETUP=<path_to_your_env_script>
sbatch --nodes=8 slurm/sbatch.sh "quantize.sh nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 NVFP4_DEFAULT_CFG"
```

The process count is decided differently here. The wrapper launches ranks
through Slurm rather than `torchrun`, so the number of processes comes from the
node count and the tasks per node, not from the parallelism variables. Choose
parallel sizes in the setup script whose product matches the ranks the
allocation provides, or the job fails while setting up distributed
communication.

See
[Slurm Examples](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/post_training/modelopt/ADVANCED.md#slurm-examples)
for a worked multi-node run.

## Related Features

The same example scripts cover other Model Optimizer workflows that are outside
the scope of this page:

- **Pruning** removes structure from a model, such as layers or attention heads,
  and recovers quality with calibration data. See `prune.sh`.
- **Speculative decoding** trains a small draft model that proposes tokens for
  the larger model to verify. See
  [Speculative Decoding](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/post_training/modelopt/speculative.md).
