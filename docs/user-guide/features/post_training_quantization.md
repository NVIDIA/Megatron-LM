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

Both examples on this page use `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`. The
same commands apply to the other models listed in the support matrix in the
[ModelOpt examples README](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/post_training/modelopt/README.md),
including the Super and Ultra sizes of Nemotron 3.

## Prerequisites

Install the Model Optimizer library:

```bash
pip install -U nvidia-modelopt
```

Download the pretrained Hugging Face checkpoint you intend to quantize and point
`HF_MODEL_CKPT` at it. The scripts convert it to Megatron-LM format on the fly.

Each script takes a configuration name as its first positional argument. That
name selects a file under
`examples/post_training/modelopt/conf/`, which supplies the model architecture
arguments. For Nemotron 3 Nano the name is
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.

Parallelism and paths are passed as environment variables rather than flags:

| Variable | Meaning |
| --- | --- |
| `HF_MODEL_CKPT` | Path to the pretrained Hugging Face checkpoint. |
| `MLM_MODEL_SAVE` | Where to write the resulting Megatron-LM checkpoint. |
| `MLM_MODEL_CKPT` | Megatron-LM checkpoint to load for a later step. |
| `EXPORT_DIR` | Where to write the exported deployment checkpoint. |
| `TP`, `PP`, `EP`, `ETP`, `CP` | Tensor, pipeline, expert, expert-tensor, and context parallel sizes. |
| `MLM_EXTRA_ARGS` | Any additional Megatron-LM arguments. |

For the full list, see
[Advanced Topics](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/post_training/modelopt/ADVANCED.md).

## Post-Training Quantization

Run `quantize.sh` with the configuration name and a quantization format. The
script loads the pretrained checkpoint, runs a calibration pass, and saves a
Megatron-LM distributed checkpoint that can be resumed for inference, further
training, or export.

```bash
cd examples/post_training/modelopt

TP=8 \
EP=8 \
HF_MODEL_CKPT=<path_to_pretrained_checkpoint> \
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

To quantize the key-value cache as well, add the precision you want through
`MLM_EXTRA_ARGS`:

```bash
MLM_EXTRA_ARGS="--export-kv-cache-quant fp8"
```

### Using a Prepared Recipe

Model Optimizer ships recipes that bundle a format, an algorithm, and a
key-value cache setting, some of them tuned for a specific model. Pass a recipe
name in place of the format and the script forwards it as `--recipe`:

```bash
./quantize.sh nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 models/Nemotron-3-Super-120B-A12B/super-nvfp4
```

The script recognizes a recipe by the `/` in the name, or by a `.yaml` or `.yml`
suffix if you pass a path to your own file. A recipe takes precedence over
`--export-quant-cfg` and `--export-kv-cache-quant`, which are then ignored.

### Letting the Search Choose Per-Layer Formats

Some layers lose more accuracy than others when quantized. Auto Quantize searches
for a per-layer assignment of formats subject to a target average bit width,
which is usually more accurate than applying one format everywhere.

Pass `auto` as the format and give the target width through `MLM_EXTRA_ARGS`:

```bash
TP=8 \
EP=8 \
HF_MODEL_CKPT=<path_to_pretrained_checkpoint> \
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
TP=8 \
EP=8 \
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
teacher is the checkpoint you quantized.

Quantize first, then fine-tune with a teacher attached:

```bash
TP=8 \
EP=8 \
MLM_MODEL_CKPT=/tmp/nemotron3-nano-nvfp4 \
MLM_MODEL_SAVE=/tmp/nemotron3-nano-nvfp4-qad \
MLM_EXTRA_ARGS="--export-kd-teacher-load <path_to_bf16_checkpoint>" \
./finetune.sh nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

Because the teacher and student share an architecture here, no teacher config
file is needed. Distillation only requires one when the two differ; in that case
supply it through `--export-kd-teacher-model-config`, or place a
`model_config.yaml` in the root of the teacher checkpoint directory.

To change the loss weighting or distill from intermediate layers as well as the
final logits, pass a distillation config through `--export-kd-distill-cfg`. The
fields are documented in
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
HF_MODEL_CKPT=<path_to_pretrained_checkpoint> \
MLM_MODEL_CKPT=/tmp/nemotron3-nano-nvfp4-qad \
EXPORT_DIR=/tmp/nemotron3-nano-export \
./export.sh nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

Export runs with tensor parallelism forced to 1; use pipeline parallelism to
spread a large model across GPUs. The script overrides `TP` and warns if you set
it to anything else.

## Running on Multiple Nodes

Models that do not fit on one node run through Slurm with the wrapper in
`examples/post_training/modelopt/slurm/sbatch.sh`. Under Slurm, variables are
not forwarded automatically, so they must be collected in a setup script and
passed through `SANDBOX_ENV_SETUP`:

```bash
export USER_FSW=<path_to_scratch_space>
export CONTAINER_IMAGE=<path_to_container_image>
export SANDBOX_ENV_SETUP=<path_to_your_env_script>
sbatch --nodes=8 slurm/sbatch.sh "quantize.sh nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 NVFP4_DEFAULT_CFG"
```

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
