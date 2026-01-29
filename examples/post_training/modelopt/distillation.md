# Megatron-LM ModelOpt Distillation Integration

## How To

### Prerequisites

In order to perform soft-label Knowledge Distillation between two models on a specific dataset,
we take a larger teacher model which has already been fully trained and use its logits as
labels for a smaller student model.

We require the following pieces of data:
* Teacher model weights
* Student model weights (unless starting from scratch)
* NeMo-format config file for teacher model
* Tokenizer
* Dataset

And optionally:
* Distillation run config file

### Teacher checkpoint format

We enforce the use of a config yaml in [NeMo](https://github.com/NVIDIA/NeMo) checkpoint-format style to define the arguments to the teacher model.
The normal command-line arguments go toward constructing the student, thus the values in this file
override the student arguments before being handed to the teacher constructor. This file must be either passed in via
`--export-kd-teacher-model-config` or be named `model_config.yaml` in the root of the teacher model checkpoint folder.
Unlike NeMo-generated checkpoints, Megatron-LM checkpoints do not contain this file by default and must be manually created.

> NOTE: Not all keys in the NeMo-style yaml correspond 1:1 to the argument names for Megatron-LM. These
are converted in `megatron/post_training/model_builder.py`.

### Distillation config format

Configuring the distillation run is done via a separate YAML file with the following fields:

```yaml
logit_layers: ["output_layer", "output_layer"]
intermediate_layer_pairs:
  - ["decoder.layers.0.input_layernorm", "decoder.layers.0.input_layernorm"]
  - ["decoder.final_layernorm", "decoder.layers.30.input_layernorm"]
skip_lm_loss: true
kd_loss_scale: 10.0
logit_kl_temperature: 1.0
```

* `logit_layers` defines the names of the student and teacher submodules, respectively, whose outputs are the logits.
* `intermediate_layer_pairs` defines the potentially multiple – or zero – pairs of intermediate activation layers to also perform loss on.
* `skip_lm_loss` decides whether or not to compute and combine the original training LM loss with the KD loss.
* `kd_loss_scale` will scale the KD loss before adding it to the LM loss, if `skip_lm_loss` is `False`.
* `logit_kl_temperature` is the temperature smoothing factor to multiply the logits by prior to softmax and loss.

Without this configuration file, the default logits-only distillation with scale and temperatures of 1.0 will be performed.

### Training

Distillation is triggered by calling `pretrain_gpt.py` or `pretrain_mamba.py` with the following arguments:

```bash
--export-kd-teacher-load <path-to-teacher-checkpoint>
--export-te-mcore-model
```

optionally alongside the additional following arguments:

```bash
--export-kd-distill-cfg <path-to-distill-config-yaml-file>
--export-kd-teacher-model-config <path-to-teacher-model-config-file>
```

> NOTE: If the teacher checkpoint happens to be in a different format from the student's (whose format is specified via `--ckpt-format`), it can
be distinguished separately using the additional flag `--export-kd-teacher-ckpt-format`.

## Distillation API and design

Knowledge Distillation is done via the [NVIDIA Model Optimizer library](https://github.com/NVIDIA/Model-Optimizer).

The model creation step wraps the base model as the student in a
`modelopt.torch.distill.DistillationModel` wrapper which also contains the teacher model.

Model Optimizer modifies the model using the loss criterion present in the distillation config yaml file, which
defines a loss function between two module attribute names of the teacher and student model, respectively.

Default loss function used between logits is a KL-Divergence Loss and loss used among intermediate tensors is Cosine-Similarity,
both defined in `modelopt.torch.distill.plugins.megatron`.

## Restrictions

* Interleaved Pipeline Parallel is unsupported for Distillation.

## Known Issues

* An unknown memory allocation (a few megabytes per microbatch) takes place when the model is converted to a
`modelopt.torch.distill.DistillationModel`. If `--manual-gc` is enabled, it can easily lead to an OOM after some iterations.

* A CUDA kernel issue is occurring where student's forward latency is severly prolonged compared to running student forward
without a teacher model. This means the total time per iteration may be up to 40% longer than ideally expected.
