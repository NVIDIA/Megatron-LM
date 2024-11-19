# Megatron-LM ModelOpt Distillation Integration

## Table of Contents

[[_TOC_]]

## How To

### Prerequisites

In order to perform soft-label Knowledge Distillation between two models on a specific dataset,
we take a larger teacher model which has already been fully trained and use its logits as
labels for a smaller student model.

We require the following pieces of data:
* Teacher model weights
* Student model weights (unless starting from scratch)
* NeMo-format config file for teacher model
* Distillation run config file
* Tokenizer
* Dataset

It also requires the installation of the [NVIDIA Model Optimizer library](https://github.com/NVIDIA/TensorRT-Model-Optimizer) (minimum version 0.15)

### Teacher checkpoint format

We enforce the use of a config yaml in [NeMo](https://github.com/NVIDIA/NeMo) checkpoint-format style to define the arguments to the teacher model.
The normal command-line arguments go toward constructing the student, thus the values in this file
override the student arguments before being handed to the teacher constructor. This file must be
named `model_config.yaml` and be placed in the root of the teacher model checkpoint folder.
Unlike NeMo-generated checkpoints, Megatron-LM checkpoints do not contain these files by default and must be manually created.

> NOTE: Not all keys in the NEMO-style yaml correspond 1:1 to the argument names for Megatron-LM. These
are converted in `megatron/inference/gpt/model_provider.py`.

### Distillation config format

Configuring the distillation run is done via a separate YAML file with the following fields:

```yaml
logit_layers: ["output_layer", "output_layer"]
intermediate_layer_pairs:
  - ["decoder.layers.0.input_layernorm", "decoder.layers.0.input_layernorm"]
  - ["decoder.final_layernorm", "decoder.layers.30.input_layernorm"]
skip_lm_loss: true
kd_loss_scale: 10.0
```

* `logit_layers` defines the names of the student and teacher submodules, respectively, whose outputs are the logits.
* `intermediate_layer_pairs` defines the potentially multiple – or zero – pairs of intermediate activation layers to also perform loss on.
* `skip_lm_loss` decides whether or not to compute and combine the original training LM loss with the KD loss
* `kd_loss_scale` will scale the KD loss before adding it to the LM loss, if `skip_lm_loss` is `True`.

### Training

Distillation is triggered by calling `megatron/inference/pretrain_gpt_modelopt.py` while the `--kd-teacher-load` argument is not empty.

Use the regular arguments you would for `pretrain_gpt.py` in addition to the following:

```bash
--kd-teacher-load <path-to-teacher-checkpoint>
--kd-distill-cfg <path-to-distill-config-yaml-file>
--export-te-mcore-model
```

## Distillation API and design

Knowledge Distillation is done via the [NVIDIA Model Optimizer library](https://github.com/NVIDIA/TensorRT-Model-Optimizer).

The model creation step wraps the base model as the student in a
`modelopt.torch.distill.DistillationModel` wrapper which also contains the teacher model.

Model Optimizer modifies the model using the loss criterion present in the distillation config yaml file, which
defines a loss function between two module attribute names of the teacher and student model, respectively.

Default loss function used between logits is a KL-Divergence Loss and loss used among intermediate tensors is Cosine-Similarity,
both defined in `megatron/inference/algos/distillation.py`.

## Restrictions

* Pipeline Parallel is currently unsupported for Distillation.

* Only Megatron-Core (not legacy Megatron-LM) is supported for Distillation.

## Known Issues

* An unknown memory allocation (a few megabytes per microbatch) takes place when the model is converted to a
`modelopt.torch.distill.DistillationModel`. If `--manual-gc` is enabled, it can easily lead to an OOM after some iterations.

* A CUDA kernel issue is occurring where student's forward latency is severly prolonged compared to running student forward
without a teacher model. This means the total time per iteration may be up to 40% longer than ideally expected.
