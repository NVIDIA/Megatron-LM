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
are converted in `megatron/post_training/model_config.py`.

### Distillation config format

Configuring the distillation loss is done via a separate YAML file, passed with `--export-kd-cfg`, with the following fields:

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

The YAML is read once, when the CLI is translated, into the fields of the same names on the run's
`DistillationConfig` (`PretrainConfigContainer.distillation`). A run config built directly can set those
fields itself; `DistillationConfig.kd_cfg` (a path to this YAML) is deprecated.

### Training

Distillation is triggered by calling `pretrain_gpt.py` or `pretrain_hybrid.py` with the following arguments:

```bash
--export-kd-teacher-load <path-to-teacher-checkpoint>
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

### Configuration-based construction

Distillation is a run-level setting: `PretrainConfigContainer.distillation`, next to the
student's `checkpoint`, rather than part of the student's model config.
`distillation_config_from_args(args, model_config)` resolves the teacher architecture into
`DistillationConfig.teacher_model` and reads the loss settings, at CLI translation time.
The teacher, its checkpoint path/format, and the loss are serialized with the run config;
the ModelOpt setup options and independent MTP specification are serialized with the stock
GPT/Hybrid model config. Rebuilding a deserialized config reconstructs setup hooks
automatically; it does not need to reread the teacher architecture YAML or retain the
original argparse namespace.

`DistillationConfig.teacher_model_config` (argparse-name overrides of the student's
architecture) exists only to carry `--export-kd-teacher-model-config` into
`teacher_model`. A YAML run config should instead give the teacher directly as
`teacher_model`, a full `ModelConfig` of its own.

Setup runs in this order: restore ModelOpt state, apply QAD freezing and LSQ
setup, register checkpoint key remapping, run user pre-wrap hooks (including
training freezes), then convert for KD. The training loop passes the run's
`checkpoint` and `distillation` to `build_distributed_models(..., checkpoint=...,
distillation=...)`; a custom runner using the builders directly must do the same. Teacher
weights are loaded after wrapping by `load_kd_teacher_checkpoint(models, cfg.distillation)`,
which such a runner must also call.

The run container owns the stock model config and student checkpoint.
`model.transformations` contains only `ModelTransformConfig` options (QAD, LSQ,
TE key mapping); it does not duplicate checkpoint locations. The teacher's own checkpoint reference is in
`DistillationConfig`. Inference containers have no distillation section.

Post-training CLI entry points use `modelopt_inference_config_from_args(args)`
once, before runtime initialization. After tokenizer setup,
`resolve_tokenizer_vocab_size(cfg, args.padded_vocab_size)` finalizes the same
model config. Call `build_inference_models(cfg)` to construct it without DDP.
CPU export passes `device="cpu"`; conversion disables the precision wrapper with
`mixed_precision=False`.

There are no ModelOpt-specific model-config subclasses or model builders.
Whole-run YAML loading preserves stock builder metadata and transformation
specs. Container validation checks KD compatibility and placement conflicts
before construction.

Megatron is still migrating from CLI arguments. Unmigrated runtime services
continue using `get_args()`; `load_checkpoint` still reads the load path and
format from args, which are the values the run config's `checkpoint` is built
from. The pre-wrap restore hook reads them from `cfg.checkpoint`.

Builders and configuration validation keep explicit inputs so independent configs can
be built and tested without installing global state.
