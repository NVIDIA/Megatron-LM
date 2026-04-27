<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Tensor Inspection

Megatron-LM integrates with TransformerEngine's debug / tensor inspection features via
[NVIDIA DLFW Inspect](https://github.com/NVIDIA/nvidia-dlfw-inspect). This
integration enables logging tensor statistics and altering GEMM precision for
debugging and analysis during training.

**NOTE: TELinear modules only.** This integration covers only TELinear
modules in the model architecture. In a standard
Megatron-Core GPT model these are surfaced as `linear_qkv`, `linear_proj`,
`linear_fc1`, and `linear_fc2`.

For a complete reference on configuration syntax, available features, and
advanced usage, see the upstream docs:

- [TransformerEngine Debug Documentation](https://github.com/NVIDIA/TransformerEngine/tree/main/docs/debug)
- [NVIDIA DLFW Inspect Documentation](https://github.com/NVIDIA/nvidia-dlfw-inspect/tree/main/docs)

## Enabling in a Training Run

Tensor inspection is controlled by four arguments on the training script:

| Argument | Description |
| --- | --- |
| `--tensor-inspect` | Enable tensor inspection. |
| `--tensor-inspect-features` | Path to a feature YAML file|
| `--tensor-inspect-feature-dirs` | Directories containing feature modules. Defaults to TransformerEngine's `transformer_engine/debug/features` directory if unset. |
| `--tensor-inspect-log-dir` | Root directory for inspection logs. Defaults to `--save` if unset. |


## Available Features

TransformerEngine ships two categories of debug features in
[`transformer_engine/debug/features`](https://github.com/NVIDIA/TransformerEngine/tree/main/transformer_engine/debug/features),
all applied only to TELinear modules matched by the layer selectors:

- **Statistics logging** (`LogTensorStats`, `LogFp8TensorStats`,
  `LogNvfp4TensorStats`) — emit per-tensor stats at a configurable frequency
  (`min`/`max`/`mean`/`std`, norms, dynamic range, FP8/NVFP4
  `underflows%`/`overflows%`/`mse` and more).
- **Precision overrides** (`DisableFP8GEMM`, `DisableFP8Layer`,
  `DisableQuantizationGEMM`, `DisableQuantizationLayer`, `PerTensorScaling`,
  `FakeQuant`) — selectively disable quantization on specific GEMMs or
  entire layers, force a particular scaling recipe, or run fake quantization
  for precision experiments.

## Feature Configuration

Feature configuration is supplied via a YAML file. Each top-level key names a feature
section. Within a section, the `layers` selector chooses which modules the
feature applies to, and the `transformer_engine` block names the feature
classes and their parameters.

### Example: high-precision tensor statistics on TELinear modules

```yaml
high_precision_tensor_stats:
  enabled: true
  layers:
    layer_name_regex_pattern: ".*linear_(fc1|fc2|qkv|proj)"
  transformer_engine:
    LogTensorStats:
      enabled: true
      tensors: [activation, gradient, weight]
      stats: [min, max, mean, std, l1_norm]
      freq: 5
      start_step: 1
      end_step: 10000
```

### Example: FP8 underflow tracking for the tensorwise recipe

```yaml
fp8_tensor_stat_collection:
  enabled: true
  layers:
    layer_name_regex_pattern: ".*"
  transformer_engine:
    LogFp8TensorStats:
      enabled: true
      tensors: [weight, activation, gradient]
      stats: ["underflows%"]
      freq: 5
      start_step: 1
      end_step: 10000
    LogTensorStats:
      enabled: true
      tensors: [wgrad, dgrad]
      stats: [min, max, mean, dynamic_range]
      freq: 5
      start_step: 1
      end_step: 10000
```

### Layer Selection

Features apply **only to TELinear modules** matched by
selectors in the `layers` section — non-TE modules in the same regex match
are silently ignored. Common patterns:

- `layer_name_regex_pattern: ".*"` – every matched TELinear layer.
- `layer_name_regex_pattern: ".*linear_(fc1|fc2)"` – MLP projections only.
- `layer_name_regex_pattern: ".*decoder\.layers\.(1|2|3).*linear_(fc1|fc2|qkv|proj)"` –
  TELinear modules in the first three transformer layers.

Tensor-level selectors control which tensor roles
are logged: `activation`, `gradient`, `weight`, `output`, `wgrad`, `dgrad`.

## Output

Two log files are written under `--tensor-inspect-log-dir`, one pair per global
rank:

- `<log_dir>/nvdlfw_inspect_logs/nvdlfw_inspect_globalrank-{N}.log` – main
  debug log: config loaded, layer-name assignments, per-GEMM precision
  decisions.
- `<log_dir>/nvdlfw_inspect_statistics_logs/nvdlfw_inspect_globalrank-{N}.log`
  – per-iteration tensor statistics.

### Main debug log

The main log records which feature sections loaded, how layer names were
resolved, and the per-GEMM precision decisions for each matched layer. The
following excerpt is from a 4-layer GPT pretraining run:

```text
INFO - Default logging to file enabled at ./logs/tensor_inspect
INFO - Reading config from ./examples/configs/high_prec_stats.yaml.
INFO - Loaded configs for ['high_precision_tensor_stats'].
INFO - Assigned layer name: model
INFO - Assigned layer name: model.module
INFO - Assigned layer name: model.module.module
INFO - Assigned layer name: model.module.module.embedding
INFO - Assigned layer name: model.module.module.embedding.word_embeddings
INFO - Assigned layer name: model.module.module.decoder.layers.1
INFO - Assigned layer name: model.module.module.decoder.layers.1.self_attention
INFO - Assigned layer name: model.module.module.decoder.layers.1.self_attention.linear_qkv
INFO - Assigned layer name: model.module.module.decoder.layers.1.self_attention.linear_proj
INFO - Assigned layer name: model.module.module.decoder.layers.1.mlp
INFO - Assigned layer name: model.module.module.decoder.layers.1.mlp.linear_fc1
INFO - Assigned layer name: model.module.module.decoder.layers.1.mlp.linear_fc2
...
```

### Statistics log

Each entry is a single scalar keyed by `{layer_name}_{tensor_role}_{stat}` and
tagged with the training iteration. The excerpt below is from the same run
with `LogTensorStats` collecting `[min, max, mean, std, l1_norm]` on
`activation`, `gradient`, and `weight` tensors at `freq: 2`:

```text
INFO - model.module.module.decoder.layers.1.self_attention.linear_qkv_activation_min      iteration=000002    value=-4.7812
INFO - model.module.module.decoder.layers.1.self_attention.linear_qkv_activation_max      iteration=000002    value=4.6875
INFO - model.module.module.decoder.layers.1.self_attention.linear_qkv_activation_mean     iteration=000002    value=0.0003
INFO - model.module.module.decoder.layers.1.self_attention.linear_qkv_activation_std      iteration=000002    value=0.9882
INFO - model.module.module.decoder.layers.1.self_attention.linear_qkv_activation_l1_norm  iteration=000002    value=3309568.0000
INFO - model.module.module.decoder.layers.1.self_attention.linear_qkv_weight_min          iteration=000002    value=-0.0957
INFO - model.module.module.decoder.layers.1.self_attention.linear_qkv_weight_max          iteration=000002    value=0.0923
INFO - model.module.module.decoder.layers.1.self_attention.linear_qkv_weight_mean         iteration=000002    value=-0.0000
INFO - model.module.module.decoder.layers.1.self_attention.linear_qkv_weight_std          iteration=000002    value=0.0200
INFO - model.module.module.decoder.layers.1.self_attention.linear_qkv_weight_l1_norm      iteration=000002    value=12544.0000
INFO - model.module.module.decoder.layers.1.self_attention.linear_proj_activation_min     iteration=000002    value=-1.5391
INFO - model.module.module.decoder.layers.1.self_attention.linear_proj_activation_max     iteration=000002    value=1.4844
INFO - model.module.module.decoder.layers.1.mlp.linear_fc1_activation_min                 iteration=000002    value=-4.7500
INFO - model.module.module.decoder.layers.1.mlp.linear_fc1_activation_max                 iteration=000002    value=4.6875
INFO - model.module.module.decoder.layers.1.mlp.linear_fc1_activation_l1_norm             iteration=000002    value=3309568.0000
```

### TensorBoard and Weights & Biases

When TensorBoard or W&B are configured on the training run (`--tensorboard-dir`
and/or `--wandb-project`), statistics are forwarded as scalar metrics in
addition to the text log. Each metric is named
`{layer_name}_{tensor_role}_{stat}` and plotted against the training step, so
series can be filtered by layer, tensor, or statistic in the usual way.
