<div align="center">

# Model Optimizer Integrated Examples


[Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) |
[Local Examples](#getting-started-in-a-local-environment) |
[Configuration](./ADVANCED.md#advanced-configuration) |
[Slurm Examples](./ADVANCED.md#slurm-examples) |
[Speculative Decoding](./speculative.md) |
[Knowledge Distillation](./distillation.md) |
[Advanced Topics](./ADVANCED.md)

</div>

[Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) (**ModelOpt**, `nvidia-modelopt`)
provides end-to-end model optimization for NVIDIA hardware including quantization (real or simulated),
knowledge distillation, pruning, speculative decoding, and more.


## Major Features

- Start from Hugging Face pretrained model checkpoint with on-the-fly conversion to Megatron-LM checkpoint format.
- Support all kinds of model parallelism (TP, EP, ETP, PP).
- Export to TensorRT-LLM, vLLM, and SGLang ready unified checkpoint.

## Support Matrix {Model}x{Features}

| Model (`conf/`) | Quantization | EAGLE3 | Pruning (PP only) | Distillation |
| :---: | :---: | :---: | :---: | :---: |
| `deepseek-ai/DeepSeek-R1` | ✅ | ✅ | - | - |
| `meta-llama/Llama-{3.1-8B, 3.1-405B, 3.2-1B}-Instruct` | ✅ | ✅ | ✅ | ✅ |
| `meta-llama/Llama-4-{Scout,Maverick}-17B-{16,128}E-Instruct` | ✅ | ✅ | - | - |
| `moonshotai/Kimi-K2-Instruct` | ✅ | ✅ | - | - |
| `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | ✅ | - | ✅ | ✅ |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | ✅ | - | ✅ | ✅ |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | ✅ | - | ✅ | ✅ |
| `openai/gpt-oss-{20b, 120b}` | ✅ | **Online** | ✅ | ✅ |
| `Qwen/Qwen3-{0.6B, 8B}` | ✅ | ✅ | ✅ | ✅ |
| `Qwen/Qwen3-{30B-A3B, 235B-A22B}` | **WAR** | ✅ | ✅ | ✅ |

## Getting Started in a Local Environment

Install `nvidia-modelopt` from [PyPI](https://pypi.org/project/nvidia-modelopt/):
```sh
pip install -U nvidia-modelopt
```
Alternatively, you can install from [source](https://github.com/NVIDIA/Model-Optimizer)
to try our latest features.

> **❗ IMPORTANT:** The first positional argument (e.g. `meta-llama/Llama-3.2-1B-Instruct`) of each script
> is the config name used to match the supported model config in `conf/`. The pretrained HF checkpoint should
> be downloaded and provided through `${HF_MODEL_CKPT}`.


### ⭐ NVFP4 Quantization, Qauntization-Aware Training, and Model Export

Provide the pretrained checkpoint path through variable `${HF_MODEL_CKPT}` and provide variable
`${MLM_MODEL_SAVE}` which stores a resumeable Megatron-LM distributed checkpoint. To export
Hugging Face-Like quantized checkpoint for TensorRT-LLM, vLLM, or SGLang deployement,
provide `${EXPORT_DIR}` to `export.sh`.

> **📙 NOTE:** ModelOpt supports different quantization formats which are listed in the [ModelOpt quant configs](https://github.com/NVIDIA/Model-Optimizer/blob/7971fff05882da7eae16eae6bc927d1481dcd63f/modelopt/torch/quantization/config.py#L626).
> The quant config is specified by the full config name in all-caps, e.g. NVFP4_DEFAULT_CFG.
> By default, we simulate the low-precision numerical behavior (fake-quant) which can be run on GPUs with compute > 80.
> Real low-precision paramters (e.g. `E4M3` or `E2M1`)
> and low-precision compute (e.g. `FP8Linear`) are also supported depending on GPU compute capability.
> **See [Advanced Topics](./ADVANCED.md) for details**.

```sh
\
    TP=1 \
    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
    MLM_MODEL_SAVE=/tmp/Llama-3.2-1B-Instruct_quant \
    ./quantize.sh meta-llama/Llama-3.2-1B-Instruct NVFP4_DEFAULT_CFG 

\
    PP=1 \
    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Instruct_quant \
    EXPORT_DIR=/tmp/Llama-3.2-1B-Instruct_export \
    ./export.sh meta-llama/Llama-3.2-1B-Instruct
```

To export the model for vLLM fakequant example in `modelopt/examples/vllm_serve/vllm_serve_fakequant.py`,
export the model with flag `--export-vllm-fq`:
```sh
\
    PP=1 \
    MLM_EXTRA_ARGS=--export-vllm-fq \
    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Instruct_quant \
    EXPORT_DIR=/tmp/Llama-3.2-1B-Instruct_export \
    ./export.sh meta-llama/Llama-3.2-1B-Instruct
```

For KV cache quantization, add a flag like `MLM_EXTRA_ARGS="--export-kv-cache-quant fp8"` while specifying your desired KV cache precision (see `KV_QUANT_CFG_CHOICES` in `quantize.py`).

### ⭐ Online BF16 EAGLE3 Training

Online EAGLE3 training has both the target (frozen) and draft models in the memory where the `hidden_states`
required for training is generated on the fly. Periodically, acceptance length (AL, the higher the better) is
evaluated on MT-Bench prompts. Use the same `export.sh` script to export the EAGLE3 checkpoint for
deployment.

```sh
\
    TP=1 \
    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
    MLM_MODEL_SAVE=/tmp/Llama-3.2-1B-Eagle3 \
    ./eagle3.sh meta-llama/Llama-3.2-1B-Instruct

\
    PP=1 \
    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Eagle3 \
    EXPORT_DIR=/tmp/Llama-3.2-1B-Eagle3-Export \
    ./export.sh meta-llama/Llama-3.2-1B-Instruct
```

See [Advanced Topics](./ADVANCED.md) for a `moonshotai/Kimi-K2-Instruct` EAGLE3 training example using `slurm`.

### ⭐ Offline BF16 EAGLE3 Training
Unlike online EAGLE3 training, offline workflow precomputes target model `hidden_states` and dumps to disk.
Then only the draft model is called during training. AL is no longer reported during training. After training,
`export.sh` is used to export EAGLE3 checkpoint.

```sh
\
    # Convert to online eagle3 model for base model feature extraction
    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
    MLM_MODEL_SAVE=/tmp/Llama-3.2-1B-Eagle3 \
    MLM_EXTRA_ARGS="--algorithm eagle3" \
    ./convert.sh meta-llama/Llama-3.2-1B-Instruct

\
    # Dump base model feature to disk
    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Eagle3 \
    MLM_EXTRA_ARGS="--output-dir /tmp/offline_data" \
    ./offline_feature_extrach.sh meta-llama/Llama-3.2-1B-Instruct

\
    # Convert to offline eagle3 model
    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
    MLM_MODEL_SAVE=/tmp/Llama-3.2-1B-Eagle3-offline \
    MLM_EXTRA_ARGS="--algorithm eagle3 --export-offline-model" \
    ./convert.sh meta-llama/Llama-3.2-1B-Instruct

\
    # Train the offline eagle3 model using extracted features
    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Eagle3-offline \
    MLM_MODEL_SAVE=/tmp/Llama-3.2-1B-Eagle3-offline \
    MLM_EXTRA_ARGS="--export-offline-model --offline-distillation-data /tmp/offline_data" \
    ./finetune.sh meta-llama/Llama-3.2-1B-Instruct

\
    # Export the trained eagle3 checkpoint
    PP=1 \
    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Eagle3-offline \
    EXPORT_DIR=/tmp/Llama-3.2-1B-Eagle3-Export \
    MLM_EXTRA_ARGS="--export-offline-model" \
    ./export.sh meta-llama/Llama-3.2-1B-Instruct
```

### ⭐ Pruning

Pruning is supported for GPT and Mamba models in Pipeline Parallel mode. The `prune.sh` script
prunes a model by passing `--prune-export-config '<json_without_spaces>'` to `prune.py` via `MLM_EXTRA_ARGS`.
The JSON describes the target pruned architecture; calibration data is used to compute importance
scores that drive the dimension reduction.

Supported hyperparameters (any subset can appear as keys in `--prune-export-config`):
`hidden_size`, `ffn_hidden_size`, `num_attention_heads`, `num_query_groups`, `mamba_num_heads`,
`mamba_head_dim`, `num_moe_experts`, `moe_ffn_hidden_size`, `moe_shared_expert_intermediate_size`,
`num_layers`.

Example for depth pruning Qwen3-8B from 36 to 24 layers:

```sh
PP=1 \
MLM_EXTRA_ARGS='--prune-export-config {"num_layers":24}' \
HF_MODEL_CKPT=<pretrained_model_name_or_path> \
MLM_MODEL_SAVE=Qwen3-8B-Pruned \
./prune.sh Qwen/Qwen3-8B
```

The default calibration dataset is `nemotron-post-training-dataset-v2` (gated, requires
`hf auth login`). Override it by adding `--calib-dataset <hf_dataset_name_or_local_jsonl>`
to `MLM_EXTRA_ARGS` (e.g. `cnn_dailymail` for an ungated alternative).

> [!TIP]
> If number of layers in the model is not divisible by pipeline parallel size (PP), you can configure uneven
> PP by adding `--decoder-first-pipeline-num-layers <X> --decoder-last-pipeline-num-layers <Y>` to `MLM_EXTRA_ARGS`.

> [!TIP]
> You can reuse intermediate pruning scores when pruning the same model again to a different config
> by adding `--prune-intermediate-ckpt <path_to_cache_dir>` to `MLM_EXTRA_ARGS`.

> [!NOTE]
> When loading pruned M-LM checkpoint for subsequent steps, make sure overwrite the pruned parameters in the
> default `conf/` by setting `MLM_EXTRA_ARGS`. E.g.: for loading above pruned Qwen3-8B checkpoint for mmlu, set:
> `MLM_EXTRA_ARGS="--num-layers 24"`

For NAS-based automatic pruning (search across many candidate architectures and pick the best via
MMLU scoring), see the [Megatron-Bridge pruning example](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/megatron_bridge#pruning).
Checkout pruning getting started and general guidelines in the [ModelOpt pruning README](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/pruning).

### ⭐ Inference and Training

The saved Megatron-LM distributed checkpoint (output of above scripts) can be resumed for inference
(generate or evaluate) or training (SFT or PEFT). To read more about these features, see
[Advanced Topics](./ADVANCED.md).

```sh
\
    TP=1 \
    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Instruct_quant \
    ./generate.sh meta-llama/Llama-3.2-1B-Instruct

\
    TP=1 \
    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Instruct_quant \
    ./mmlu.sh meta-llama/Llama-3.2-1B-Instruct

\
    TP=1 \
    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Instruct_quant \
    ./finetune.sh meta-llama/Llama-3.2-1B-Instruct
```

## Advanced Usage
To contribute, please ping [@NVIDIA/post-training](https://github.com/orgs/NVIDIA/teams/post-training) team members. We format the examples with
```
uvx black@24.10.0 .
uvx isort .
```
