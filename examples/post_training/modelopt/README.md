<div align="center">

# Model Optimizer Integrated Examples


[Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) |
[Local Examples](#getting-started-in-a-local-environment) |
[Configuration](./ADVANCED.md#advanced-configuration) |
[Slurm Examples](./ADVANCED.md#slurm-examples) |
[Speculative Decoding](./speculative.md) |
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

> **📙 NOTE:** ModelOpt supports different quantization formats. By default, we simulate the
> low-precision numerical behavior (fake-quant) which can be run on GPUs with compute > 80.
> Real low-precision paramters (e.g. `E4M3` or `E2M1`)
> and low-precision compute (e.g. `FP8Linear`) are also supported depending on GPU compute capability.
> **See [Adanvanced Topics](./ADVANCED.md) for details**.

```sh
\
    TP=1 \
    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
    MLM_MODEL_SAVE=/tmp/Llama-3.2-1B-Instruct_quant \
    ./quantize.sh meta-llama/Llama-3.2-1B-Instruct nvfp4

\
    PP=1 \
    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Instruct_quant \
    EXPORT_DIR=/tmp/Llama-3.2-1B-Instruct_export \
    ./export.sh meta-llama/Llama-3.2-1B-Instruct
```

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

See [Adanvanced Topics](./ADVANCED.md) for a `moonshotai/Kimi-K2-Instruct` EAGLE3 training example using `slurm`.

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

Checkout pruning getting started section and guidelines for configuring pruning parameters in the [ModelOpt pruning README](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/pruning).

Pruning is supported for GPT and Mamba models in Pipeline Parallel mode. Available pruning dimensions are:

- `TARGET_FFN_HIDDEN_SIZE`
- `TARGET_HIDDEN_SIZE`
- `TARGET_NUM_ATTENTION_HEADS`
- `TARGET_NUM_QUERY_GROUPS`
- `TARGET_MAMBA_NUM_HEADS`
- `TARGET_MAMBA_HEAD_DIM`
- `TARGET_NUM_MOE_EXPERTS`
- `TARGET_MOE_FFN_HIDDEN_SIZE`
- `TARGET_MOE_SHARED_EXPERT_INTERMEDIATE_SIZE`
- `TARGET_NUM_LAYERS`
- `LAYERS_TO_DROP` (comma separated, 1-indexed list of layer numbers to directly drop)

Example for depth pruning Qwen3-8B from 36 to 24 layers:

```sh
PP=1 \
TARGET_NUM_LAYERS=24 \
HF_MODEL_CKPT=<pretrained_model_name_or_path> \
MLM_MODEL_SAVE=Qwen3-8B-Pruned \
./prune.sh Qwen/Qwen3-8B
```

> [!TIP]
> If number of layers in the model is not divisible by pipeline parallel size (PP), you can configure uneven
> PP by setting `MLM_EXTRA_ARGS="--decoder-first-pipeline-num-layers <X> --decoder-last-pipeline-num-layers <Y>"`

> [!TIP]
> You can reuse pruning scores for pruning same model again to different architectures by setting
> `PRUNE_ARGS="--pruning-scores-path <path_to_save_scores>"`

> [!NOTE]
> When loading pruned M-LM checkpoint for subsequent steps, make sure overwrite the pruned parameters in the
> default `conf/` by setting `MLM_EXTRA_ARGS`. E.g.: for loading above pruned Qwen3-8B checkpoint for mmlu, set:
> `MLM_EXTRA_ARGS="--num-layers 24"`

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
TBD
