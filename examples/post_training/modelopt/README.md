<div align="center">

# TensorRT Model Optimizer Integrated Examples


[TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) |
[Local Examples](#getting-started-in-a-local-environment) |
[Configuration](ADVANCED.md#learn-more-about-configuration) |
[Slurm Examples](ADVANCED.md#slurm-examples) |
[Speculative Decoding](speculative.md) |
[Advanced Topics](ADVANCED.md)

</div>

[TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) (**ModelOpt**, `nvidia-modelopt`)
provides end-to-end model optimization for
NVIDIA hardware including quantization (real or simulated), sparsity, knowledge distillation, pruning,
neural architecture search, and speulative decoding.


## Major Features

- Start from Hugging Face pretrained model checkpoint with on-the-fly conversion.
- Support all kinds of model parallelism (TP, EP, ETP, PP).
- Export to TensorRT-LLM, vLLM, and SGLang ready unified checkpoint.

## Support Matrix {Model}x{Features}

| Model (`conf/`) | Quantization | EAGLE3 | Pruning (PP only) | Distillation |
| :---: | :---: | :---: | :---: | :---: |
| `moonshotai/Kimi-K2-Instruct` | ✅ | ✅ | - | - |
| `Qwen/Qwen3-{30B-A3B, 235B-A22B}` | **WAR** | ✅ | - | - |
| `Qwen/Qwen3-{0.6B, 8B}` | ✅ | ✅ | ✅ | ✅ |
| `deepseek-ai/DeepSeek-R1` | ✅ | ✅ | - | - |
| `meta-llama/Llama-{3.1-8B, 3.1-405B, 3.2-1B}-Instruct` | ✅ | ✅ | ✅ | ✅ |

## Getting Started in a Local Environment

Install `nvidia-modelopt` from [PyPI](https://pypi.org/project/nvidia-modelopt/):
```sh
pip install -U nvidia-modelopt
```
Alternatively, you can install from [source](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
to try our latest features.


### ⭐ NVFP4 Quantization, Qauntization-Aware Training, and Model Export

Provide the pretrained checkpoint path through variable `${HF_MODEL_CKPT}` and provide variable
`${MLM_MODEL_SAVE}` which stores a resumeable Megatron-LM distributed checkpoint. To export
Hugging Face-Like quantized checkpoint for TensorRT-LLM, vLLM, or SGLang deployement,
provide `${EXPORT_DIR}` to `export.sh`.

> **📙 NOTE:** ModelOpt supports different quantization formats. By default, we simulate the
> low-precision numerical behavior (fake-quant) which can be run on GPUs with compute > 80.
> Real low-precision paramters (e.g. `E4M3` or `E2M1`)
> and low-precision compute (e.g. `FP8Linear`) are also supported depending on GPU compute capability.
> **See [Adanvanced Topics](advanced.md) for details**.

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

> **❗ IMPORTANT:** The first positional arugment (e.g. `meta-llama/Llama-3.2-1B-Instruct`) of each script
> is the config name used to match the supported model config in `conf/`. The pretrained checkpoint should
> be downloaded and provided through `${HF_MODEL_CKPT}`.

Loading the saved distributed checkpoint, the quantized Megatron model can be resumed for inference
(generate or evaluate) or training (SFT or PEFT). To read more about these features, see
[Adanvanced Topics](advanced.md). To learn more about the design, see our [Design]() document [WIP].

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

See [Adanvanced Topics](ADVANCED.md) for a `moonshotai/Kimi-K2-Instruct` EAGLE3 training example using `slurm`.

### ⭐ Pruning

Pruning is supported for GPT and Mamba models. Available pruning options are:
- `TARGET_FFN_HIDDEN_SIZE`
- `TARGET_HIDDEN_SIZE`
- `TARGET_NUM_ATTENTION_HEADS`
- `TARGET_NUM_QUERY_GROUPS`
- `TARGET_MAMBA_NUM_HEADS`
- `TARGET_MAMBA_HEAD_DIM`
- `TARGET_NUM_LAYERS`
- `LAYERS_TO_DROP` (comma separated, 1-indexed list of layer numbers to directly drop)

```sh
PP=1 \
TARGET_NUM_LAYERS=24 \
HF_MODEL_CKPT=<pretrained_model_name_or_path> \
MLM_MODEL_SAVE=/tmp/Qwen3-8B-DPruned \
./prune.sh qwen/Qwen3-8B
```

## Advanced Usage
TBD
