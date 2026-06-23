# Llama-3.1-Nemotron-Nano-VL-8B-V1

See [Hugging Face](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1) for details.

# Checkpoints

[HuggingFace version](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1)

[Megatron-Core version](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1-mcore)

# Setup

## Docker image

See `examples/multimodal/llama_3p1_nemotron_nano_vl_8b_v1/Dockerfile`.

## Dataset preparation

We use [Megatron Energon](https://github.com/NVIDIA/Megatron-Energon) for multimodal dataloading.

## Model

You can download trained tensor parallel size 1 and 4 Megatron checkpoints [here](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1-mcore).
Alternatively, you can follow the steps in [Model conversion](#model-conversion) and [Training](#training) below to prepare a model
and run pretraining and SFT from scratch using a prepared dataset.

### Model conversion

#### Language model conversion

We start from [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) from HuggingFace.
Please download it and run the following command to convert it to Megatron format.
```
export LLAMA_DOWNLOAD_DIR=<downloaded hf model directory>
CUDA_DEVICE_MAX_CONNECTIONS=1 python tools/checkpoint/convert.py --bf16 --model-type GPT --loader llama_mistral --saver core \
    --target-tensor-parallel-size 4 --checkpoint-type hf \
    --load-dir $LLAMA_DOWNLOAD_DIR --save-dir llama3p1 --tokenizer-model $LLAMA_DOWNLOAD_DIR \
    --saver-transformer-impl transformer_engine --model-size llama3
```

#### Vision model conversion

You can run the following command to convert RADIO to an mcore compatible format:
```
python examples/multimodal/model_converter/radio_converter.py --output radio_tp_4 --tensor-parallel-size 4 --use-te \
    --version c-radio_v2-vlm-h --model-type radio_v2.5-h
```

#### Combined checkpoint

Combine the language and vision model by running:
```
examples/multimodal/combine_lm_vision_checkpoints.sh <language model directory> <vision model directory> <output directory>
```

# Training

1. Pretraining: we provide an example pretraining script at `examples/multimodal/llama_3p1_nemotron_nano_vl_8b_v1/pretraining_llama_3p1_nemotron_nano_vl_8b_v1.sh`.
2. SFT: we provide an example SFT script at `examples/multimodal/llama_3p1_nemotron_nano_vl_8b_v1/sft_llama_3p1_nemotron_nano_vl_8b_v1.sh`.

# Inference and evaluation

To run a simple inference example:
```
export LLAMA_NEMOTRON_NANO_VL_PATH=<path to the megatron tp=4 checkpoint>
examples/multimodal/llama_3p1_nemotron_nano_vl_8b_v1/text_generation.sh --model-path $LLAMA_NEMOTRON_NANO_VL_PATH \
    --task inference --output-path inference-example --tensor-model-parallel-size 4
```

To evaluate the model, you can change `--task` to `MMMU` or `TextVQA`, for example.
