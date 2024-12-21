NVLM
====

Please refer to the [NVLM paper](https://arxiv.org/pdf/2409.11402) for details.

*NOTE: VLMs in Megatron are under active development and are expected to change.*

# Checkpoints

NVLM 1.0 model weights are publicly available in HuggingFace and Megatron format.

- NVLM-1.0-D 72B [HuggingFace version](https://huggingface.co/nvidia/NVLM-D-72B)
- NVLM-1.0-D 72B [Megatron-Core version](https://huggingface.co/nvidia/NVLM-D-72B-mcore) 

# Setup

## Docker image

Please use `examples/multimodal/Dockerfile`.

## Dataset preparation

Please refer to Tables 4 and 6 in the [NVLM paper](https://arxiv.org/pdf/2409.11402) for full list of pretrain and SFT datasets.
Please refer to https://nvidia.github.io/Megatron-Energon/data_prep.html on preparing datasets in the Megatron Energon format.

## Model conversion

### Vision model

NVLM 1.0 models use [OpenGVLab/InternViT-6B-448px-V1-5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-5) from HuggingFace.
Please download it and run the following command to convert it to Megatron format.
```
python examples/multimodal/model_converter/internvit_converter.py --output-dir <some output dir> --use-te --tensor-parallel-size 8
```

### 34B Language model

NVLM 1.0 34B starts from [NousResearch/Nous-Hermes-2-Yi-34B](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B) from HuggingFace.
Please download it and run the following command to convert it to Megatron format.
```
python tools/checkpoint/convert.py --bf16 --model-type GPT --loader llama_mistral --saver mcore --target-tensor-parallel-size 8 --checkpoint-type hf \
    --load-dir <hf model directory> --save-dir <output dir> --tokenizer-model <hf model name/directory> \
    --saver-transformer-impl transformer_engine --model-size yi-34B --make-vocab-size-divisible-by 1
```

### 72B Language model

NVLM 1.0 72B starts from [Qwen/Qwen2-72B-Instruct](https://huggingface.co/Qwen/Qwen2-72B-Instruct) from HuggingFace.
Please download it and run the following command to convert it to Megatron format.
```
python tools/checkpoint/convert.py --bf16 --model-type GPT --loader llama_mistral --saver mcore --target-tensor-parallel-size 8 --checkpoint-type hf \
    --load-dir <hf model directory> --save-dir <output directory> --tokenizer-model <hf model name/directory> \
    --saver-transformer-impl transformer_engine --model-size qwen2.5-72Bf
```

### Combined checkpoint

Combine the vision model checkpoint from [InternVit](#internvit) with the [34B](#34b-language-model) or [72B](#72b-language-model) language model by running:
```
examples/multimodal/combine_lm_vision_checkpoints.sh <language model directory> <vision model directory> <output directory> nvlm
```

# Training

## 34B

1. Pretraining: please run `examples/multimodal/nvlm/pretrain_yi_34b_internvit_6b.sh`. Please use the InternViT + 34B [combined checkpoint](#combined-checkpoint) and tokenizer from HuggingFace.
2. SFT: please run `examples/multimodal/nvlm/sft_34b_internvit.sh` using the checkpoint from 1.

## 72B

1. Pretraining: please run `examples/multimodal/nvlm/pretrain_qwen20_72b_internvit_6b.sh`. Please use the InternViT + 72B [combined checkpoint](#combined-checkpoint) and tokenizer from HuggingFace.
2. Convert the pretraining checkpoint from 1. to have pipeline parallel size = 4 for SFT. Please run
```
examples/multimodal/nvlm/pp_checkpoint_converter.py --input <pretrained checkpoint directory> \
--input-pipeline-parallel 1 --output <some output dir> --output-pipeline-parallel 4 \
--tensor-parallel 8
```
3. SFT: please run `examples/multimodal/nvlm/sft_qwen20_72b_internvit_6b.sh` using the checkpoint from 2.
4. To convert the checkpoint with pipeline parallel size = 4 back to 1 for evaluation, please run
```
examples/multimodal/nvlm/pp_checkpoint_converter.py --input <sft checkpoint directory> \
--input-pipeline-parallel 4 --output <some output dir> --output-pipeline-parallel 1 \
--tensor-parallel 8
```

# Evaluation

Run the text generation script.
- 34B
```
examples/multimodal/nvlm/run_text_generation_yi_34b_internvit_6b.sh --input-image-path /path/to/input/images --output-path /some/output/directory \
    --model-path /path/to/model.pt --gt-path /path/to/groundtruth/file --task generation-task-name --use-tiling
```
- 72B
```
examples/multimodal/nvlm/run_text_generation_qwen20_72b_internvit_6b.sh --input-image-path /path/to/input/images --output-path /some/output/directory \
    --model-path /path/to/model.pt --gt-path /path/to/groundtruth/file --task generation-task-name --use-tiling
```

where `--task generation-task-name` is the name of the evaluation benchmark such as `captioning`, `MMMU` or `TextVQA`.

Then, run one of the evaluation scripts from `examples/multimodal`. For example

```
python examples/multimodal/evaluate_mmmu.py --input-path /output/directory/from/generation
```
