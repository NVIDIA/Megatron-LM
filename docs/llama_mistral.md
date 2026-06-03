<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Llama, Mistral and other Llama-like model support in Megatron-LM

NOTE: In order to simplify code we now only support converting llama-3.x and mistral checkpoints downloaded from Hugging Face. For converting other models, see [Megatron Bridge](models/index.md).

The Llama-2 and Llama-3.x family of models are an open-source set of pretrained & finetuned (for chat) models that have achieved strong results across a wide set of benchmarks. At their times of release, both Llama-2 and Llama-3 models achieved among the best results for open-source models, and were competitive with leading closed-source models (see <https://arxiv.org/pdf/2307.09288.pdf>).

Similarly, [Mistral-7b](https://mistral.ai/news/announcing-mistral-7b/) is an open-source model with pretrained and finetuned (for chat) variants that achieve strong benchmark results.

Architecturally Llama-2, Llama-3 and Mistral-7b are very similar. As such Megatron can support loading checkpoints from all three for inference and finetuning. Converting the checkpoints and loading them is slightly different for each model and is detailed for each below.

# Contents

- [Llama, Mistral and other Llama-like model support in Megatron-LM](#llama-mistral-and-other-llama-like-model-support-in-megatron-lm)
- [Contents](#contents)
- [Llama-2](#llama-2)
  - [Download Huggingface checkpoints](#download-huggingface-checkpoints)
  - [Convert checkpoint format](#convert-checkpoint-format)
    - [Huggingface format](#huggingface-format)
  - [Launch model](#launch-model)
    - [Launch Megatron](#launch-megatron)
    - [Launch Huggingface](#launch-huggingface)
  - [Benchmark results](#benchmark-results)
    - [Big Bench](#big-bench)
    - [Multilingual](#multilingual)
    - [LM Evaluation Harness](#lm-evaluation-harness)
    - [MMLU](#mmlu)
- [Llama-3.x](#llama-3x)
  - [Download Huggingface checkpoints](#download-huggingface-checkpoints)
  - [Convert checkpoint format](#convert-checkpoint-format)
    - [Huggingface format](#huggingface-format)
  - [Launch model](#launch-model)
- [Mistral-7b](#mistral-7b)
  - [Download Huggingface checkpoints](#download-huggingface-checkpoints)
  - [Convert checkpoint format](#convert-checkpoint-format)
  - [Launch model](#launch-model)
- [Other Llama-like model support](#other-llama-like-model-support)
- [Known numerical differences](#known-numerical-differences)

# Llama-2

Llama-2 checkpoints can be loaded into Megatron for inference and for finetuning. Loading these checkpoints consists of three steps:

1. Get access to download the checkpoints.
2. Convert the checkpoints from Huggingface format to Megatron format.
3. Setup arguments for launching the model.

The following sections detail these steps. The final section lists benchmark result comparisons between: 1) Llama-2 inference code running the Meta-format checkpoints, and 2) Megatron inference code running the converted checkpoints.

## Download Huggingface checkpoints

Users must first apply for access to download the Llama-2 checkpoints either directly [Huggingface](https://huggingface.co/docs/transformers/main/model_doc/llama2) (HF). The checkpoints are available in HF's format (available only from HF). HF format can be converted to Megatron, as detailed next.

## Convert checkpoint format

We recommend passing `--dtype bf16` for training or finetuning. Inference can be done in bfloat16 or float16.

### Huggingface format

The HF checkpoints can be converted to Megatron format by using Megatron-Bridge's checkpoint converter for HF format [see script](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/convert_checkpoints.py).

```
python Megatron-Bridge/examples/conversion/convert_checkpoints.py import \
  --hf-model meta-llama/Llama-2-7B \
  --megatron-path ./checkpoints/llama2_7b \
  --torch-dtype bfloat16 \
  --device-map auto
```

After this conversion, we are ready to load the checkpoints into a Megatron GPT model.

## Launch model

### Launch Megatron

If loading for either inference or finetuning, use the following arguments:

```
--tensor-model-parallel-size ${TP} \
--pipeline-model-parallel-size 1 \
--seq-length 4096 \
--max-position-embeddings 4096 \
--tokenizer-type Llama2Tokenizer \
--tokenizer-model ${TOKENIZER_MODEL} \
--load ${CHECKPOINT_DIR} \
--exit-on-missing-checkpoint \
--use-checkpoint-args \
--no-load-optim \
--no-load-rng \
--untie-embeddings-and-output-weights \
--use-rotary-position-embeddings \
--normalization RMSNorm \
--no-position-embedding \
--no-masked-softmax-fusion \
--attention-softmax-in-fp32
```

### Launch Huggingface

Huggingface checkpoints can be launched with: <https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py>

## Benchmark results

The tables below list the benchmark comparisons between native Llama-2 (using Meta's checkpoint and Meta's inference code) and Megatron (using a converted HF checkpoint and Megatron's inference code).

The values are the percent error between Megatron and Llama-2, calculated using the formula: `|<llama_score> - <megatron_score>| / <llama_score>`, where the type of score is detailed before each table. Across all tests (80 total per model size), the mean error is 0.15%. The small difference in benchmark scores between the two models is due to minor arithmetic differences in implementation that alter the numerics slightly. Some of the factors that influence this difference include:

- Megatron performs batch matrix multiplications in a couple places, such as within self attention and in SwiGLU, that Llama performs separately.
- Megatron uses `torch.baddbmm` within self attention, versus Llama using `torch.matmul`.
- Megatron uses a `sin`/`cos` implementation for rotary position embeddings, versus Llama using a `polar`/`complex` implementation.
- Llama calls `torch.set_default_dtype(torch.float16)` during initialization, which Megatron does not.

### Big Bench

Score type: multiple choice grade.

| bigbench / standard | 7b | 13b | 70b |
| -- | -- | -- | -- |
| date_understanding | 0.29% | 0.13% | 0.12% |
| general_knowledge | 0.00% | 0.00% | 0.00% |
| human_organs_senses | 0.00% | 0.00% | 0.00% |
| intent_recognition | 0.00% | 0.11% | 0.00% |
| riddle_sense | 0.00% | 0.00% | 0.00% |
| similarities_abstraction | 0.00% | 0.58% | 0.00% |
| simple_arithmetic_json_multiple_choice | 0.00% | 0.00% | 0.00% |
| undo_permutation | 0.19% | 0.19% | 0.18% |

### Multilingual

Score type: multiple choice grade.

| multilingual / xcopa | 7b  | 13b  | 70b |
| -- | -- | -- | -- |
| en-template-mGPT-remove-punctuation | 0.08% | 0.00% | 0.00% |
| et-template-mGPT-remove-punctuation | 0.00% | 0.13% | 0.25% |
| ht-template-mGPT-remove-punctuation | 0.26% | 0.13% | 0.26% |
| id-template-mGPT-remove-punctuation | 0.11% | 0.00% | 0.19% |
| it-template-mGPT-remove-punctuation | 0.00% | 0.10% | 0.09% |
| qu-template-mGPT-remove-punctuation | 0.00% | 0.00% | 0.27% |
| sw-template-mGPT-remove-punctuation | 0.14% | 0.13% | 0.13% |
| th-template-mGPT-remove-punctuation | 0.25% | 0.13% | 0.13% |
| tr-template-mGPT-remove-punctuation | 0.26% | 0.00% | 0.34% |
| vi-template-mGPT-remove-punctuation | 0.00% | 0.11% | 0.00% |
| zh-template-mGPT-remove-punctuation | 0.00% | 0.10% | 0.09% |

### LM Evaluation Harness

Score type: multiple choice grade.

| lm-eval | 7b  | 13b  | 70b |
| -- | -- | -- | -- |
| boolq | 0.04% | 0.04% | 0.07% |
| hellaswag | 0.02% | 0.03% | 0.03% |
| piqa | 0.00% | 0.00% | 0.07% |
| winogrande | 0.00% | 0.11% | 0.20% |

### MMLU

Score type: multiple choice grade.

Note: the number in brackets is the number of sub-tasks for each supercategory.

| mmlu | 7b  | 13b  | 70b |
| -- | -- | -- | -- |
| stem [18]  | 0.79% | 0.05% | 0.01% |
| humanities [13]  | 0.19% | 0.01% | 0.02% |
| other (business, health, misc.) [14]  | 0.08% | 0.06% | 0.12% |
| social sciences [12]  | 0.37% | 0.21% | 0.01% |

# Llama-3.x

Llama-3.x checkpoints can be loaded into Megatron for inference and for finetuning. Loading these checkpoints consists of several steps:

1. Get access to download the checkpoints (weights and tokenizer).
2. Convert the checkpoints from Huggingface format to Megatron format.
3. (Optional) Validate converted checkpoints
4. Setup arguments for launching the model.

The following sections detail these steps.

## Download Huggingface checkpoints

Users must first apply for access to download the Llama-3.x checkpoints from [Huggingface](https://huggingface.co/meta-llama).

## Convert checkpoint format

We recommend passing `--dtype bf16` for training or finetuning. Inference can be done in bfloat16 or float16.

### Huggingface format

The HF checkpoints can be converted to Megatron format by using Megatron-Bridge's checkpoint converter for HF format [see script](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/convert_checkpoints.py).

```
python Megatron-Bridge/examples/conversion/convert_checkpoints.py import \
  --hf-model meta-llama/Llama-3.2-1B \
  --megatron-path ./checkpoints/llama3_2_1b \
  --torch-dtype bfloat16 \
  --device-map auto
```

After this conversion, we are ready to load the checkpoints into a Megatron GPT model.

## Launch model

If loading for either inference or finetuning, use the following arguments for Llama 3.0:

```
--tensor-model-parallel-size ${TP} \
--pipeline-model-parallel-size 1 \
--seq-length 8192 \
--max-position-embeddings 8192 \
--tokenizer-type HuggingFaceTokenizer \
--tokenizer-model ${TOKENIZER_MODEL} \
--load ${CHECKPOINT_DIR} \
--exit-on-missing-checkpoint \
--use-checkpoint-args \
--no-load-optim \
--no-load-rng \
--untie-embeddings-and-output-weights \
--normalization RMSNorm \
--position-embedding-type rope \
--no-masked-softmax-fusion \
--attention-softmax-in-fp32 \
--disable-bias-linear \
--transformer-impl transformer_engine \
--group-query-attention 8 \
--attention-dropout 0.0 \
--hidden-dropout 0.0 \
--rotary-base 500000 \
--rotary-percent 1.0 \
--ffn-hidden-size 14336 \
--num-attention-heads 32 \
--swiglu \
--bf16 \
```

For Llama3.1 please use the following arguments:

```
--tensor-model-parallel-size ${TP} \
--pipeline-model-parallel-size 1 \
--seq-length 8192 \
--max-position-embeddings 131072 \
--tokenizer-type HuggingFaceTokenizer \
--tokenizer-model ${TOKENIZER_MODEL} \
--load ${CHECKPOINT_DIR} \
--exit-on-missing-checkpoint \
--use-checkpoint-args \
--no-load-optim \
--no-load-rng \
--untie-embeddings-and-output-weights \
--normalization RMSNorm \
--position-embedding-type rope \
--no-masked-softmax-fusion \
--attention-softmax-in-fp32 \
--disable-bias-linear \
--transformer-impl transformer_engine \
--group-query-attention 8 \
--attention-dropout 0.0 \
--hidden-dropout 0.0 \
--rotary-base 500000 \
--rotary-percent 1.0 \
--use-rope-scaling \
--ffn-hidden-size 14336 \
--num-attention-heads 32 \
--swiglu \
--bf16 \
```

# Mistral-7b

Megatron currently supports loading the v0.3 release of Mistral-7b (which does not use sliding window attention and offers a larger 32768 vocabulary) for inference and finetuning. Loading these checkpoints consists of several steps:

1. Get access to download the checkpoints (weights and tokenizer).
2. Convert the checkpoints from HuggingFace format to Megatron format.
3. (Optional) Validate converted checkpoints
4. Setup arguments for launching the model.

The following sections detail these steps.

## Download Huggingface checkpoints

Users must first apply for access to download the Mistral-7b checkpoints through Huggingface. Two variants are available: the base model ([Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3)) and the instruct model ([Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)).

## Convert checkpoint format

The HF checkpoints can be converted to Megatron format by using Megatron-Bridge's checkpoint converter for HF format [see script](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/convert_checkpoints.py).

```
python Megatron-Bridge/examples/conversion/convert_checkpoints.py import \
  --hf-model mistralai/Mistral-7B-Instruct-v0.3 \
  --megatron-path ./checkpoints/mistral_7b \
  --torch-dtype bfloat16 \
  --device-map auto
```

After this conversion, we are ready to load the checkpoints into a Megatron GPT model.

## Launch model

If loading for either inference or finetuning, use the following arguments:

```
--tensor-model-parallel-size ${TP} \
--pipeline-model-parallel-size 1 \
--seq-length 4096 \
--max-position-embeddings 4096 \
--tokenizer-type HuggingFaceTokenizer \
--tokenizer-model ${TOKENIZER_MODEL} \
--load ${CHECKPOINT_DIR} \
--exit-on-missing-checkpoint \
--use-checkpoint-args \
--no-load-optim \
--no-load-rng \
--untie-embeddings-and-output-weights \
--normalization RMSNorm \
--position-embedding-type rope \
--no-masked-softmax-fusion \
--attention-softmax-in-fp32
--apply-layernorm-1p \
--transformer-impl transformer_engine \
--group-query-attention 8 \
--disable-bia-linear \
--rotary-base 1000000 \
--rotary-percent 1.0 \
--swiglu \
--ffn-hidden-size 14336 \
--num-attention-heads 32
```

# Other Llama-like model support

*Note: Experimental*

Many models such as Yi-34B and Qwen2.x use the Llama architecture and may be converted from HuggingFace to Megatron using the commands in [Llama-3.x](#llama-3x).

# Known numerical differences

It is not expected that the megatron and Huggingface implementations of llama3.x and mistral models will produce numerically identical results. There are multiple points where small numerical differences are expected. This is a non-exhaustive list:

1. TransformerEngine (TE) uses the model params_dtype inside RMSNorm whereas the Huggingface implementation uses fp32. See for details: <https://github.com/NVIDIA/TransformerEngine/issues/1132>
2. Huggingface `transformers` implements the q, k and v projections in self-attention as separate GEMMs whereas Megatron core combines them into a single GEMM for efficiency. This leads to small numerical differences.
