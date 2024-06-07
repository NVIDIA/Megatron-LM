# Llama and Mistral support in Megatron-LM

NOTE: Llama-3 and Mistral support in Megatron is currently experimental and we are still evaluting benchmark results to confirm model conversion, training and inference correctness.

The [Llama-2](https://ai.meta.com/llama/) and [Llama-3](https://llama.meta.com/) family of models are an open-source set of pretrained & finetuned (for chat) models that have achieved strong results across a wide set of benchmarks. At their times of release, both Llama-2 and Llama-3 models achieved among the best results for open-source models, and were competitive with leading closed-source models (see https://arxiv.org/pdf/2307.09288.pdf and https://ai.meta.com/blog/meta-llama-3/).

Similarly, [Mistral-7b](https://mistral.ai/news/announcing-mistral-7b/) is an open-source model with pretrained and finetuned (for chat) variants that achieve strong benchmark results.

Architecturally Llama-2, Llama-3 and Mistral-7b are very similar. As such Megatron can support loading checkpoints from all three for inference and finetuning. Converting the checkpoints and loading them is slightly different for each model and is detailed for each below.

# Llama-2

Llama-2 checkpoints can be loaded into Megatron for inference and for finetuning. Loading these checkpoints consists of three steps:

1. Get access to download the checkpoints.
2. Convert the checkpoints from Meta/Huggingface format to Megatron format.
3. Setup arguments for launching the model.

The following sections detail these steps. The final section lists benchmark result comparisons between: 1) Llama-2 inference code running the Meta-format checkpoints, and 2) Megatron inference code running the converted checkpoints.

## Contents
  * [Download Meta or Huggingface checkpoints](#download-meta-or-huggingface-checkpoints)
  * [Convert checkpoint format](#convert-checkpoint-format)
    * [Meta format](#meta-format)
    * [Huggingface format](#huggingface-format)
  * [Launch model](#launch-model)
    * [Megatron](#launch-megatron)
    * [Meta](#launch-meta)
    * [Huggingface](#launch-hf)
  * [Benchmark results](#benchmark-results)

## Download Meta or Huggingface checkpoints

Users must first apply for access to download the Llama-2 checkpoints either directly from [Meta](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) or through [Huggingface](https://huggingface.co/docs/transformers/main/model_doc/llama2) (HF). The checkpoints are available in two formats, Meta's native format (available from both the Meta and HF links), and HF's format (available only from HF). Either format can be converted to Megatron, as detailed next.

## Convert checkpoint format

We recommend passing `--dtype bf16` for training or finetuning. Inference can be done in bfloat16 or float16.

### Meta format

The Meta format checkpoints are converted to HF format as an intermediate step before converting to Megatron format. The `transformers` package is required, and must have version >=4.31.0 (e.g., `pip install transformers>=4.31.0`). (**Note**: we have specifically tested with versions `4.31.0` and `4.32.0`; your experience may vary with newer versions.) Assuming the downloaded checkpoints are in `$CHECKPOINT_DIR` (with separate sub-directories for 7B, 13B, 70B, etc.), the following example command can be used to convert from Llama-2 format to HF format in bfloat16:

```
python tools/checkpoint/convert.py --model-type GPT \
>   --loader llama_mistral \
>   --saver megatron \
>   --checkpoint-type meta \
>   --model-size llama2-7B \
>   --load-dir $LLAMA_META_FORMAT_DIR \
>   --save-dir ${MEGATRON_FORMAT_DIR} \
>   --tokenizer-model ${TOKENIZER_MODEL} \
>   --target-tensor-parallel-size ${TP} \
>   --target-pipeline-parallel-size ${PP} \
>   --bf16
```

Valid values for `--model-size` are `llama2-7B`, `llama2-13B`, and `llama2-70B` (for pretrained-only models), and `llama2-7Bf`, `llama2-13Bf`, and `llama2-70Bf` (for chat-finetuned models).

### Huggingface format

The HF checkpoints can be converted to Megatron format by using Megatron's own Llama-2 checkpoint converter for HF format (see script `tools/checkpoint/loader_llama_mistral.py`). One important argument that must be set correctly is the tensor parallel size (`TP`) for each model. The following table shows these values:

| Model size | Tensor parallel size (`TP`) |
| ---------- | --------------------------- |
|  7B        | 1                           |
| 13B        | 2                           |
| 70B        | 8                           |

Using these values for `TP`, along with the path to the Llama-2 tokenizer model (automatically downloaded with original checkpoint download; see `${TOKENIZER_MODEL}` below), run the following command from the root of your Megatron source code to convert from HF format to Megatron format:

```
$>: python tools/checkpoint/convert.py \
 >    --model-type GPT \
 >    --loader llama_mistral \
 >    --saver megatron \
 >    --target-tensor-parallel-size ${TP} \
 >    --checkpoint-type hf
 >    --load-dir ${HF_FORMAT_DIR} \
 >    --save-dir ${MEGATRON_FORMAT_DIR} \
 >    --tokenizer-model ${TOKENIZER_MODEL}
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

### Launch Meta

Meta checkpoints can be launched with: https://github.com/facebookresearch/llama

### Launch Huggingface

Huggingface checkpoints can be launched with: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

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

# Llama-3

Llama-3 checkpoints can be loaded into Megatron for inference and for finetuning. Loading these checkpoints consists of several steps:

1. Get access to download the checkpoints (weights and tokenizer).
2. Clone the llama3 loading code from Meta.
3. Install the llama package from source.
4. Convert the checkpoints from Meta/Huggingface format to Megatron format.
5. Setup arguments for launching the model.

The following sections detail these steps.

## Contents
  * [Download Meta or Huggingface checkpoints](#download-meta-or-huggingface-checkpoints)
  * [Install tiktoken](#install-tiktoken)
  * [Install llama package from Meta](#install-llama-package)
  * [Convert checkpoint format](#convert-checkpoint-format)
    * [Meta format](#meta-format)
    * [Huggingface format](#huggingface-format)
  * [Launch model](#launch-model)
    * [Megatron](#launch-megatron)
    * [Meta](#launch-meta)
    * [Huggingface](#launch-hf)
  * [Benchmark results](#benchmark-results)

## Download Meta or Huggingface checkpoints

Users must first apply for access to download the Llama-3 checkpoints either directly from [Meta](https://llama.meta.com/llama-downloads) or through [Huggingface](https://huggingface.co/meta-llama) (HF). The checkpoints are available in two formats, Meta's native format (available from both the Meta and HF links), and HF's format (available only from HF). Either format can be converted to Megatron, as detailed next.

## Install tiktoken

The Llama-3 tokenizer relies on the availability of the `tiktoken` module which can be installed through `pip`.

## Install llama package from Meta

1. In a location outside of the megatron-lm source directory, e.g `~`: `git clone https://github.com/meta-llama/llama3.git`
2. `cd $LLAMA3_SOURCE_DIR`
4. `pip install -e .`

## Convert checkpoint format

We recommend passing `--dtype bf16` for training or finetuning. Inference can be done in bfloat16 or float16.

### Meta format

The Meta format checkpoints are converted to HF format as an intermediate step before converting to Megatron format. The `transformers` package is required, and must have version >=4.31.0 (e.g., `pip install transformers>=4.31.0`). (**Note**: we have specifically tested with versions `4.31.0` and `4.32.0`; your experience may vary with newer versions.) Assuming the downloaded checkpoints are in `$CHECKPOINT_DIR` (with separate sub-directories for 8B, 70B, etc.), the following example command can be used to convert from Llama-3 format to HF format in bfloat16:

```
python tools/checkpoint/convert.py \
>   --model-type GPT \
>   --loader llama_mistral \
>   --saver mcore \
>   --checkpoint-type meta \
>   --model-size llama3-8B \
>   --load-dir $LLAMA_META_FORMAT_DIR \
>   --save-dir ${MEGATRON_FORMAT_DIR} \
>   --tokenizer-model ${TOKENIZER_MODEL} \
>   --target-tensor-parallel-size ${TP} \
>   --target-pipeline-parallel-size ${PP} \
>   --bf16
```

Valid values for `--model_size` are `llama3-8B` and `llama3-70B` (for pretrained-only models), and `llama3-8Bf` and `llama3-70Bf` (for chat-finetuned models).

### Huggingface format

The HF checkpoints can be converted to Megatron format by using Megatron's own Llama-3 checkpoint converter for HF format (see script `tools/checkpoint/loader_llama_mistral.py`). One important argument that must be set correctly is the tensor parallel size (`TP`) for each model. The following table shows these values:

| Model size | Tensor parallel size (`TP`) |
| ---------- | --------------------------- |
|  8B        | 1                           |
| 70B        | 8                           |

Using these values for `TP`, along with the path to the Llama-3 tokenizer model (automatically downloaded with original checkpoint download; see `${TOKENIZER_MODEL}` below), run the following command from the root of your Megatron source code to convert from HF format to Megatron format:

```
$>: python tools/checkpoint/convert.py \
 >    --model-type GPT \
 >    --loader llama_mistral \
 >    --saver mcore \
 >    --target-tensor-parallel-size ${TP} \
 >    --checkpoint-type hf
 >    --load-dir ${HF_FORMAT_DIR} \
 >    --save-dir ${MEGATRON_FORMAT_DIR} \
 >    --tokenizer-model ${TOKENIZER_MODEL}
 >    --model-size llama3-8B \
```

Valid values for `--model-size` are `llama3-8B` and `llama3-70B` (for pretrained-only models), and `llama3-8Bf` and `llama3-70Bf` (for chat-finetuned models).

After this conversion, we are ready to load the checkpoints into a Megatron GPT model.

## Launch model

### Launch Megatron

If loading for either inference or finetuning, use the following arguments:

```
--tensor-model-parallel-size ${TP} \
--pipeline-model-parallel-size 1 \
--seq-length 4096 \
--max-position-embeddings 4096 \
--tokenizer-type Llama3Tokenizer \
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
```

### Launch Meta

Meta checkpoints can be launched with: https://github.com/meta-llama/llama3

### Launch Huggingface

Huggingface checkpoints can be launched by following the instructions here: https://huggingface.co/blog/llama3

## Benchmark results

Llama-3 support in Megatron is currently experimental and we are still carrying out benchmark evaluations.

# Mistral-7b

Megatron currently supports loading the v.03 release of Mistral-7b (which does not use sliding window attention and offers a larger 32768 vocabulary) for inference and finetuning. Loading these checkpoints consists of several steps:

1. Get access to download the checkpoints (weights and tokenizer).
2. Install the `mistral-common` package
3. Convert the checkpoints from HuggingFace format to Megatron format.
4. Setup arguments for launching the model.

The following sections detail these steps.

## Contents
  * [Download Huggingface checkpoints](#download-huggingface-checkpoints)
  * [Install mistral-common packgage](#install-mistral-common)
  * [Convert checkpoint format](#convert-checkpoint-format)
  * [Launch model](#launch-model)
  * [Benchmark results](#benchmark-results)

## Download Huggingface checkpoints

Users must first apply for access to download the Mistral-7b checkpoints through [Huggingface](https://huggingface.co/mistralai/Mistral-7B-v0.3) (HF). Megatron does not currently support the v0.1 or v0.2 checkpoints, ensure you download v0.3. Megatron also does not currently support using the raw weights directly from [Mistral](https://docs.mistral.ai/getting-started/open_weight_models/).

## Install the mistral-common package

`pip install mistral-common`

## Convert checkpoint format

The HF checkpoints can be converted to Megatron format by using Megatron's own Mistral checkpoint converter for HF format (see script `tools/checkpoint/loader_llama_mistral.py`).

Using the path to the Mistral tokenizer model (downloaded alongside the HF checkpoint), run the following command from the root of your Megatron source code to convert from HF format to mcore format:

```
$>: python tools/checkpoint/convert.py \
 >    --model-type GPT \
 >    --loader llama_mistral \
 >    --saver mcore \
 >    --target-tensor-parallel-size ${TP} \
 >    --checkpoint-type hf \
 >    --load-dir ${HF_FORMAT_DIR} \
 >    --save-dir ${MEGATRON_FORMAT_DIR} \
 >    --tokenizer-model ${TOKENIZER_MODEL} \
 >    --model-size mistral-7B \
```

Valid values for `--model-size` are mistral-7B for the pretrained model or mistral-7Bf for the chat fine-tuned model.

After this conversion, we are ready to load the checkpoints into an mcore GPT model.

## Launch model

If loading for either inference or finetuning, use the following arguments:

```
--tensor-model-parallel-size ${TP} \
--pipeline-model-parallel-size 1 \
--seq-length 4096 \
--max-position-embeddings 4096 \
--tokenizer-type MistralTokenizer \
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
```

# Benchmark results

Mistral-7B support in Megatron is currently experimental and we are still carrying out benchmark evaluations.
