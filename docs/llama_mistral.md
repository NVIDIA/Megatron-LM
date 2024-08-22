# Llama, Mistral and other Llama-like model support in Megatron-LM

NOTE: In order to simplify code we now only support converting llama-3.x and mistral checkpoints downloaded from Huggingface.

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
2. Convert the checkpoints from Huggingface format to Megatron format.
3. (Optional) Validate converted checkpoints
4. Setup arguments for launching the model.

The following sections detail these steps.

## Contents
  * [Download Huggingface checkpoints](#download-huggingface-checkpoints)
  * [Convert checkpoint format](#convert-checkpoint-format)
    * [Huggingface format](#huggingface-format)
  * [Validate checkpoint](#optional-validate-checkpoint)
  * [Launch model](#launch-model)

## Download Huggingface checkpoints

Users must first apply for access to download the Llama-3 checkpoints from [Huggingface](https://huggingface.co/meta-llama).

## Convert checkpoint format

We recommend passing `--dtype bf16` for training or finetuning. Inference can be done in bfloat16 or float16.

### Huggingface format

The HF checkpoints can be converted to Megatron format by using Megatron's own Llama-3 checkpoint converter for HF format (see script `tools/checkpoint/loader_llama_mistral.py`). One important argument that must be set correctly is the tensor parallel size (`TP`) for each model. The following table shows these values:

| Model size | Tensor parallel size (`TP`) |
| ---------- | --------------------------- |
|  8B        | 1                           |
| 70B        | 8                           |

Using these values for `TP`, along with the path to the Llama-3 tokenizer model (automatically downloaded with original checkpoint download; see `${TOKENIZER_MODEL}` below), run the following command from the root of your Megatron source code to convert from HF format to Megatron format:

```
$>: python tools/checkpoint/convert.py \
 >    --bf16 \
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

## (Optional) Validate checkpoints

A Megatron-LM text generation server for Llama3 can be launched using the script `examples/llama_mistral/run_text_generation_llama3.sh <PATH_TO_CONVERTED_MCORE_CHECKPOINT> <PATH_TO_DOWNLOADED_HUGGINGFACE_CHECKPOINT>`.

Once running, query the server with `curl 'http://<TEXT_GENERATION_SERVER_IP>:5000/api' -X 'PUT' -H 'Content-Type: application/json; charset=UTF-8'  -d '{"prompts":["<SOME_PROMPT>"], "tokens_to_generate":100, "top_k":1}'`.

A reference generation for comparison can be obtained from the Huggingface transformers library by running `python examples/llama_mistral/huggingface_reference.py --model_path <PATH_TO_DOWNLOADED_HUGGINGFACE_CHECKPOINT> --prompt <SOME_PROMPT>`.

## Launch model

If loading for either inference or finetuning, use the following arguments:

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

# Mistral-7b

Megatron currently supports loading the v0.3 release of Mistral-7b (which does not use sliding window attention and offers a larger 32768 vocabulary) for inference and finetuning. Loading these checkpoints consists of several steps:

1. Get access to download the checkpoints (weights and tokenizer).
2. Convert the checkpoints from HuggingFace format to Megatron format.
3. (Optional) Validate converted checkpoints
4. Setup arguments for launching the model.

The following sections detail these steps.

## Contents
  * [Download Huggingface checkpoints](#download-huggingface-checkpoints)
  * [Convert checkpoint format](#convert-checkpoint-format)
  * [(Optional) Validate checkpoint](#optional-validate-checkpoint)
  * [Launch model](#launch-model)

## Download Huggingface checkpoints

Users must first apply for access to download the Mistral-7b checkpoints through [Huggingface](https://huggingface.co/mistralai/Mistral-7B-v0.3) (HF).

## Convert checkpoint format

The HF checkpoints can be converted to Megatron format by using Megatron's own Mistral checkpoint converter for HF format (see script `tools/checkpoint/loader_llama_mistral.py`).

Using the path to the Mistral tokenizer model (downloaded alongside the HF checkpoint), run the following command from the root of your Megatron source code to convert from HF format to mcore format:

```
$>: python tools/checkpoint/convert.py \
 >    --bf16 \
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

## (Optional) Validate checkpoints

A Megatron-LM text generation server for Mistral-7B can be launched using the script `examples/llama_mistral/run_text_generation_mistral.sh <PATH_TO_CONVERTED_MCORE_CHECKPOINT> <PATH_TO_DOWNLOADED_HUGGINGFACE_CHECKPOINT>`.

Once running, query the server with `curl 'http://<TEXT_GENERATION_SERVER_IP>:5000/api' -X 'PUT' -H 'Content-Type: application/json; charset=UTF-8'  -d '{"prompts":["<SOME_PROMPT>"], "tokens_to_generate":100, "top_k":1}'`.

A reference generation for comparison can be obtained from the Huggingface transformers library by running `python examples/llama_mistral/huggingface_reference.py --model_path <PATH_TO_DOWNLOADED_HUGGINGFACE_CHECKPOINT> --prompt <SOME_PROMPT>`.

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

Many models such as Yi-34B use the Llama architecture and may be converted from HuggingFace to Megatron using the commands in [Llama3](#llama-3).
