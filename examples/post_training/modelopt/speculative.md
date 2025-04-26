# Speculative Decoding

[Medusa](https://arxiv.org/abs/2401.10774) and [EAGLE](https://arxiv.org/pdf/2401.15077) 
training and model export are supported (fast decoding is supported through TensorRT-LLM).
To run the examples, follow [README.md](README.md) to setup the containerized environment
and `NGC_CLI_API_KEY`, then
```sh
TP=8 bash medusa_sft.sh meta-llama/Llama-3.1-8B-Instruct
```
EAGLE training is similar. Just replace `medusa_sft.sh` with `eagle_sft.sh`
(requires `nvidia-modelopt>=0.20.0`).

Medusa head top-1 accuracy is reported per step (**NOTE:** the accuracy here does not
translate to the acceptance rate described in the writeup. The top-1 of the 1st head
can however signal whether the training is converged). By the end of the example, the
end results are stored in the following locations.
```sh
/tmp/megatron_workspace/meta-llama/
├── Llama-3.1-8B-Instruct_medusa
│   ├── iter_0000001
│   └── ...
├── Llama-3.1-8B-Instruct_medusa_quant
│   ├── iter_0000001
│   └── ...
└── Llama-3.1-8B-Instruct_medusa_quant_trtllm_export
```
`Llama-3.1-8B-Instruct_medusa_quant_trtllm_export` is the TensorRT-LLM checkpoint. To
deploy, check the TensorRT-LLM section below. 

> **IMPORTANT:** The sample flow `medusa_sft.sh` does not contain synthetic data generation.
> To achieve the best acceptance rate, check the whole receipt and options in the following sections.

## Table of Contents

[[_TOC_]]

## Training and Export Workflow

In practice, speculative decoding should be combined with quantization (weights and kv-cache)
to achieve the the highest tokens-per-second-per-user (or TPS) without changing the quality of
the model. We provide quantization-aware training (QAT) receipt with self-distillation in the following.


### Model Convertion

To ensure no quality degredation, base model is frozen and the draft model is attached as a
transformation. By providing `--export-num-medusa-heads` or `--export-num-eagle-layers`,
the resulting model stored in `${MLM_MODEL_SAVE}` will have randomly initialized draft model weights.

```
python examples/post_training_opt/convert_gpt.py \
    --export-num-medusa-heads 4 \
    --load ${MLM_MODEL_CKPT} --save ${MLM_MODEL_SAVE} ${OTHER_MLM_ARGS}
```

> **NOTE:** `MLM_MODEL_SAVE=Llama-3.1-8B-Instruct_medusa` in the example.

### Synthetic Data Generation

Rather than learning the language and syntax, the draft model is trained to mimic the base
model output. As a result, self-synthesized data is crucial for the draft model accuracy
and acceptance rate (AR). In EAGLE training, hidden state and logits distillation are also
applied.

For simplicity and efficiency, we use `vllm serve --quantization modelopt` to host an quantized
endpoint and we feed multi-turn conversation data to synthesize the assistant output.
See ModelOpt's example (https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/speculative_decoding)
for more details. The final output is stored as jsonlines in an OpenAI chat completion format.


### Quantization-Aware Training (QAT)

For quantize-aware training (QAT), the process is `bf16 training`, `fake quantization`, `qat`.
Since the base model weights are frozen, the initial training is mainly to get an more accurate
range of the draft model activation and weights. We store a new checkpoint where the model
now has additional quantization scalars for both the base and draft models. We launch the
finetuning again to continue the training with fake quantization until convergence.

```sh
python examples/post_training_opt/finetune_gpt.py \
    --export-num-medusa-heads 4 \
    --load ${MLM_MODEL_SAVE} --save ${MLM_MODEL_SAVE} ${OTHER_MLM_ARGS}
python examples/post_training_opt/text_generation_ptq.py \
    --export-quant-cfg fp8 \
    --decoder llama \
    --export-num-medusa-heads 4 \
    --load ${MLM_MODEL_SAVE} --save ${MLM_QUANT_SAVE} ${OTHER_MLM_ARGS}
python examples/post_training_opt/finetune_gpt.py \
    --export-num-medusa-heads 4 \
    --load ${MLM_QUANT_SAVE} --save ${MLM_QUANT_SAVE} ${OTHER_MLM_ARGS}
```

> **NOTE:** `MLM_QUANT_SAVE=Llama-3.1-8B-Instruct_medusa_quant` in the example.

### Export TensorRT-LLM Checkpoint

To finally export a TensorRT-LLM checkpoint, we leverage the same script by providing
`${TRTLLM_CKPT}` and the inference `${TP}`.

```sh
python examples/post_training_opt/text_generation_ptq.py \
    --export-dir ${TRTLLM_CKPT} \
    --inference-tensor-parallel ${TP} \
    --export-quant-cfg None \
    --decoder llama \
    --export-num-medusa-heads 4 \
    --load ${MLM_QUANT_SAVE} ${OTHER_MLM_ARGS}
```

> **NOTE:** `TRTLLM_CKPT=Llama-3.1-8B-Instruct_medusa_quant_trtllm_export` in the example.

**TensorRT-LLM deployment:** To build (`trtllm-build`) and run TensorRT-LLM engine, follow the steps here 
https://github.com/NVIDIA/TensorRT-Model-Optimizer#installation--docker to prepare the container.

For `tensorrt-llm>0.12`, the builder can detect this is a Medusa checkpoint directly
```sh
trtllm-build --checkpoint_dir Llama-3.1-8B-Instruct_medusa_quant_trtllm_export --output_dir /tmp/trtllm_engine ${other args}
```

The `run.py` (https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/run.py) and `gptManagerBenchmark` (https://github.com/NVIDIA/TensorRT-LLM/tree/main/benchmarks/cpp)
both support Medusa decoding by supplying argument `--medusa_choices`. This argument describes the sparse attention tree structure used in the Medusa writeup. For examples,
the following option is tree with 63 nodes which represent 63 draft tokens proposed by the 4 Medusa heads.
```sh
--medusa_choices="[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]"
```

> **ADVANCED USAGE:** When training, we typically train `4` heads if memory is sufficient and by default the max draft length is `63`.
> Optionally, users can change these values something smaller in TensorRT-LLM checkpoint's `config.json` before calling `trtllm-build`.
> For example, it is possible to only use 2 heads with maximum draft tokens 7 if this is a sweet spot. You must also change
> `--medusa_choices` to make sure you are not accessing draft tokens from the 3rd and 4th heads as well as shorting the list to have
> length 7.
