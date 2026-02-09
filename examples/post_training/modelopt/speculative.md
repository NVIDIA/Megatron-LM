<div align="center">

# Speculative Decoding

</div>

[Medusa](https://arxiv.org/abs/2401.10774) and [EAGLE](https://arxiv.org/pdf/2401.15077) 
training and model export are supported (fast decoding is supported through TensorRT-LLM).

Medusa head top-1 accuracy is reported per step (**NOTE:** the accuracy here does not
translate to the acceptance rate described in the writeup. The top-1 of the 1st head
can however signal whether the training is converged).


## Training and Export Workflow

In practice, speculative decoding should be combined with quantization (weights and kv-cache)
to achieve the the highest tokens-per-second-per-user (or TPS) without changing the quality of
the model. We provide quantization-aware training (QAT) receipt with self-distillation in the following.


### Model Convertion

To ensure no quality degredation, base model is frozen and the draft model is attached as a
transformation. For Medusa, set `--export-algorithm medusa` and provide `--export-num-medusa-heads`.
For EAGLE, set `--export-algorithm eagle` and provide `--export-eagle-algorithm`.
the resulting model stored in `${MLM_MODEL_SAVE}` will have randomly initialized draft model weights.

```
python examples/post_training/modelopt/convert_model.py \
    --export-algorithm eagle \
    --export-eagle-algorithm eagle3 \
    --load ${MLM_MODEL_CKPT} --save ${MLM_MODEL_SAVE} ${MLM_EXTRA_ARGS}
```


### Synthetic Data Generation

Rather than learning the language and syntax, the draft model is trained to mimic the base
model output. As a result, self-synthesized data is crucial for the draft model accuracy
and acceptance rate (AR).

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
python examples/post_training/modelopt/finetune.py \
    --load ${MLM_MODEL_SAVE} --save ${MLM_MODEL_SAVE} ${MLM_EXTRA_ARGS}
python examples/post_training/modelopt/quantize.py \
    --export-quant-cfg fp8 \
    --load ${MLM_MODEL_SAVE} --save ${MLM_QUANT_SAVE} ${MLM_EXTRA_ARGS}
python examples/post_training/modelopt/finetune.py \
    --load ${MLM_QUANT_SAVE} --save ${MLM_QUANT_SAVE} ${MLM_EXTRA_ARGS}
```

### Export Checkpoint

Last, we export the Medusa heads or EAGLE module so that it can be deployed on runtime framework (i.e., TensorRT-LLM). 

```sh
python examples/post_training/modelopt/export.py \
    --export-dir ${CKPT_DIR} \
    -export-extra-modules \
    --load ${MLM_QUANT_SAVE} ${MLM_EXTRA_ARGS}
```

### TensorRT-LLM Deployment

To serve the exported checkpoint with [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), follow the sample commands below with the TensorRT-LLM GitHub repo:

```sh
trtllm-serve <exported checkpoint> --host 0.0.0.0 --port 8000 --backend pytorch --max_batch_size 32 --max_num_tokens 8192 --max_seq_len 8192 --tp_size 8 --extra_llm_api_options extra-llm-api-config.yml
```

`extra-llm-api-config.yml` is like this
```sh
enable_attention_dp: false
disable_overlap_scheduler: true
enable_autotuner: false

cuda_graph_config:
    max_batch_size: 1

speculative_config:
    decoding_type: Eagle
    max_draft_len: 3
    speculative_model_dir: <eagle3 checkpoint>

kv_cache_config:
    enable_block_reuse: false
```



