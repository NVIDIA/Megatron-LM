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
to achieve the highest tokens-per-second-per-user (or TPS) without changing the quality of
the model. We provide quantization-aware training (QAT) receipt with self-distillation in the following.


### Model Convertion

To ensure no quality degredation, base model is frozen and the draft model is attached as a
transformation. For Medusa, set `--export-algorithm medusa` and provide `--export-num-medusa-heads`.
For EAGLE, set `--export-algorithm eagle` and provide `--export-eagle-algorithm`.
the resulting model stored in `${MLM_MODEL_SAVE}` will have randomly initialized draft model weights.

```
python examples/post_training/modelopt/convert_speculative.py \
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
See ModelOpt's example (https://github.com/NVIDIA/Model-Optimizer/tree/main/speculative_decoding)
for more details. The final output is stored as `jsonlines` in an OpenAI chat completion format.


### Stream teacher features to EAGLE3 (experimental)

Use a [ModelOpt vLLM teacher server](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/speculative_decoding#megatron-eagle3-teacher-features)
and an offline EAGLE3 GPTModel checkpoint prepared with
`convert_speculative.py --algorithm eagle3 --export-offline-model`. The checkpoint and server
must use the same teacher weights and tokenizer. Install a ModelOpt build with
`MegatronEagleVllmStreamingDataset` and NIXL in the trainer environment.

The following one-step Qwen3-0.6B smoke test assumes the server is already running on a separate
GPU at port 18876 under the name `qwen3-teacher`. Set `TEACHER_MODEL` to the local teacher
tokenizer directory and `OFFLINE_EAGLE_CKPT` to the converted checkpoint directory.

```sh
cat > train.jsonl <<'JSONL'
{"conversation_id":"smoke","messages":[{"role":"user","content":"What is speculative decoding?"},{"role":"assistant","content":"A drafter proposes tokens and a teacher verifies them."}],"chat_template_kwargs":{"enable_thinking":false}}
JSONL

torchrun --standalone --nproc-per-node=1 examples/post_training/modelopt/finetune.py \
    --num-layers 28 --hidden-size 1024 --ffn-hidden-size 3072 \
    --num-attention-heads 16 --group-query-attention --num-query-groups 8 --kv-channels 128 \
    --normalization RMSNorm --norm-epsilon 1e-6 --position-embedding-type rope \
    --rotary-base 1000000 --swiglu --disable-bias-linear --make-vocab-size-divisible-by 1187 \
    --seq-length 256 --max-position-embeddings 1024 --bf16 \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$TEACHER_MODEL" \
    --export-offline-model --export-model-type GPTModel --ckpt-format torch_dist \
    --load "$OFFLINE_EAGLE_CKPT" --finetune --no-load-optim --no-load-rng \
    --save eagle-smoke --save-interval 1 --no-save-optim --no-save-rng \
    --micro-batch-size 1 --global-batch-size 1 --train-iters 1 --lr 1e-4 --lr-decay-style constant \
    --no-gradient-accumulation-fusion --no-masked-softmax-fusion --no-bias-swiglu-fusion \
    --mock-data --dataloader-type single --num-workers 0 --eval-iters 0 --log-interval 1 \
    --eagle-streaming-server-urls http://127.0.0.1:18876 \
    --eagle-streaming-model qwen3-teacher --eagle-streaming-train-data train.jsonl
```

For a checkpoint with zero-based auxiliary layer IDs `[1, 13, 24]`, configure captures
`[2, 14, 25, 28]`. The server counts completed layers. Keep auxiliary captures in checkpoint
order and append the final layer as a distinct pre-norm capture. The adapter emits `S+1` token IDs
and an unshifted loss mask, plus `S` BF16 rows for each feature tensor. Megatron shifts the IDs
and mask once to build next-token training batches and passes the final residual to ModelOpt with
`base_hidden_prenorm=True` so ModelOpt applies the saved teacher final norm before the LM head.

Use BF16, micro-batch size 1, CP=PP=1, and one conversation per sample. The trainer reads the
JSONL through the streaming flags. Here `--mock-data` satisfies Megatron's generic data-source
validation. The capture slot must hold the full prompt and the server must allow one extra
generation token. For same-host UCX, set `NIXL_BACKENDS=UCX` and `UCX_TLS=tcp,sm,self,cuda_copy`
on both processes. Supply `--eagle-streaming-valid-data` when using nonzero `--eval-iters`.
Optional test data uses `--eagle-streaming-test-data`. To supervise only assistant tokens, use
`--eagle-streaming-answer-only-loss` with a compatible chat template.

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



