# K2-Horizon-0.9B

[IFM/K2-Horizon-0.9B](https://huggingface.co/IFM/K2-Horizon-0.9B) is a dense Llama-style
decoder, so it runs on the stock `GPTModel` with no new model code:

| Component | Setting |
| --- | --- |
| Layers / hidden / FFN | 28 / 1536 / 5120 |
| Attention | GQA, 32 query heads, 8 KV groups, head_dim 64 (attention width 2048 > hidden 1536) |
| MLP | SwiGLU, no biases |
| Norm | RMSNorm (pre-norm), eps 1e-6 |
| Embeddings | untied, vocab 64,256 |
| Parameters | 1.08B total, ~0.88B excluding embeddings |

## Files

- `k2_horizon_0.9b_model_args.sh`: the architecture flags (`K2_HORIZON_0P9B_ARCH_ARGS`) and
  one RoPE flag set per training stage.
- `convert_hf_to_mcore.py`: loads a Hugging Face checkpoint into Megatron, checks logit
  parity against the Hugging Face model, and optionally saves a Megatron checkpoint.

## Position encoding by training stage

Only the position encoding changes between stages. Each revision tag of the Hugging Face
repo carries its own `config.json`; use the flag set that matches the checkpoint's stage.

| Stage | Revision tags | Context | Flag set |
| --- | --- | --- | --- |
| Pretraining | `pretrain_*` | 8K | `K2_HORIZON_ROPE_PRETRAIN_ARGS` (RoPE, base 500,000) |
| Midtraining 1 | `mid_1_*` | 40K | `K2_HORIZON_ROPE_MIDTRAIN1_ARGS` (RoPE, base 1,000,000) |
| Midtraining 2 | `mid_2_*` | 128K | `K2_HORIZON_ROPE_MIDTRAIN2_ARGS` (RoPE, base 1,000,000) |
| RL, merge, MOPD | `rl_*`, `rl-mopd_249`, `main` | 128K | `K2_HORIZON_ROPE_FINAL_ARGS` (YaRN, factor 16 over 8K) |

To pretrain from scratch, use the pretraining flag set.

## Pretraining from scratch

```bash
source examples/k2_horizon/k2_horizon_0.9b_model_args.sh
torchrun --nproc-per-node 8 pretrain_gpt.py \
    "${K2_HORIZON_0P9B_ARCH_ARGS[@]}" "${K2_HORIZON_ROPE_PRETRAIN_ARGS[@]}" \
    --tokenizer-model /path/to/K2-Horizon-0.9B \
    --seq-length 8192 --data-path <prefix>_text_document \
    <training args>
```

Tokenize data with the same tokenizer. It does not add BOS; `--append-eod` appends
`<|endoftext|>` (id 1) after each document:

```bash
python tools/preprocess_data.py --input data.jsonl --json-keys text \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model /path/to/K2-Horizon-0.9B \
    --append-eod --workers 32 --output-prefix <prefix>
```

## Converting a released checkpoint

```bash
source examples/k2_horizon/k2_horizon_0.9b_model_args.sh
torchrun --nproc-per-node 1 examples/k2_horizon/convert_hf_to_mcore.py \
    "${K2_HORIZON_0P9B_ARCH_ARGS[@]}" "${K2_HORIZON_ROPE_PRETRAIN_ARGS[@]}" \
    --hf-path /path/to/K2-Horizon-0.9B --tokenizer-model /path/to/K2-Horizon-0.9B \
    --seq-length 4096 --micro-batch-size 1 --train-iters 1 \
    --save /path/to/k2_horizon_mcore --no-save-optim --no-save-rng
```

Add `--bf16` to convert in bf16. The parity check reads `README.md` from `--hf-path` as test
text; revisions without a README need `--verify-text-file`. Load the result with
`--load <dir> --finetune`. The converter runs at TP=1/PP=1 and does not pad the vocabulary,
so convert with `--make-vocab-size-divisible-by 128`; the saved checkpoint can be loaded at
other TP/PP sizes.

## Verification

Parity against the Hugging Face model on 4,096 tokens of text, fp32:

| Checkpoint | Flag set | Loss (HF / Megatron) | Top-1 agreement | KL(HF‖Megatron) |
| --- | --- | --- | --- | --- |
| `main` | final (YaRN) | 0.43933 / 0.43933 | 100% | 2.1e-7 |
| `pretrain_600000` | pretrain | 0.44395 / 0.44394 | 99.98% | 9.3e-8 |
| `pretrain_600000` | final (YaRN), wrong stage | 0.44395 / 0.50515 | 94.6% | 6.6e-2 |
| `main` | final with default YaRN betas, wrong | 0.43933 / 0.43658 | 97.9% | 8.9e-3 |

The last two rows are negative controls: wrong position-encoding flags are clearly detected.
