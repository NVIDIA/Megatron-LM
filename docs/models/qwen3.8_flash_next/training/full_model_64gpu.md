# The 48-layer model on 64 GPUs

The configuration that has been run end to end: **BF16, expert parallelism only**
(TP1 / PP1 / CP1 / EP64), `alltoall` dispatcher, no activation recompute, no CUDA graphs,
seq 4096, micro-batch 1, on 16 × 4 GB300 (≈ 277 GiB each).

## Why EP64

Single-GPU fake-process-group sizing at the real dimensions (48 layers + MTP + PLE, seq 4096,
mbs 1, no recompute), confirmed by the real 64-rank run:

| Layout | Peak allocated per GPU | Verdict |
|---|---|---|
| EP16 | out of memory | — |
| EP32 | 251 GiB | does not fit: the real run shows ≈ 42 GiB of non-PyTorch device memory (NCCL communicators and all-to-all buffers, CUDA context, workspaces) on top of the allocator peak |
| **EP64** | **204.4 GiB** measured on every rank (fake-pg predicted 204.5), 206.2 GiB reserved, **248.6 GiB device memory used** | **29 GiB headroom** |

Activations are ≈ 130 GiB at any EP (≈ 15 GiB of them vocab-sized cross-entropy intermediates);
parameters + optimizer are 5 B replicated + 172 B / EP sharded (experts 120.6 B + PLE 51.2 B),
16 bytes per sharded parameter because expert-DP is 1. The PLE table is 1.49 GiB per rank at
EP64 (fp32 master 2.98 GiB, Adam 5.96 GiB). Recompute for the n-gram memory is available since
the 2026-09-15 branch (`engram.md`), so EP32 can now be re-measured; CUDA graphs are still
rejected by the gated-residual and Engram modules.

## Measured step (mock data, GBS 512 = 2 M tokens/step, 50 iterations)

| Metric | Value |
|---|---|
| iteration 1 | 182 s (Triton / inductor warm-up) |
| steady state | median 16.0 s, mean 17.6 s per iteration → **119 k tokens/s** on 64 GPUs |
| losses | lm loss 7.36 → 0.009 and MTP loss 7.40 → 0.014 over 50 steps at lr 1e-5 on the deterministic mock pattern (mechanics only) |

No performance work has been done on this layout (see the "Not yet done" row in the
[README](../README.md)).

## Argument list (weights-only resume from the converted checkpoint)

Multi-node launch: one `torchrun` per node with `--nnodes 16 --nproc_per_node 4`, `MASTER_ADDR`
resolved outside the container (`scontrol show hostnames` on SLURM). Everything below is the
Megatron command line; the model block is identical to
[`config_mapping.md`](../architecture/config_mapping.md).

```bash
pretrain_hybrid.py \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model Qwen/Qwen3.8-Flash-Next \
    --hidden-size 2560 --num-attention-heads 24 --kv-channels 256 --max-position-embeddings 262144 \
    --make-vocab-size-divisible-by 1940 \
    --hybrid-layer-pattern GEGEGEQEGEGEGEQEGEGEGEQEGEGEGEQEGEGEGEQEGEGEGEQEGEGEGEQEGEGEGEQEGEGEGEQEGEGEGEQEGEGEGEQEGEGEGEQE/QE \
    --spec megatron.core.models.hybrid.hybrid_layer_specs gated_residual_hybrid_stack_spec \
    --group-query-attention --num-query-groups 2 --qk-layernorm --attention-output-gate \
    --position-embedding-type rope --rotary-percent 0.25 --rotary-base 10000000 \
    --qsa-indexer-n-heads 4 --qsa-indexer-head-dim 128 --qsa-indexer-budget 2048 \
    --qsa-indexer-compress-ratio 4 --qsa-indexer-loss-coeff 0.01 \
    --linear-conv-kernel-dim 4 --linear-key-head-dim 128 --linear-value-head-dim 128 \
    --linear-num-key-heads 16 --linear-num-value-heads 48 --gdn-output-gate-activation sigmoid \
    --enable-mhc-connections --mhc-connection-variant gated_residual \
    --mhc-num-residual-streams 4 --hc-lowrank 320 \
    --num-experts 512 --moe-ffn-hidden-size 640 --moe-shared-expert-intermediate-size 640 \
    --moe-shared-expert-gate --moe-router-topk 10 --moe-router-score-function softmax \
    --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 1e-3 --moe-router-dtype fp32 \
    --engram-variant qwen --engram-layer-ids 3 --engram-vocab-sizes 20000000 20000000 \
    --engram-max-ngram-order 3 --engram-num-hash-heads 8 --engram-memory-dim 1280 \
    --engram-kernel-size 4 --engram-unigram-vocab-size 248320 \
    --engram-hash-seed 1234 --engram-eos-token-id 248044 \
    --mtp-num-layers 1 --mtp-loss-scaling-factor 0.1 \
    --normalization RMSNorm --apply-layernorm-1p --norm-epsilon 1e-6 --swiglu \
    --disable-bias-linear --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 --hidden-dropout 0.0 --no-weight-decay-cond-type apply_wd_to_qk_layernorm \
    --mock-data --seq-length 4096 --moe-router-force-load-balancing \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 64 \
    --context-parallel-size 1 --expert-tensor-parallel-size 1 \
    --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
    --micro-batch-size 1 --global-batch-size 512 \
    --moe-token-dispatcher-type alltoall --moe-grouped-gemm --moe-permute-fusion \
    --bf16 --transformer-impl transformer_engine --enable-experimental \
    --cross-entropy-loss-fusion --cross-entropy-fusion-impl native \
    --lr 1e-5 --min-lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --clip-grad 1.0 \
    --train-iters 50 --log-interval 1 --log-memory-interval 1 --eval-iters 0 --eval-interval 1000 \
    --load <dist_checkpoint_dir> --no-load-optim --no-load-rng --finetune --ckpt-format torch_dist
```

`--no-load-optim --no-load-rng --finetune` because the converted checkpoint holds weights only
(iteration counter restarts at 0). Environment used: `CUDA_DEVICE_MAX_CONNECTIONS=1`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## From scratch

Same argument list without the `--load …` line, plus a pretraining schedule and checkpointing.
The values used in the prepared recipe (placeholders for a ≈ 200 B-token run at 2 M tokens/step;
not yet executed):

```bash
    --train-iters 100000 --lr 3.0e-4 --min-lr 3.0e-5 --lr-decay-style cosine \
    --lr-warmup-iters 2000 --lr-warmup-init 0.0 --adam-beta1 0.9 --adam-beta2 0.95 \
    --init-method-std 0.02 --seed 1234 \
    --save <ckpt_dir> --load <ckpt_dir> --save-interval 500 --ckpt-format torch_dist \
    --dist-ckpt-optim-fully-reshardable --exit-duration-in-mins 230
```

Re-submitting the same command resumes from the newest checkpoint under `--load`; on an empty
directory Megatron says so and starts from random init. A checkpoint is 335 GiB of weights plus
≈ 1 TiB of optimizer state. `--mock-data` only exercises the mechanics; for text replace it with
`--data-path … --split 99,1,0` and remove `--moe-router-force-load-balancing`.
