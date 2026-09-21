# Proxy model on one node

A proxy that keeps **every per-layer, per-expert and PLE dimension of the real model** and shrinks
only the things that set the total size, chosen so that per-rank behaviour on 4 GPUs (EP4) matches
the 64-GPU full model:

| Dimension | Full model | Proxy | Why |
|---|---|---|---|
| Qwen layers | 48 = 12 × `GEGEGEQE`, + 1 MTP | **8** = 2 × `GEGEGEQE`, + 1 MTP | PLE stays on hybrid layer 3 (Qwen layer 2, a GDN layer) |
| experts | 512 at EP64 → 8 local per rank | **32** at EP4 → 8 local per rank | same grouped-GEMM shapes and dispatch fan-out per rank |
| PLE table base per n-gram order | 20,000,000 | **1,000,000** | ≈ 2.6 B table parameters instead of 51 B; still EP row-sharded |

≈ 5.5 B parameters total (≈ 2.4 B active per token). It fits one GB300 node at seq 4096 with a
large margin and is a *functional / recipe-development* proxy, not a memory proxy for EP64. It
trains on Megatron's mock data with `NullTokenizer` at the real vocabulary size (EOD = vocab − 1,
so `--engram-eos-token-id 248044` stays in range) — no tokenizer download.

## Command (4 GPUs, EP4, BF16, from scratch)

Run from the Megatron-LM root inside a container with TE, fla and Triton available.

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m torch.distributed.run --nproc_per_node 4 --nnodes 1 pretrain_hybrid.py \
    --hybrid-layer-pattern GEGEGEQEGEGEGEQE/QE \
    --spec megatron.core.models.hybrid.hybrid_layer_specs gated_residual_hybrid_stack_spec \
    --tokenizer-type NullTokenizer --vocab-size 248320 --make-vocab-size-divisible-by 1940 \
    --hidden-size 2560 --num-attention-heads 24 --kv-channels 256 --max-position-embeddings 262144 \
    --group-query-attention --num-query-groups 2 --qk-layernorm --attention-output-gate \
    --position-embedding-type rope --rotary-percent 0.25 --rotary-base 10000000 \
    --qsa-indexer-n-heads 4 --qsa-indexer-head-dim 128 --qsa-indexer-budget 2048 \
    --qsa-indexer-compress-ratio 4 --qsa-indexer-loss-coeff 0.01 \
    --linear-conv-kernel-dim 4 --linear-key-head-dim 128 --linear-value-head-dim 128 \
    --linear-num-key-heads 16 --linear-num-value-heads 48 --gdn-output-gate-activation sigmoid \
    --enable-mhc-connections --mhc-connection-variant gated_residual \
    --mhc-num-residual-streams 4 --hc-lowrank 320 \
    --num-experts 32 --moe-ffn-hidden-size 640 --moe-shared-expert-intermediate-size 640 \
    --moe-shared-expert-gate --moe-router-topk 10 --moe-router-score-function softmax \
    --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 1e-3 --moe-router-dtype fp32 \
    --moe-token-dispatcher-type alltoall --moe-grouped-gemm --moe-permute-fusion \
    --engram-variant qwen --engram-layer-ids 3 --engram-vocab-sizes 1000000 1000000 \
    --engram-max-ngram-order 3 --engram-num-hash-heads 8 --engram-memory-dim 1280 \
    --engram-kernel-size 4 --engram-unigram-vocab-size 248320 \
    --engram-hash-seed 1234 --engram-eos-token-id 248044 \
    --mtp-num-layers 1 --mtp-loss-scaling-factor 0.1 \
    --normalization RMSNorm --apply-layernorm-1p --norm-epsilon 1e-6 --swiglu \
    --disable-bias-linear --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 --hidden-dropout 0.0 --no-weight-decay-cond-type apply_wd_to_qk_layernorm \
    --mock-data --seq-length 4096 --moe-router-force-load-balancing \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 4 \
    --context-parallel-size 1 --expert-tensor-parallel-size 1 \
    --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
    --micro-batch-size 1 --global-batch-size 16 \
    --bf16 --transformer-impl transformer_engine --enable-experimental \
    --cross-entropy-loss-fusion --cross-entropy-fusion-impl native \
    --train-iters 200 --lr 3.0e-4 --min-lr 3.0e-5 --lr-decay-style cosine \
    --lr-warmup-iters 20 --lr-warmup-init 0.0 --weight-decay 0.1 --clip-grad 1.0 \
    --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.02 --seed 1234 \
    --log-interval 1 --log-memory-interval 1 --log-throughput --eval-iters 0 --eval-interval 100
```

Later flags win in Megatron's argparse, so append overrides after the block rather than editing
it. `--eval-interval` must stay set: iteration-based data sizing divides by it. `--mock-data` needs
`--moe-router-force-load-balancing` for meaningful timing (random tokens give a skewed router);
drop both together for real data (`--data-path ... --split 99,1,0`) and switch to
`--tokenizer-type HuggingFaceTokenizer --tokenizer-model Qwen/Qwen3.8-Flash-Next`.

## Verified

The command above was run on 4 GB300s (2026-09-15, `lit/main_qwen4`): 8 iterations, finite
`lm loss` / `mtp_1 loss` / `load_balancing_loss` / `indexer loss`, **71.9 GiB peak allocated** per
GPU at seq 4096. The same command with flag overrides is what produced the
[support matrix](../validation/support_matrix.md) — run it before trusting a new branch.

## What to expect

- Iteration 1 is slow (Triton/inductor JIT for GDN, QSA and the PLE hash kernels), after that a
  few hundred milliseconds per step at this size.
- Every iteration line must show finite `lm loss`, `mtp_1 loss`, `load_balancing_loss` and
  `indexer loss`. With random init the first `lm loss` is ≈ `ln(248320) = 12.4`; on mock data it
  then drops far faster than on text (the mock dataset is a deterministic pattern) — mechanics, not
  quality.
- Gradient-level proof that the memory is really in the graph: `--engram-verify-training` prints
  per-iteration `num_tables=N zero_grad_tables=0 nonfinite_tables=0 changed_tables=N` (N = 16
  tables × EP ranks, so 64 at EP4). This is the check that caught two silent no-ops during
  integration (memory built but never applied). Two things to know: the flag reads full gradients
  from `main_grad` and therefore **refuses `--use-distributed-optimizer`**, and a warmup schedule
  starting at lr 0 makes iteration 1 report `changed_tables=0` — use a constant lr for the check.
- Parameter counts with and without each of `--engram-*`, `--enable-mhc-connections …`,
  `--mtp-num-layers` must differ; if not, the flag did not reach model construction.

## Smaller variants used by the tests

The whole-model parity harness uses a 4-layer `GEQE/QE` proxy with hidden 512, 8 experts and PLE
base 20,000 (parameter counts in
[`../architecture/config_mapping.md`](../architecture/config_mapping.md)); the Engram functional
run uses `pretrain_hybrid.py` with pattern `*-*-`, memory on layer 2, on 2 GPUs. Both are
described in [`../validation/parity_and_acceptance.md`](../validation/parity_and_acceptance.md).
