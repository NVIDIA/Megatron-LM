# Proxy model on one node

A proxy that keeps **every per-layer, per-expert and PLE dimension of the real model** and shrinks
only the things that set the total size, chosen so that per-rank behaviour on 4 GPUs (EP4) matches
the 64-GPU full model:

| Dimension | Full model | Proxy | Why |
|---|---|---|---|
| Qwen layers | 48 = 12 × `GEGEGEQE`, + 1 MTP | **8** = 2 × `GEGEGEQE`, + 1 MTP | PLE stays on hybrid layer 3 (Qwen layer 2, a GDN layer) |
| experts | 512 at EP64 → 8 local per rank | **32** at EP4 → 8 local per rank | same grouped-GEMM shapes and dispatch fan-out per rank |
| PLE table base per n-gram order | 20,000,000 | **1,000,000** | ≈ 2.6 B table parameters instead of 51 B; still EP row-sharded |
| vocabulary (`LAYOUT=thd32k` only) | 248,320 | **40,832** | ≈ 1/6, scaled with the layers (8 of 48) so the vocabulary-sized share of the step (embedding, output layer, cross-entropy) matches the 48-layer model at 32K tokens per rank; a multiple of 128. `bshd4k` keeps the real vocabulary |

≈ 5.5 B parameters total at the real vocabulary (≈ 2.4 B active per token). It fits one GB300
node and is a *functional / recipe-development* proxy, not a memory proxy for EP64. It trains on
Megatron's mock data with `NullTokenizer` (EOD = vocab − 1, so `--engram-eos-token-id` is 40831
in `thd32k` and 248044 in `bshd4k`) — no tokenizer download.

## Command (4 GPUs, EP4, BF16, from scratch)

Run from the Megatron-LM root inside the image built from
[`../../Dockerfile.gb300`](../../Dockerfile.gb300):

```bash
./examples/qwen4exp/train_proxy_1node.sh                  # LAYOUT=thd32k, the default
LAYOUT=bshd4k ./examples/qwen4exp/train_proxy_1node.sh    # BSHD baseline, any container with TE,
                                                       # fla, Triton and emerging-optimizers
```

The script's full command for `LAYOUT=bshd4k`:

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
    --attention-dropout 0.0 --hidden-dropout 0.0 --apply-wd-to-qk-layernorm \
    --mock-data --seq-length 4096 --moe-router-force-load-balancing \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 4 \
    --context-parallel-size 1 --expert-tensor-parallel-size 1 \
    --optimizer muon --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
    --micro-batch-size 1 --global-batch-size 16 \
    --bf16 --transformer-impl transformer_engine --enable-experimental \
    --cross-entropy-loss-fusion --cross-entropy-fusion-impl native \
    --train-iters 200 --lr 3.0e-4 --min-lr 3.0e-5 --lr-decay-style cosine \
    --lr-warmup-iters 20 --lr-warmup-init 0.0 --weight-decay 0.1 --clip-grad 1.0 \
    --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.02 --seed 1234 \
    --log-interval 1 --log-memory-interval 1 --log-throughput --eval-iters 0 --eval-interval 100
```

`LAYOUT=thd32k` replaces the vocabulary, sequence and evaluation flags of that block and adds the
packed-THD data path and the training kernels (it also exports `FLA_DISABLE_TENSOR_CACHE=1`, which
GDN requires for THD attention capture, `NVTE_FUSED_ATTN=1`, `NCCL_NVLS_ENABLE=0`,
`NCCL_GRAPH_REGISTER=0` and `NUM_OF_TOKENS_PER_CHUNK_{DISPATCH,COMBINE,PREPROCESSING}_API=128`):

```bash
    --vocab-size 40832 --make-vocab-size-divisible-by 128 \
    --engram-unigram-vocab-size 40832 --engram-eos-token-id 40831 \
    --seq-length 32768 --eval-iters 1 --split 98,2,0 \
    --sft --pad-packed-seq-alignment max --max-seqlen-per-dp-cp-rank 32768 \
    --thd-max-packed-sequences 1 \
    --sft-mock-dataset-config-json '{"mode":"distribution","type":"lognormal","format":"thd","min_seq_len":32768,"max_seq_len":32768,"mean_seq_len":32768,"lognormal_sigma":1.1}' \
    --cuda-graph-impl transformer_engine --cuda-graph-modules attn moe_router \
    --cuda-graph-warmup-steps 2 --cuda-graph-capture-engram-host --te-rng-tracker \
    --moe-megakernel-backend mok \
    --moe-megakernel-backend-config '{"fwd_num_comm_sms":48,"bwd_num_comm_sms":56,"minibatch_size":8192,"macrobatch_size":65536,"schedule_capacity_multiplier":0.0625,"all_gather_top_experts_chunk_bytes":2048}' \
    --moe-router-fusion \
    --gdn-kernel-backend cudnn --gdn-pre-gated-delta-rule-fusion \
    --qsa-use-sparse-attention --qsa-indexer-loss-impl fused --qsa-indexer-loss-chunk 1024 \
    --manual-gc --manual-gc-interval 100 --rerun-mode disabled \
    --no-check-for-nan-in-loss-and-grad --num-workers 6 --log-device-memory-used
```

The `--sft` path builds a validation loader, so `thd32k` needs `--eval-iters` ≥ 1.

Later flags win in Megatron's argparse, so append overrides after the block rather than editing
it. `--eval-interval` must stay set: iteration-based data sizing divides by it. `--mock-data` needs
`--moe-router-force-load-balancing` for meaningful timing (random tokens give a skewed router);
drop both together for real data (`--data-path ... --split 99,1,0`) and switch to
`--tokenizer-type HuggingFaceTokenizer --tokenizer-model Qwen/Qwen3.8-Flash-Next`.
The optimizer is Muon on the layer-wise distributed optimizer, with Megatron's Muon defaults
(momentum 0.9, 5 quintic Newton-Schulz steps, spectral scale, split QKV); everything that is not
a 2-D matrix goes to Adam, configured by the `--adam-*` flags. Muon rejects
`--use-precision-aware-optimizer`. `--optimizer adam` restores Adam for every parameter.

## Verified

`LAYOUT=thd32k`: its arguments were run on 4 GB300s for 40 iterations each with Muon and with Adam
plus `--use-precision-aware-optimizer` (constant learning rate instead of the script's cosine
schedule): finite losses, no skipped iterations, iteration-1 `lm loss` 10.7386, peak allocated
**126.9 GiB** per GPU with Muon and **122.3 GiB** with Adam.

Performance of `LAYOUT=thd32k` on one node of 4 × GB300, with the dependencies pinned in
[`../../Dockerfile.gb300`](../../Dockerfile.gb300); mean of iterations 21–40 of a 40-iteration run,
TFLOP/s as reported by `--log-throughput` (model FLOPs only):

| Optimizer | ms/step | TFLOP/s/GPU | Peak allocated |
|---|---:|---:|---:|
| `--optimizer muon` (default) | 2677 | 384 | 126.9 GiB |
| `--optimizer adam --use-precision-aware-optimizer --main-grads-dtype fp32 --main-params-dtype fp32 --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16` | 2394 | 430 | 122.3 GiB |

Muon's Newton-Schulz work is not counted as model FLOPs, so it shows up as lower TFLOP/s.
Back-to-back runs on one node agree within 0.1 %; across nodes within 0.5 %.

`LAYOUT=bshd4k`: run on 4 GB300s with Adam (before Muon became the default): 8 iterations, finite
`lm loss` / `mtp_1 loss` / `load_balancing_loss` / `indexer loss`, **71.9 GiB peak allocated** per
GPU at seq 4096. The same command with flag overrides is what produced the
[support matrix](../validation/support_matrix.md) — run it before trusting a new branch.

## What to expect

- Iteration 1 is slow (Triton/inductor JIT for GDN, QSA and the PLE hash kernels); later iterations
  are much faster.
- Every iteration line must show finite `lm loss`, `mtp_1 loss`, `load_balancing_loss` and
  `indexer loss`. With random init the first `lm loss` is ≈ `ln(vocab)`: 10.6 for `thd32k`, 12.4 for
  `bshd4k`; on mock data it
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
