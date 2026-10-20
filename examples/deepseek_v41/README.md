# V4.1 text training draft

This composition uses main's HybridModel, MoE, distributed optimizer and packed CP
path. Each logical block is `VE`: CSA2 attention followed by MoE. It does not
include vision, DSpark, FP4 QAT or a Sinkhorn optimizer. The fused attention uses
main's 512-dimensional KV path, with BF16 indexer H32/H64 and D128.

A small GPU smoke configuration (requires the normal Megatron CI container,
FlashMLA and cuDNN Frontend):

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 uv run python -m torch.distributed.run \
  --nproc-per-node=1 pretrain_hybrid.py \
  --dsv4-version v4.1 --hybrid-layer-pattern VEVEVE \
  --num-layers 6 --hidden-size 256 --num-attention-heads 64 \
  --multi-latent-attention --q-lora-rank 128 --v-head-dim 512 \
  --qk-pos-emb-head-dim 64 --output-projection-groups 8 \
  --output-projection-lora-rank 32 --qk-layernorm --normalization RMSNorm \
  --csa-compress-ratios '[2,0,2,0,2,0]' --csa-window-size 128 \
  --csa2-kv-source-layers 0 --csa2-index-source-layers 0 2 \
  --dsa-indexer-n-heads 64 --dsa-indexer-head-dim 128 --dsa-indexer-topk 128 \
  --no-dsa-indexer-rotate-activation --dsa-kernel-backend cudnn \
  --dsa-indexer-loss-coeff 0.01 --enable-mhc-connections --mhc-single-pass \
  --num-experts 8 --moe-router-topk 2 --moe-ffn-hidden-size 256 \
  --moe-token-dispatcher-type alltoall --swiglu --disable-bias-linear \
  --attention-dropout 0 --hidden-dropout 0 --bf16 \
  --seq-length 1024 --max-position-embeddings 1024 \
  --micro-batch-size 1 --global-batch-size 8 --train-iters 10 --lr 0.0001 \
  --mock-data --tokenizer-type NullTokenizer --vocab-size 256 \
  --eval-interval 1000 --eval-iters 1 --num-workers 0 \
  --no-create-attention-mask-in-dataloader --untie-embeddings-and-output-weights
```

Source IDs and compression ratios above use **physical** V/E layer positions.
`DeepSeekV41Config.from_hf()` converts the released logical block IDs to that
representation. For the 40-block schedule, Full source IDs are `[4,16,28,40]`,
index sources are `[4,16,28,40,48,56,64,72]`, and candidate source is `40`.
The first two logical blocks use window-only attention, the next 18 use r2,
and the final 20 use r1. Insert zero ratios at all E positions.

Engram is enabled with `--engram-config options.json --engram-token-map map.pt`.
The JSON contains the `EngramConfig` fields. Layer IDs here are **logical block
IDs**. `map.pt` must contain the int64 tensor produced from the exact released
tokenizer by `build_compressed_token_map`; an identity map is not a substitute.
The table counts must match the released distinct-prime layout.

For PP/VPP, insert `|` at explicit stage/chunk boundaries. The builder creates
one typed pipeline control group per replica and reuses it for virtual chunks.
Packed batches are staged before P2P; ordinary CP1 batches remain lazy. CP uses
`--attention-cp-layout contiguous --linear-cp-layout contiguous` and an existing
SFT packed loader or `--dataloader-inter-document-masking`. Dynamic hybrid-CP and
the separate sequence-packing scheduler are not enabled in this draft.

Full recomputation uses `--recompute-granularity full --recompute-method uniform
--recompute-num-layers 1`. Selective mHC recomputation keeps main's existing
interface. Graph capture is an explicit model API (`capture_layers`,
`graph_slots`, `graph_backend`), not automatic CLI warmup. It excludes Engram
layers and full-recompute profiles. Reserve a slot for every in-flight
microbatch. Variable-token MoE execution remains eager; capture attention layers
without Engram attached.

Local checks use real CPU PyTorch/Gloo. HybridStack parity tests substitute CPU
projections and RoPE storage for TE/CUDA, and compare eager, checkpoint and split
PP outputs/parameter gradients. H100 qualification also covers real FlashMLA/cuDNN attention forward/backward,
native/fused and CP gradient parity, Torch/TE state graphs, PP2/VPP2 overlap,
DP2/EP2/PP2/CP2 training, fused mHC and full recomputation, plus checkpoint save/load.
These runs use small proxy models and synthetic data, not released-model convergence
or a production throughput qualification. Blackwell validation is left to CI.
