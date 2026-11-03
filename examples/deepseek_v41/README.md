# V4.1 text training draft

This composition uses main's HybridModel, MoE, distributed optimizer and packed CP
path. Each logical block is `VE`: CSA2 attention followed by MoE. It does not
include vision, DSpark or FP4 QAT. The fused attention uses
main's 512-dimensional KV path, with BF16 or MXFP8 index selection, H32/H64 and D128. MXFP8 selection
requires SM100 or later and the cuDNN backend; indexer supervision retains its
original BF16/FP32 path.

A small GPU smoke configuration (requires the normal Megatron CI container,
FlashMLA and cuDNN Frontend):

```bash
uv run python -m torch.distributed.run \
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
SFT packed loader, `--dataloader-inter-document-masking`, or static varlen
scheduling with `--use-varlen-dataset --sequence-packing-scheduler dp_balanced`.
The scheduler path preserves logical and padded prefixes separately and slices
queries contiguously. Dynamic hybrid-CP remains unsupported.

Full recomputation uses `--recompute-granularity full --recompute-method uniform
--recompute-num-layers 1`. Selective mHC recomputation keeps main's existing
interface. Partial graphs use the common `StatefulModule`/`StatefulGraphs`
interfaces. With `--cuda-graph-impl transformer_engine --cuda-graph-modules attn
moe_router`, the existing warmup boundary arms capture on the next real batch;
the pipeline schedule determines each chunk's required graph slots. The explicit
model API (`capture_layers`, `graph_slots`, `graph_backend`) remains available.
Physical V layers capture attention and shared CSA2 outputs; E layers capture
router probabilities and maps. Normalization on E layers, dispatch and experts
remain eager. Engram and mHC aggregation/post-mixing run outside the consumer
graphs, preserving selective mHC checkpoints. Full recompute uses a separate
execution profile. Explicit capture callers must reserve a slot for every
in-flight microbatch. Capture profiles fix geometry, dtype and gradient activity;
packed prefix contents and tokens may change at that geometry.

Local checks use real CPU PyTorch/Gloo. HybridStack parity tests substitute CPU
projections and RoPE storage for TE/CUDA, and compare eager, checkpoint and split
PP outputs/parameter gradients. H100 qualification also covers real FlashMLA/cuDNN attention forward/backward,
native/fused and CP gradient parity, Torch/TE state graphs, PP2/VPP2 overlap,
DP2/EP2/PP2/CP2 training, fused mHC and full recomputation, plus checkpoint save/load.
These checks use synthetic data and do not establish released-model convergence.

Blackwell validation additionally covers real MXFP8 selection, parameter gather,
recompute, explicit layer graphs and distributed training. The integrated stack
(with the MoK dependency from #8210) passed 16-GPU PP2/VPP4/EP8/CP4, 16K packed
training with both BF16 and MXFP8, including overlapping gradient reduction and
parameter gather. Each precision completed 12 iterations without skipped or
non-finite logged steps. Full cross-branch, same-weight gradient parity and
released-model convergence are separate from these training checks.


## MXFP8 configuration

Model precision and indexer precision are independent. For the validated MXFP8
path, add these options to a compatible distributed model configuration:

```text
--bf16
--fp8-recipe mxfp8
--fp8-format e4m3
--fp8-param-gather
--reuse-grad-buf-for-mxfp8-param-ag
--moe-router-padding-for-quantization
--dsa-kernel-backend cudnn
--dsa-indexer-precision mxfp8
```

Use `--use-distributed-optimizer` for distributed parameter updates; the tested
recipe uses Muon with `--muon-coefficient-type deepseekv4`, ten Newton-Schulz
steps, `--overlap-grad-reduce` and `--overlap-param-gather`. BF16-only runs omit
the FP8 recipe, parameter-gather reuse and quantization-padding options and set
`--dsa-indexer-precision bf16`.

For V4.1 head-wise Muon, enable `--muon-split-qkv-per-head` with the default
QKV splitting enabled and `--muon-tp-mode duplicated`. The attention Q projection
and Full/Reindex indexer Q projections split by their own head counts, including
the full content and RoPE width of each head. SWA KV and the Full owner's indexer
K each have one shared head and remain whole matrices. Reuse layers have no
indexer weights. Output, compressor and other linear weights retain whole-matrix
Muon. Without the per-head flag, V4.1 retains whole-projection updates.

This also applies with `--use-layer-wise-distributed-optimizer`, which distributes
whole parameters across DP ranks. The separate `--muon-tp-mode layer_sharded`
algorithm currently rejects split-QKV/per-head Muon. GPU validation of V4.1
per-head updates with distributed optimizer, FP8 parameter gather and checkpoint
resume is still required; the training results above predate per-head support.

For the paper's embedding/prediction-head update, add `--muon-use-sinkhorn` to
`--optimizer muon` (also supported with `adaptive_muon`). This requires EO's
`emerging_optimizers.orthogonalized_optimizers.sinkhorn_utils.sinkhorn_balance`;
early builds with the same `0.5.0a0` version string may not include it. Linear
weights continue to use Muon; norms, biases and other excluded parameters continue
to use `--muon-scalar-optimizer`. Per-head Muon is independent of this flag.

Sinkhorn uses EMA/Nesterov momentum 0.95, 11 alternating row/column L2
normalizations ending on rows, a zero-row threshold of 1e-3, epsilon 1e-20 and
update RMS 0.18 (`--sinkhorn-momentum`, `--sinkhorn-num-normalization-steps`,
`--sinkhorn-zero-row-threshold`, `--sinkhorn-eps`, `--sinkhorn-extra-scale-factor`).
Its groups always have zero weight decay, including under the weight-decay
scheduler, and otherwise follow the embedding/output learning-rate schedule.

V4.1 Engram lookup tables use a 5x learning-rate multiplier by default. Set
`embedding_lr_multiplier` in the Engram JSON to change it (for example, `1.0`
disables the multiplier). Projections and normalization weights keep the base
schedule. The multiplier scales the selected parameter-group schedule, including
nonzero warmup and minimum LR, and is retained in optimizer checkpoints.

Sinkhorn requires TP=1 and no GTP-sharded embeddings. Engram tables carry their
explicit EP row partition: near-zero-row masking uses the global mean row norm,
and column normalization reduces squared norms across all row shards. Empty or
temporarily inactive shards participate without gathering the full table. With
`--use-distributed-optimizer`, each local table shard and its FP32 master/momentum
belong to one replica owner through LayerWiseDistributedOptimizer. Compact and
padded layouts choose the same replica coordinate for every EP shard of a table;
padded layouts also pad peer shards to fit the largest local matrix. This can be
expensive for large vocabularies. Direct MCore callers
must pass `use_sinkhorn_for_embeddings=True` to `tag_params_for_buffer_routing`
before constructing DDP, and set `OptimizerConfig.muon_use_sinkhorn=True` when
constructing the optimizer. Existing LayerWise checkpoint support is reused;
switching an existing Adam embedding checkpoint to Sinkhorn requires a fresh
optimizer state. GPU/distributed training validation is still required.

The paper's remaining optimizer recipe is not selected automatically: set Muon
Nesterov momentum (`--muon-nesterov`), momentum 0.95, weight decay 0.1 and update
scale explicitly. `--muon-scale-mode spectral --muon-extra-scale-factor 0.18`
uses shape-based scaling; it does not explicitly normalize each update to RMS
0.18. Use `--muon-scale-mode rms --muon-extra-scale-factor 0.18` to normalize each
nonzero Newton-Schulz output to actual RMS 0.18 before applying the learning rate.
Per-head Muon normalizes each head independently, and zero updates stay zero.
This mode supports ordinary Muon with TP=1 and no GTP weight sharding, including
LayerWiseDistributedOptimizer; AdaptiveMuon and `layer_sharded` are unsupported.
The default remains `spectral`, and the existing `unit_rms_norm` mode also uses
shape-based scaling. No optimizer state is added; resume with the same scale mode
and factor. AdamW uses beta1=0.9, beta2=0.95 and epsilon=1e-20 in the paper. Add
`--apply-wd-to-norm --weight-decay 0.1` to include normalization weights in weight
decay, including standalone RMSNorm/LayerNorm and TE fused normalization weights.
Engram's fused `q_weight` and `k_weight` carry the generic `is_norm_weight` tag,
so this option also includes them while keeping their AdamW optimizer routing.
This takes precedence over `--apply-wd-to-qk-layernorm`; biases and other vector
parameters retain their standard exemptions, and Sinkhorn groups keep zero decay.
The flag defaults to off. Without `--muon-use-sinkhorn`, embeddings/output heads
retain the scalar-optimizer fallback.

The 16K proxy keeps 24 logical blocks / 48 physical V/E layers, hidden size 5120,
48 experts with Top-6 routing, MBS1 and GBS128. It uses selective
`--recompute-modules mla_up_proj mhc`, `--mhc-recompute-layer-num 2`, and no
CUDA graphs or Engram. It does not measure the full model's 64K candidate-pruning
ratio. The validated environment is GB200, PyTorch nv26.09, Transformer Engine
2.20.2 and cuDNN Frontend 1.29. MoK is an optional, separate dependency; the
individual V4.1 component PRs do not install or embed that backend.
