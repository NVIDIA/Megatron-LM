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

The same JSON controls row parallelism and the optional lookup runtime. For a
128-rank physical stage (TP1), the following fields shard Engram over all 128
ranks independently of MoE EP64:

```json
{
  "parallel_size": 128,
  "prefetch": true,
  "defer_grad_transfer": true,
  "buffer_budget_mb": 1024,
  "communication_backend": "hybridep",
  "hybridep_chunk_size": 32768
}
```

Merge these fields into the existing architecture JSON. `parallel_size` must
divide the physical stage's DP×CP size; the remaining factor is the number of
table replicas. The builder creates the groups once, and the parameter domain
carries them through DDP, optimizer ownership, Sinkhorn statistics and checkpoint
replica metadata. Independent row parallelism currently requires TP1, no GTP
remapping, standard Megatron DDP and one distributed optimizer instance. Use
LayerWise for Muon/Sinkhorn, or DistributedOptimizer for Adam/SGD. When changing
from the legacy expert optimizer domain to the new `engram` domain, load model
weights with a fresh optimizer; automatic migration of legacy optimizer groups
is not implemented.

Omitting `parallel_size` preserves the existing EP/expert-DP partition and
optimizer domain. `prefetch` and `defer_grad_transfer` default to false and can
be enabled independently. The original lookup remains available when both are
off with the NCCL or HybridEP backend. Prefetch stages token/THD metadata before
the PP schedule, shares it across VPP chunks, deduplicates each batch's requests,
and carries a ticket to the matching consumer. Lookup runs outside full
recomputation and partial CUDA graphs; projection and gating retain their normal
recompute behavior. Deferred backward coalesces queued microbatches per table,
delivers row gradients to their owners, then lets DDP reduce replicas.

`buffer_budget_mb` bounds the runtime's reserved row-payload/gradient capacity,
shared across local tables and VPP chunks. Pressure flushes pending gradients
and evicts optional prefetch payloads, which can be fetched again. It does not
bound model activations, saved request routes, transient tensor operations,
DDP/optimizer storage or opaque NCCL/HybridEP allocations. A single lookup that
cannot fit raises an error requesting a larger budget; rows are never dropped.

The experimental `communication_backend: "symmetric"` binds NCCL4Py + CuTe DSL.
Within an NCCL LSA domain it reads registered table rows directly. Cross-domain
requests, packed FP8 bytes/scales and gradients use GIN. Two receive slots per
sender bound staging memory; GPU completion signals protect slot reuse, without
per-chunk host launches or collectives. The existing Triton activation repack is
reused; sorted duplicate gradients are accumulated by one Triton writer per row.
Count planning and table-registration metadata still use host collectives per
lookup. Registration is reused while parameter storage is unchanged,
and collectively refreshed if an optimizer/DDP operation replaces an allocation.

This requires `nccl4py >= 0.6.0`, CuTe DSL, and matching NCCL headers/runtime/device
bitcode. Prefer CuTe DSL 4.8.1: its
[release notes](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md#481-2026-10-15)
confirm the fix for zero-argument `cute.extern` calls across compilations.
Unmodified 4.8.0 can fail repeated NCCL barrier compilation with an MLIR context
mismatch. The isolated extern recompilation check passes with unmodified 4.8.1.
Four-GPU B300 smoke tests have passed with CuTe DSL 4.7.0,
NCCL4Py 0.6.0 and NCCL 2.32.3.
If setting `NCCL_HOME`, it must identify that same NCCL installation with
`lib/libnccl_device.bc`.
Cross-domain operation requires full-peer GIN connectivity; LSA-only groups do
not require GIN. Table allocations must be registerable CUDA VMM memory, e.g.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; tables are never shadow-copied.
PyTorch storage starts need not be 4-KiB aligned. Window registration queries the
underlying CUDA mapping, registers from its base, and records each table's full
offset from that window so LSA reads handle different suballocations on peers.
The borrowed window buffer retains the original storage;
allocation queries and registration are cached until storage changes.
Workspace is included in `buffer_budget_mb`; NCCL's internal allocations are not.
Transfers split each peer's rows across up to 32 independent stripes, using
separate credits and slots within the same workspace budget. Small budgets use
fewer stripes. Barrier and exchange executables are cached explicitly so steady
state calls avoid CuTe auto-JIT argument processing; row counts and row widths
remain dynamic. These changes require no training arguments.
Use `EngramRuntime.close()` before destroying the row groups when explicitly
tearing down a model. Single-node LSA exchange, direct reads, storage replacement,
and MXFP8 prefetch/deferred-gradient/DDP integration have GPU smoke coverage.
128-shard training and end-to-end performance still need qualification.
The striped transport passes four-GPU LSA tests and
eight-GPU tests across two NVLink domains on GB300 with NCCL4Py 0.6.0,
NCCL 2.32.3 and CuTe DSL 4.8.1. These cover mixed LSA/GIN reads and transfers,
empty/skewed requests, slot reuse, storage replacement and byte-exact replay.

`benchmark_engram.py` isolates count exchange, cached registration, barriers,
FP8 row lookup, gradient transfer and repeated-row accumulation. Its runtime
mode includes deduplication, FP8 activation repacking, autograd and deferred
gradient delivery for two embedding modules. It excludes Engram projections,
DDP and the optimizer, so its results are not training iteration times.

Run inside the training container, with registerable CUDA allocations:

```bash
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  examples/deepseek_v41/benchmark_engram.py --output=/tmp/engram-transport.json
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  examples/deepseek_v41/benchmark_engram.py --mode=runtime \
  --backend=symmetric --microbatches=32 --output=/tmp/engram-runtime.json
```

Defaults are 4096 tokens, 24 tables, 256 columns and 125024 rows **per shard**.
The per-GPU table working set stays fixed when changing world size; smaller jobs
therefore have fewer global rows. Runtime mode also accepts `nccl` and `hybridep`
for comparisons. JSON contains warm steady-state samples, the maximum rank's
wall/submission/CUDA times, rank-0 memory statistics, and the transport's actual
LSA/GIN topology. Transport mode records cold initialization and lookup separately
and checks returned bytes against exact references before timing. Use nodes from
different NVLink domains to qualify GIN; node count alone does not establish the
communication path.

For PP/VPP, insert `|` at explicit stage/chunk boundaries. The builder creates
one typed pipeline control group per replica and reuses it for virtual chunks.
Packed batches are staged before P2P; ordinary CP1 batches remain lazy unless
Engram prefetch is enabled. CP uses
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

Engram follows the model's FP8 configuration. `--fp8-recipe mxfp8 --fp8-format
e4m3` uses TE Linear for its fused key/value projection. Adding `--fp8-param-gather`
also stores the lookup table in rowwise MXFP8; use
`--reuse-grad-buf-for-mxfp8-param-ag` with the existing LayerWise optimizer path.
No additional Engram precision flag is needed. Layers explicitly kept in BF16
keep BF16 Engram storage and projections too.

FP8 lookups exchange E4M3 values and their E8M0 scales together over EP and retain
those bytes as the projection's rowwise GEMM input. The table stores only one
scaling direction. Projection weight gradients need columnwise quantization of
the retrieved batch. Triton kernels gather values/scales directly into the EP
payload, then fuse receive reordering, head concatenation and columnwise
requantization into TE's GEMM buffers without an intermediate BF16 activation.
The rowwise values/scales are preserved exactly; columnwise quantization rounds
through the model dtype as TE does. The single-table path skips table sorting.
A native fallback remains available when Triton is absent, generating only the
needed columnwise quantization. There is no persistent BF16 table or full-table
columnwise copy. Lookup gradients
remain high precision, with repeated rows accumulated in FP32 before casting to
the model gradient dtype. Normalization and gate arithmetic remain FP32.

FP32 master parameters and Sinkhorn momentum follow the existing optimizer path.
Local table storage is padded to multiples of 32; padding is excluded from
Sinkhorn statistics and the logical model/master/momentum checkpoint shapes.
The checkpoint factory supports whole row shards, including LayerWise ownership;
the flattened optimizer checkpoint format is unsupported. Parameter
synchronization among replicas still uses existing BF16 staging and FP8
requantization; the FP8 wire format described above applies to EP lookup results.
Projection scaling follows the existing DSv4 TE MXFP8 recipe, rather than the
inference reference's 32-by-32 weight scaling layout.

CPU/Gloo tests cover FP8 byte transport, row gradients, padding, logical checkpoint
metadata and EP/replica-DP Sinkhorn parity. Blackwell tests cover real TE storage,
fused/native payload and activation parity, projection gradients, recompute,
optimizer copy-back and EP1/EP2 bit-exact replay (including fused scale padding).
The latter tests require GPU qualification; they were not run on the CPU
development host.

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


## Performance switches

The optimizations below speed up the 16-GPU 64K proxy on GB300: 40 V/E blocks, hidden size
5120, 48 experts with Top-6 routing, Engram, 64K-token packed sequences, CP16 and EP8, MXFP8,
MoK experts, and transformer_engine CUDA graphs of the attention layers. A training step takes
24.36 s (579 TFLOP/s per GPU) without them and 18.02 s (783 TFLOP/s per GPU) with all of them,
including the run settings below and the two opt-in switches.

Each switch is an environment variable, read once at import. Every optimization that keeps
training results bit-identical is on by default, and its variable is a kill switch. The one
exception is `MCORE_ROUTER_SAVE_FP32_INPUT`, which trades memory for time and stays opt-in. The
debugging and profiling aids (`MCORE_CSA2_DEVICE_CHECKS`, `MCORE_ROUTER_CHECK_STATIC_TOKENS` and
`MCORE_COMPONENT_NVTX`) are off by default. The cuDNN sparse indexer loss and the cuTile mHC
projection change numerics, so they are opt-in and off by default. The router switches
(`MCORE_ROUTER_*`, `MCORE_AUX_LOSS_SCALE_0DIM` and `MCORE_RANDOM_STE_EMPTY_LIKE`) apply to every
MoE model, with or without MoK; the `MCORE_MOK_*` switches apply to the MoK backend only.
Boolean switches accept `1`, `true`, `yes` or `on` and `0`, `false`, `no` or `off` (any case);
any other value, including an empty one, raises a `ValueError` naming the variable. Integer
switches raise one for non-integer values too; `MCORE_CSA2_CANDIDATE_CHUNK_MIB` and
`MCORE_CSA2_PAD_ROW_CAPACITY` must also be positive.

| Variable | Default | Kill switch or opt-in | Effect | Bit-identical |
|---|---|---|---|---|
| `MCORE_CSA2_DEVICE_CHECKS` | off | `1` turns the checks on | Device-side validation of packed THD metadata: 17 checks per CSA2 layout build, each a compare/reduce plus a `torch._assert_async` launch, also inside CUDA graphs, where a failing check aborts the CUDA context without a Python stack. It also gates the Q/KV agreement asserts of `validate_packed_inputs`, which the DSv4 and V4.1 attention paths share. Metadata on the CPU is always validated. Off saves about 0.5 s per step at 64K; turn the checks on for new data pipelines and in CI. | Yes: the checks change no tensor |
| `MCORE_ROUTER_FAST_CAST` | on | `0` | Widens the BF16/FP16 router GEMM input and weight to FP32 with an exact, vectorized Triton kernel instead of PyTorch's casting copy. | Yes: the widening is exact |
| `MCORE_ROUTER_FAST_CAST_BLOCK`, `_WARPS`, `_SPLIT` | `4096`, `8`, off | launch tuning | Launch shape of that kernel: elements per program (a power of two from 256 to 65536), warps per program, and the store variant (`_SPLIT` writes each 16-byte input chunk as two 16-byte FP32 stores). An invalid block/warps pair falls back to 4096/8 with a warning. | Yes |
| `MCORE_ROUTER_SAVE_FP32_INPUT` | off | `1` opts in | Saves the router input cast to FP32 in forward for backward instead of casting again there, at the cost of one FP32 copy of the router input per MoE layer and microbatch in flight. | Yes |
| `MCORE_MOK_SKIP_UNROUTED_GRAD_MASK` | on | `0` | Lets MoK skip zeroing the router-weight gradients of unrouted top-k slots when no slot can be unrouted (no capacity factor, hash routing or routing replay) and only Megatron's route adapter reads them. Inactive with a MoK build whose `functional.backward` lacks `zero_unrouted_router_grads`. | Yes: the skipped values are never read |
| `MCORE_ROUTER_STATIC_TOKENS` | on | `0` | Takes the token count of the per-token aux-loss scale from the host (local tokens times the static group sizes) instead of an all-reduce, when every rank holds the same Python-int count and the aux loss is FP32 or FP64. | Yes: the same count |
| `MCORE_ROUTER_CHECK_STATIC_TOKENS` | off | `1` (debug) | Still all-reduces that count and asserts that it equals the host value; adds a host sync per call outside CUDA-graph capture. | Yes |
| `MCORE_MOK_ROUTE_NO_MATERIALIZE_GRADS` | on | `0` | The routing-map conversion for MoK does not materialize zero gradients for its non-differentiable outputs. | Yes |
| `MCORE_AUX_LOSS_SCALE_0DIM` | on | `0` | Forms the aux-loss scale gradient as a 0-dim tensor instead of a `[1]` tensor that autograd reduces. | Yes |
| `MCORE_RANDOM_STE_EMPTY_LIKE` | on | `0` | RandomSTE draws its random logits into `torch.empty_like` instead of a clone. | Yes: the same draws |
| `MCORE_CSA2_SPARSE_PAD_ZERO` | on | `0` restores `masked_fill` | CSA2 zeroes only the THD padding rows in place instead of building masked copies of whole tensors: the attention query, key and output (and, through hooks, their gradients) in forward, and the incoming gradient and LSE rows in backward, which it restores after the kernel. | Yes: the same `+0` values |
| `MCORE_CSA2_PAD_ROW_CAPACITY` | `512` | positive integer | Static number of padding rows one zeroing call handles, which keeps the row list capturable in CUDA graphs; the zeroing work grows with it. More padding rows in one call trip a device assert; raise it or set `MCORE_CSA2_SPARSE_PAD_ZERO=0`. Data with many short documents per microbatch, each padded to the dataset's granularity (a multiple of 2 x CP rows), can exceed 512 rows on one CP rank. | Yes |
| `MCORE_CSA2_TOPK_PAD_SEGMENTS` | on | `0` | Compact Top-K gives THD padding rows empty key spans, so their tied scores no longer stall the selection launch on the last CP rank. | Yes: real rows keep their spans |
| `MCORE_MHC_SP_FUSED_GRAD_ADD` | `exact` | `0` | One kernel sums the three gradients of each single-pass mHC residual input, rounding after each add in the order autograd accumulates them. `fp32` rounds the FP32 sum once instead (opt-in). | `exact`: yes; `fp32`: no |
| `MCORE_CSA2_CANDIDATE_CHUNK_MIB` | `152` | `32` restores the old budget | Score budget per dense-scorer launch of the candidate-source layer, in MiB: fewer, fuller launches for about 120 MiB more transient memory per call. | Yes: rows are scored independently |
| `MCORE_CSA2_MXFP8_NATIVE_H32` | on | `0` pads H32 to H64 | cuDNN's MXFP8 selection scorers take H32 indexer heads natively instead of zero-padded to H64. | Yes, by test: scores and selections match bit for bit with the validated cuDNN Frontend |
| `MCORE_DSV4_GROUP_PROJ_BMM` | on | `0` restores `einsum` | The non-TE grouped output projection runs as strided-batched GEMMs on the token-major tensors, without layout copies. TE BatchedLinear is unchanged. | Yes, by test: cuBLAS picks the same GEMM algorithms |
| `MCORE_CSA2_CUDNN_INDEXER_LOSS` | off | `1` opts in | Computes the sparse indexer loss with cuDNN Frontend's DSA kernels instead of Triton, about 0.6 s per step faster at 64K. Inputs it does not support keep Triton, with one warning per reason; it needs an SM10x GPU, cuDNN Frontend's DSA wrappers, a BF16 H32/D128 indexer and at most 2048 Top-K slots, among other checks. | No: the score gradient is rounded to BF16, and dK accumulates with unordered FP32 adds (not reproducible run to run) |
| `MCORE_MHC_SP_CUTILE_PROJ_RMS` | off | `1` opts in | Single-pass mHC computes its mapping projection and RMS with cuTile split-K kernels instead of Inductor and TF32 cuBLAS on an FP32 copy, where the backend policy selects cuTile and the residual qualifies (aligned, K = n * hidden >= 8192); other residuals warn and keep Inductor. | No: TF32 MMAs on the residual dtype and merged FP32 gradients |
| `MCORE_COMPONENT_NVTX` | off | `1` opts in | NVTX ranges per model component (`comp:<name>`) and layer (`layer:<n>`), forward and backward, also recorded during CUDA-graph capture. | Yes: host-only |

"By test" means the equality was checked on GB300 rather than following from the code: re-run
the determinism tests under `tests/unit_tests/determinism/kernels/` after a cuDNN Frontend or
cuBLAS upgrade and on other GPUs. Other changes in the same work need no switch and are
bit-identical too: the V4.1 state codec reuses the packed layouts after attention graph replays
and restores the chunk-start state once per forward, packed THD batches are built with fewer
host syncs, `pretrain_hybrid.get_batch` builds its transformer config once per argument
namespace, and the Engram JSON is parsed once per path. Full-iteration CUDA graphs can also
capture V4.1 training now (`--cuda-graph-impl full_iteration`), without Engram; with Engram that
option is rejected, because Engram plans its EP lookups on the host.

The numbers above also use these run settings, outside Megatron-LM's switches:

```text
--moe-megakernel-backend mok
--moe-megakernel-backend-config '{"fwd_num_comm_sms": 34, "bwd_num_comm_sms": 34, "minibatch_size": 8192, "macrobatch_size": 32768, "schedule_capacity_multiplier": 0.0625, "all_gather_top_experts_chunk_bytes": 2048}'
--recompute-granularity selective --recompute-modules mhc --mhc-recompute-layer-num 2
```

The backend config replaces MoK's whole configuration, so keys left out take MoK's defaults (for
example a `schedule_capacity_multiplier` of 0.5); give all six. The baseline ran 28/28
communication SMs and minibatch 4096 with the same other values. MoK is built from its PR #7
(commit `ad5e5b4`) with per-call host-overhead trims on top; both live in the MoK repository,
and `MCORE_MOK_SKIP_UNROUTED_GRAD_MASK` needs at least that build. Recomputing only `mhc`
instead of the baseline's `mla_up_proj mhc` trades activation memory for time; recomputation
reruns the same kernels on the same inputs, so values are expected to match. The measured runs
also set `MCORE_CSA2_PAD_ROW_CAPACITY=32`: the proxy data has at most one padding row per
microbatch. The MoK build and its schedule options are outside Megatron-LM's bit-identity tests:
keep them fixed when comparing switch settings, and keep `MCORE_CSA2_CUDNN_INDEXER_LOSS` and
`MCORE_MHC_SP_CUTILE_PROJ_RMS` off when the comparison must be bit-identical.
