# Engram

Engram adds trainable, deterministic n-gram memory to selected GPT transformer layers. This
document defines the first supported MCore milestone. The numerical definition follows the
official DeepSeek Engram implementation at commit
`fb7f84a21f91223715394a33a1dc24bbfb7f788e` and Engram paper v2 (arXiv 2601.07372).

## Variants

Two hashing/layout variants share the module, the EP-sharded tables, and every distributed
mechanism; `--engram-variant` selects between them. The per-variant conventions are declared
once in `megatron/core/models/engram/variants.py` as an `EngramVariant` descriptor, so adding
a variant is a matter of adding a descriptor rather than auditing the package for
variant-specific conditionals:

| | `deepseek` (default) | `qwen` (Qwen3.8-Flash-Next PLE) |
|---|---|---|
| Token preprocessing | offline tokenizer compression artifact | raw token IDs (no artifact) |
| Multipliers | PCG64, from the artifact | splitmix64(seed + 10007·ple_layer_index), in-process |
| Table primes | next unused prime above each order budget | n-th prime after `ngram_vocab_size_base − 1`, per head |
| n-gram boundary | left-pad with the compressed pad token | window resets at every EOS (`--engram-eos-token-id`) |
| Projection bias / gamma | biased projections, ordinary gamma | bias-free, zero-centered group-norm gamma |
| Packed (THD) rows | rejected | supported via the `--sft` family; schedulers incl. `--use-varlen-dataset` rejected; with PP > 1 requires `--pad-packed-seq-alignment max` (fixed-capacity rows; the prefetched tokens are zero-padded identically) |

The qwen variant additionally requires `--engram-unigram-vocab-size` (the HF
`config.vocab_size` bounding the multipliers; the padded vocabulary must not exceed it) and
`--engram-vocab-sizes` entries all equal to `ngram_vocab_size_base`. Both variants use one
fused key projection and one group RMSNorm per role across all residual streams — the exact
official Qwen layout, mathematically identical to per-stream projections and norms.

The official convolution takes zero left context at the sequence start and does **not**
segment at document boundaries: only the hash windows reset at EOS, while the 9-position
convolution mixes across boundaries exactly as the reference implementation does.
Packed rows shorter than the capacity are zero-padded end to end (tokens by the prefetch,
hidden states by ``pad_sequence_for_thd``); the padded tail is dummy sequence and the causal
convolution cannot leak it into valid positions. Packed training with PP >= 3 is currently
blocked upstream: middle pipeline stages receive no ``cu_seqlens``/``max_seqlen`` from the TP
batch broadcast and fail identically without Engram.
`tools/engram/convert_qwen_ple.py` converts HF PLE weights (fused padded table, whole or
dim-0-sharded) into per-head prime tables and back-loads them into an initialized module.

## Configuration and validation

Engram is enabled when `--engram-vocab-sizes` is present. The Engram namespace contains the
global table budget for every n-gram order, selected 1-based global layer IDs, maximum n-gram
order, hash-head count, memory dimension per n-gram order, convolution kernel size, hash seed,
raw tokenizer pad ID, tokenizer-map artifact, sparse-table LR multiplier, and sparse-table weight
decay.

Startup validates the following before model allocation:

- there is one positive global vocabulary budget for every order from 2 through the configured
  maximum order;
- layer IDs are unique and fall in `[1, num_layers]`;
- the memory dimension is divisible by the positive number of hash heads;
- the versioned tokenizer-map artifact matches the configured tokenizer vocabulary, pad ID,
  selected layers, maximum order, and hash seed;
- CP is one and VPP, activation recomputation, CUDA graphs, and FSDP are off;
- packed (THD) sequences are off, except for variants that reset n-gram windows at a document
  boundary token (see the variant table above).

EP is legal without MoE experts when Engram is enabled. If MoE and Engram coexist they share the
same EP dimension and `ProcessGroupCollection.ep`.

Engram issues its own variable-split all-to-all and does not go through the MoE token dispatcher,
so it neither requires nor restricts the dispatcher the MoE layers use: `--moe-token-dispatcher-type
flex` with the `deepep` or `hybridep` backend keeps working unchanged alongside it, and both are
covered by a smoke test. The one MoE communication feature that is rejected is
`--overlap-moe-expert-parallel-comm`, whose fine-grained attention callable does not forward
`input_ids`; that is a separate 1F1B overlap scheduler, not a dispatcher backend.

## Offline tokenizer map and hashing

`tools/engram/generate_tokenizer_map.py` is the only component that constructs a Hugging Face
tokenizer. It scans raw token IDs in ascending order and reproduces the official normalization
sequence: NFKC, NFD, accent stripping, lowercase, whitespace collapse, preservation of the
single-space token, and surrounding-space stripping. Undecodable replacement-character tokens
use their tokenizer token string as the canonical key. The first occurrence of each key receives
the next compressed ID.

The artifact also contains the official NumPy-PCG64-generated odd multiplier for every selected
layer and n-gram position. Keeping these constants in the offline artifact lets model construction
register Torch buffers without importing NumPy or reproducing PCG64 in the training process.

For input `tokens[B,S]`, the model maps nonnegative token IDs through `remap[V]`, left-pads each
suffix with the compressed pad ID, and computes the official signed-int64 multiplicative-XOR mix.
For each order and head it applies the corresponding distinct prime modulus. Hashing finishes on
the full local input sequence before any SP selection, so n-grams at an SP boundary include tokens
from the preceding TP partition.

## Tensor layout and residual semantics

The canonical layouts are:

| Value | Standard residual | Native mHC |
| --- | --- | --- |
| Transformer input | `[S_local,B,H]` | `[S_local,B,N*H]` |
| Hash IDs before SP | `[B,S,(max_order-1)*K]` | same |
| Retrieved memory | `[S_local,B,D_mem]` | shared across branches |
| Gate | `[S_local,B,1]` | `[S_local,B,N,1]` |
| Engram output | `[S_local,B,H]` | `[S_local,B,N*H]` |

Each n-gram order has `K` prime-sized embedding tables. A head returns
`memory_dim / K` values; heads and orders are concatenated. One value projection and all sparse
tables are shared across mHC branches. Every branch has its own key projection, key RMSNorm,
query RMSNorm, scalar gate, convolution normalization, and convolution channels. The gate uses
the official signed-square-root transform before sigmoid.

The depthwise causal convolution has dilation equal to the maximum n-gram order. Its weight is
zero-initialized, so the convolution branch is initially zero. Engram output is added directly to
the residual tensor before attention. For native mHC this addition occurs before the layer's
`H_pre` mapping and preserves all real branches; Engram never creates temporary fake branches or
contracts them by averaging.

## EP ownership and communication

Every prime-sized table is independently divided into contiguous, possibly uneven row ranges.
For global row count `R`, EP rank `r` owns:

```text
base = R // EP
remainder = R % EP
start(r) = r * base + min(r, remainder)
rows(r) = base + (r < remainder)
```

No padding contributes to the logical shape. Each local weight is
`[rows(r), memory_dim / K]`; the full table is never allocated on an EP rank. Shard
initialization runs on a forked RNG stream seeded by the globally unique prime table size and
the EP rank: the uneven shard shapes therefore never desynchronize the default generator across
EP ranks (which would silently diverge later replicated parameters across data-parallel peers),
while EP shards stay decorrelated and TP/expert-DP replicas stay identical.

A module batches requests for all of its head tables into one routing exchange:

1. flatten `(table, row)` requests in token/head order;
2. calculate the owner and owner-local row, stable-sort by owner, exchange per-peer counts, and
   exchange integer requests with native `all_to_all_single`;
3. perform the appropriate local table lookups in received request order;
4. return embeddings with MCore's differentiable variable-split all-to-all;
5. invert the owner sort and restore `[B,S_local,num_tables,head_dim]`.

The protocol permits duplicate rows, unequal request counts, empty peer splits, uneven tables,
and EP=1. The distributed reference test runs at both EP2 and EP4 and checks outputs, sparse and
dense gradients, and one Adam step against complete-table references. Backward through the return
all-to-all sends gradients to the owning lookup operations,
where embedding accumulation combines duplicates. In deterministic mode, the owner sorts local row
IDs, sums each repeated-row gradient with a segmented reduction, and writes only unique rows. This
avoids CUDA embedding atomic-add ordering differences across checkpoint restarts without deduplicating
or changing the forward request protocol.

Sparse table weights have `allreduce=False`, so DDP synchronizes matching shards over expert-DP,
not across EP owners. The weights are replicated over TP in this milestone. With SP, dense Engram
parameter gradients are summed across TP by the existing sequence-parallel finalizer, and the much
larger sparse-table gradients are summed on a dedicated unflattened all-reduce so no full-table
flatten copy is materialized. All other Engram parameters use normal dense DP synchronization.

The opt-in training verifier logs per-rank ordered token checksums plus global sparse-table and
FP64 full-model checksums before and after every optimizer step, in addition to gradient, update,
and peak-memory evidence. Its collectives run symmetrically on every rank, including pipeline
stages that own no Engram layer, so verification can never desynchronize the job. Failure is
global — a non-finite gradient, a globally zero Engram gradient, or no table updating at all; a
single quiet row shard with zero hash hits in one step is reported but is not an error. The flag
is rejected together with the distributed optimizer, whose reduce-scattered gradient buffer does
not expose full per-parameter gradients. Normal training does not compute these diagnostics.

## TP, SP, and PP data flow

TP does not shard Engram tables or dense Engram projections. Engram requires
`expert_tensor_parallel_size == tensor_model_parallel_size` so that the EP table-sharding
dimension stays orthogonal to the dense TP replication dimension. SP hashes full `tokens[B,S]`,
then selects the same contiguous sequence interval used by MCore's sequence-parallel hidden
state. The causal short convolution takes its left context explicitly: zeros at the true
sequence start, and under SP a differentiable one-hop halo exchange that fetches the previous
TP rank's trailing `(kernel_size - 1) * dilation` normalized positions, so SP output is
bitwise-equivalent in structure to the full-sequence computation. Each SP slice must be at
least as long as that receptive field; shorter slices are rejected at runtime.

Before each training or evaluation pipeline schedule, the first PP stage's TP source prefetches
all raw microbatches, broadcasts token IDs across its TP group, and then each TP coordinate
broadcasts through the matching PP group. The scheduler consumes a short-lived replay iterator,
so the source sees the prefetched raw batches in their original order and rerun-state-machine
rewinds remain exact. A PP group fixes the data shard and TP/EP coordinates, so every stage
receives the same microbatch tokens without a model-forward collective or mutable module cache.
Layers use their
global 1-based `layer_number`; therefore selected layers on middle and last stages work without
stage-specific layer specs or saved per-forward state.

## Optimizer policy

Only sparse table weights carry `is_engram_embedding=True`. A `ParamKey` override selects those
parameters, sets both maximum and minimum LR schedule endpoints to the model endpoint multiplied by
the configured factor (default 5), and applies the configured fixed weight decay (default zero).
Value/key projections, gates, norms, and convolution remain in the model's ordinary
LR/weight-decay policy.

The override deliberately does not select an optimizer, so no parameter-group bucketing is added to
the shared optimizer factory and a run without Engram builds exactly the optimizers it did before.
The tables are flagged `is_embedding_or_output_parameter`, so an emerging base optimizer such as
Muon already routes them to Adam through its own registered `default_param_overrides`. The only
other change is that the decoupled-LR override skips Engram tables, since matching both overrides
would raise on a conflicting `max_lr`; with no Engram parameters present that predicate is exactly
the previous attribute match.

## Distributed checkpointing

Each table weight is represented by an irregular `ShardedTensor` whose global shape is exactly
`[global_prime_rows, head_dim]`, whose local shape is `[owned_rows, head_dim]`, and whose global
offset is `[owned_start, 0]`. `axis_fragmentations=None` records a nonuniform grid. TP and
expert-DP copies are replicas; EP ranks are distinct global row slices. There is no EP-dependent
padding in checkpoint metadata.

Torch distributed checkpoint save/load therefore transfers intersecting slices directly between
the saved and target layouts when EP changes. Optimizer state uses the model parameter's same
sharded metadata, enabling Adam moment and master-weight resharding without gathering a full table
onto one rank. Optimizer restore already keys parameter groups on every `ParamGroupOverride`
field, so the Engram group stays distinct from an otherwise identical ordinary expert group
through its recorded LR and weight-decay endpoints.

### Checkpoint compatibility

Enabling Engram switches the whole transformer block to layer-numbered (non-homogeneous)
checkpoint keys, because Engram exists only at selected global layers and cannot be represented
as a hole in a homogeneous stacked tensor. A checkpoint saved without Engram therefore cannot be
loaded directly into an Engram-enabled model or vice versa; converting an existing pretrained
checkpoint requires an offline key migration in addition to initializing the new Engram
parameters.

For the BF16 proxy, allocation reports count 14 bytes of scalable sparse payload per owned
parameter: 2 bytes for the model value plus 12 bytes for the FP32 master value and two Adam
moments. They report this payload for every global table and each Engram module at EP8 and EP4;
fixed-size Adam step scalars are excluded because they do not scale with a table's row shard.

## Profiling evidence

When `--nvtx-ranges` is enabled, Engram emits ranges for hashing, owner mapping, request-count
exchange, request all-to-all, owner-local embedding, differentiable return all-to-all in both
forward and backward, deterministic embedding-gradient accumulation, gate/value projection, and
short convolution. With `--profile`, these ranges follow its selected iteration window. Without
`--profile`, selected `--profile-ranks` emit the ranges for the complete training loop, allowing
one outer nsys session to trace every torchrun worker without coordinating CUDA-profiler API
start/stop calls across ranks.

`--record-memory-history` starts allocator-history capture before legacy or config-container model
construction. When `--profile-ranks` is nonempty, every selected rank writes a distinct
`*_rank-<global-rank>.pickle` snapshot instead of racing on one output path. Engram profiling runs
also write per-rank JSONL records after model/DDP construction, optimizer construction, the first
optimizer step, and the final steady-state step. The records contain exact BF16 table, FP32 master
parameter, and named Adam-state tensor bytes plus CUDA allocated/reserved/peak counters and the
TP/PP/EP/dense-DP/expert-DP group coordinates. This separates EP row ownership from any further
distributed-optimizer sharding over expert-DP.

## Supported and deferred combinations

The current milestone supports GPT training with BF16 parameters (FP8/MXFP8 main-model recipes
are permitted; Engram modules are plain torch modules outside the Transformer Engine autocast
regions and stay in BF16), standard residuals or native mHC, EP, TP, PP, SP, MoE coexistence,
multi-token prediction (MTP layers never build Engram), native all-to-all, torch distributed
checkpoints, and — for the qwen variant — packed THD rows with EOS document boundaries.
CP greater than one, VPP, activation recomputation, CUDA graphs, FSDP, inference serving,
offload, DeepEP, communication overlap, request deduplication, FP8 table storage, and fused
Engram kernels are intentionally deferred and rejected during startup.
