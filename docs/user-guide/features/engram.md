<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
-->

# Distributed Engram

Engram adds conditional memory before selected Hybrid attention layers. It compresses
raw token IDs, forms suffix n-grams, hashes them into embedding tables, and combines
retrieved values with the residual stream through a query/key gate and a causal
short convolution. GPTModel is unchanged.

## Layer ownership and explicit inputs

| Component | Responsibility |
| --- | --- |
| `engram_layer_ids` | Stable nonnegative memory identities; a nonempty list enables Engram |
| `engram_target_layer_indices` | Corresponding zero-based positions in the main Hybrid pattern |
| `EngramHybridProvider` | Selects local layer specifications and prepares raw token context |
| `EngramAttentionLayer.engram` | Registers its own addressing, table and fusion modules |
| HybridModel / HybridStack | Forward optional token context through generic layer orchestration |
| `row_a2a` backend | Shards logical table rows over TP × DP × CP and exchanges requested rows |
| `local` backend | Shards table width over TP, with ordinary dense table gradients |
| `RowSparseAdam` | Maintains moments and FP32 master values only for touched rows |
| DDP | Excludes tagged sparse parameters from dense parameter/gradient buffers |

Each consumer owns its parameters. The provider retains construction inputs and no
mutable current-batch cache. Normal forward and activation recomputation receive the
same explicit microbatch token tensor. Middle pipeline stages and virtual chunks
request token data only when they contain a consumer; their native data iterators
preserve the same sample sequence as the corresponding input stage. No additional
pipeline collective is inserted into the forward schedule.

Hash capacities are resolved against the complete ordered memory-ID configuration
before selecting locally owned parameters. Pipeline segmentation therefore does not
change hash moduli. The provider converts token context to the attention layout;
Engram restores sequence/context partitions for n-gram and convolution history, then
returns hidden states in the attention layout. This follows the upstream constraints
on supported context-parallel attention layouts.

## Placement

For eight logical blocks with one dense MLP and seven MoE MLPs, use `*-` followed
by seven `*E` pairs. A single attention/MoE MTP module adds the suffix `/*E`.
To place memory ID 1 immediately before the second attention:

```text
--engram-layer-ids 1
--engram-target-layer-indices 2
```

Both lists must have the same length and unique nonnegative entries. Every target
must select a standard attention (`*`) in the main stack. Without explicit positions,
memory ID `i` selects the `i`-th standard attention occurrence. Pipeline separators,
MTP suffixes, and intervening non-attention layers do not count as occurrences.
Other Hybrid layer families may coexist; memory is not automatically attached to
those layers or to MTP.

## Optimization and native checkpoints

Row-sharded tables use touched-row Adam, with an independent learning-rate multiplier
of 5 and no weight decay. Other parameters use the selected Megatron optimizer.
Sparse gradients participate in the same global clipping coefficient as dense and
expert gradients. The sparse communication path applies TP replication correction
and the same DP/CP loss normalization used by the dense training path.

Native distributed checkpoints store table tensors by logical table coordinates and
sparse moments by stable memory ID, table ID and row. Optimizer hyperparameters are
also saved per memory ID so relocating a layer between pipeline stages does not
change the required checkpoint metadata. Checkpoint resharding and continuation are
validated separately: changing topology can preserve state without guaranteeing
bitwise-identical subsequent floating-point reductions. Historical GPT Engram
checkpoint conversion is outside this feature.

## Training scope

The integration targets FP32/BF16 training with TP, PP/VPP, CP, EP, ordinary/full
recomputation and MTP1. Existing Hybrid constraints, such as valid head/parallel
sizes and pipeline patterns, still apply. The `row_a2a` backend requires unity loss
scaling and `torch_dist` checkpoints. Each supported combination must have explicit
unit or functional validation; implementation support alone is not a convergence
result.

Engram rejects inference decoding, packed sequences, mHC, FP8/FP4, FSDP, ModelOpt,
CUDA graphs, CPU activation/optimizer offloading, and fine-grained expert-overlap schedules.
Restrictions apply only when Engram is enabled. The local backend requires a hash
head width divisible by TP.

## Validation and numerical limits

The unit and distributed suites exercise separate parts of the supported training
contract. A passing topology does not establish every combination of TP, PP/VPP,
CP, EP, recomputation and optional Hybrid backends.

| Area | Regression coverage |
| --- | --- |
| Addressing and fusion | Independent hash/prime layout, lookup, convolution and fusion references in `test_addressing.py` and `test_memory_fusion.py` |
| Row-sharded tables | Empty requests, repeated rows, sparse updates, recomputation and distributed checkpoints in `test_row_a2a.py` |
| Token context | Resumed sample identity, pipeline microbatches and feature-off routing in `test_batch_routing.py` |
| Pipeline execution | PP/VPP, selected table backends, MTP and checkpoint round trips in `test_pipeline.py` |
| Context parallelism | Complete first-attention models and fixed-input later-attention comparisons in `test_context_parallel_model.py` |
| Heterogeneous stacks | Optional GDN/Mamba backends, standard attention, dense/MoE layers and MTP in `test_heterogeneous_model.py` |
| Optimizer and clipping | Sparse topology and independent global-clipping checks in `test_sparse_topology.py` and the optimizer suites |

The files above are under `tests/unit_tests/models/engram/`. Dependency skips are
not passing support evidence. Optional Hybrid backends require their own working,
compatible dependencies; Engram does not disable backend correctness guards.
MLA and experimental attention variants are outside the current Engram scope.

Full-model CP1/CP2 execution need not be bitwise identical. A preceding BF16
attention kernel can perturb the hidden state entering a later Engram gate, whose
square-root transform can amplify rounding near zero. Fixed-input comparisons
isolate layout and reduction correctness; they do not prove arbitrary-placement
full-model gradient equivalence. The first-attention model regression checks
main CE within 2e-4 and per-parameter gradients within 2% relative L2 error.

Checkpoint checks distinguish exact state serialization from subsequent numerical
execution. Same-topology state, including RNG and sample progress, is compared
before another update. Resharding tests preserve logical table coordinates and
optimizer state but do not promise bitwise-identical floating-point trajectories.
TP/PP resharding requires a fixed padded vocabulary and compatible expert tensor
parallelism; arbitrary optimizer-group classification changes are not supported.
Empty sparse stages and PP-to-VPP transitions have dedicated regression coverage.

## Offline experiment

The [Engram examples](../../../examples/engram/README.md) use a local GPT-2 tokenizer
and pre-tokenized FineWeb IndexedDataset files. They provide one shared data/token
budget for a baseline and a memory-augmented model, while keeping experiment-specific
learning-rate/MTP schedules outside the core layers. Parameter counts separately
identify embeddings, the LM head and MTP. TensorBoard and W&B record the same progress
and metric definitions; offline W&B records retain their synchronization status.

Complete validation/test CE uses `final/valid_ce` and `final/test_ce`, separate from
periodic `lm loss validation`. TensorBoard uses the evaluated checkpoint step. W&B
uses `final/checkpoint_step` as the custom horizontal axis for `final/*`, allowing
its internal history counter to advance without resubmitting an older step. Local
journal validation and online synchronization are recorded separately.

Report convergence only from completed comparable runs. A short smoke test checks
correctness, logging, memory use and throughput; it cannot establish a loss benefit.
