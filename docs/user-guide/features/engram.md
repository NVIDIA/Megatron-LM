{/*
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
*/}

# Engram n-gram memory

<Note>
Engram is experimental. Its flags, checkpoint layout and defaults may change.
</Note>

Engram adds a trainable, deterministic n-gram memory to selected layers of a hybrid model
(`pretrain_hybrid.py`, `HybridModel`). At each selected layer, the token IDs of the preceding
n-grams are hashed into large embedding tables. The rows that are looked up are gated against
the hidden state and added to the residual stream before the layer runs. The numerics follow
the official DeepSeek Engram implementation (commit `fb7f84a21f91223715394a33a1dc24bbfb7f788e`)
and the Engram paper v2 (arXiv 2601.07372). The `qwen` variant follows the Qwen per-layer
embedding (PLE) layout. The `deepseek_v41` variant follows the Engram module of DeepSeek-V4.1
as [NVIDIA/Megatron-LM#7224](https://github.com/NVIDIA/Megatron-LM/pull/7224) implements it,
which has no short convolution.

The tables dominate the memory: one table set can hold hundreds of millions of rows. They are
always sharded by rows, in the EP layout: rows are sharded over the expert-parallel (EP) group
and replicated over expert-data-parallel. Lookups use NCCL all-to-alls.

## Quick start

EP layout (a MoE model with EP 8; Engram on layers 2 and 10):

```bash
python pretrain_hybrid.py ... \
    --expert-model-parallel-size 8 \
    --engram-vocab-sizes 129280 129280 --engram-layer-ids 2 10 \
    --engram-variant qwen --engram-eos-token-id 151643 --engram-unigram-vocab-size 151936
```

## Variants

`--engram-variant` selects the hashing conventions and how the memory enters the hidden states.
All variants share the tables and every distributed mechanism.

| | `deepseek` (default) | `deepseek_v41` | `qwen` |
|---|---|---|---|
| Token preprocessing | Offline tokenizer compression artifact (`--engram-tokenizer-map`) | As `deepseek` | Raw token IDs, no artifact |
| Hash multipliers | PCG64, stored in the artifact | As `deepseek` | splitmix64, computed in process |
| Table sizes | Next unused prime above each order's budget | As `deepseek` | Per-head n-th prime after the shared budget |
| n-gram boundary | Left padding with the compressed pad token (`--engram-pad-token-id`) | As `deepseek` | Windows reset at every EOS (`--engram-eos-token-id`) |
| Projections | Biased key and value projections | One bias-free projection for the keys and the value | Bias-free key and value projections |
| Gate | Group RMSNorm of key and hidden state | Per-channel gate weights, FP32 arithmetic | As `deepseek`, with zero-centered gamma |
| Short convolution | Yes | No | Yes |
| Positions of a packed row that hold no token | Gated like every position | Gate closed | Gated like every position |

The `deepseek_v41` variant computes the forward of the DeepSeek-V4.1 module and uses its
parameter names (`wkv`, `q_weight`, `k_weight`). It ignores `--engram-kernel-size`. With BF16
parameters it adds the memory to the hidden states in FP32 and rounds once. Its tables remain one
per hash head, whereas the DeepSeek-V4.1 module stores one fused table per layer.

The positions of a packed row that hold no token are those after the last document boundary
and, when the row says how many tokens each document holds, the padding of every document. Only
a context-parallel row in the contiguous layout says so: its `cu_seqlens_q` counts the tokens
and its `cu_seqlens_q_padded` holds the positions. The `--sft` path counts the padding at the
end of a row as part of the last document, so its rows have no such position and their padding
is gated like every position.

The `qwen` variant also needs `--engram-unigram-vocab-size`, the Hugging Face `vocab_size` that
bounds the multipliers. The padded vocabulary must not exceed it, and every
`--engram-vocab-sizes` entry must be equal.

`tools/engram/generate_tokenizer_map.py` builds the artifact of the `deepseek` and
`deepseek_v41` variants. It is the only
component that loads a Hugging Face tokenizer. `tools/engram/convert_qwen_ple.py` converts
Hugging Face PLE weights into per-head prime tables and loads them into Engram modules.

## Configuration

Engram is enabled by `--engram-vocab-sizes`.

| Flag | Default | Meaning |
|---|---|---|
| `--engram-vocab-sizes` | – | Table budget for each n-gram order, from 2 through the maximum order |
| `--engram-layer-ids` | – | 1-based global layer IDs that get a memory |
| `--engram-max-ngram-order` | 3 | Largest n-gram order hashed |
| `--engram-num-hash-heads` | 8 | Hash heads (tables) per order |
| `--engram-memory-dim` | 512 | Memory width per order, split evenly over its heads |
| `--engram-kernel-size` | 4 | Kernel size of the causal short convolution, dilated by the maximum order; unused by `deepseek_v41` |
| `--engram-hash-seed` | 0 | Seed of the hash multipliers |
| `--engram-embedding-lr-multiplier` | 5.0 | Multiplier on the tables' maximum and minimum learning rates |
| `--engram-embedding-weight-decay` | 0.0 | Weight decay of the tables |
| `--engram-freeze-embedding` | off | Keep the tables fixed; the dense part still trains |

Layer IDs follow `--hybrid-layer-pattern` with the pipeline separators (`|`) removed, so the
same IDs work for any pipeline split. A selected layer must be a transformer layer (attention,
MLA, MLP, MoE and similar); a Mamba layer cannot carry the memory. MTP layers never get one.
The memory is added to the residual stream before the layer runs. With the hybrid
hyper-connection wrapper (`--enable-mhc-connections`), it is added to every residual stream
before the wrapper's read gate.

Configuration errors are raised at startup, before any table is allocated.

## Table layouts

Each table is split into contiguous, possibly uneven row ranges. With `R` rows over `N` owners,
owner `r` holds `R // N` rows, plus one for `r < R % N`. The full table is never allocated, and
no padding enters the logical shape.

### EP layout

The tables are expert parameters (`allreduce=False`). DDP reduces their gradients over
expert-data-parallel, and the distributed optimizer shards their state with the experts'. EP
works without MoE layers when Engram is enabled. When MoE and Engram coexist, they share the
EP group. Engram issues its own all-to-alls over the EP group, separately from the MoE token
dispatcher, so it works with any dispatcher. `--moe-expert-rank-capacity-factor` is rejected
(see the limitations).

## Lookup

The owner of each row is computed in closed form. Requests are exchanged with
`all_to_all_single`, the owners gather their rows, and the rows return through a differentiable
all-to-all. Backward sends gradients back to the owners, which accumulate them. In
deterministic mode, the owner sums repeated rows with a segmented reduction instead of atomics.

## Known issue: cross-node all-to-all with the SPCX network plugin

The lookup sends all-to-alls with uneven splits across nodes. The SPCX network plugin
(`NCCL_NET_PLUGIN=spcx`) can drop part of such grouped sends. The receiver then reads stale data
or hangs. This is reported for SPCX 1.4, as shipped in NGC PyTorch 26.07-era images. No fix has
been confirmed for later versions. See
[NVIDIA-NeMo/Megatron-Bridge#5462](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/5462)
and the Megatron Core 0.19 known issues. Engram is affected when the ranks that shard a
table span NVLink domains. NCCL logs `Using network SPCX` when the plugin is active.

Each of these workarounds avoids it:

- `NCCL_NET_SHARED_COMMS=0`;
- `NCCL_PXN_DISABLE=1`;
- `NCCL_NET_PLUGIN=none NCCL_NET=IB`.

Under Slurm with Pyxis, variables that the image or enroot already sets keep their image values
unless they are passed with `srun --container-env=NAME,...`. Check the NCCL INFO lines
(`... set by environment to ...`) to confirm that a workaround applied.

## Optimizer

Only the table weights carry `is_engram_embedding=True`. A parameter override multiplies their
maximum and minimum learning rate by `--engram-embedding-lr-multiplier` and sets their weight
decay. The projections, gates, norms and convolution follow the model's ordinary settings. With
Muon, the tables are embedding parameters and go to Adam.

`--engram-freeze-embedding` keeps the tables fixed. They still serve lookups but take no
gradient, so they get no gradient buffer and no optimizer state.

## Checkpointing

Each table is saved as an irregular `ShardedTensor` whose global shape is the full table and
whose local shard is the owned row range. Replicas are marked as such. The model weights
therefore load under a different EP size.

The tables' optimizer state follows the checkpoint format:

- In the default format, it loads at another data- or context-parallel size with the same EP size.
- To change the EP size, save it fully reshardable
  (`--dist-ckpt-optim-fully-reshardable`). It then loads when the table optimizer sits at the same
  position in the chained optimizer of both runs. Otherwise, resume with `--no-load-optim`.

## Pipelines, recomputation and CUDA graphs

- Pipeline and virtual pipeline parallelism: before each schedule, the first stage prefetches
  the microbatch token IDs and broadcasts them to every stage. Every model chunk then reads its
  memory's tokens without a forward-time collective.
- Sequence parallelism: hashing runs on the full sequence before the sequence-parallel split,
  and the short convolution, in the variants that have it, fetches its left context from the
  previous tensor-parallel rank.
- Context parallelism: the memory follows the layout its layer runs in, zigzag or contiguous
  (`--attention-cp-layout`, `--linear-cp-layout`). It restores the token
  row from the shards of the context-parallel ranks, hashes it, and keeps the rows of its own
  positions. The short convolution takes its left context from the neighboring ranks, so the
  rows of a memory layer must be long enough to hold that context: in one zigzag chunk of an
  unpacked row for a zigzag layer, and in one contiguous slice for a contiguous layer.
- Packed sequences (THD) are supported through the `--sft` path. Under context parallelism a
  packed row is split per document in the zigzag layout and like an unpacked row in the
  contiguous one, where documents straddle ranks. With `--linear-cp-layout` and
  `--attention-cp-layout` set to different layouts the memory must then sit on contiguous
  layers: a zigzag layer of such a stack gets a packed row in a padded per-sequence zigzag,
  which the memory does not follow. With pipeline parallelism packed rows require
  `--pad-packed-seq-alignment max` and `--max-seqlen-per-dp-cp-rank` equal to the packed row
  width (`--max-seq-length-per-sample` when it is set, `--seq-length` otherwise).
  `--dataloader-inter-document-masking` with context parallelism requires `--sft`.
- Activation recomputation replays the memory from the token IDs, on every rank of the EP group
  together. A model that calls the module itself can replay the fusion alone, see
  [Lookup ahead of the fusion](#lookup-ahead-of-the-fusion).
- CUDA graphs: only Transformer Engine per-layer capture restricted to the MoE scopes
  (`--cuda-graph-modules moe`, `moe_router` or `moe_preprocess`) is supported, and the memory
  must then sit on a non-MoE layer. The lookup has data-dependent shapes and always runs eagerly.

## Lookup ahead of the fusion

The layers call the module once:
`engram(hidden_states, input_ids, packed_seq_params, add_to_hidden_states=True)` hashes the
token IDs, looks the rows up and fuses them into the hidden states. A model that drives the
module itself can run the first two ahead of the third:

```python
result = engram.lookup(input_ids, packed_seq_params)
...
hidden_states = engram(hidden_states, add_to_hidden_states=True, lookup=result)
```

`lookup` returns an `EngramLookupResult`, which holds all the fusion needs from the row: the
rows of this rank's positions (`memory`, `[S, B, memory width]`, which carries the gradient edge
to the tables), the positions that hold a token (`valid`, see [Variants](#variants)) and the
positions of the documents of a packed row (`cu_seqlens`). The two calls compute what the
single call computes, bit for bit. `input_ids` and `packed_seq_params` are those of the layer,
in its context-parallel layout.

- The lookup is a collective over the ranks that share the tables, and so is the return of its
  gradient. Make every result with gradients enabled when the tables train, and consume it in
  exactly one call, the same one on every rank, with the calls in the same order on every rank.
- To recompute a region that holds the fusion, pass `result.memory` to the region as one of its
  tensor arguments and rebuild the result inside it, as
  `EngramLookupResult(memory, result.valid, result.cu_seqlens)`. The region then replays the
  fusion only, and returns the gradient of the rows to the one lookup made outside it.
- The lookup has data-dependent shapes and must not run while a CUDA graph is captured.

## Limitations

Rejected at startup, unless noted:

- GPT models (Engram is integrated into hybrid models only; `GPTModel` rejects it) and
  `hybrid_layer_config_list`;
- inference and generation (`--transformer-impl inference_optimized` is rejected at startup;
  any other inference or generation call raises in the model's forward);
- FP16 parameters and FP4 training (BF16 and FP32 parameters work, and FP8 or MXFP8 recipes run
  the rest of the model);
- Megatron-FSDP and FSDP2;
- ModelOpt model building, including ModelOpt distillation (rejected when the hybrid model is
  built). Offline logits distillation (`--logits-save-dir`, `--logits-load-dir`) is not
  rejected and has not been validated with Engram;
- whole-layer and full-iteration CUDA graphs;
- sequence-packing schedulers (including the one `--use-varlen-dataset` selects) and dynamic
  context parallelism;
- packed sequences under context parallelism with the memory on a zigzag layer of a stack whose
  `--linear-cp-layout` and `--attention-cp-layout` differ;
- `--overlap-moe-expert-parallel-comm`, `--moe-shortcut-connection`,
  `--moe-expert-rank-capacity-factor`, wide residuals and muP;
- hash-routed MoE layers together with sequence parallelism.

Engram tables are plain PyTorch parameters in the model's parameter dtype. Request
deduplication across microbatches, prefetching the rows of a later microbatch and FP8 table
storage are not supported.

## Credits

The package comes from the Engram support in
[NVIDIA/Megatron-LM#7231](https://github.com/NVIDIA/Megatron-LM/pull/7231) and
[NVIDIA/Megatron-LM#7553](https://github.com/NVIDIA/Megatron-LM/pull/7553) by Li Tao.
The fusion of the `deepseek_v41` variant follows the DeepSeek-V4.1 Engram module by Deyu Fu and
Hongxiao Bai in [NVIDIA/Megatron-LM#7224](https://github.com/NVIDIA/Megatron-LM/pull/7224).
