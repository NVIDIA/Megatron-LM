<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Engram n-gram memory

```{note}
Engram is experimental. Its flags, checkpoint layout and defaults may change.
```

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
always sharded by rows, and you choose the layout:

- **EP layout** (the default). Rows are sharded over the expert-parallel (EP) group and
  replicated over expert-data-parallel. Lookups use NCCL all-to-alls.
- **DP x CP layout** (`--engram-parallel-size P`). Rows are sharded over `P` consecutive ranks
  of each pipeline stage's data-parallel times context-parallel group, which contains the EP
  group. Each shard is replicated over the remaining ranks. Lookups use a symmetric device
  transport: direct NVLink loads inside an NVLink domain, and GPU-initiated networking (GIN)
  across domains. NCCL all-to-alls remain available as a reference backend.

## Quick start

EP layout (a MoE model with EP 8; Engram on layers 2 and 10):

```bash
python pretrain_hybrid.py ... \
    --expert-model-parallel-size 8 \
    --engram-vocab-sizes 129280 129280 --engram-layer-ids 2 10 \
    --engram-variant qwen --engram-eos-token-id 151643 --engram-unigram-vocab-size 151936
```

DP x CP layout with every row owned by one rank of a 128-rank data-parallel group:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python pretrain_hybrid.py ... \
    --use-distributed-optimizer \
    --engram-vocab-sizes 129280 129280 --engram-layer-ids 2 10 \
    --engram-variant qwen --engram-eos-token-id 151643 --engram-unigram-vocab-size 151936 \
    --engram-parallel-size 128
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
| `--engram-parallel-size` | – | Use the DP x CP layout with row groups of this many ranks |
| `--engram-communication-backend` | see below | `symmetric` or `nccl` |
| `--engram-symmetric-workspace-mb` | 64 | Workspace of the symmetric transport, per rank |

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

### DP x CP layout

`--engram-parallel-size P` reads each pipeline stage's data-parallel times context-parallel
group (dp_cp) in rank order as a `[replica, shard]` mesh. Rows are sharded over `P` consecutive
ranks and replicated over the ranks at the same position in the other shards. `P` must divide
the size of dp_cp. With tensor and expert-tensor parallel sizes of 1, dp_cp contains every EP
group as a contiguous block, so:

- `P = size(dp_cp)` gives every row a single owner, which minimizes table memory;
- `P = size(EP)` reproduces the EP layout's row ranges, initialization and checkpoint shards.

The tables form their own parameter domain (`engram`), with their own DDP buffers and their own
distributed optimizer over the replica group. When the replica group has one rank, it is never
reduced, and no communicator is created for it.

Requirements:

- `--use-distributed-optimizer` (also with Muon, in its default layer-wise parameter layout, not
  `--no-use-layer-wise-param-layout`) and `--num-distributed-optimizer-instances 1`;
- tensor-parallel and expert-tensor-parallel sizes of 1, and no GTP;
- a distributed checkpoint format;
- no Megatron-FSDP or FSDP2, no `--overlap-param-gather-with-optimizer-step`, and no
  `--distrib-optim-fully-reshardable-mem-efficient` together with
  `--dist-ckpt-optim-fully-reshardable`.

## Lookup backends

`--engram-communication-backend` defaults to `symmetric` when `--engram-parallel-size` is above
1, and to `nccl` otherwise. The EP layout supports `nccl` only.

### `nccl`

The owner of each row is computed in closed form. Requests are exchanged with
`all_to_all_single`, the owners gather their rows, and the rows return through a differentiable
all-to-all. Backward sends gradients back to the owners, which accumulate them. In
deterministic mode, the owner sums repeated rows with a segmented reduction instead of atomics.

### `symmetric`

The symmetric transport is a CuTe DSL and NCCL4Py device transport from
[NVIDIA/Megatron-LM#7224](https://github.com/NVIDIA/Megatron-LM/pull/7224). Each row group gets
one NCCL4Py communicator, which serves only the device API. It is created with one CTA and
NVLS disabled, as no NCCL kernel ever runs on it. One lookup:

1. finds the unique rows and their owners;
2. exchanges one row of counts and registration flags per peer;
3. reads the rows: owners in the same NVLink (LSA) team are read directly from their registered
   table windows, and owners elsewhere receive the requests and send the rows back through GIN.

Backward sums the gradients of duplicate rows in FP32, sends them to their owners in one
exchange, and adds them row-sparse into DDP's `main_grad`. No dense table gradient is built.

Requirements:

- CUDA, Triton, `nvidia-cutlass-dsl >= 4.8.1`, and `nccl4py >= 0.6.0` with the same NCCL version
  as the runtime library and the device bitcode (`libnccl_device.bc`) linked into the kernels;
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, so the tables can be registered as NCCL
  windows. With Transformer Engine CUDA graphs, Megatron then also requires
  `NCCL_GRAPH_REGISTER=0`;
- multi-node NVLink for LSA teams that span nodes (`NCCL_MNNVL_ENABLE=1`, IMEX channels), and a
  GIN-capable network plugin for row groups that span NVLink domains;
- no `--fake-process-group`, `--moe-use-upcycling` or `--inprocess-restart`.

The transport's barrier and exchange kernels spin-wait on their peers and cannot time out.
Every rank of a row group issues the same calls in the same order, including ranks with no
rows. A rank that fails aborts its transports and exits, and the launcher (torchrun, or
`srun --kill-on-bad-exit`) then stops the peers. A silent peer death needs a hang detector, such
as the NVIDIA Resiliency Extension's, or the job's time limit.

Use `--engram-communication-backend nccl` with the DP x CP layout for a numerical reference, or
on systems without the symmetric dependencies.

## Memory

The table, gradient and optimizer bytes scale with the rows each rank owns. In the DP x CP
layout with `P = size(dp_cp)`, the parameter and gradient buffers are half of what EP 64 x
expert-DP 2 holds. The optimizer state is the same in both layouts. With 300B table parameters
on 128 GPUs, `P = 128` takes 13.1 GiB less per GPU: 4.4 GiB of bf16 parameters and 8.7 GiB of
FP32 gradients.

The lookup backends differ in the device memory that NCCL holds outside PyTorch's allocator,
per GPU, measured on GB300 after two training steps:

| Row group | `symmetric` | `nccl` |
|---|---|---|
| `P = 4`, one node | About 125 MiB | About 1.1 GiB |
| `P = 8`, two NVLink domains | About 205 MiB | About 650 MiB |
| `P = 128`, two NVLink domains | About 560 MiB | About 2.1 GiB |

The `symmetric` cost is all allocated at start-up: the 64 MiB workspace
(`--engram-symmetric-workspace-mb`), a communicator of 16 to 20 MiB, and the device-API
resources, including the GIN contexts across NVLink domains. `NCCL_MAX_CTAS` in the environment
overrides the communicator's one-CTA configuration and brings back about 6 MiB per extra
channel. The `nccl` cost is mostly peer-to-peer buffers that the first all-to-alls set up. It
depends on the number of NVLink peers and on the number of channels NCCL gives each peer, which
is smaller in larger groups. Peers in another NVLink domain use shared network pools instead.

## Known issue: cross-node all-to-all with the SPCX network plugin

The `nccl` backend sends all-to-alls with uneven splits across nodes. The SPCX network plugin
(`NCCL_NET_PLUGIN=spcx`) can drop part of such grouped sends. The receiver then reads stale data
or hangs. This is reported for SPCX 1.4, as shipped in NGC PyTorch 26.07-era images. No fix has
been confirmed for later versions. See
[NVIDIA-NeMo/Megatron-Bridge#5462](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/5462)
and the Megatron Core 0.19 known issues. Engram is affected when the EP layout or the `nccl`
backend spans NVLink domains. NCCL logs `Using network SPCX` when the plugin is active.

Each of these workarounds avoids it:

- `NCCL_NET_SHARED_COMMS=0`;
- `NCCL_PXN_DISABLE=1`;
- `NCCL_NET_PLUGIN=none NCCL_NET=IB`.

Under Slurm with Pyxis, variables that the image or enroot already sets keep their image values
unless they are passed with `srun --container-env=NAME,...`. Check the NCCL INFO lines
(`... set by environment to ...`) to confirm that a workaround applied. The `symmetric` backend
issues no all-to-all.

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
therefore load under a different EP size or `--engram-parallel-size`, and across the EP and
DP x CP layouts.

The tables' optimizer state follows the checkpoint format:

- In the default format, it loads at another data- or context-parallel size with the same layout
  and the same EP size or `--engram-parallel-size`.
- To change the layout, the EP size or `--engram-parallel-size`, save it fully reshardable
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
  `--pad-packed-seq-alignment max` and `--max-seqlen-per-dp-cp-rank` equal to `--seq-length`.
  `--dataloader-inter-document-masking` with context parallelism requires `--sft`.
- Activation recomputation replays the memory from the token IDs, on every rank of a row group
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
[NVIDIA/Megatron-LM#7553](https://github.com/NVIDIA/Megatron-LM/pull/7553) by Li Tao. The
parameter replica domains and the symmetric transport come from
[NVIDIA/Megatron-LM#7224](https://github.com/NVIDIA/Megatron-LM/pull/7224) by Hongxiao Bai.
The fusion of the `deepseek_v41` variant follows the DeepSeek-V4.1 Engram module of the same
pull request, by Deyu Fu and Hongxiao Bai.
