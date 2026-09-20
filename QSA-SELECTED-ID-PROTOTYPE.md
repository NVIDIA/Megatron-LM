# QSA selected-ID sparse kernel: isolated prototype

This branch contains a Triton forward/backward kernel for QSA core attention.
It consumes compact document-relative selected complete-block IDs and includes
the last incomplete block directly. It is now connected to `qsa.py` through
the **default-off** `id_sparse` backend. A synthetic single-QSA-layer 256K
forward/backward has run; Relax training and full-model 256K have not.
Keep this branch isolated until real-model training and long-CP behavior are
reviewed.

## Current integration and scale gates (2026-09-20)

The `QSAIndexer` emits IDs directly when `id_sparse` is selected, without
first materializing a bitset. The default Flex and explicit dense-masked
paths still emit bits. Packed THD uses document-relative positions; CP2
reconstructs full hidden, selected IDs and Q/K/V in causal order, performs a
differentiable K/V gather, and splits output back to local zigzag order.

On two H800 ranks, BF16 `Hq=24, Hkv=2, D=256, K=512, R=4, S=2064`, with
packed documents of 2056 and 8 tokens and per-token synthetic mRoPE64,
both dense-masked and ID paths completed forward/backward. The first document
has 514 complete blocks, so selection is genuinely sparse. With identical
weights, inputs and a rank-0-only loss, CP2 reconstructed outputs were
bit-equal to CP1 for both backends. CP2 reconstructed hidden gradients differed
from CP1 by at most 0.001953 (dense) and 0.0078125 (ID); the sum of the two
CP2 QKV-weight gradients differed from CP1 by relative L2 0.001917 (dense)
and 0.001916 (ID), with no element outside `0.08 + 0.08*|reference|`.
Rank 1 had zero upstream gradient, zero Q/gate weight gradients and nonzero
K/V gradients, confirming the remote K/V path. Recomputing the whole QSA
attention call with `torch.utils.checkpoint(..., use_reentrant=False)` also
passed on both ranks. This is a single-layer model-path gate, not a trained VL
model or a TP4/EP8 gate.

For the same synthetic packed single-document layer on one H800, measured
one-shot times and PyTorch CUDA peak memory were:

| Tokens | Route seconds | Forward seconds | Backward seconds | Peak allocated GiB | Peak reserved GiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16,384 | 0.499 | 2.183 | 1.561 | 2.318 | 3.193 |
| 65,536 | 0.662 | 3.167 | 5.887 | 8.782 | 12.107 |
| 262,144 | 2.236 | 8.829 | 27.131 | 34.648 | 48.027 |

All three runs had finite outputs and hidden gradients. The route tensor at
256K is `[262144,512]` int32, exactly 512 MiB. These times are single runs
with an uncontrolled Triton compilation/cache state, not throughput or
production training measurements. The script allocates the model and inputs
before resetting peak statistics; `initial_allocated_gib` in each log records
that baseline. Run it with:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 QSA_BENCH_SEQ_LEN=262144 \
  python tests/unit_tests/transformer/experimental_attention_variant/bench_qsa_id_layer.py
```

The 256K run repeated with this script: route 2.227 s, forward 7.015 s,
backward 26.280 s, peak allocated 34.648 GiB and reserved 48.027 GiB.
All four parameter-gradient tensors present were finite and nonzero. The
three indexer parameters had no gradient, consistent with the current
no-grad discrete selection and frozen-indexer RL recipe; this is a
training-objective gap, not evidence that the indexer trains.

The CP1/CP2 comparison is reproducible through `QSA_CP_SAVE_DIR`: run the
`cp1_real` test on one GPU, `cp2_real` under two-rank torchrun, then run
`saved_real_geometry_parity` with the same directory. The latter checks the
QKV weight SHA, both complete outputs and hidden gradients, and the summed
CP2 weight gradient. Each saved run uses fixed model/data seeds.

```bash
QSA_CP_SAVE_DIR=/tmp/qsa-cp-parity CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  python -m pytest -q tests/unit_tests/transformer/experimental_attention_variant/test_qsa_id_cp.py \
  -k cp1_real --override-ini addopts=''
QSA_CP_SAVE_DIR=/tmp/qsa-cp-parity CUDA_VISIBLE_DEVICES=1,0 PYTHONPATH=. \
  python -m torch.distributed.run --standalone --nproc-per-node=2 --module pytest -q \
  tests/unit_tests/transformer/experimental_attention_variant/test_qsa_id_cp.py \
  -k cp2_real --override-ini addopts=''
QSA_CP_SAVE_DIR=/tmp/qsa-cp-parity PYTHONPATH=. python -m pytest -q \
  tests/unit_tests/transformer/experimental_attention_variant/test_qsa_id_cp.py \
  -k saved_real_geometry_parity --override-ini addopts=''
```

The 256K measurement is limited to **one QSA layer, CP1, one document**, with
synthetic mRoPE. The current CP path still reconstructs full S activations on
each rank, and `_pool_keys` allocates `[n_docs,max_doc_blocks,D]`, which can
approach quadratic memory for mixed long/short packed documents. Indexer
routes are discrete/no-grad: the RL recipe still freezes the indexer; training
it needs a separately reviewed sparse-KL auxiliary objective. Full 8L VL
training, long CP, optimizer/checkpoint and Relax RL gates remain open.

## Kernel contract

`qsa_sparse_attention_id(Q, K, V, block_ids, positions, ratio=4)` expects:

- Contiguous CUDA `Q[B,Hq,S,D]`, `K/V[B,Hkv,S,D]`, with `Hq % Hkv == 0`,
  power-of-two `D <= 256`;
- `block_ids[B,S,topk]` int32, containing exactly
  `min((positions+1)//ratio, topk)` distinct document-relative complete-block
  IDs per query and `-1` in all other slots;
- `positions[B,S]` int32, restarting at zero for each packed document;
- `ratio` dividing 32. The current Qwen setting is 4.

Every selected block expands to its `ratio` original KV tokens. For each
query, the kernel also includes positions from
`((q_position+1)//ratio)*ratio` through the query, which is the still-open
tail. Each query/head program loops over 32 token slots at a time and uses
online softmax. Backward recomputes scores from saved log-sum-exp, computes
dQ locally, and uses FP32 atomic additions for dK/dV. Atomics are
race-safe but their floating-point accumulation order is not deterministic.

`validate=True` checks the exact ID count, range and uniqueness, including
the otherwise-empty row when a query closes a block but receives no selected
ID. Validation sorts and synchronizes CUDA, so it is **debug-only**; the
default hot path relies on the producer contract. The kernel does not
differentiate through selected IDs. It provides gradients only to Q/K/V.

## Correctness evidence

Environment: PyTorch 2.9.1, Triton supplied by that environment, one local
H800 GPU. The 8-H20 pod was not used.

```bash
CUDA_VISIBLE_DEVICES=1 python -m pytest -q \
  tests/unit_tests/transformer/experimental_attention_variant/test_qsa_id_sparse.py \
  --override-ini addopts=''
```

13 CUDA tests passed. The tests compare output and Q/K/V gradients against
exact dense masked SDPA (MATH backend) for BF16 and FP32 GQA, batch 1/2,
packed documents, partial tails, and random distinct selected blocks.
Additional cases exercise multiple 32-token chunks, ratios 2/4/8, head
dimensions 64/128, three finite differences, and rejection of an empty
complete-block row and duplicate IDs. An independent reviewer rechecked
the updated implementation and approved retaining it as an isolated
prototype, while confirming the integration and 256K limits below.

## Single-GPU synthetic benchmark

The script below warms both kernels and times one full forward/backward
invocation after compilation. It uses BF16, `B=1`, `Hq=4`, `Hkv=2`,
`topk=512`, `ratio=4`; inputs and ID lists are allocated before peak-memory
measurement. `recent` IDs are contiguous recent blocks; `strided` IDs are
unique spread-out blocks. Both patterns are synthetic and checked by the
debug validator outside the timed region.

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python \
  tests/unit_tests/transformer/experimental_attention_variant/bench_qsa_id_sparse.py
PYTHONPATH=.:tests/unit_tests/transformer/experimental_attention_variant \
  CUDA_VISIBLE_DEVICES=1 python -c \
  'from bench_qsa_id_sparse import measure; [measure(n, "strided", dim=128) for n in (4096, 8192, 16384)]'
```

| Tokens | D | Route | Forward + backward ms | Extra peak MiB | Preallocated IDs MiB |
| ---: | ---: | --- | ---: | ---: | ---: |
| 4,096 | 64 | recent / strided | 8.6 / 8.6 | 10.06 | 8 |
| 8,192 | 64 | recent / strided | 17.6 / 17.6 | 20.12 | 16 |
| 16,384 | 64 | recent / strided | 35.7 / 35.8 | 40.25 | 32 |
| 4,096 | 128 | strided | 17.7 | 20.06 | 8 |
| 8,192 | 128 | strided | 36.4 | 40.12 | 16 |
| 16,384 | 128 | strided | 74.2 | 80.25 | 32 |

The historical standalone-kernel estimate from 16K D128 was roughly 1.19
seconds and 1.25 GiB of extra memory at 256K. The integrated D256/Hq24 layer
measurement above supersedes that projection; its much larger activation
cost shows why the standalone estimate cannot represent model training.

## Integration boundary and next gate

`qsa.py` `_select_blocks` obtains compact `top.indices` and `valid` for each
tiled query batch. The opt-in producer retains those int32 IDs with `-1`
invalid slots; the default Flex/dense paths still scatter a bitset. Avoid
unpacking a 256K bitset into a token-by-block or token-by-token matrix.
The new kernel receives the same full-sequence selection and full Q/K/V
order after CP zigzag reconstruction; `QSACoreAttention.forward`
gathers differentiable full K/V and splits output back to CP-local tokens.
Packed THD maps to `[1,H,S,D]` and uses document-relative `positions`.

The CPU probe below checks that retained IDs reproduce the bitset and exact
mask for packed/unpacked layouts, including short documents, partial tails
and all-selected rows. GPU dense-masked comparison, CP2 remote K/V gradients
and full-call activation recompute have since passed as described above.
Model optimizer steps and save/resume remain open. The ID path remains
opt-in; dense/Flex fallback behavior and QSA indexer freezing stay explicit.

### CPU producer-contract probe after the kernel commit

The integrated opt-in producer adds `output_format="ids"` to
`QSAIndexer._select_blocks`. It returns the already-computed `top.indices`
as padded int32 IDs, avoiding the bitset allocation and int32 scatter in
that mode. Existing callers keep the default `output_format="bits"` and
their original return contract. The CPU probe calls both modes on the same
scores, rebuilds the bitset from IDs, and compares the exact token mask in
single-document, packed, uniform batch, all-selected, partial-tail, and
ratios 2/4/8 cases. This still computes tiled indexer scores over the
compressed key sequence; only the selected output storage is O(S*K).
The existing `_pool_keys` workspace remains `[n_docs, max_doc_len/R, D]`
plus validity metadata. A packed batch containing one very long document
and many short documents can therefore have near-quadratic pooling memory
despite compact selected IDs. This probe does not establish arbitrary packed
256K support.

The opt-in model wiring chooses the output format before
`QSAIndexer.forward`: `flex` and `dense_masked` use bits, while `id_sparse`
uses IDs. It adds an optional ID field to `QSASelection` and passes the same
document-relative positions. For packed THD, it reshapes full-sequence Q/K/V
to `[1,H,S,D]` and IDs to `[1,S,K]`; for BSHD, permute to `[B,H,S,D]`.
Under CP, it retains the existing full-sequence zigzag reconstruction for
selection and Q/K/V, differentiable KV gather, and local output split.
The CPU test establishes ordering and mask equivalence; the CP2 GPU gradient
test described above checks the model-path attention integration.

Preserve the current all-selected dense paths when sparse forcing is off.
`flex` and `dense_masked` remain explicit fallback choices that build bits.
An `id_sparse` runtime kernel error should fail the step rather than silently
allocate the quadratic dense mask. The selected IDs stay hard/no-grad;
Q/K/V gradients flow through the new autograd function, while training the
indexer requires a separate KL auxiliary loss. Deterministic tie-breaking
for `torch.topk` routes is another prerequisite for multi-rank recompute.

Design references: [official Triton fused-attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
for online softmax and backward derivatives; [TileLang DeepSeek-V3.2 sparse
MLA examples](https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/README.md)
for selected-index KV loading and atomic sparse gradients. Those MLA
kernels use different attention layouts and were not copied into this GQA
prototype.

### Public upstream follow-up found after this prototype

The separate [Megatron-LM QSA reference PR #7234](https://github.com/NVIDIA/Megatron-LM/pull/7234)
at `9e17c43032b2d8df3b4156d9bbe2be1b7d6e29a9` defines stable
score-descending/block-ID-ascending TopK and an indexer-only KL reference;
it deliberately excludes fused kernels and THD/TP/CP. The author's public
[QSA development branch](https://github.com/AllenFeiZZ/Megatron-LM/tree/codex/qsa-sparse-gqa)
at `a904109c3a67a0c0535262ce4eb02bd83d933d6b` does contain TileLang
forward/backward, streaming indexer, deterministic radix TopK, sparse KL,
and THD/CP tests. Its self-reported
[implementation status](https://github.com/AllenFeiZZ/Megatron-LM/blob/a904109c3a67a0c0535262ce4eb02bd83d933d6b/QSA_IMPLEMENTATION_STATUS.md)
explicitly leaves whole-model 256K benchmark and convergence open. This
branch is a stronger production candidate than the slow Triton prototype
here, but its alternate QSA module, CP layout, RoPE and training hooks need
reconciliation with our pinned MCore and multimodal path before adoption.
