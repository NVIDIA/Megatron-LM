# QSA selected-ID sparse kernel: isolated prototype

This branch contains a standalone Triton forward/backward kernel for QSA
core attention. It consumes a compact list of document-relative selected
complete-block IDs per query and includes the last incomplete block directly.
It is **not connected to `qsa.py` or Relax training**, and no 256K attention
or RL step has been run. Keep it isolated until the indexer output contract,
CP reconstruction, packed layouts, and real-model training are reviewed.

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

At 256K, the selected-ID input alone is exactly 512 MiB for one batch row
with 512 int32 IDs per token. Multiplying the 16K D128 measured values by
16 gives roughly 1.19 seconds and 1.25 GiB extra memory, **only as a naive
linear arithmetic projection**. It is not a bound or a measured 256K result:
cache behavior, atomic contention, actual model heads/dimensions,
indexer scoring/top-k, CP gathering, activation storage, optimizer state,
and RL orchestration can dominate or change scaling. The current QSA
indexer emits a bitset, not IDs; at 256K that bitset itself is about 2 GiB
per batch row, so this kernel alone does not unlock 256K training.

## Integration boundary and next gate

The current `qsa.py` `_select_blocks` already obtains compact `top.indices`
and `valid` for each tiled query batch, then scatters them to `selected_bits`.
The narrow producer change is to retain those int32 IDs with `-1` invalid
slots and preserve the existing bitset only for Flex/dense fallbacks. Avoid
unpacking a 256K bitset into a token-by-block or token-by-token matrix.
The new kernel must receive the same full-sequence selection and full Q/K/V
order after CP zigzag reconstruction; `QSACoreAttention.forward` already
gathers differentiable full K/V and splits output back to CP-local tokens.
Packed THD should map to `[1,H,S,D]` and use document-relative `positions`.

The CPU probe below checks that retained IDs reproduce the current bitset
and exact mask for packed/unpacked layouts, including short documents,
partial tails and all-selected rows. Remaining gates are comparison with
the existing Flex/dense paths on GPU, CP2 remote dK/dV, selective activation
recompute, model optimizer steps and save/resume.
The selected-ID path should be opt-in until those gates pass; dense/Flex
fallback behavior and QSA indexer freezing remain explicit.

### CPU producer-contract probe after the kernel commit

The follow-up candidate adds `output_format="ids"` to
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

The proposed opt-in model wiring should choose the output format before
`QSAIndexer.forward`: `flex` and `dense_masked` use bits, while `id_sparse`
uses IDs. Add an optional ID field to `QSASelection` and pass the same
document-relative positions. For packed THD, reshape full-sequence Q/K/V
to `[1,H,S,D]` and IDs to `[1,S,K]`; for BSHD, permute to `[B,H,S,D]`.
Under CP, retain the existing full-sequence zigzag reconstruction for
selection and Q/K/V, differentiable KV gather, and local output split.
The CPU test establishes ordering and mask equivalence only; a CP2 GPU
gradient test remains required.

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
