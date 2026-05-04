# DeepSeek-V4 hybrid attention (CSA / HCA) in the hybrid model

This directory holds an example slurm script for pretraining a Mamba-based hybrid
model that mixes DeepSeek-V4's two new attention variants:

* **CSA — Compressed Sparse Attention** (Section 2.3.1 of the DSv4 tech report).
  Compresses every 4 tokens (with overlap = 2) into one KV entry and uses a
  learned lightning indexer to pick the top-k relevant compressed entries. Each
  query then attends to a small sliding window plus the selected compressed
  positions.
* **HCA — Heavily-Compressed Attention** (Section 2.3.2). Compresses every 128
  tokens (no overlap) into one KV entry and applies dense attention over all
  valid compressed positions, again concatenated with a sliding window.

Both share the same module (`DSv4HybridSelfAttention` / `CompressedSparseAttention`);
the per-layer compression ratio selects the behaviour.

## Pattern symbols

The hybrid layer pattern grows two new symbols in addition to the existing
`M / G / * / D / - / E`:

| Symbol | Meaning |
| ------ | --------------------------------------------------------- |
| `M`    | Mamba                                                     |
| `G`    | Gated DeltaNet                                            |
| `*`    | Standard self-attention (GQA)                             |
| `D`    | DeepSeek Sparse Attention (DSA, MLA-style + indexer)      |
| `C`    | DSv4 Compressed Sparse Attention (CSA, ratio = 4)         |
| `H`    | DSv4 Heavily-Compressed Attention (HCA, ratio = 128)      |
| `-`    | Dense MLP                                                 |
| `E`    | MoE                                                       |

Only one of `*` and the MLA-like family (`D`, `C`, `H`) may appear in the same
model — they share the MLA-style q/kv projection setup.

## Required CLI flags

A model that uses `C` or `H` layers should set:

* `--hybrid-layer-pattern` containing `C` and/or `H`.
* The MLA-related flags: `--q-lora-rank`, `--qk-head-dim`, `--qk-pos-emb-head-dim`,
  `--v-head-dim`, plus `--rope-type rope|yarn`.
* DSA indexer flags (used only by `C`): `--dsa-indexer-n-heads`,
  `--dsa-indexer-head-dim`, `--dsa-indexer-topk`,
  optionally `--dsa-indexer-loss-coeff` for KL training of the indexer.

CSA/HCA-specific flags:

* `--csa-window-size N` — sliding-window length (default 128).
* `--csa-compress-ratio-for-c N` — ratio used by every `C` layer (default 4).
* `--csa-compress-ratio-for-h N` — ratio used by every `H` layer (default 128).
* `--csa-compress-ratios "[...]"` — explicit per-layer ratios. Overrides the
  pattern-derived defaults; length must equal `num_layers`.
* `--csa-compress-rotary-base FLOAT` — RoPE base for compressed KV positions
  (default 40000).
* `--csa-dense-mode` — run all `C` layers in dense (no-indexer) mode. Useful as
  a warmup phase before sparse training.
* `--csa-no-attention-sink` — disable the per-head learnable sink logit.
* `--o-groups N`, `--o-lora-rank N` — grouped output projection geometry.
  ``num_attention_heads * v_head_dim`` must be divisible by `o_groups`.

## Prototype scope

This is a CP=1, TP=1 prototype that uses the unfused RoPE path. MTP, packed
sequences, FP8/FP4, and fine-grained activation offloading are not supported
yet. Inference is also disabled for `C`/`H` layers.

## Running

```
bash examples/dsv4_hybrid/train_dsv4_hybrid.sh
```

The provided script trains a 2B-class hybrid model with the pattern

```
M-M-MCM-MHM-MDM-MCM-MHM-MDM-
```

(24 layers, mostly Mamba + MLP, with CSA / HCA / DSA layers sprinkled in).

Adjust `IMAGE`, `BASE_DIR`, `BLEND_PATH`, `TOKENIZER_MODEL_PATH`, etc. to your
environment. Tests must run inside the docker image — this is a slurm login
node so GPUs are not directly available.

## Tests

* `tests/unit_tests/transformer/experimental_attention_variant/test_csa_hca_hybrid.py`
  — unit tests covering the helpers, the `Compressor`, `CSAIndexer` and the
  `CompressedSparseAttention` core attention (CSA, HCA and window-only paths,
  forward + backward).
* `tests/unit_tests/ssm/test_hybrid_layer_allocation.py`
  — extended to exercise the new `C` and `H` symbols in the hybrid pattern.
