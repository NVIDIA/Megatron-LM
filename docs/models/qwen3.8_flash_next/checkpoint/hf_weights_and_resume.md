# Released weights → Megatron checkpoint → resume

The released `Qwen/Qwen3.8-Flash-Next` checkpoint (131 safetensors shards, 335 GiB, text tower +
MTP head + vision tower) has been converted strictly into this model definition and trained from.
The converter itself lives in the companion toolkit
(`mcore_devtoolkit.ckpt_tools`, adapter `models/qwen4_exp.py`, native writer
`convert/package2dist.py`); this page records the **mapping rules and the checkpoint contract**
that any converter must satisfy, and what was measured.

## Pipeline

```
HF shards ──(1) strict tensor mapping, CPU, streaming──▶ logical Megatron tensors + per-rank shards
          ──(2) build the model at the target EP on GPUs, copy shards in, dist_checkpointing.save──▶ native torch_dist (weights only, iter_0000000)
          ──(3) pretrain_hybrid.py --load … --no-load-optim --no-load-rng --finetune──▶ training
```

Measured on the full model (2026-09-14): (1) 2 h 09 min and 41 GiB host RAM on one CPU node,
`present=1658 matched=1322 missing=0 unexpected=0`, 333 vision tensors dropped by prefix;
(2) 61 s on 64 ranks for the 335 GiB write; a fresh EP64 model loads it with
`--dist-ckpt-strictness raise_unexpected` and **109,632 / 109,632 parameters are exactly equal**;
(3) 50 iterations on 64 GPUs, numbers in
[`../training/full_model_64gpu.md`](../training/full_model_64gpu.md). **Re-verified 2026-09-16**
after the branch was rebuilt on a newer dev base, the n-gram import was swapped and the router
moved to post-softmax: the same checkpoint loads exactly and trains 20 iterations with no NaN at
204.0 GiB peak, so neither the parameter names nor the checkpoint layout moved. The native checkpoint
reshards across EP (EP1 ↔ EP4 verified on a 4-layer truncation).

## Mapping rules

Text tower (`model.language_model.*` / `model.layers.{i}.*` in HF → `decoder.layers.{j}` with
`j = 2i` for the attention-type symbol and `2i+1` for the MoE symbol of Qwen layer `i`):

| Component | Rule | Reference |
|---|---|---|
| Embedding / output | `embed_tokens` → `embedding.word_embeddings`, `lm_head` → `output_layer`; vocab padded to 248320 exactly (`--make-vocab-size-divisible-by 1940`) | |
| GDN layer | `in_proj_qkvz`, `in_proj_ba` concatenated into the fused `in_proj` (q, k, v, z, b, a order); `conv1d`, `A_log`, `dt_bias`, `norm` (gated RMSNorm, zero-centered gamma → `+1`), `out_proj` | stock GDN |
| QSA layer | `q_proj` (query and output gate interleaved per head) → `linear_q`, `k_proj`/`v_proj` → `linear_kv`, `q_norm`/`k_norm`; indexer `index_qk_proj` split into 4 query heads + 1 key head, per-head norms | [`../../qsa.md`](../../qsa.md) §"Checkpoint mapping" |
| MoE | 3-D packed experts `experts.gate_up_proj [E, 2I, H]` / `experts.down_proj [E, H, I]` → per-expert `linear_fc1` / `linear_fc2`; `gate` → `router`; shared expert + `shared_expert_gate` | note: transformers' `save_pretrained` writes per-expert tensors, the released files are packed |
| Gated residual | `hc_pre_fc_*` / `hc_post_fc_*` / `hc_norm` per sublayer → `hyper_connection.*`; `hyper_connection_mixer` → `hc_exit_contract`; **no** `final_norm` is expected or produced | [`../../gated_residual.md`](../../gated_residual.md) §"HF ↔ Megatron weight mapping" |
| PLE / n-gram | HF's one fused, padded table per order → 16 prime-sized per-head tables cut at the prime offsets, streamed shard by shard (never materialize the 102 GiB table); `ple_key_proj`, `ple_value_proj`, group norms | `engram.md` §"Distributed checkpointing"; EP row ranges = `EPShardedEmbeddingTable` |
| MTP head | `mtp.pre_fc_norm_embedding` → `enorm`, `mtp.pre_fc_norm_hidden [10240]` → `hnorm` (grouped `[n·h]`), `mtp.fc` → `eh_proj`, `mtp.layers.0.*` → nested hybrid stack `mtp_model_layer.layers.{0,1}`, `mtp.hyper_connection_mixer` → MTP exit contract | HF has no MTP forward: the MTP mapping is verified by exact load only |
| Vision tower | `model.visual.*` dropped by prefix (333 tensors) | not modelled |

Strictness: every HF tensor must be consumed exactly once and every Megatron parameter must have
a source (`missing=0 unexpected=0 skipped_rules=0`). Loading the real MTP weights is what
exposed the `hnorm` width; converting only the proxy would not have.

## Checkpoint contract

- Format `torch_dist`, weights only: `iter_0000000/` with `latest_checkpointed_iteration.txt = 0`,
  no optimizer or RNG state. Resume with `--load <dir> --no-load-optim --no-load-rng --finetune
  --ckpt-format torch_dist`; `--finetune` restarts the iteration counter.
- `--dist-ckpt-strictness raise_unexpected` is the right level to verify a conversion
  (`raise_all` trips on `rerun_state_machine_state`, which a weights-only checkpoint lacks).
- Writing the checkpoint needs GPUs: `initialize_megatron` asserts CUDA and `get_model` moves
  every component to the device. The full model was written with EP64 on 16 nodes (7.77 B
  parameters per rank); a 4-layer truncation fits one GPU at EP1.
- Cross-EP loads work for the weights; optimizer state written later by training reshards only if
  it was saved with `--dist-ckpt-optim-fully-reshardable`.

## Disk budget seen in practice

| Artifact | Size |
|---|---|
| HF shards | 335 GiB |
| logical Megatron tensors (bf16, vision dropped) | 335 GiB |
| per-rank shards at EP64 | 926 GiB (the ≈ 10 GiB of replicated non-expert parameters appears 64 times) |
| native `torch_dist` | 335 GiB (replicas deduplicated) |

## What is verified about the converted model

Parity of the real weights against the HF implementation on the 4-layer truncation, the exact
native load, and the 50-step resume are in
[`../validation/parity_and_acceptance.md`](../validation/parity_and_acceptance.md). Loss quality
on real text has **not** been checked (mock data only).
