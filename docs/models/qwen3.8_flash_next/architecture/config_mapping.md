# HF `config.json` → Megatron arguments

Source of truth for the numbers: `Qwen/Qwen3.8-Flash-Next` `config.json`, `text_config`. Every
row below was validated by the whole-model parity harness (see
[`../validation/parity_and_acceptance.md`](../validation/parity_and_acceptance.md)); the rows
marked ⚠ are the ones where a literal reading of the HF field produced a wrong model.

## Layer schedule

One Qwen decoder layer is one attention-type symbol followed by one MoE symbol, so the 48-layer
model is 96 hybrid-layer symbols. The MTP depth (`mtp.hybrid=true`: the MTP layer runs on the
residual streams) is one more QSA + MoE pair after `/`.

```
--hybrid-layer-pattern GEGEGEQEGEGEGEQEGEGEGEQEGEGEGEQEGEGEGEQEGEGEGEQEGEGEGEQEGEGEGEQEGEGEGEQEGEGEGEQEGEGEGEQEGEGEGEQE/QE
--spec megatron.core.models.hybrid.hybrid_layer_specs gated_residual_hybrid_stack_spec
--mtp-num-layers 1 --mtp-loss-scaling-factor 0.1
```

`G` = Gated DeltaNet, `Q` = QSA, `E` = MoE. `--num-layers` is derived from the pattern. PLE is
declared in **hybrid** layer numbers: HF `ple_layer_ids [2]` (1-based Qwen layer 2) is the second
attention-type symbol, i.e. hybrid layer 3 → `--engram-layer-ids 3`.

## Field mapping

| HF (`text_config`) | Value | Megatron | Note |
|---|---|---|---|
| `hidden_size` | 2560 | `--hidden-size 2560` | |
| `num_attention_heads` / `num_key_value_heads` | 24 / 2 | `--num-attention-heads 24 --group-query-attention --num-query-groups 2` | |
| `head_dim` | 256 | `--kv-channels 256` | |
| `attn_output_gate` | true | `--attention-output-gate` | query and gate interleaved per head in `q_proj` |
| `use_qk_norm` | true | `--qk-layernorm` | |
| `partial_rotary_factor`, `rope_theta` | 0.25, 1e7 | `--rotary-percent 0.25 --rotary-base 10000000 --position-embedding-type rope` | |
| `max_position_embeddings` | 262144 | `--max-position-embeddings 262144` | |
| `vocab_size` | 248320 | `--make-vocab-size-divisible-by 1940` (with the HF tokenizer) or `--vocab-size 248320` with `NullTokenizer` | keeps the padded vocab at exactly 248320 |
| `indexer_n_heads`, `indexer_head_dim`, `indexer_budget`, `indexer_compress_ratio` | 4, 128, 2048, 4 | `--qsa-indexer-n-heads 4 --qsa-indexer-head-dim 128 --qsa-indexer-budget 2048 --qsa-indexer-compress-ratio 4` | see [`../../qsa.md`](../../qsa.md); `--qsa-indexer-loss-coeff 0.01` trains the indexer (HF has no such loss — omit it for parity runs) |
| `linear_num_value_heads`, `linear_num_key_heads`, `linear_key_head_dim`, `linear_value_head_dim`, `linear_conv_kernel_dim` | 48, 16, 128, 128, 4 | `--linear-num-value-heads 48 --linear-num-key-heads 16 --linear-key-head-dim 128 --linear-value-head-dim 128 --linear-conv-kernel-dim 4` | |
| ⚠ `output_gate_type` | `sigmoid` | `--gdn-output-gate-activation sigmoid` | Qwen3-Next used silu; only the GDN output gate changes, the causal conv and the MLP stay on `hidden_act` (silu). The knob exists for exactly this reason |
| `hc_count`, `hc_lowrank` | 4, 320 | `--enable-mhc-connections --mhc-connection-variant gated_residual --mhc-num-residual-streams 4 --hc-lowrank 320` | see [`../../gated_residual.md`](../../gated_residual.md) |
| `num_experts`, `num_experts_per_tok`, `moe_intermediate_size` | 512, 10, 640 | `--num-experts 512 --moe-router-topk 10 --moe-ffn-hidden-size 640` | |
| `shared_expert_intermediate_size` + gated shared expert | 640 | `--moe-shared-expert-intermediate-size 640 --moe-shared-expert-gate` | |
| ⚠ `norm_topk_prob` | true | **nothing** — Megatron's default post-softmax routing (`moe_router_pre_softmax=False`): top-k on logits, softmax over the selected k | HF does softmax over all experts, top-k, divide by the selected sum. The softmax denominator cancels, so the two are the same function (as for Qwen3-Next / Qwen3.5). Setting `--moe-router-pre-softmax` instead gives a router **without** renormalization and scales the MoE output by the top-10 probability mass (parity showed `rel_l2 0.7` on the MoE output) |
| `router_aux_loss_coef` | 1e-3 | `--moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 1e-3 --moe-router-score-function softmax --moe-router-dtype fp32` | |
| `ple_layer_ids` | [2] | `--engram-layer-ids 3` | hybrid numbering, see above |
| `ngram_vocab_size_base` | 20000000 | `--engram-vocab-sizes 20000000 20000000` | one base per n-gram order; the qwen variant derives the 8 per-head primes of each order from it |
| `ple_ngram_size`, `ple_num_heads` | 3, 8 | `--engram-max-ngram-order 3 --engram-num-hash-heads 8 --engram-variant qwen` | |
| ⚠ `ple_embed_dim` | 2560 | `--engram-memory-dim 1280` | Megatron's `memory_dim` is the width of **one** n-gram order; HF's `ple_embed_dim` is the concatenation over all orders: `2560 / (3 − 1) = 1280`, head_dim 160 |
| `ple_conv_kernel_size` | 4 | `--engram-kernel-size 4` | |
| `seed`, `eos_token_id` | 1234, 248044 | `--engram-hash-seed 1234 --engram-eos-token-id 248044 --engram-unigram-vocab-size 248320` | the splitmix64 multipliers and the EOS window reset must match the released weights |
| `rms_norm_eps` | 1e-6 | `--normalization RMSNorm --apply-layernorm-1p --norm-epsilon 1e-6` | HF stores zero-centered gammas |
| `hidden_act`, biases, tied embeddings | silu, none, untied | `--swiglu --disable-bias-linear --untie-embeddings-and-output-weights` | |
| — | — | `--no-weight-decay-cond-type apply_wd_to_qk_layernorm` | training-side choice used in all runs |

## Things the mapping implies about the model definition

- **No final norm after the exit contract.** HF feeds `hyper_connection_mixer` straight into
  `lm_head`; the checkpoint has no `model.norm` / `mtp.norm`. Under
  `mhc_connection_variant=gated_residual` Megatron builds no `final_norm` on the GPT block, the
  hybrid stack or the MTP head — the exit contract's group norm is the last normalization.
- **MTP `hnorm` is a grouped norm over the streams.** The released `mtp.pre_fc_norm_hidden.weight`
  is `[hc_count × hidden] = [10240]`; the gated-residual MTP layer uses
  `GroupedRMSNorm(n·h, group_size=h)` on the flat multi-stream hidden (mHC keeps its shared `[h]`
  gamma).
- **The n-gram memory runs inside the hyper-connection wrapper.** `HyperConnectionHybridLayer`
  adds the memory's output to the n-stream tensor before its read gate and tells the wrapped
  `TransformerLayer` to skip its own injection (`skip_engram`). MTP layers never build a memory
  (`is_mtp_layer` is passed for every hybrid layer type).
- **Parameter counts to expect** on the 4-layer parity proxy (hidden 512, 8 experts): 42,069,232
  without GR/PLE/MTP; +GR 44,507,888; +PLE 67,702,896; +MTP 55,089,392; all three 78,284,400. If
  a flag does not change the count, it did not reach model construction.
