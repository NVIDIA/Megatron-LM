# How `lit/qwen4` is built

`lit/qwen4` is an **integration branch**: it is rebuilt from feature branches rather than edited
in place, so that each feature keeps a readable history of its own and the integration branch
stays short. On dev `bb5dfd08f` (2026-09-14) it is five commits:

```
fix(hybrid): compose the QSA layer with the gated-residual stack spec   <- the one glue commit
feat(gdn): add a separate activation switch for the GDN output gate     <- standalone (lit/gdn_output_gate)
feat(engram): squash-merge the n-gram memory and its Qwen PLE variant  <- lit/n_gram (15 commits)
feat(transformer): squash-merge the gated-residual variant             <- lit/gated_residual (4 commits)
feat(qsa): squash-merge QSA sparse attention                           <- lit/qsa (8 commits)
```

| Feature | Branch | Owns | Doc |
|---|---|---|---|
| QSA | `lit/qsa` | `experimental_attention_variant/qsa.py`, `dsa_layout.py`, `ops/triton_qsa.py`, the `Q` hybrid symbol, `hybrid_qsa_stack_spec` | [`../../qsa.md`](../../qsa.md) |
| Gated residual | `lit/gated_residual` | `transformer/gated_residual.py`, `mhc_connection_variant`, `gated_residual_hybrid_stack_spec`, the GR branches of `HyperConnectionHybridLayer`, no-final-norm on all three paths, the grouped MTP `hnorm` | [`../../gated_residual.md`](../../gated_residual.md) |
| n-gram memory / PLE | `lit/n_gram` | `megatron/core/models/engram/`, GPT and hybrid integration, qwen variant + HF PLE converter, packed-document boundaries, recompute, VPP prefetch, CP (unpacked rows), memory inside the hyper-connection wrapper, MTP on the hybrid path, fp32 | [`../../engram.md`](../../engram.md) |
| GDN output gate | `lit/gdn_output_gate` | `gdn_output_gate_activation` (`ssm/gated_delta_net/common.py`) + test | [`../architecture/config_mapping.md`](../architecture/config_mapping.md) |

## The glue commit

`hybrid_qsa_stack_spec` and `gated_residual_hybrid_stack_spec` each deep-copy the default hybrid
stack and mutate it, so selecting one drops the other's symbol: a pattern with `Q` built with the
GR spec would leave `qsa_layer = IdentityOp`. The glue makes the GR builder start from the
QSA-extended stack when `experimental_attention_variant == "qsa"` and strip the QSA layer's
`input_layernorm` along with the other explicit-norm layers. It needs both features, which is why
it lives here and not in either branch. Any third config-aware spec builder will hit the same
problem; the composition has to be explicit (or the builders have to accept a base spec).

## Rebuild procedure

A squash merge records no merge relationship, so the integration branch cannot be updated
incrementally — a second `merge --squash` of the same feature replays its whole diff. Every round:

```bash
git tag qwen4/r<N> lit/qwen4                      # keep the working round
git branch -f lit/qwen4 <new dev base>
git merge --squash lit/qsa            && git commit -s -S
git merge --squash lit/gated_residual && git commit -s -S
git merge --squash lit/n_gram         && git commit -s -S
git cherry-pick -x <gdn output gate>  # until it is upstream
git cherry-pick -x <the glue commit>
```

Conflicts seen on the 2026-09-15 rebuild, all "keep both": the import block of
`models/gpt/experimental_attention_variant_module_specs.py` (QSA vs GR), the two sibling builders
in `models/hybrid/hybrid_layer_specs.py`, the import block of `training/models/hybrid.py` (GR vs
Engram). Rebasing the feature branches onto a newer dev touched `pretrain_gpt.py`
(`prepare_packed_seq_params` replaced `finalize_packed_seq_params`), the utils import blocks, and
the `HyperConnectionHybridLayer` constructor (upstream offload fields around the module switch).

## Five defects to re-check after every rebuild

All of them are "the code runs and the loss goes down but the feature is not there":

1. **Mutually exclusive spec builders** (above) — build the spec for the real pattern and assert
   every symbol has a non-`IdentityOp` entry.
2. **`pretrain_gpt` and `pretrain_hybrid` use different builder paths** — `pretrain_hybrid` goes
   through `HybridModelConfig` / `HybridModelBuilder`, so a feature attached only in
   `hybrid_builders.py` is silently ignored. Check: parameter count with and without the flag.
3. **The hyper-connection wrapper bypasses `TransformerLayer.forward`** — anything injected there
   never runs under hyper connections. Check: `--engram-verify-training` gradient counters.
4. **Global guards for per-layer features** — a startup guard that rejects recompute / CP / VPP for
   the whole model because one layer type cannot do it.
5. **Config fields with HF names but different semantics** — `memory_dim` vs `ple_embed_dim`,
   `norm_topk_prob` vs the routing path.

Acceptance after a rebuild is the matrix in
[`../validation/parity_and_acceptance.md`](../validation/parity_and_acceptance.md) §5.

## Fixes that used to be glue

Nine integration-round fixes were absorbed into their feature branches on 2026-09-15 (GR: norm-free
spec flag, no final norm, MTP `hnorm`; n-gram: hybrid path, config-driven builder, hyper-connection
wrapper, MTP on the hybrid path, fp32; GDN gate as its own branch). A tenth — a
`moe_router_norm_topk_prob` knob — was retired instead: HF `norm_topk_prob` is Megatron's default
post-softmax routing, so the model configuration simply stops asking for pre-softmax routing.
