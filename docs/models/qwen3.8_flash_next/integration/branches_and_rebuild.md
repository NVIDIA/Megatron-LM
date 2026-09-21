# How `lit/main_qwen4` is built

`lit/main_qwen4` is an **integration branch**: it is rebuilt from feature branches rather than
edited in place, so that each feature keeps a readable history of its own and the integration
branch stays short. On `origin/main` `5f4c9ac90` (2026-09-20) it is seven commits:

```
fix(examples): let the Qwen3.8 proxy script accept a short --train-iters
examples/docs: port the parity harness and model docs
fix(hybrid): forward --sft-mock-dataset-config-json to the dataset config
feat(transformer): gated-residual variant       <- lit/main_gr       (4 commits + 1 port commit)
feat(engram): hashed n-gram memory              <- lit/main_n_gram  (15 commits + 1 port commit)
feat(gdn): separate GDN output-gate activation  <- lit/main_gdn_gate (1 commit)
feat(qsa): Qwen Sparse Attention                <- lit/main_qsa      (8 commits + 1 port commit)
```

| Feature | Branch | Owns | Doc |
|---|---|---|---|
| QSA | `lit/main_qsa` | `experimental_attention_variant/qsa.py`, `qsa_module_specs.py`, `qsa_layer_config.py`, `dsa_layout.py`, `ops/triton_qsa.py`, the `Q` hybrid symbol, the static `qsa_layer` / `qsa_qk_layernorm_layer` entries of `hybrid_stack_spec` | [`../../qsa.md`](../../qsa.md) |
| Gated residual | `lit/main_gr` | `transformer/gated_residual.py`, `mhc_connection_variant`, `gated_residual_hybrid_stack_spec`, the GR branch of `HyperConnectionHybridLayer`, no-final-norm on the HybridStack and MTP exits, the grouped MTP `hnorm` | [`../../gated_residual.md`](../../gated_residual.md) |
| n-gram memory / PLE | `lit/main_n_gram` | `megatron/core/models/engram/`, GPT and hybrid integration, qwen variant + HF PLE converter, packed-document boundaries, recompute, VPP prefetch, CP (unpacked rows), memory inside the hyper-connection wrapper, MTP on the hybrid path, fp32 | [`../../engram.md`](../../engram.md) |
| GDN output gate | `lit/main_gdn_gate` | `gdn_output_gate_activation` (`ssm/gated_delta_net/common.py`) + test | [`../architecture/config_mapping.md`](../architecture/config_mapping.md) |

## There is no QSA-into-GR glue commit on this base

On the `dev`-based lineage, `gated_residual_hybrid_stack_spec(config)` and
`hybrid_qsa_stack_spec(config)` were **config-aware factories** that each deep-copied the default
hybrid stack and mutated it, so selecting one dropped the other's symbol: a pattern with `Q`
built with the GR spec would leave `qsa_layer = IdentityOp`. A glue commit had to teach the GR
builder to start from the QSA-extended stack.

`main` forbids that shape. `megatron/core/models/hybrid/CLAUDE.md` requires module specs to be
comptime-available, and `hybrid_builder` enforces it (`--spec must refer to a static ModuleSpec`).
So on this base:

* QSA is wired **statically into `hybrid_stack_spec`** as `qsa_layer` / `qsa_qk_layernorm_layer`,
  exactly the way `main` wires `csa_layer` / `csa_qk_layernorm_layer`;
* `gated_residual_hybrid_stack_spec` is a **module-level object**, built once at import by
  `_strip_input_norms(hybrid_stack_spec)`.

Because the strip runs over the one stack that already contains the QSA layers, the composition
the glue commit used to perform happens by construction. `--spec … gated_residual_hybrid_stack_spec`
is unchanged for callers; it now resolves to an object rather than a function.

## Rebuild procedure

A squash merge records no merge relationship, so the integration branch cannot be updated
incrementally — a second `merge --squash` of the same feature replays its whole diff. Every round:

```bash
git tag qwen4/r<N> lit/main_qwen4                      # keep the working round
git branch -f lit/main_qwen4 <new dev base>
git merge --squash lit/main_qsa            && git commit -s -S
git merge --squash lit/main_gr && git commit -s -S
git merge --squash lit/main_n_gram         && git commit -s -S
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
