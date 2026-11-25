# Native LoRA training (draft)

This is the initial PEFT 1 implementation for the consolidation tracked in #9363.
It is not ready to merge. The native recipe entrypoint requires #9625, and
adapter-only save/resume requires #9426. Neither prerequisite is copied into this PR.

## Configuration

Starting from a native training recipe:

```python
from megatron.training.peft import LoRA

cfg.peft = LoRA(
    target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
    dim=16,
    alpha=32,
)
cfg.checkpoint.pretrained_checkpoint = "/checkpoints/base"
cfg.checkpoint.save = "/checkpoints/adapters"
# To resume the same adapter run:
cfg.checkpoint.load = "/checkpoints/adapters"
```

Leave `checkpoint.finetune=False`: base loading already uses weights-only
semantics, while the adapter resume must restore optimizer and training progress.
The run configuration is not changed to switch between these two loads.

Use `canonical_qkv_projections=True` for independent Q/K/V adapters. Targets may
be the fused name `linear_qkv` (all three) or logical names `linear_q`,
`linear_k`, and `linear_v`. `canonical_mlp_projections=True` similarly selects
separate gate/up adapters for a dense gated MLP. These options do not change
the base model's fused projections. There is no separate CanonicalLoRA class.

`validate()` checks settings without mutation. Matching happens when the model
is built, so edits to target patterns after constructing the recipe take effect.
Base parameters are frozen; adapters are attached before DDP wrapping and
optimizer construction. Recompute input hooks preserve adapter gradients
without unfreezing embeddings.

## Ownership

- `megatron/core/transformer/peft`: low-rank modules, wrappers, expert gradient
  synchronization, tensor layouts and mathematical weight merging. No training
  or Bridge imports, and process groups are passed explicitly.
- `megatron/training/peft`: optional configuration, module selection, model
  setup, recompute integration and checkpoint selection.
- Bridge is reserved for HF conversion and model-specific integration; no
  generic PEFT package is added there.

The Core modules are adapted from Megatron-Bridge revision
`c031e61d82dafbe4655a2e2d7e742b88f7e81c3a`, retaining its Apache-2.0 notices.
Adapter construction does not modify the shared TransformerConfig. Per-expert
adapters initially use the explicit per-expert implementation, not private TE
FP8 APIs or the optional grouped-GEMM optimization.

## Before marking ready

- Integrate #9625 and #9426, then demonstrate native model construction, a short
  training run, adapter-only checkpointing and resumed/uninterrupted equivalence.
- Validate TP/SP/PP/EP, distributed optimizer, shared and per-expert adapters,
  BF16 and recompute behavior against Bridge. Focused tests are not a claim
  that this full distributed matrix has passed.
- Add kernel determinism replay coverage for the supported adapter execution
  paths before marking this draft ready.
- Complete HF adapter import/export and checkpoint merge integration, or agree
  and track a separately scoped follow-up before calling PEFT 1 complete.
- Add compatibility for supported old Bridge config class paths and adapter
  checkpoint layouts. Missing historical router-bias state cannot be recreated.
- Settle the canonical MLP option and unsupported canonical expert layouts.
- Evaluate FSDP, meta initialization and quantized base models separately;
  the initial native training path rejects these combinations.

Per-expert adapters currently require expert TP=1. Canonical QKV requires at
least one query group per TP rank; canonical gate/up requires a dense gated MLP.
The mathematical merge helper is present, but a complete checkpoint-to-HF merge
workflow is not yet provided.

DoRA and MultiLoRA are not included. MultiLoRA is PEFT 2. Router LoRA, TE operation
fuser, serving-specific shared-outer layouts and LoRA+ recipe wiring are not
part of this initial draft. Generic wildcard targeting does not mean every
model-specific module or every combination is supported.
