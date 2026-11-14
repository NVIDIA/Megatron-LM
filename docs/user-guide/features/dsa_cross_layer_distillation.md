# Cross-layer indexer distillation for sparse DSA

IndexShare reuses one layer's Top-K selection in the following layers. Cross-layer
indexer distillation (CLI) additionally trains that owner indexer against the
attention targets of every layer in its group. It is opt-in and leaves the main
model's language-model gradient path unchanged.

The objective is the **sum** of per-layer KL losses, as in equation (8) of
[LongCat Sparse Attention, version 2](https://arxiv.org/abs/2608.01662v2):

$$\mathcal L_{\mathrm{CLI}}=\sum_{i=0}^{N-1}
D_{\mathrm{KL}}(p^{(l+i)}\Vert q^{(l)}).$$

Each teacher is normalized over the owner's selected tokens. The loss coefficient
applies to the sum, so changing the group size also changes its total weight.
Loss logging records the complete group loss at the owner layer.

## Configuration

Add these options to an otherwise working absorbed-MLA DSA training configuration:

```bash
--experimental-attention-variant dsa \
--dsa-kernel-backend cudnn \
--dsa-indexer-use-sparse-loss \
--dsa-indexer-topk 256 \
--dsa-indexer-topk-freq 2 \
--dsa-indexer-skip-topk-offset 1 \
--dsa-indexer-cross-layer-distillation \
--dsa-indexer-loss-coeff 0.001 \
--tensor-model-parallel-size 1 \
--micro-batch-size 1 \
--recompute-granularity full \
--recompute-method uniform \
--recompute-num-layers 1
```

CLI needs TileLang for target accumulation and a cuDNN Frontend build whose
`DSA.sparse_attention_backward_wrapper` accepts `return_cli_target` and
`cli_target_head_count`. This extension is required only with CLI enabled; ordinary
DSA does not request its output. No modified FlashMLA forward is needed.

## Initial support contract

- SM90 only; cuDNN backward receives one physical 64-head block. Fewer real heads
  use the existing padding, with padded LSE values set to positive infinity.
- TP=1 and microbatch size 1; absorbed latent V dimension 512 and QK dimension
  512 or 576, as required by the existing sparse-attention path.
- Sparse loss, flat indexer selection, positive Top-K divisible by 256, and full
  uniform activation recomputation of one layer at a time.
- Each pipeline stage must begin on a computing layer; groups cannot cross a
  pipeline boundary. CP and packed layouts retain DSA's existing layout checks.
  Validate each distributed layout before using it in production.
- Dense warm-up, pooled selection, cross-depth MTP distillation and
  `deterministic_mode` are outside this initial implementation.

The original no-grad forward stores the selected indices on the same per-forward
carrier as IndexShare. Reuse-layer attention backward reconstructs and exports
the head-reduced teacher. The accumulator normalizes each teacher independently
and retains its entropy. When the owner is recomputed, its indexer is scored on
the **original indices**, so ties and ordering cannot pair targets with different
tokens. The owner merges the target statistics and executes one indexer backward.
Single-layer prefix or trailing groups use ordinary DSA loss.

This scheduling requires the reuse-layer backward to run before the owner's
recomputation. Supporting a different recomputation scheme requires a separate
state/gradient scheduling design. Temporary targets are not checkpoint weights.

## Validation

Run the focused tests in the standard GPU development container:

```bash
uv run python -m torch.distributed.run --nproc-per-node 1 -m pytest -q \
  tests/unit_tests/ops/test_dsa_cli.py \
  tests/unit_tests/determinism/kernels/test_dsa_cli.py \
  tests/unit_tests/transformer/experimental_attention_variant/test_dsa_native_parity.py
```

Also run the existing DSA regressions and a representative multi-step training
run with multiple microbatches and checkpoint/resume. Record source revisions,
GPU and compiler/library versions, the parallel/recompute configuration, loss,
step time and peak memory. Older downstream results do not validate a newly
rebased source revision automatically.
