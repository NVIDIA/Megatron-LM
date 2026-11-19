# Cross-layer indexer distillation for sparse DSA

## Why use cross-layer supervision?

IndexShare saves indexer computation by selecting Top-K tokens in an owner layer
and reusing those indices in subsequent layers. With ordinary DSA supervision,
only the owner's attention distribution trains that retained indexer. Reuse
layers consume its selection but contribute no teacher signal. Their attention
patterns can differ from the owner's, so a selection useful to the owner may
omit tokens needed by other layers in the group.

Cross-layer indexer distillation (CLI) closes this supervision gap: the retained
indexer learns from every attention layer that consumes its selection. It keeps
the index-sharing structure and adds auxiliary indexer supervision. The teachers
are detached; the language-model gradient still follows ordinary attention
backward. The feature is opt-in and disabled by default.

CLI is useful when training with cross-layer Top-K reuse and the quality of the
shared selection matters. There is no cross-layer reuse to supervise when every
layer has its own indexer. Ordinary owner-only supervision remains appropriate when its
quality is sufficient or the additional dependency/recompute constraints are
undesirable. CLI is not required for ordinary DSA training.

The [LongCat Sparse Attention cross-layer distillation ablation, Section 5.3.3
and Fig. 7(c)](https://arxiv.org/html/2608.01662v2#S5.SS3.SSS3) reports:

| Configuration | Sharing group size | 128K retrieval accuracy |
| --- | ---: | ---: |
| Index sharing without cross-layer distillation | 2 | 70% |
| Index sharing with cross-layer distillation | 2 | 96% |

The 26-percentage-point improvement motivates supervision from reuse layers.
These are published LongCat-Flash-Lite results, not measurements of this
implementation or guarantees for other models and sharing frequencies.

Private training evaluations also showed quality gains on the order of ten
score points (0-100 scale) across evaluated benchmarks versus CLI-disabled
index-sharing DSA at sharing frequency 4. This is a coarse description, not an
average or a guaranteed gain; detailed scores are omitted. These evaluations
are separate from the focused validation of this revision.

## Objective and loss scaling

The sparse CLI objective follows Section 3.2, equation (8) of
[LongCat Sparse Attention, version 2](https://arxiv.org/abs/2608.01662v2):

$$
\mathcal L_{\mathrm{CLI}} = \sum_{i=0}^{N-1}
D_{\mathrm{KL}}(p^{(l+i)} \Vert q^{(l)}).
$$

Here, $q^{(l)}$ is the owner's indexer distribution and each $p^{(l+i)}$ is a
detached, head-reduced attention teacher, normalized over that owner's selected
valid tokens. The group includes the owner's own teacher and all reuse-layer
teachers. This implementation supports sparse training, not dense warm-up.

The loss is a **sum**, not a group average. The configured loss coefficient
multiplies that sum, so its effective weight grows with group size. Prefix
singletons and truncated tail groups use their actual sizes. Group loss is
logged at the owner. Changing the sharing frequency may therefore require
retuning the loss coefficient.

## Why this training and operator design?

### Reuse probabilities already computed by attention backward

cuDNN sparse attention backward already reconstructs the probabilities needed
for its gradients. The optional teacher output from
[cuDNN Frontend PR #2556](https://github.com/NVIDIA/cudnn-frontend/pull/2556)
exports their head-reduced FP32 mass in the original Top-K slot order. Collecting
this output avoids a separate teacher attention-score recomputation pass for
each reuse layer. The attention forward computation is unchanged; no CLI-specific
FlashMLA forward modification is required.

### Keep sufficient statistics instead of per-layer loss graphs

For reuse-layer teachers define $T = \sum_i p_i$ and
$E = \sum_i \sum_k p_{i,k}\log p_{i,k}$. Their loss contribution is

$$
\sum_i D_{\mathrm{KL}}(p_i \Vert q) = E - \sum_k T_k\log q_k.
$$

A custom TileLang kernel normalizes each masked teacher, accumulates its target
in FP32, and computes per-row entropy terms in the same kernel invocation.
The group retains one target sum and one entropy sum; chunking bounds temporary
per-call work. It does not retain a separate teacher target or student loss graph
for every reuse layer. Invalid slots and empty rows contribute zero.

When the owner is recomputed, its own teacher is combined with these statistics.
One grouped indexer backward uses the averaged target with the coefficient
multiplied by the actual group size. On a valid row this gives the summed
student-logit gradient $Nq - \sum_i p_i$. The entropy sum also preserves the
reported scalar KL loss. Averaging teachers and scaling the gradient alone
would not reconstruct that scalar loss without the individual entropy terms.

### Match the schedule to full activation recomputation

1. The original no-grad forward stores the owner's Top-K selection on the
   existing per-forward IndexShare carrier.
2. During backward, each reuse layer is recomputed and its attention backward
   contributes a detached teacher to that owner's accumulator.
3. After the reuse-layer backwards complete, owner recomputation consumes the
   statistics, performs the grouped indexer backward, and releases the saved
   selection and accumulation state.

Each checkpoint closure retains its forward's carrier, even when multiple
forwards receive the same packed sequence metadata. This allows backward in
either order. The owner checks that exactly the expected number of reuse
teachers has arrived.

The initial implementation requires full uniform recomputation with **one layer
per chunk**. That order makes reuse teachers available before owner recomputation.
Recomputing multiple layers together can run the owner's forward before the
reuse-layer backwards have supplied their teachers. Supporting it requires a
different state/gradient schedule, not just relaxing the configuration guard.

### Re-score the original selection

Owner recomputation scores only the saved Top-K slots. This preserves the
teacher's token order, including ties, and avoids another global selection.
A Triton kernel gathers selected keys and produces FP32 logits in that order.
Masked log-softmax then preserves the KL contribution
of valid tokens whose probabilities would underflow in FP32.

## Why not the alternatives?

| Alternative | Tradeoff and reason for the current choice |
| --- | --- |
| Keep owner-only supervision | Lowest auxiliary cost and still the default. It leaves reuse-layer attention patterns outside the retained indexer's objective. |
| Compute an indexer in every layer | Removes cross-layer selection reuse and its computation savings. CLI trains the existing shared indexer while retaining that structure. |
| Recompute each teacher in a separate score pass | Useful as a reference or for other backends, but adds a teacher-score pass when attention backward already reconstructs the probabilities. This path uses the fused output instead. |
| Keep a separate student KL graph for every teacher | A valid reference formulation, but requires additional retained graph/target state or repeated student work. The sufficient-statistics formulation supports one grouped backward under the selected recompute schedule. |
| Use a fixed group-average loss | Changes the auxiliary loss weight relative to the paper's sum, especially for partial groups. This implementation keeps explicit summed-loss scaling and entropy terms. |

These are design tradeoffs, not measured head-to-head performance results for
all alternatives. A different backend or recomputation strategy may justify a
different implementation; it must preserve token alignment, masking, loss
scaling and microbatch state isolation.

A matched-Top-K SM90 microbenchmark of the earlier cuDNN score recompute path
(128K global sequence, 16K local queries,
Top-K 2048, sharing frequency 4, BF16, chunk size 4096) measured an incremental
3.165 +/- 0.196 ms per timed attention/indexer group, approximately 0.32%, and
0.25 GiB additional peak allocated memory. Both arms perform the same Top-K
work. This isolates additional CLI group cost; it is not full-step timing or a
measurement of the separate benefit from avoiding owner reselection.

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

CLI requires TileLang, Triton, and the teacher output in cuDNN Frontend
[1.33.0](https://github.com/NVIDIA/cudnn-frontend/releases/tag/v1.33.0) or later.
`DSA.sparse_attention_backward_wrapper` must expose `return_cli_target` and
`cli_target_head_count`. FlashMLA must be built with SM90 support.

## When not to enable this path

- The initial implementation supports SM90 cuDNN sparse backward with one
  physical 64-head block. Fewer real heads use the existing padding, with padded
  LSE set to positive infinity so they contribute zero probability.
- TP=1 and microbatch size 1 are required. Absorbed latent V dimension is 512 and
  QK dimension is 512 or 576, following the existing sparse-attention path.
- Selection must be flat with Top-K in increments of 256 up to 2048. The SM90
  sparse attention score recompute limit is 2176 physical slots.
- CLI supports the GPT DSA stack. Hybrid layer layouts, MTP, dense warm-up,
  and pooled selection are unsupported.
- Recompute must be full/uniform with one layer per chunk. Each pipeline stage
  must begin on a computing layer, and groups cannot cross stage boundaries.
- `deterministic_mode` is unsupported: target replay checks do not establish
  deterministic model gradients for the existing atomic backward operations.
- CP and packed inputs retain DSA's existing layout checks. The synthetic CP=8
  query shape in the microbenchmark does not validate distributed execution.
  Qualify the intended distributed layout before production use.

Temporary targets are not checkpoint weights. Changing the scheduling, backend
or parallel layout requires separate validation of teacher delivery and state
lifetime. The reported evaluation and microbenchmark results do not replace a
multi-step integration check of the exact configuration being deployed.

## Validation

In an SM90 development environment with CUDA PyTorch, Transformer Engine, and
the normal Megatron training dependencies, install the released Frontend and
build the FlashMLA revision pinned in `uv.lock`. The standard image's FlashMLA
build disables SM90. These commands assume pip build tools including Ninja.

```bash
python -m pip install 'nvidia-cudnn-frontend==1.33.0' 'nvidia-cutlass-dsl==4.8.1' 'tilelang==0.1.12'
git clone https://github.com/deepseek-ai/FlashMLA.git /tmp/FlashMLA-cli
git -C /tmp/FlashMLA-cli checkout b7643bd54521f563b839b98289b5cd048c062ba2
git -C /tmp/FlashMLA-cli submodule update --init --recursive
env -u FLASH_MLA_DISABLE_SM90 FLASH_MLA_DISABLE_SM100=1 \
  python -m pip install --no-build-isolation --no-deps --force-reinstall /tmp/FlashMLA-cli
```

Run the focused tests, including teacher replay, from the Megatron checkout.

```bash
python -m torch.distributed.run --standalone --nproc-per-node 1 -m pytest -q -rs \
  tests/unit_tests/ops/test_dsa_cli.py \
  tests/unit_tests/determinism/kernels/test_dsa_cli.py
bash tests/functional_tests/test_cases/gpt/gpt_dsa_cli_sm90/run_smoke.sh
```

The SM90 tests should run without skips. They cover reverse-order backward with
shared packed metadata, summed KL under probability underflow, masking, and
bit-exact replay. The synthetic smoke test saves at iteration 2 and resumes to
iteration 4 with the same scheduler configuration.
