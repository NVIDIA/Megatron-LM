# Scaling protocol

How this repo plans to test architecture and optimizer changes across model
sizes without burning the cluster on uninformative runs. Pedagogic on purpose:
this doc assumes you have trained a transformer, but does not assume you have
read the scaling-laws literature.

## 1. What "scaling laws" actually are

Throughout this doc:

- **N** = number of model parameters (e.g. 350M, 1.3B). Counts the embedding
  table, the attention and MLP weight matrices, and the LayerNorms. Does not
  count optimizer state, activations, or gradients.
- **D** = number of training tokens seen during pre-training (each token is one
  position in the input sequence; one "sample" of length 2048 contains 2048
  tokens, so D is roughly batch_size * seq_len * num_steps).
- **C** = total training compute. For dense transformers the standard
  approximation is `C ~ 6 * N * D` FLOPs (Kaplan et al. 2020,
  https://arxiv.org/abs/2001.08361). The factor 6 is 2 for the forward and 4
  for the backward, summed over the dominant matmuls; embedding and attention
  contributions are small at the sizes we run.
- **L(N, D)** = test-set cross-entropy loss after training a model of size N on
  D tokens. The "scaling law" is the empirical observation that L(N, D) is
  smooth and well-fit by a parametric form.

The most-cited form (Hoffmann et al. 2022, "Chinchilla",
https://arxiv.org/abs/2203.15556):

```
L(N, D) = E + A / N^alpha + B / D^beta
```

`E` is the irreducible loss (data entropy you cannot beat), `A/N^alpha` is the
parameter-bottleneck term, `B/D^beta` is the data-bottleneck term. Fitted
exponents are roughly `alpha ~ 0.34` and `beta ~ 0.28` for dense decoder-only
transformers on web text.

The two predictions that matter for us:

1. **Compute-optimal allocation.** For a fixed compute budget C, the (N, D)
   pair that minimises L is on a curve with `D ~ 20 * N`. This is the
   "Chinchilla-optimal" recipe. Training a smaller model on more tokens, or a
   bigger model on fewer tokens, gives strictly worse loss per FLOP.
2. **Over-training is rational at deployment time.** If you intend to serve the
   model many times, inference cost dominates training cost. You then want the
   smallest N that hits your loss target, which means D far above 20*N.
   Llama-3 8B used D/N ~ 1875 (15T tokens / 8B params). This is the
   "over-trained" regime.

These are different regimes and they reward different architectural choices.
That is the whole point of this doc.

## 2. The current ladder, and the regime drift in it

Existing baseline table (from `README.md`):

| size | tokens | D/N |
| ---: | ---: | ---: |
| 350M | 15B | 43 |
| 760M | 30B | 39 |
| 1.3B | 100B | 77 |
| 2.7B | 300B | 110 |

D/N drifts from ~40 at 350M to ~110 at 2.7B. Neither end is Chinchilla-optimal
(20), neither end is fully over-trained (1000+). This is a problem: an
architecture that wins at D/N = 40 may lose at D/N = 110, because at higher
D/N the data term `B/D^beta` is small and the gain comes from anything that
helps the model exploit more passes over similar data (regularisation,
optimiser, tokenisation), whereas at D/N = 40 the data term still dominates
and the gain comes from anything that improves sample efficiency (better
init, better attention, better depth-vs-width).

**Decision needed: pick one D/N ratio and hold it across the ladder.** Two
defensible choices:

- **D/N = 20 (Chinchilla-optimal).** Cheapest, most comparable to the
  literature, most informative for "is this architecture better per FLOP."
  GPU-h cost shrinks roughly by D/N_current at each rung.
- **D/N = 100 (mildly over-trained).** Closer to current baselines and to
  deployable models. More expensive but matches the regime real production
  runs will operate in.

Recommendation: **D/N = 20 for the ablation ladder, D/N = 100+ for the one or
two candidates we promote to "production scaling".** Run the cheap regime to
pick winners, then re-run the winner at the deployable regime to confirm the
ranking holds. If it does not, that is itself a result worth a leaderboard
note.

## 3. Why a single 350M ablation is not enough

The 350m-ablation board ranks candidates at one (N, D) point. Two failure
modes:

- **Crossover.** Architecture A beats B at 350M, then B beats A at 1.3B. This
  is common for changes that affect signal propagation or optimiser
  conditioning, because the conditioning of the network at width 1024 is
  qualitatively different from at width 2048. Pre-LN vs Post-LN is the
  textbook example; deep Pre-LN suffers worse variance growth than wide
  Pre-LN, so the loss gap moves with depth.
- **Vanishing delta.** The win at 350M is real but its size shrinks with N at
  a faster rate than the noise floor, so by 1.3B it is below run-to-run
  variance. This is the "free at 350M, useless at scale" failure.

The cheapest robust check is a **3-point ladder**: run the candidate at three
sizes along the same D/N line and look at the loss-vs-N curve. If the gap
versus baseline is roughly constant or grows, the architecture transfers. If
it shrinks toward zero, you have a 350M-only win. If it inverts, you have a
crossover.

Three points are the minimum that lets you fit `L = E + A/N^alpha` for that
candidate (two free parameters fit two points exactly, leaving zero residual;
three points give one residual to detect a misfit).

## 4. The 3-point ladder, concretely

For each candidate architecture, fix D/N (call it 20 for the cheap regime)
and run:

| rung | N | D | est GPU-h |
| ---: | ---: | ---: | ---: |
| 1 | 350M | 7B | ~14 |
| 2 | 760M | 15B | ~58 |
| 3 | 1.3B | 26B | ~155 |

Total ~230 GPU-h per candidate at D/N = 20. (Costs scale with D/N: at D/N =
100 the same ladder is roughly 5x more expensive.)

Read the result by plotting `log(L - E_estimate)` vs `log(N)`. A straight
line means the candidate fits the same L(N) form as the baseline; the slope
is `alpha` and the intercept is `log(A)`. Compare:

- **Same alpha, lower A**: candidate is uniformly better at all N. Promote.
- **Steeper alpha (more negative slope), candidate gets better with N**:
  promote enthusiastically; this is what we want.
- **Shallower alpha, candidate gets worse with N**: 350M-only win, do not
  promote.
- **Crossover (lines cross)**: complicated; record but do not promote.

This is the cheap version of the methodology in Hoffmann et al. (Chinchilla
fits hundreds of runs) and is sufficient for ranking, not for forecasting
absolute loss at unseen N.

## 5. Picking which experiments to run (the "Spend Less, Fit Better" idea)

Sabbaqi et al. 2026, "Spend Less, Fit Better" (https://arxiv.org/abs/2604.22753)
points out that even a 3-point ladder over-spends if you already have prior
runs at nearby (N, D) points. They formalise the question "given a budget,
which (N, D) runs reduce uncertainty most in the target region (e.g. at 7B,
1T tokens)?" and show that uncertainty-aware sequential allocation reaches
near-full-set extrapolation accuracy at ~10% of the full budget.

For this repo the practical takeaway is short: **after a few candidates have
ladders, the next candidate may only need rung 3 (the largest)**, because the
small-N points are already constrained by the family of curves we have. Do
not run the full ladder by reflex once we have ~5 candidates with ladders.

## 6. Learning-rate transfer (and why μP is not a free lunch here)

The single most expensive hyperparameter at scale is the peak learning rate.
A standard ablation re-tuning of LR at 1.3B would dominate the budget.
Three options for transferring a small-scale-tuned LR up the ladder.

### 6a. μP (Maximal Update Parametrisation)

Yang and Hu 2021 (https://arxiv.org/abs/2011.14522) and follow-ups. The idea:
re-parametrise the model so that the optimal LR is independent of width.
Concretely you scale the initialisation, the per-layer learning rates, and
the attention softmax temperature so that activations, gradients, and updates
all stay O(1) as width grows. With μP, the LR you tune at width 256 transfers
exactly to width 8192.

**Caveat that matters here:** μP was derived for Adam-style optimisers, where
the update is `lr * m / (sqrt(v) + eps)`. The derivation assumes the second
moment v approximately rescales the gradient by the gradient's RMS, so the
effective per-coordinate step size is lr-times-O(1) and the analysis follows.
Muon's update is **spectral** (Newton-Schulz orthogonalisation of the
momentum matrix), so the effective per-coordinate step has different scaling
in width. μP-for-Adam does not directly apply.

References for "μP at scale is not bulletproof anyway":

- Lingle et al. 2025, "How to Set the Learning Rate for Large-Scale
  Pre-training" (https://arxiv.org/abs/2601.05049). Shows μTransfer breaks
  for MoE, depth scaling, weight decay, and long-token regimes. Argues for a
  "fitting paradigm" instead.

### 6b. Fitting paradigm (recommended for Muon)

Run a small LR sweep at each rung, fit the optimal-LR-vs-N curve, and
extrapolate. With Muon, the spectral update has its own width-scaling rule
(roughly matrix-LR ~ 1/sqrt(fan_in * fan_out) at fixed Newton-Schulz iters,
empirically modified by Aurora's row-rescale and NorMuon's per-row
second-moment), so the right move is to measure the curve rather than assume
a closed-form transfer.

Concretely at each ladder rung:

1. Pick three matrix-LRs spaced 0.5x, 1x, 2x around the predicted optimum.
2. Run all three at the cheap rung (350M, 7B tokens), one at the expensive
   rung. Fit a quadratic in log(LR) to find the minimum.
3. Use the rung-1 and rung-2 minima to predict rung-3, and verify with a
   single 1x rung-3 run.

The scalar group (embeddings, norms, biases) is on AdamW, so for that group
μP transfer applies. Set scalar-LR by μP, sweep matrix-LR by fitting.

### 6c. The Aurora-row-imbalance flag

Tilde's Aurora post (https://blog.tilderesearch.com/blog/aurora) documents
that Muon's polar update on tall matrices does not balance row L2 norms.
Some MLP rows (i.e. neurons) get persistently small updates, become "dead",
and stop contributing. The effect is mild at 350M, growing at 1.3B+. The
fork already logs `row_cv` (per-matrix coefficient of variation of row
norms) and optional `neuron_stats` (forward activation moments per neuron).
At each ladder rung, eyeball these:

- AdamW baseline: `row_cv` should be ~constant or weakly drifting.
- Aurora / NorMuon: `row_cv` should be near AdamW; if it climbs through
  training the row-rescale is not pulling its weight.
- Plain Muon: `row_cv` is expected to drift up; this is the pathology
  Aurora was designed to fix.

If a candidate's `row_cv` slope is materially worse than baseline, the
candidate may have a hidden Muon-interaction problem and the ladder result is
suspect.

## 7. MoE-specific scaling

Mixture-of-Experts replaces a single MLP block with E parallel "expert" MLPs
plus a router that sends each token to the K experts with the highest router
score. Total parameters = N_dense + (E - 1) * N_mlp_per_layer; **active**
parameters per token = N_dense + K * N_mlp_per_layer / E (roughly). Compute
per token scales with active parameters, not total. So an MoE gives you the
representation capacity of a big model at the FLOP cost of a small model,
provided the router does its job.

Three things that go wrong:

1. **Expert-batch fragmentation.** Per Noumena
   (https://noumena.com/research/0000-why-training-moes-is-hard/), the
   per-expert effective batch size is `T * K / E` where T is tokens per
   batch. As E grows without T or K growing, each expert's GEMM gets thinner
   and underutilised. At GH200 with T = 524288 (GBS=128, seq=4096), K=2,
   E=8: per-expert batch is 131072 tokens, fine. At E=64, K=2: 16384, the
   GEMM starts to leak utilisation. Plan E around your batch, not the other
   way around.
2. **Routing collapse under low-precision optimiser state.** FP8 / NVFP4
   training drops router gradients below the quantisation floor. Once a
   handful of experts are slightly favoured, others stop receiving useful
   gradient and starve. This is a stability failure, not a slowdown: the
   loss curve looks fine until it does not. Noumena's mitigations: μP-style
   embedding and logit rescale; disable global gradient clipping (it kills
   the router-update direction); a learnable scalar at each expert output as
   a "bungee" to keep gradients in range; aux-loss-free token-choice
   routing.
3. **Muon-on-experts interaction (open).** Newton-Schulz on a thin matrix
   (when each expert sees a small batch) sees fewer effective rows; the
   spectral update may behave differently. Untested at our scale.

For our entry into MoE, the proposed plumbing run is:

- N_active = 350M, E = 8, K = 2 (so total params ~ 1.5B, active ~ 350M).
- D/N_active = 20, so 7B tokens.
- Optimiser: Aurora on matrices (including expert MLPs), AdamW on scalars
  and the router.
- Logging additions (Section 8) on by default for this run.

This is a debugging run, not an ablation. The deliverable is "router stays
healthy, no collapse, all 8 experts utilised, throughput within 20% of
dense-350M". Once that holds, raise E and run an actual ladder.

## 8. Instrumentation we need before the next ladder

Already in `_research/logging_patch/`:

- per-step `train_loss`, `lr`, `grad_norm`, `params_norm`, `tput`, MFU
- per-block FP8-stability activation stats (amax, l2, frac_outlier, rms)
- top-1 next-token accuracy
- per-parameter `row_cv` (the Aurora row-imbalance probe)
- optional MLP per-neuron pre-activation stats

To add for the scaling/MoE work:

- **Smoothed slope of `row_cv` over last K steps** (so a regression check at
  scale is one number, not a plot). Cheap, append-only.
- **Router entropy per layer**, normalised by `log(E)` so values are in
  [0, 1]. Healthy aux-loss-free routing sits near 1.0; collapse drops it.
- **Expert utilisation per layer**: tokens routed to expert e, fraction over
  uniform 1/E. Min, mean, max, CV across experts.
- **Top-1 router-confidence histogram per layer** (10 bins). Saturated
  routing (most mass at top-1=1.0) signals a stuck router.
- **Effective per-expert batch size**: `T * K / E` recorded each step (it
  varies with token routing, so this is the actual not nominal).

All flag-gated, default off for dense runs, default on for MoE runs.

## 9. Decision summary

What is being adopted:

- Hold D/N constant across the ladder. **D/N = 20** for ablations,
  **D/N >= 100** for the candidate we promote.
- **3-point ladder** at 350M, 760M, 1.3B per candidate, ~230 GPU-h each at
  D/N = 20.
- **Fitting-paradigm LR** for Muon-family: 3-point matrix-LR sweep at each
  rung, scalar-LR via μP. Do not blindly trust μTransfer for matrix
  parameters.
- After ~5 candidates have ladders, switch to **uncertainty-aware
  allocation** (Sabbaqi et al. style); skip ladder rungs whose uncertainty
  is already constrained by the existing data.
- For MoE: a single non-ladder plumbing run first (350M-active, E=8, K=2,
  Aurora on matrices, AdamW on router). Ladder only after routing
  stability is confirmed.

What is open:

- The ranking is for sample efficiency at fixed D/N. Inference latency,
  KV-cache footprint, and parameter efficiency are separate axes; we are
  not currently scoring those.
- Muon-on-thin-experts has no measured behaviour at our scale; the MoE
  plumbing run will be the first datapoint.
- "How many candidates per quarter" is a budget question, not answered
  here.

## References

- Kaplan et al. 2020, "Scaling Laws for Neural Language Models",
  https://arxiv.org/abs/2001.08361
- Hoffmann et al. 2022, "Training Compute-Optimal Large Language Models"
  (Chinchilla), https://arxiv.org/abs/2203.15556
- Yang and Hu 2021, "Tensor Programs IV: Feature Learning in
  Infinite-Width Neural Networks" (μP),
  https://arxiv.org/abs/2011.14522
- Yang et al. 2022, "Tensor Programs V: Tuning Large Neural Networks via
  Zero-Shot Hyperparameter Transfer" (μTransfer),
  https://arxiv.org/abs/2203.03466
- Lingle et al. 2026, "How to Set the Learning Rate for Large-Scale
  Pre-training", https://arxiv.org/abs/2601.05049
- Sabbaqi et al. 2026, "Spend Less, Fit Better",
  https://arxiv.org/abs/2604.22753
- Jordan 2024, "Muon: An optimizer for hidden layers in neural networks",
  https://kellerjordan.github.io/posts/muon/
- Tilde Research 2025, "Aurora: a Muon-derivative that fixes row
  imbalance", https://blog.tilderesearch.com/blog/aurora
- Noumena 2025, "Why training MoEs is hard",
  https://noumena.com/research/0000-why-training-moes-is-hard/
- Fedus et al. 2022, "Switch Transformer: Scaling to Trillion Parameter
  Models with Simple and Efficient Sparsity",
  https://arxiv.org/abs/2101.03961
- Muennighoff et al. 2024, "OLMoE: Open Mixture-of-Experts Language
  Models", https://arxiv.org/abs/2409.02060
