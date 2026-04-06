# CP-MoE Experiments

## Notation

All formulas in this document use the following symbols consistently. Full definitions are in `CLAUDE.md` — brief reference here:

| Symbol | Meaning |
|---|---|
| `r_{i,t}` | Raw router logit for expert `i`, token `t`. `r_{i,t} = (W_r x_t)_i`. Unbounded scalar. |
| `s_{i,t}` | Soft routing probability. `s_{i,t} = softmax(r_{·,t})_i ∈ (0,1)`. Differentiable surrogate for hard dispatch. |
| `λ_i` | Dual variable / shadow price for expert `i`. Non-negative scalar, updated via tâtonnement. |
| `usage_i` | Hard dispatch count: number of tokens routed to expert `i` in the current batch. Non-differentiable. |
| `Σ_t s_{i,t}` | Soft usage surrogate: differentiable approximation of `usage_i`. Used in loss computation. |
| `C_i` | Expert capacity (`T/E * capacity_factor`). |
| `α` | Capacity slack. Target: `usage_i ≤ α * C_i`. |

`r_{i,t}` and `s_{i,t}` carry the same information — `s_{i,t} = softmax(r_{i,t})` — and produce the same top-k ranking. `s_{i,t}` is used in the loss because it has a meaningful unit (probability / expected count); `r_{i,t}` is used in the routing offset because it is the natural logit space for top-k selection.

---

## Overview

Two experiments. The first validates the core framework claim. The second answers the fundamental question about whether load balancing is a learned inductive bias or a persistent structural requirement.

---

## Experiment 1: Four-Way Ablation

### Purpose

Compare four routing/balancing configurations on the same small MoE LM to isolate the contribution of each component of the CP-MoE framework.

### Ablation Design

| ID | Name | Routing | Balancing Loss | λ update |
|---|---|---|---|---|
| A | No Balancing | `top-k(r_{i,t})` | none | none |
| B | Fixed Aux Loss | `top-k(r_{i,t})` | GShard: `α * Σ f_i P_i` | none |
| C | CP-MoE Loss Only | `top-k(r_{i,t})` | `Σ λ_i · Σ s_{i,t}` | tâtonnement |
| D | CP-MoE Full | `top-k(r_{i,t} - λ_i)` | `Σ λ_i · Σ s_{i,t}` | tâtonnement |

The comparison B vs C isolates **adaptive λ vs fixed α** (same routing, same loss structure, different coefficient). The comparison C vs D isolates **routing offset** (same loss, different dispatch mechanism).

### Model

Small MoE transformer trainable on a single A100 in under 24 hours:

- 6 layers, 512 hidden dim, 8 heads
- 8 experts per MoE layer, top-2 routing
- FFN dim: 2048 per expert
- ~150M total parameters, ~40M activated per token
- Dataset: wikitext-103, GPT-2 tokenizer
- Training: 50k steps, batch 128, seq 512 (~3.3B tokens)

### Primary Research Questions

**RQ1 (B vs C):** Does adaptive λ provide more stable expert utilization than fixed α, with less sensitivity to the coefficient hyperparameter?

**RQ2 (C vs D):** Does the routing offset provide token-level heterogeneity beyond what the loss term alone achieves?

**RQ3 (A as baseline):** How quickly does routing collapse without any balancing? What is the Gini coefficient floor for each method?

### Metrics and Logging

Log every 50 steps to WandB. Primary curves to produce:

**Curve 1 — Expert Gini over training steps (all 4 ablations)**
The main quantitative result. Shows convergence speed and final balance level for each method.

**Curve 2 — Per-expert λ convergence (Ablations C and D)**
Shows that λ converges to stable per-expert values. Experts with structurally higher demand will have higher converged λ*. This is a core theoretical prediction.

**Curve 3 — LM loss over training steps (all 4 ablations)**
Checks that balancing does not significantly hurt language modeling performance. If C/D have meaningfully higher LM loss than A, the balancing penalty is too strong.

**Curve 4 — Routing entropy over training steps (all 4 ablations)**
Complementary to Gini. Higher entropy = more uniform routing.

### Expected Results

| Comparison | Expected finding | Reasoning |
|---|---|---|
| A (no balancing) | Gini increases rapidly, expert collapse within ~5k steps | No incentive to distribute |
| B vs A | Gini stays low, slight LM loss penalty | Standard result, establishes baseline |
| C vs B | Similar final Gini, but C more stable across seeds / less sensitive to α_coeff | Adaptive λ vs proportional control |
| D vs C | Lower Gini than C at same λ strength; routing entropy higher per token type | Routing offset adds token-level discrimination |

If C and B are indistinguishable: the adaptive coefficient adds no practical benefit; the framework's main contribution reduces to the routing offset in D.

If D and C are indistinguishable: the routing offset has no measurable effect beyond the loss term; the framework reduces to adaptive aux loss.

Both null results are publishable — they sharpen the understanding of what each component contributes.

### Hyperparameter Sensitivity Analysis (optional, after main ablation)

For Ablation B: sweep `α_coeff ∈ {0.001, 0.01, 0.1}`. For Ablation C/D: sweep `η ∈ {0.001, 0.01, 0.1}`. The claim is that C/D are less sensitive to their hyperparameter than B is to `α_coeff`. Measure variance in final Gini across sweep values.

---

## Experiment 2: Lambda Inference Ablation

### Purpose

Answer the fundamental question: after training with CP-MoE Full (Ablation D), does the router depend on λ at inference time, or has it internalized the price signal?

This directly probes whether load balancing should be understood as a **learned inductive bias** (router adapts its weights to be naturally balanced) or a **persistent structural constraint** (router always needs external price signal to stay balanced).

### Setup

Use the best checkpoint from Ablation D (lowest validation LM loss). Extract the converged λ* from `price_mech.state_dict()`.

Evaluate on the validation set under two conditions:

**Condition 1 — λ = λ* (training prices)**
Set all MoE layers' λ to the converged training values. Run standard inference.

**Condition 2 — λ = 0 (prices removed)**
Set all MoE layers' λ to zero. Run standard inference.

### Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| Routing KL | KL(P_{λ*} ‖ P_{λ=0}) per layer | How much routing changes when λ is removed |
| Gini (λ*) | Gini of dispatch counts under λ* | Balance with prices |
| Gini (λ=0) | Gini of dispatch counts under λ=0 | Balance without prices |
| PPL (λ*) | Perplexity under λ* | Baseline performance |
| PPL (λ=0) | Perplexity under λ=0 | Performance without prices |
| PPL delta | PPL(λ=0) − PPL(λ*) | Cost of removing prices |

### Interpretation Framework

**Case 1: Small KL + small PPL delta (< 0.5 PPL points)**

Router internalized the price signal. Load balancing is a learned inductive bias. λ is redundant at inference. The framework is analogous to dropout: pressure applied during training, removed at inference, but the learned weights are more robust.

This is the **stronger result** for the paper. It means the model genuinely learned to route more uniformly, not just that it is forced to at runtime.

**Case 2: Large KL or large PPL delta (> 1.0 PPL points)**

Router still depends on λ at inference. The price signal encodes structural information about the data distribution that the router's weights alone cannot capture. λ* should be retained as a fixed inference-time parameter (zero storage overhead, just a vector of E scalars).

This is also a valid and interesting result: it means the converged λ* contains meaningful information about which expert domains are structurally over-demanded in the training data.

**Case 3: Small KL but large PPL delta**

Routing distribution looks similar but model performance degrades. This would suggest the routing offset is important for expert output weighting (via gates), not just dispatch. Worth investigating per-expert gate weight distributions.

### Per-Layer Analysis

Run the KL and Gini analysis per MoE layer, not just aggregated. Early layers and late layers may differ significantly in their dependence on λ. This granularity is cheap to add and provides richer analysis.

### Connection to Training Dynamics

After the inference ablation, produce a scatter plot:
- x-axis: converged λ_i* value for each expert
- y-axis: Δ routing frequency when λ removed (how much that expert's usage changes)

If the relationship is monotone (high λ* experts shift the most routing when removed), this confirms that λ* correctly identified the structurally over-demanded experts.

---

## Running the Experiments

### Ablation Study

```bash
# All 4 ablations × 3 seeds
bash experiments/run_ablation.sh

# Single ablation (for debugging)
python -m src.training.trainer --config configs/ablation_D.yaml training.seed=42
```

### Inference Ablation

Run after Ablation D has completed:

```bash
bash experiments/run_inference_ablation.sh
```

### Analysis Plots

```bash
# After all runs complete
python analysis/plot_utilization.py   --runs_dir checkpoints/ --output figures/gini.pdf
python analysis/plot_lambda.py        --runs_dir checkpoints/ --output figures/lambda.pdf
python analysis/plot_routing_drift.py --runs_dir checkpoints/ --output figures/drift.pdf
```

---

## Compute Budget

| Experiment | Runs | Est. time (A100) | Total |
|---|---|---|---|
| Ablation A–D × 3 seeds | 12 | ~6 hrs each | ~3 days |
| Inference ablation | 3 | ~1 hr each | ~3 hrs |
| Hyperparameter sweep (optional) | 18 | ~6 hrs each | ~4.5 days |

All experiments fit on a **single A100 80GB**. Ablations can be parallelized across multiple GPUs if available.

---

## Failure Modes to Watch

**λ oscillation:** If λ oscillates rather than converging (check `lambda_norm` curve), increase EMA β (e.g., 0.999) or decrease η (e.g., 0.001).

**LM loss significantly worse in C/D vs A:** λ values are too large, over-penalizing. Decrease η or increase `lambda_max` threshold.

**No difference between B and C:** Adaptive λ adds no benefit over fixed α at this scale. Consider testing at larger scale (more experts, longer training) where the benefit of per-expert adaptation is more pronounced.

**Expert collapse in D but not C:** The routing offset is destabilizing training. Try annealing the routing offset (reduce λ contribution to routing logits over time) while keeping the loss term.
