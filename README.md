# Capacity-Priced MoE — Project Guide

## Project Goal

Implement and empirically validate the **Capacity-Priced Mixture-of-Experts (CP-MoE)** routing framework. The core claim: replacing fixed auxiliary loss coefficients with adaptive per-expert dual variables (λ) updated via tâtonnement provides more stable load balancing and requires less hyperparameter tuning than standard approaches.

This is a **research validation project**. The primary deliverable is a clean four-way ablation study on a small MoE language model, producing expert utilization curves, λ convergence curves, and a λ=0 vs λ=λ* inference comparison.

---

## Theoretical Background

### Primal-Dual Formulation

The framework reformulates MoE load balancing as a constrained optimization problem:

```
min_θ  L_LM(θ)
s.t.   E[usage_i(θ)] ≤ α * C_i   ∀ i
```

The Lagrangian introduces per-expert dual variables λ_i ≥ 0:

```
L(θ, λ) = L_LM(θ) + Σ_i λ_i * (E[usage_i(θ)] - α * C_i)
```

**Primal step** (update θ, hold λ fixed):
```
θ ← θ - η_θ * (∇_θ L_LM + Σ_i λ_i ∇_θ E[usage_i])
```

**Dual step** (gradient ascent on dual — this is the tâtonnement rule):
```
λ_i ← max(0, λ_i + η_λ * (usage_i - α * C_i))
```

The `max(0,·)` projection enforces dual feasibility (KKT complementary slackness): λ_i > 0 only when the capacity constraint is active.

### Notation Reference

All symbols used throughout this document and the codebase:

| Symbol | Shape | Definition | Code name |
|---|---|---|---|
| `x_t` | `[d_model]` | Hidden state of token `t` from the previous layer | `x` |
| `W_r` | `[E, d_model]` | Learned router linear projection | `self.W_r` |
| `r_{i,t}` | scalar | **Raw router logit**: unnormalized score of expert `i` for token `t`. `r_{i,t} = (W_r x_t)_i`. Unbounded, scale depends on weight magnitude. | `affinity[t, i]` |
| `s_{i,t}` | scalar ∈ (0,1) | **Soft routing probability**: `s_{i,t} = softmax(r_{·,t})_i = exp(r_{i,t}) / Σ_j exp(r_{j,t})`. Differentiable surrogate for the hard dispatch indicator. Sums to 1 over experts for each token. | `s_it[t, i]` |
| `λ_i` | scalar ≥ 0 | **Dual variable / shadow price** for expert `i`. External state, not `nn.Parameter`. Updated via tâtonnement after each optimizer step. | `price_mech.lambda_[i]` |
| `C_i` | scalar | **Expert capacity**: maximum tokens per expert per batch. `C_i = (T / E) * capacity_factor`. | `price_mech.capacity` |
| `α` | scalar ∈ (0,1) | **Capacity slack**: target usage is `α * C_i`, not `C_i`. Leaves headroom to avoid hard token dropping. | `price_mech.alpha` |
| `η_λ` | scalar | **Dual learning rate**: step size for the tâtonnement update. | `price_mech.eta` |
| `T` | int | Number of tokens in the current batch (`batch_size * seq_len`). | `T` |
| `E` | int | Total number of experts per MoE layer. | `num_experts` |
| `k` | int | Top-k: number of experts activated per token. | `top_k` |

**Key distinction: `r_{i,t}` vs `s_{i,t}`**

`r_{i,t}` and `s_{i,t}` carry the same information — `s_{i,t}` is just `r_{i,t}` passed through softmax. They are related by a monotone transformation, so top-k selection on either produces the same ranking. The distinction matters for two reasons:

1. **Loss surrogate**: `Σ_t s_{i,t}` is used (not `Σ_t r_{i,t}`) because it approximates `usage_i` as a probability-weighted count, interpretable as "expected tokens assigned to expert `i`". Raw logit sums have no meaningful unit.
2. **Routing offset scale**: subtracting `λ_i` from `r_{i,t}` is scale-dependent (logit magnitude varies with training). See the scale discussion in the Routing Offset section.

---

### Three Perspectives on λ

| Perspective | Interpretation of λ |
|---|---|
| Market / Tâtonnement | Shadow price of expert capacity |
| Primal-Dual Optimization | Lagrange multiplier for capacity constraint |
| Control Theory | Integral controller on expert load |

All three describe the same update rule. λ acts as an **integral controller**: it accumulates excess demand over time, responding to sustained imbalance rather than instantaneous fluctuations. Fixed α (standard aux loss) is a **proportional controller** — it applies the same pressure every step regardless of history.

### Differentiable Surrogate for Usage

`usage_i = Σ_t 1[token t → expert i]` is non-differentiable (argmax / indicator). Substitute a soft surrogate:

```
usage_i ≈ Σ_t s_{i,t},   where s_{i,t} = softmax(r_{i,t})_i
```

This recovers the standard auxiliary loss structure but with **adaptive per-expert coefficients** λ_i derived from first principles rather than tuned as hyperparameters.

### Training Objective (at each step, λ held fixed)

```
L(θ) = L_LM(θ) + Σ_i λ_i · Σ_t s_{i,t}
```

where `s_{i,t} = softmax(affinity)_i` uses **pre-offset** router logits. λ is always **detached** — it is a fixed coefficient for the current step, not a learnable parameter.

### Routing Offset (Ablation D only)

In addition to the loss term, CP-MoE Full introduces a token-level price signal at dispatch time:

```
top-k(r_{i,t} - λ_i)
```

This provides token-level heterogeneity that standard aux loss cannot: high-affinity tokens (large `r_{i,t}`) can still access congested experts because `r_{i,t} - λ_i` remains large; marginal tokens (small `r_{i,t}`) are priced out because `r_{i,t} - λ_i` goes negative. λ is **detached** here — it affects dispatch but receives no gradients through the routing path.

**Scale note.** Subtracting `λ_i` from raw logit `r_{i,t}` is scale-dependent: the effective strength of the price signal depends on the current magnitude of `r_{i,t}`, which changes throughout training. A more scale-robust alternative is to normalize by the running standard deviation of router logits:

```
top-k(r_{i,t} - σ_r · λ_i)
```

where `σ_r = std(r_{·,·})` over the current batch. This keeps `λ_i` in units commensurate with the logit distribution. Whether this matters empirically is an ablation worth running.

Note: using `log(s_{i,t}) - λ_i = r_{i,t} - log(Z_t) - λ_i` (log-probability space) is equivalent to the raw offset up to a token-level constant `log(Z_t)` that cancels in top-k ranking — so it is not more scale-agnostic in practice.

**Critical design note for Ablation D:** The loss surrogate always uses `softmax(affinity)` — the pre-offset logits `r_{i,t}` — not `softmax(affinity - λ)`. Using post-offset logits would double-count λ's effect and artificially suppress the usage estimate for already-penalized experts, making the gradient signal unreliable.

---

## Repository Structure

```
capacity-priced-moe/
├── CLAUDE.md                     ← This file (primary Claude Code guide)
├── EXPERIMENTS.md                ← Detailed experiment specs and expected results
├── requirements.txt
├── configs/
│   ├── base_moe.yaml             ← Shared model + training hyperparameters
│   ├── ablation_A.yaml           ← No balancing
│   ├── ablation_B.yaml           ← Fixed aux loss (GShard-style)
│   ├── ablation_C.yaml           ← CP-MoE loss only (no routing offset)
│   └── ablation_D.yaml           ← CP-MoE full (loss + routing offset)
├── src/
│   ├── model/
│   │   ├── price_mechanism.py    ← PriceMechanism: λ state + tâtonnement update
│   │   ├── router.py             ← TopKRouter: standard and price-adjusted modes
│   │   ├── moe_layer.py          ← MoE FFN layer (dispatch + combine)
│   │   └── transformer.py        ← Small transformer with MoE layers
│   ├── training/
│   │   ├── trainer.py            ← Main training loop (primal-dual update order)
│   │   ├── losses.py             ← Loss variants: none / fixed_aux / cpmoe
│   │   └── optimizer.py          ← AdamW setup
│   ├── evaluation/
│   │   ├── metrics.py            ← Gini, routing entropy, KL divergence
│   │   └── inference_ablation.py ← λ=0 vs λ=λ* comparison
│   └── utils/
│       ├── data.py               ← Dataset + dataloader (wikitext-103)
│       └── logging.py            ← WandB logging helpers
├── experiments/
│   ├── run_ablation.sh           ← Runs all 4 ablations × 3 seeds
│   └── run_inference_ablation.sh ← λ* vs λ=0 routing comparison
└── analysis/
    ├── plot_utilization.py       ← Expert utilization Gini curves over training
    ├── plot_lambda.py            ← Per-expert λ convergence curves
    └── plot_routing_drift.py     ← KL divergence P(λ*) vs P(λ=0)
```

---

## Core Components

### PriceMechanism (`src/model/price_mechanism.py`)

Maintains per-expert dual variables λ as **external state** — never a `nn.Parameter`. All updates use `torch.no_grad()` and are called **after** `optimizer.step()`.

Key fields:
- `lambda_`: `[E]` float tensor, non-negative, initialized to zero
- `_ema_usage`: `[E]` EMA-smoothed usage for noise reduction

Update rule:
```
û_i  ← β * û_{i-1} + (1-β) * usage_i       # EMA smooth (β=0.99)
λ_i  ← clamp(λ_i + η * (û_i - α*C_i), 0, λ_max)
```

The `get_lambda()` method returns `lambda_.detach()` — always safe to use in forward passes.

Must be saved alongside model weights in checkpoints (needed for the inference ablation).

### TopKRouter (`src/model/router.py`)

Two modes controlled by config field `routing_mode`:

**`standard`** — Ablations A, B, C:
```
affinity           = W_r(x)                  # [T, E] learned scores
gates, indices     = top-k(softmax(affinity))
```

**`price`** — Ablation D:
```
affinity           = W_r(x)                  # [T, E]
adjusted           = affinity - lambda_.detach()
gates, indices     = top-k(softmax(adjusted))
```

The router exposes both `affinity` (pre-offset) and `adjusted_logits` to the caller. The loss always uses `affinity`.

### Loss Functions (`src/training/losses.py`)

| Ablation | `loss_mode` | Formula |
|---|---|---|
| A | `none` | `L_LM` |
| B | `fixed_aux` | `L_LM + α_coeff * Σ_i f_i * P_i` |
| C | `cpmoe` | `L_LM + Σ_i λ_i * Σ_t softmax(affinity)_{i,t}` |
| D | `cpmoe` | same as C, but routing uses `affinity - λ` |

For B: `f_i` is hard dispatch fraction (stop-gradient), `P_i` is mean soft router probability. For C/D: `λ_i = price_mech.get_lambda()` (detached).

Gradient path for C/D:
```
λ_i (const) * Σ_t s_{i,t}  →  softmax(affinity)  →  W_r  →  θ
```

### Training Loop (`src/training/trainer.py`)

The primal-dual update order must be preserved exactly:

```
1. forward pass        → gates, indices, affinity, lm_logits
2. compute loss        → L_LM + balancing term (λ detached)
3. loss.backward()     → ∇_θ L
4. optimizer.step()    → primal step: update θ
5. price_mech.update(hard_usage_counts)   → dual step: update λ
6. log metrics
```

Step 5 uses **hard dispatch counts** from `router.compute_usage(indices)`, not the soft surrogate. The soft surrogate is only for differentiability in step 2.

---

## Configuration Reference

### `configs/base_moe.yaml`

```yaml
model:
  d_model: 512
  n_layers: 6
  n_heads: 8
  n_experts: 8
  top_k: 2
  ffn_dim: 2048
  vocab_size: 32000

training:
  batch_size: 128
  seq_len: 512
  max_steps: 50000        # ~3.3B tokens with above settings
  lr: 3.0e-4
  warmup_steps: 2000
  grad_clip: 1.0
  eval_interval: 500
  log_interval: 50
  seeds: [42, 43, 44]

data:
  dataset: wikitext-103
  tokenizer: gpt2

balancing:
  capacity_factor: 1.0   # C_i = (T/E) * capacity_factor tokens/expert
  alpha: 0.5             # target: usage_i ≤ α * C_i
  eta: 0.01              # λ learning rate (dual step size)
  ema_beta: 0.99         # EMA smoothing for usage statistics
  lambda_max: 10.0       # λ clipping (stability during early training)
  alpha_coeff: 0.01      # fixed coefficient for Ablation B only
```

### Ablation Configs (override two fields each)

```yaml
# ablation_A.yaml
routing_mode: standard
loss_mode: none

# ablation_B.yaml
routing_mode: standard
loss_mode: fixed_aux

# ablation_C.yaml
routing_mode: standard
loss_mode: cpmoe

# ablation_D.yaml
routing_mode: price
loss_mode: cpmoe
```

---

## Metrics Reference

### Training Metrics (logged every `log_interval` steps)

| Metric | Description | Good sign |
|---|---|---|
| `lm_loss` | Language modeling cross-entropy | Decreasing |
| `expert_gini` | Gini coefficient of dispatch counts | Decreasing (→ balanced) |
| `routing_entropy` | Shannon entropy of routing distribution | Increasing (→ uniform) |
| `lambda_norm` | L2 norm of λ vector | Plateaus (→ converged) |
| `lambda/exp...

[//]: # (Megatron-LM and Megatron Core)

[//]: # (=============================)

[//]: # ()
[//]: # (<h4>GPU-optimized library for training transformer models at scale</h4>)

[//]: # ()
[//]: # ([![Documentation]&#40;https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat&#41;]&#40;https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html&#41;)

[//]: # ([![version]&#40;https://img.shields.io/badge/release-0.15.0-green&#41;]&#40;./CHANGELOG.md&#41;)

[//]: # ([![license]&#40;https://img.shields.io/badge/license-Apache-blue&#41;]&#40;./LICENSE&#41;)

[//]: # ()
[//]: # (<div align="left">)

[//]: # ()
[//]: # (## About)

[//]: # ()
[//]: # (This repository contains two components: **Megatron-LM** and **Megatron Core**.)

[//]: # ()
[//]: # (**Megatron-LM** is a reference example that includes Megatron Core plus pre-configured training scripts. Best for research teams, learning distributed training, and quick experimentation.)

[//]: # ()
[//]: # (**Megatron Core** is a composable library with GPU-optimized building blocks for custom training frameworks. It provides transformer building blocks, advanced parallelism strategies &#40;TP, PP, DP, EP, CP&#41;, mixed precision support &#40;FP16, BF16, FP8, FP4&#41;, and model architectures. Best for framework developers and ML engineers building custom training pipelines.)

[//]: # ()
[//]: # (**[Megatron Bridge]&#40;https://github.com/NVIDIA-NeMo/Megatron-Bridge&#41;** provides bidirectional Hugging Face ↔ Megatron checkpoint conversion with production-ready recipes.)

[//]: # ()
[//]: # (## Getting Started)

[//]: # ()
[//]: # (**Install from PyPI:**)

[//]: # ()
[//]: # (```bash)

[//]: # (uv pip install megatron-core)

[//]: # (```)

[//]: # ()
[//]: # (**Or clone and install from source:**)

[//]: # ()
[//]: # (```bash)

[//]: # (git clone https://github.com/NVIDIA/Megatron-LM.git)

[//]: # (cd Megatron-LM)

[//]: # (uv pip install -e .)

[//]: # (```)

[//]: # ()
[//]: # (> **Note:** Building from source can use a lot of memory. If the build runs out of memory, limit parallel compilation jobs by setting `MAX_JOBS` &#40;e.g. `MAX_JOBS=4 uv pip install -e .`&#41;.)

[//]: # ()
[//]: # (For NGC container setup and all installation options, see the **[Installation Guide]&#40;https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/install.html&#41;**.)

[//]: # ()
[//]: # (- **[Your First Training Run]&#40;https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html&#41;** - End-to-end training examples with data preparation)

[//]: # (- **[Parallelism Strategies]&#40;https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html&#41;** - Scale training across GPUs with TP, PP, DP, EP, and CP)

[//]: # (- **[Contribution Guide]&#40;https://docs.nvidia.com/megatron-core/developer-guide/latest/developer/contribute.html&#41;** - How to contribute to Megatron Core)

[//]: # ()
[//]: # (# Latest News)

[//]: # ()
[//]: # (- **[2026/03]** **Deprecating Python 3.10 support:** We're officially dropping Python 3.10 support with the upcoming 0.17.0 release. Downstream applications must raise their lower boundary to 3.12 to stay compatible with MCore.)

[//]: # (- **[2026/01]** **[Dynamic Context Parallelism]&#40;https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/&#41;** - Up to 1.48x speedup for variable-length sequence training with adaptive CP sizing.)

[//]: # (- **[2025/12]** **Megatron Core development has moved to GitHub!** All development and CI now happens in the open. We welcome community contributions.)

[//]: # (- **[2025/10]** **[Megatron Dev Branch]&#40;https://github.com/NVIDIA/Megatron-LM/tree/dev&#41;** - early access branch with experimental features.)

[//]: # (- **[2025/10]** **[Megatron Bridge]&#40;https://github.com/NVIDIA-NeMo/Megatron-Bridge&#41;** - Bidirectional converter for interoperability between Hugging Face and Megatron checkpoints, featuring production-ready recipes for popular models.)

[//]: # (- **[2025/08]** **[MoE Q3-Q4 2025 Roadmap]&#40;https://github.com/NVIDIA/Megatron-LM/issues/1729&#41;** - Comprehensive roadmap for MoE features including DeepSeek-V3, Qwen3, advanced parallelism strategies, FP8 optimizations, and Blackwell performance enhancements.)

[//]: # (- **[2025/08]** **[GPT-OSS Model]&#40;https://github.com/NVIDIA/Megatron-LM/issues/1739&#41;** - Advanced features including YaRN RoPE scaling, attention sinks, and custom activation functions are being integrated into Megatron Core.)

[//]: # (- **[2025/06]** **[Megatron MoE Model Zoo]&#40;https://github.com/yanring/Megatron-MoE-ModelZoo&#41;** - Best practices and optimized configurations for training DeepSeek-V3, Mixtral, and Qwen3 MoE models with performance benchmarking and checkpoint conversion tools.)

[//]: # (- **[2025/05]** Megatron Core v0.11.0 brings new capabilities for multi-data center LLM training &#40;[blog]&#40;https://developer.nvidia.com/blog/turbocharge-llm-training-across-long-haul-data-center-networks-with-nvidia-nemo-framework/&#41;&#41;.)

[//]: # ()
[//]: # (<details>)

[//]: # (<summary>Previous News</summary>)

[//]: # ()
[//]: # (- **[2024/07]** Megatron Core v0.7 improves scalability and training resiliency and adds support for multimodal training &#40;[blog]&#40;https://developer.nvidia.com/blog/train-generative-ai-models-more-efficiently-with-new-nvidia-Megatron-Core-functionalities/&#41;&#41;.)

[//]: # (- **[2024/06]** Megatron Core added supports for Mamba-based models. Check out our paper [An Empirical Study of Mamba-based Language Models]&#40;https://arxiv.org/pdf/2406.07887&#41; and [code example]&#40;https://github.com/NVIDIA/Megatron-LM/tree/ssm/examples/mamba&#41;.)

[//]: # (- **[2024/01 Announcement]** NVIDIA has released the core capabilities in **Megatron-LM** into [**Megatron Core**]&#40;https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core&#41; in this repository. Megatron Core expands upon Megatron-LM's GPU-optimized techniques with more cutting-edge innovations on system-level optimizations, featuring composable and modular APIs.)

[//]: # ()
[//]: # (</details>)

[//]: # ()
[//]: # (# Project Structure)

[//]: # ()
[//]: # (```)

[//]: # (Megatron-LM/)

[//]: # (├── megatron/)

[//]: # (│   ├── core/                    # Megatron Core &#40;kernels, parallelism, building blocks&#41;)

[//]: # (│   │   ├── models/              # Transformer models)

[//]: # (│   │   ├── transformer/         # Transformer building blocks)

[//]: # (│   │   ├── tensor_parallel/     # Tensor parallelism)

[//]: # (│   │   ├── pipeline_parallel/   # Pipeline parallelism)

[//]: # (│   │   ├── distributed/         # Distributed training &#40;FSDP, DDP&#41;)

[//]: # (│   │   ├── optimizer/           # Optimizers)

[//]: # (│   │   ├── datasets/            # Dataset loaders)

[//]: # (│   │   ├── inference/           # Inference engines and server)

[//]: # (│   │   └── export/              # Model export &#40;e.g. TensorRT-LLM&#41;)

[//]: # (│   ├── training/                # Training scripts)

[//]: # (│   ├── legacy/                  # Legacy components)

[//]: # (│   ├── post_training/           # Post-training &#40;quantization, distillation, pruning, etc.&#41;)

[//]: # (│   └── rl/                      # Reinforcement learning &#40;RLHF, etc.&#41;)

[//]: # (├── examples/                    # Ready-to-use training examples)

[//]: # (├── tools/                       # Utility tools)

[//]: # (├── tests/                       # Comprehensive test suite)

[//]: # (└── docs/                        # Documentation)

[//]: # (```)

[//]: # ()
[//]: # (# Performance Benchmarking)

[//]: # ()
[//]: # (For our latest performance benchmarking results, please refer to [NVIDIA Megatron Bridge Performance Summary]&#40;https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html&#41;.)

[//]: # ()
[//]: # (Our codebase efficiently trains models from 2B to 462B parameters across thousands of GPUs, achieving up to **47% Model FLOP Utilization &#40;MFU&#41;** on H100 clusters.)

[//]: # ()
[//]: # (![Model table]&#40;images/model_table.png&#41;)

[//]: # ()
[//]: # (**Benchmark Configuration:**)

[//]: # ()
[//]: # (- **Vocabulary size**: 131,072 tokens)

[//]: # (- **Sequence length**: 4096 tokens)

[//]: # (- **Model scaling**: Varied hidden size, attention heads, and layers to achieve target parameter counts)

[//]: # (- **Communication optimizations**: Fine-grained overlapping with DP &#40;`--overlap-grad-reduce`, `--overlap-param-gather`&#41;, TP &#40;`--tp-comm-overlap`&#41;, and PP &#40;enabled by default&#41;)

[//]: # ()
[//]: # (**Key Results:**)

[//]: # ()
[//]: # (- **6144 H100 GPUs**: Successfully benchmarked 462B parameter model training)

[//]: # (- **Superlinear scaling**: MFU increases from 41% to 47-48% with model size)

[//]: # (- **End-to-end measurement**: Throughputs include all operations &#40;data loading, optimizer steps, communication, logging&#41;)

[//]: # (- **Production ready**: Full training pipeline with checkpointing and fault tolerance)

[//]: # (- *Note: Performance results measured without training to convergence*)

[//]: # ()
[//]: # (## Weak Scaling Results)

[//]: # ()
[//]: # (Our weak scaled results show superlinear scaling &#40;MFU increases from 41% for the smallest model considered to 47-48% for the largest models&#41;; this is because larger GEMMs have higher arithmetic intensity and are consequently more efficient to execute.)

[//]: # ()
[//]: # (![Weak scaling]&#40;images/weak_scaling.png&#41;)

[//]: # ()
[//]: # (## Strong Scaling Results)

[//]: # ()
[//]: # (We also strong scaled the standard GPT-3 model &#40;our version has slightly more than 175 billion parameters due to larger vocabulary size&#41; from 96 H100 GPUs to 4608 GPUs, using the same batch size of 1152 sequences throughout. Communication becomes more exposed at larger scale, leading to a reduction in MFU from 47% to 42%.)

[//]: # ()
[//]: # (![Strong scaling]&#40;images/strong_scaling.png&#41;)

[//]: # ()
[//]: # (# Roadmaps)

[//]: # ()
[//]: # (- **[MoE Roadmap]&#40;https://github.com/NVIDIA/Megatron-LM/issues/1729&#41;** - DeepSeek-V3, Qwen3, advanced parallelism, FP8 optimizations, and Blackwell enhancements)

[//]: # ()
[//]: # (# Resources)

[//]: # ()
[//]: # (## Getting Help)

[//]: # ()
[//]: # (- 📖 **[Documentation]&#40;https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html&#41;** - Official documentation)

[//]: # (- 🐛 **[Issues]&#40;https://github.com/NVIDIA/Megatron-LM/issues&#41;** - Bug reports and feature requests)

[//]: # ()
[//]: # (## Contributing)

[//]: # ()
[//]: # (We ❤️ contributions! Ways to contribute:)

[//]: # ()
[//]: # (- 🐛 **Report bugs** - Help us improve reliability)

[//]: # (- 💡 **Suggest features** - Shape the future of Megatron Core)

[//]: # (- 📝 **Improve docs** - Make Megatron Core more accessible)

[//]: # (- 🔧 **Submit PRs** - Contribute code improvements)

[//]: # ()
[//]: # (**→ [Contributing Guide]&#40;https://docs.nvidia.com/megatron-core/developer-guide/latest/developer/contribute.html&#41;**)

[//]: # ()
[//]: # (## Citation)

[//]: # ()
[//]: # (If you use Megatron in your research or project, we appreciate that you use the following citations:)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@article{megatron-lm,)

[//]: # (  title={Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism},)

[//]: # (  author={Shoeybi, Mohammad and Patwary, Mostofa and Puri, Raul and LeGresley, Patrick and Casper, Jared and Catanzaro, Bryan},)

[//]: # (  journal={arXiv preprint arXiv:1909.08053},)

[//]: # (  year={2019})

[//]: # (})

[//]: # (```)
