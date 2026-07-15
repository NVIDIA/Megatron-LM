---
orphan: true
---

# Determinism Glossary

Definitions for the determinism developer reference. The user guide avoids
these abbreviations where possible.

## Determinism Contract

- **Bitwise determinism**

  Given fixed inputs and seeds, a run yields identical numerical results
  every time, with everything else held constant. This includes input data
  and order, model recipe and configuration, parallelism layout (TP, PP,
  DP, EP, CP, VPP), container image and library versions (Megatron-Core,
  CUDA, cuDNN, NCCL, Transformer Engine, PyTorch, driver), NCCL settings,
  and hardware type and topology.

  The contract is about numerical results, not the computation graph.
  Kernel scheduling and execution order may vary as long as every
  floating-point reduction happens in the same order. Floating-point
  addition is not associative, and reduction order is the root cause of
  nearly all training non-determinism.

- **Verification**

  Verify determinism by tracking training metrics (loss, gradient norm,
  params norm, num-zeros) for *bitwise-identical curves* across two
  independent runs. The loss works like a checksum of the model state
  because it is a reduction over the sampled logits. A mismatch anywhere
  propagates into a bitwise difference in the loss and gradient curve
  within a few steps.

  Console logs print at limited precision, so for a strict comparison use
  the full-precision serialized metrics (for example the TensorBoard event
  files) rather than the printed values.

## Terms

| Term | Meaning |
| --- | --- |
| Deterministic mode | Execution with `--deterministic-mode`. The argument validation and environment defaults from `megatron/training/determinism.py`, plus `torch.use_deterministic_algorithms(True)`. The library code selects deterministic branches using `config.deterministic_mode` or `torch.are_deterministic_algorithms_enabled()`. |
| Default mode | Execution without `--deterministic-mode`. |
| Bit-exact / bitwise identical | Byte-identical values for the compared tensors or serialized metrics (see "Verification" above). |
| Reproducible | Same result under the same conditions (allocation, caches, environment). This guarantee is weaker than the contract above. A run can repeat within one allocation and still differ on a different physical topology. |
| Cross-allocation | Comparison of runs on independently assigned resources: a fresh scheduler allocation with generally different physical nodes and network rings. This is the determinism target and the bar a test must meet to count as a determinism certificate. Repeating work inside one allocation exercises less than the contract promises. |
| Collision-free (unique-index) write | An indexed write (`scatter`, `index_put_(accumulate=False)`) whose indices are unique, so no floating-point accumulation happens and ordering cannot change the result. Deterministic without a special kernel. |
| Fail closed | When deterministic mode encounters a feature it cannot vouch for, it rejects the configuration at validation time instead of silently running it. |

## Parallelism and infrastructure

| Term | Meaning |
| --- | --- |
| MCore | Megatron Core: the model-parallel training library in this repository. |
| DP | Data parallelism: replicas process different batches and synchronize gradients. |
| TP | Tensor parallelism: one layer is partitioned across devices. |
| PP | Pipeline parallelism: consecutive layer ranges run on different devices. |
| VPP | Virtual pipeline parallelism: interleaved pipeline chunks that reduce pipeline idle time. |
| EP | Expert parallelism: experts are partitioned across devices. |
| CP | Context parallelism: one sequence is partitioned across devices. |
| A2A | All-to-all collective (MoE token dispatch/combine). A rank-indexed permutation, not a floating-point reduction. |
| TE | Transformer Engine, NVIDIA's transformer-kernel library. |
| wgrad / dgrad | Weight gradient / input gradient of a linear layer's backward pass. |

## Model abbreviations

| Term | Meaning |
| --- | --- |
| MoE | Mixture of Experts: a layer routes tokens to one or more expert networks. |
| MLA | Multi-Latent Attention: low-rank latent q/kv projections (DeepSeek family). |
| DSV3 / DSV4 | DeepSeek-V3- / DeepSeek-V4-style model configurations. DSV3 combines MLA with fine-grained MoE. DSV4 additionally uses DSA sparse attention. |
| DSA | DeepSeek Sparse Attention: a lightning indexer scores tokens and top-k selection sparsifies core attention. |
| MTP | Multi-Token Prediction: auxiliary layers predicting additional future tokens. |
| SSM | State-space model layers (Mamba family). |
| GDN | Gated delta net, an SSM variant (`megatron/core/ssm/gated_delta_net.py`). |
