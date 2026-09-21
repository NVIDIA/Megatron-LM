# Numerical parity and acceptance

What has been checked on the `lit/main_qwen4` branch, with the numbers. Reference implementation:
HF `transformers` `modeling_qwen4_exp.py` (5.17). All fp32 comparisons run with TF32 disabled on
**both** sides (`NVIDIA_TF32_OVERRIDE=0`, `TRITON_F32_DEFAULT=ieee`, `torch.backends.*.allow_tf32=False`);
HF's "torch" gated delta rule dispatches to the fla Triton kernel when fla is importable, so the
HF process needs the same environment or the layer-0 GDN core shows a spurious 8e-4.

## 1. Module parity vs the real `transformers` implementation

Group RMSNorm, gated-residual read/write gates, the GDN output-gate switch, the PLE hashing
constants and the whole PLE layer forward (bit-exact against the official splitmix64 / prime
allocation), 14 / 14 cases. The QSA indexer is covered by the whole-model comparison.

## 2. Whole-model parity on a random-init proxy (4 layers `GEQE/QE`, hidden 512, 8 experts, PLE)

Same weights on both sides (written by HF, converted to Megatron), same tokens, fp32, TP1 and EP4
(EP2 earlier), re-run after every rebuild of the branch. **Last re-run 2026-09-20 on the current
head**, after the QSA and optimizer commits: both layouts pass, `logits max_abs 2.83e-7`,
161 / 161 gradients, step-0 loss identical to every digit.

The forward is **bit-identical between TP1 and EP4** (same `max_abs` to the last digit): expert
parallelism changes which rank holds a weight, not which experts a token passes through, and the
all-to-all is an exact move. Only the multi-step trajectory diverges, where reduction order meets
accumulated drift.

> **To reproduce this yourself**, run
> [`examples/qwen3.8_flash_next/parity/run_parity.sh`](../../../../examples/qwen3.8_flash_next/parity/run_parity.sh)
> — a self-contained harness in this repository, one GPU, about five minutes. It needs only a
> `transformers` build that provides `qwen4_exp` (see that directory's README). It exports the two
> settings the comparison depends on, `NVIDIA_TF32_OVERRIDE=0` and `TRITON_F32_DEFAULT=ieee`;
> without them TE and Triton silently run fp32 GEMMs as TF32 and every linear carries ~1e-3 of
> noise, which makes a 1e-6 comparison meaningless.
>
> That harness covers EP1 and excludes MTP. The EP4 column and the real-weight rows below come
> from the development harness, which additionally shards across ranks.

| Check | Result |
|---|---|
| forward, 55 sub-module points (GDN in/core/out, QSA q/k norms and outputs, router probabilities, MoE out, GR mixed / write gate, PLE, exit contract, `lm_head`) | all at `allclose(1e-6, 1e-5)`; **logits `max_abs 2.7e-7`, argmax 100 %**; QSA selection sets (0 / 524288 differ) and MoE top-k sets (0 / 8192) identical |
| gradients after one step, 161 parameters | 161 / 161 within `rel_l2 1e-5` / `max_abs 1e-4` (worst `2.2e-3` on GDN `A_log` / `dt_bias`, tiny-magnitude tensors); EP-sharded gradients gathered from the ranks |
| 20-step AdamW trajectory | `\|Δloss\|` peaks in the **2e-4 … 3e-4** band by step 20 across re-runs (2026-09-20: 2.8e-4 TP1 / 2.2e-4 EP4; earlier: 2.1e-4 / 3.0e-4 — the two layouts swap places between runs, so this is tail noise, not a systematic offset). The gate is **no jump**: the divergence must accumulate smoothly alongside the parameter drift (~9e-2 by step 20), not step. Steps 0–1 exact |

Four model-level defects were found only by this harness (all fixed on the branch): an extra
final RMSNorm after the gated-residual exit; a MoE router without HF's top-k renormalization
(now: post-softmax routing, no extra knob); the n-gram memory rejecting fp32; the MTP `hnorm`
width. None of them was visible to module tests or to a decreasing loss curve.

## 3. Real weights: 4-layer truncation of the released checkpoint vs HF

HF `Qwen4ExpForConditionalGeneration` (layers 0–3 + PLE + MTP, vision dropped) against the
`HybridModel` loaded from the converted package at EP4, 4 × 1024 random tokens:

| Precision | Result |
|---|---|
| fp32, TF32 off | every layer-0 point ≤ 3e-6 (GDN core 1.1e-6, PLE delta 2.4e-6); MoE top-10 tie flips per layer 0 / 2 / 16 / 34 → logits `rel_l2 7.8e-3`, argmax 99.9 %, top-5 99.9 %, CE 12.8547 vs 12.8548; QSA selections identical |
| bf16 | logits `rel_l2 7.8e-2`, argmax 88.8 %, routing flips 0.07–0.18 % |

The residual disagreement is routing ties on near-flat 512-expert distributions (random tokens),
each flip moving one token's MoE output by O(1). The fla chunked GDN kernel and HF's chunked
reference agree bitwise and are both 1.5e-4 from an fp64 sequential recurrence with the released
decays (fp32 sequential: 3.9e-6) — a shared reformulation error, not a divergence.

## 4. Full-model mechanics (64 GB300s)

Exact native load of the converted 48-layer checkpoint at EP64 (109,632 / 109,632 parameters),
50-step weights-only resume with finite losses and no skipped iterations, memory equal to the
single-GPU fake-process-group prediction to 0.1 GiB (details in
[`../training/full_model_64gpu.md`](../training/full_model_64gpu.md)). MTP loss tracking the main
loss from step 1 (7.40 vs 7.36) is the only indirect evidence for the converted MTP head; HF has no
MTP forward.

## 5. Acceptance matrix run after every rebuild of the branch

| Check | Pass condition |
|---|---|
| staged proxy `base / +GR / +PLE / +MTP / full`, 6 iterations each, then `full` with `--engram-verify-training`, `full` at PP2, `full` at TP2 + SP | all run; parameter counts differ per stage (42,069,232 / 44,507,888 / 67,702,896 / 55,089,392 / 78,284,400); `zero_grad_tables=0 changed_tables=16` |
| n-gram memory on the hybrid path under hyper connections (`pretrain_hybrid.py`, pattern `*-*-`, EP2) | 16 / 16 tables with finite non-zero gradients, updated every iteration; same counters without hyper connections; fp32 build reaches iteration 1 |
| whole-model parity TP1 and EP4 (§2) | logits `max_abs ≤ 3e-7`, 161 / 161 gradients, trajectory without jumps |
| real-weight truncation native load, EP4 | `all_ranks_equal: true`, 5,624 / 5,624 tensors equal |
| unit tests | `tests/unit_tests/transformer/test_gated_residual*.py`, `…/experimental_attention_variant/test_*qsa*.py`, `tests/unit_tests/ssm/test_hybrid_layer_allocation.py`, `tests/unit_tests/models/engram/`, `tests/unit_tests/ssm/gated_delta_net/test_gdn_output_gate.py`; failure sets equal to the base branch (multi-rank tests need `torchrun`; `pyproject` sets `-x`, run with `-o addopts=""` to see the whole file) |

## 6. Parallelism / feature coverage

Which parallel layouts and features the composed model runs under is a separate, measured
document: [`support_matrix.md`](support_matrix.md).

## 7. Not verified

- Loss quality on real text (every run used mock data).
- The MTP head numerically (no reference).
- The full-model **conversion** has only been executed on the previous dev base (`b6b2fbde8`); the
  resulting checkpoint was **re-loaded and trained on the current base** (2026-09-16: 20 iterations
  on 64 GPUs, exact load, 204.0 GiB peak — reproducing the earlier 204.4 GiB), so the load path is
  confirmed, but a fresh conversion on the current base has not been re-run.
- Performance: 16 s/step at EP64 / all-to-all is an unoptimized layout.
