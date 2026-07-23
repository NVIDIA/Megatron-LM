# Reverse checkpoint converter — validation harness

End-to-end validation for the offline **`fsdp_dtensor` → `torch_dist`** reverse
checkpoint converter
(`tools/checkpoint/checkpoint_inspector.py convert-fsdp-dtensor-to-torch-dist`).

The converter takes a Megatron-FSDP `fsdp_dtensor` checkpoint and rewrites it as a
native `torch_dist` checkpoint so a **classic (non-FSDP) N-D-parallel job can
resume from it** — weights *and* full distributed-optimizer state. This harness
trains real tiny models with Megatron-FSDP, converts their checkpoints, resumes a
classic job from the result, and proves the conversion is correct.

> **Scope.** These scripts target **1–8 GPUs on a single node**. The converter's
> CPU-memory ceiling and the multi-process (`torchrun -N`) path for larger
> checkpoints are documented in the
> [Megatron-FSDP feature guide](../../../docs/user-guide/features/megatron_fsdp.md#converting-megatron-fsdp-fsdp_dtensor-to-n-d-parallel-torch_dist-checkpoints).

---

## Requirements

- Run everything **inside the Megatron dev container** — these commands expect the
  container environment. See the
  [build & dependency skill](../../../skills/mcore-build-and-dependency/SKILL.md).
- **1 GPU** for resume-continuity and the bit-exact diff; **≥2 GPUs** for the
  load-side reshard sweep and the source-side sharding check.
- On older dev images, remove a stale `nvidia-resiliency-ext` (<0.6.0) that breaks
  `import megatron.core`:  `bash common.sh preflight`  (no-op on a fresh image).
- `gdn_hybrid` needs `flash-linear-attention`; `validate_resume.sh` installs it
  automatically (the image ships an insufficient `fla` stub).

---

## Quick start

```bash
# from anywhere inside the repo, in the dev container:
cd tools/checkpoint/fsdp_dtensor_to_torch_dist_validation

bash common.sh list-models              # what's available
bash validate_resume.sh dense           # 1 GPU: train → convert → classic resume
bash run_all.sh                         # the whole six-family coverage set
bash run_all.sh --with-bitexact --with-reshard    # add the decisive + reshard checks
bash validate_source_sharding.sh dense DP2        # ≥2 GPUs: train a SHARDED source → convert → resume
```

All outputs land under `results/<model>/` (git-ignored).

---

## What is tested — four complementary checks

The converter is **parallelism-agnostic**: it reads each tensor's *global* shape
from PyTorch DCP, independent of the source sharding, and mcore re-shards the
output on load. Checks 1–2 decide every transform's correctness on a single-rank
(unsharded) source; checks 3–4 exercise the multi-GPU paths on the *load* and
*source* sides respectively, so the parallelism-agnostic claim is validated, not
just asserted. From cheapest to most decisive:

### 1. Resume continuity — `validate_resume.sh <model>`  (1 GPU)

Train Megatron-FSDP from scratch to **iter 100** saving every 20, convert
**iter 60** and **iter 80** to `torch_dist`, then resume a **classic (non-FSDP)**
job from each and run 3 more iters.

**PASS when**, for each of iters 61 and 81:
- the classic job prints `successfully loaded checkpoint … at iteration {60|80}`
  and starts at the next iter (only the expected TE `_extra_state` keys dropped);
- its **first post-load `lm loss`** matches the FSDP training loss at that same
  iter within **bf16 tolerance**;
- its **learning rate** at that iter matches the FSDP LR **exactly**.

**Why this indicates correctness.** Two independent load points, both mid-cosine
decay (LR still moving). A matching first-step loss means the **weights** loaded
correctly; a matching LR means the **optimizer state and the LR-scheduler
bookkeeping** converted correctly. Using iters 60 *and* 80 rules out a single
lucky checkpoint. Single-rank (world size 1) has no cross-rank collective, so it
never deadlocks under GPU contention and still fully exercises the converter.

### 2. Bit-exact per-tensor diff — `validate_bitexact.py <model> --iter {60|80}`  (1 GPU)

The resume check *infers* equivalence from a scalar loss; this **proves** it. It
loads the converted checkpoint into a real classic `GPTModel` +
`DistributedOptimizer` (built via the exact `pretrain_gpt.py` init path, so the
model matches by construction), immediately **re-saves** it (no train step ⇒ no
drift), and does a strict per-tensor diff of the re-save against the converter
output.

**PASS when** the diff prints **no mismatch lines** ⇒ bit-exact **weights and
optimizer** (fp32 masters, `exp_avg`/`exp_avg_sq`, and the reconstructed
`param_groups`). This is the strongest signal available on a single GPU and is
independent of iteration count. For MoE, the `linear_fc2` optimizer tensors are
the fc2 verdict.

### 3. Load-side resharding — `validate_reshard.sh <model>`  (≥2 GPUs)

Reuses the converted `td80` and loads it into a classic job under **different
target parallel layouts** (TP/PP/EP, per the model's `RESHARD_LAYOUTS`), verifying
a clean load + iter-81 continuity vs the FSDP reference at each.

**Why this indicates correctness.** The converter's whole promise is that its
full-global-shape `torch_dist` output **reshards into any target parallelism on
load**. A `TP1` FSDP source and a `TP8/EP8` one convert *identically* (DCP
abstracts the source layout away), so what must be proven is the *load* side —
this does exactly that.

### 4. Source-side sharding — `validate_source_sharding.sh <model> [DP2|TP2|PP2|EP2]`  (≥2 GPUs)

The counterpart to check 3. Checks 1–3 all start from a source trained
**single-rank** (an *unsharded* DCP store). This driver trains the FSDP source on
≥2 GPUs under a real sharded layout — `DP2` (params + optimizer split across data-
parallel ranks; the plain "trained on >1 GPU" case), or `TP2` / `PP2` / `EP2` — then
converts (single-process, unchanged) and resumes a classic single-rank job,
checking iter-61/81 continuity against the sharded run's own log.

**Why this indicates correctness.** It exercises the converter's source-side
gather on a genuinely multi-shard checkpoint — the multi-shard DTensor
reassembly, the TP-reshard `nd_reformulated_orig_global_shape` path, the
`_stack_layers` PP layer-offset contiguity, and the expert gather across EP ranks
— none of which a single-rank source produces. This is what makes
"parallelism-agnostic" a *tested* property rather than an assumption.

**Status (2026-07-20):** `DP2` validated for **all 8 models**; `TP2` (`dense`) and
`EP2` (`moe_grouped`) validated. `PP2` cannot be validated here — Megatron-FSDP +
pipeline-parallel *training* fails at model build (`EinopsError`), so no PP2
checkpoint is produced (a training-side limitation, not a converter issue). See
**Expected results** for the numbers.

---

## Model coverage

Eight models, each gating a distinct converter transform — all trained through
`pretrain_gpt.py` (`GPTModel`). `run_all.sh` runs the first six by default; pass
`moe_mla_mtp` / `dense_fp8` explicitly via `--models`.

| Model (`models/*.sh`) | Architecture | Converter transform it gates |
|---|---|---|
| `dense`         | Dense GPT (GELU)              | dense layer-stack |
| `dense_swiglu`  | Dense + SwiGLU               | SwiGLU fc1 `_w`/`_v` merge |
| `moe_grouped`   | MoE, grouped-GEMM (mixtral)  | grouped-expert restack |
| `moe_gated`     | Non-grouped shared-expert MoE | non-grouped `local_experts` restack |
| `mtp`           | GPT + Multi-Token Prediction | MTP key-rename |
| `gdn_hybrid`    | Gated-DeltaNet + MoE (Qwen-Next) | GDN `in_proj`/`conv1d` factory split + non-grouped experts |
| `moe_mla_mtp`   | MoE + MLA + MTP (deepseek)   | MLA + MTP passthrough |
| `dense_fp8`     | Dense + FP8 (llama3)         | FP8 `_extra_state` drop |

> **Mamba-2 is not covered by this harness.** It is a different model class
> (`HybridModel`, trained via `pretrain_hybrid.py` — not the `pretrain_gpt.py`
> path every model above uses), so it is not a drop-in `models/*.sh`. The
> converter's Mamba-2 support — the `[Convert]` line's `mamba-splits=` — is
> exercised by the unit tests (`TestSplitMambaProjections` in
> [`test_reverse_convert.py`](../../../tests/unit_tests/tools/checkpoint/test_reverse_convert.py)),
> not by this end-to-end GPU harness.

---

## Expected results

**Final validation run — 2026-07-20**, single node (mcore dev container; logs
under `results/_final/`). Losses depend on hardware/kernels; the pass criterion is
**LR exact and `lm loss` within bf16 tolerance** (FP8 ~1% by design), not the
absolute value.

### Resume continuity + bit-exact (single-rank source)

`lm loss` FSDP → resumed. LR is **exact** at every point (iter 61 `5.710524e-05`,
iter 81 `2.234264e-05`). "bit-exact (iter 80)" = weights **and** optimizer
(masters, moments, `param_groups`) match mcore's native re-save; only the
intentionally-dropped `_extra_state` / `rng_state` / `rerun_state` / `common_state`
keys differ.

| Model | iter 61 | iter 81 | bit-exact (iter 80) |
|---|---|---|---|
| `dense`        | 5.796581 → 5.793622 | 4.752846 → 4.751805 | ✅ exact |
| `dense_swiglu` | 6.643600 → 6.642058 | 5.554593 → 5.558852 | ✅ exact |
| `moe_grouped`  | 5.792314 → 5.795763 | 4.303905 → 4.302549 | ✅ exact |
| `moe_gated`    | 5.771983 → 5.775371 | 4.246530 → 4.249354 | ✅ exact |
| `mtp`          | 3.944042 → 3.944121 | 2.887992 → 2.887894 | ✅ exact |
| `gdn_hybrid`   | 5.027180 → 5.025552 | 4.086754 → 4.085003 | ✅ exact |
| `moe_mla_mtp`  | 0.632209 → 0.631782 | 0.484988 → 0.485596 | ✅ exact |
| `dense_fp8`    | 6.695069 → 6.750668 | 5.630019 → 5.699869 | ⚠️ FP8 — 4 weight keys differ |

**FP8 (`dense_fp8`) is not bit-exact by design** — the `_extra_state` amax/scale
history is not round-tripped (already discarded in the `fsdp_dtensor` checkpoint),
so FP8 resume re-initializes amax and both its `lm loss` (~1%) and 4 weight tensors
drift. Expected, not a defect.

### Source-side sharding + load-side resharding

| Test | Result |
|---|---|
| Source **DP2** (all 8 models) | ✅ pass — LR exact, loss in bf16 tol (FP8 ~1%) |
| Source **TP2** (`dense`), **EP2** (`moe_grouped`) | ✅ pass |
| Source **PP2** | ⚠️ **N/A** — Megatron-FSDP + pipeline-parallel *training* fails at model build (`EinopsError`), so no PP2 checkpoint is produced. Training-side, unrelated to the converter. |
| Reshard **TP2 / PP2 / TP2SP** | ✅ pass — clean load + iter-81 continuity |
| Reshard **EP2** (`moe_grouped`, `moe_gated`) | ⚠️ **documented limitation** — optimizer state cannot load into EP>1 (`ChainedOptimizer` entry-count mismatch, `Expected 2 entries…got 4`). Resume weights-only under EP>1, or with optimizer at EP=1. |

Logic gate: **60/60** unit tests pass.

---

## Adding a new model

Drop **one file** in `models/<name>.sh` setting six variables; all drivers and
`run_all.sh` pick it up automatically (this is the single source of truth — the
bit-exact Python tool reads these same flags via `common.sh emit-args`):

```bash
# models/my_model.sh
MODEL_LABEL="Human-readable name"
MODEL_TRANSFORM="which converter transform this gates"
NUM_LAYERS=12
ARCH="--swiglu --num-experts 8 --moe-grouped-gemm --disable-bias-linear"
RESHARD_LAYOUTS="EP2 TP2SP"        # layouts validate_reshard.sh sweeps ("" to skip)
EXTRA_SETUP=""                      # optional one-time setup, e.g. a pip install
```

Then `bash validate_resume.sh my_model`. Nothing else to edit.

This covers any **GPT-family** model — anything trainable through `pretrain_gpt.py`
(including attention variants like `gdn_hybrid`). A different model *class* (e.g. a
Mamba-2 `HybridModel`, which trains via `pretrain_hybrid.py`) would first need the
drivers generalized to select the training entrypoint and model builder per model.

## Adding a new feature / transform

1. Add the transform + a pure-logic unit test in
   [`tests/unit_tests/tools/checkpoint/test_reverse_convert.py`](../../../tests/unit_tests/tools/checkpoint/test_reverse_convert.py)
   (see the [testing skill](../../../skills/mcore-testing/SKILL.md)).
2. Add a `models/<name>.sh` whose `ARCH` exercises it end-to-end.
3. If it needs shared plumbing (a new env var, flag, or layout), extend
   `common.sh` (constants / `layout_flags` in `validate_reshard.sh`).

---

## Interpreting output & troubleshooting

- **The `[Convert]` line** summarizes every transform applied:
  `mtp=… swiglu=… experts=… layer-stacks=… gdn-splits=… mamba-splits=… masters=…
  extra_state-dropped=… layout={stacked|per-layer}`. Dense/homogeneous models are
  `stacked`; MoE / GDN / MTP interleaved models stay `per-layer`.
- **`run_all.sh` roll-up** reports whether each stage *ran to completion*, not the
  numeric verdict — read each model's `VERIFICATION` block (loss + LR) for the
  resume verdict.
- **Reading the bit-exact diff:** it is *not* empty — the converter intentionally
  omits `_extra_state` / `rng_state` / `rerun_state` / `common_state`, which the
  real re-save re-adds, so those keys always show as "only in checkpoint 2". A
  clean result = **no `decoder.*` / `embedding.*` / `output_layer.*` weight or
  `optimizer.state.*` key** appears as missing or "(values differ)". (FP8 is the
  exception — a few weight tensors legitimately differ.)
- **GPU contention.** If a GPU is shared, multi-rank (2-GPU) NCCL collectives can
  deadlock (600 s ALLREDUCE timeout). Resume + bit-exact are single-rank and
  immune; the reshard sweep needs a free 2-GPU window.
- **`results/` is git-ignored** — safe to delete between runs; each
  `validate_resume.sh` wipes and recreates `results/<model>/`.

## Relationship to CI

The automated gate lives in unit tests (CPU/GPU, run by CI):
- [`test_reverse_convert.py`](../../../tests/unit_tests/tools/checkpoint/test_reverse_convert.py)
  — pure-logic coverage of every transform (**60/60 pass**).
- [`test_reverse_convert_roundtrip.py`](../../../tests/unit_tests/tools/checkpoint/test_reverse_convert_roundtrip.py)
  — standalone synthetic `torch_dist → fsdp → torch_dist` identity + a real-model
  load. **Note:** its synthetic archetypes predate two converter changes
  (all-MoE stacking; fp32-master synthesis), so the `synthetic:dense` / `synthetic:moe`
  identity checks are currently stale and need refreshing — this is a test-fixture
  gap, **not** a converter defect (the real-checkpoint bit-exact + resume checks
  above confirm those exact behaviors).

This harness is the **manual end-to-end GPU proof** on real Megatron-FSDP
checkpoints that complements those unit tests.
