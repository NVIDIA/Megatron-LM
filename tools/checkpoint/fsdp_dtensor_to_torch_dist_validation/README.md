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

**Status:** `dense`/`DP2` validated (LR exact, `lm loss` within bf16 tol at iters
61/81). The `TP2`/`PP2`/`EP2` layouts are supported by the driver but not yet run
end-to-end here.

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

Representative resume-continuity numbers from a full six-family run (single-rank,
one RTX 6000 Ada). Exact losses depend on hardware/kernels; **what matters is
FSDP ≈ resumed within bf16 tolerance and LR exact**, not the absolute value.

| Model | `[Convert]` (key fields) | iter 61 FSDP → resumed | iter 81 FSDP → resumed |
|---|---|---|---|
| `dense`        | `layout=stacked`                       | 5.787642 → 5.784644 | 4.745402 → 4.744267 |
| `dense_swiglu` | `swiglu=96 stacked`                    | 6.719251 → 6.717134 | 5.633004 → 5.637542 |
| `moe_grouped`  | `experts=96 stacked`                   | 5.777692 → 5.778813 | 4.279071 → 4.277475 |
| `moe_gated`    | `experts=96 stacked`                   | 5.759237 → 5.762474 | 4.225122 → 4.226754 |
| `mtp`          | `mtp=48 stacked`                       | 3.941385 → 3.941342 | 2.881462 → 2.881354 |
| `gdn_hybrid`   | `experts=48 gdn-splits=32 per-layer`   | 5.026937 → 5.025422 | 4.086696 → 4.084918 |

**FP8 (`dense_fp8`) tracks ~1% looser by design** — the `_extra_state` amax/scale
history is not round-tripped (already discarded in the `fsdp_dtensor` checkpoint),
so FP8 resume re-initializes amax. Not bit-exact; expected.

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
  numeric verdict — read each model's `VERIFICATION` block (loss + LR) and the
  bit-exact diff (empty = pass) for the actual pass/fail.
- **GPU contention.** If a GPU is shared, multi-rank (2-GPU) NCCL collectives can
  deadlock (600 s ALLREDUCE timeout). Resume + bit-exact are single-rank and
  immune; the reshard sweep needs a free 2-GPU window.
- **`results/` is git-ignored** — safe to delete between runs; each
  `validate_resume.sh` wipes and recreates `results/<model>/`.

## Relationship to CI

The automated gate lives in unit tests (CPU/GPU, run by CI):
- [`test_reverse_convert.py`](../../../tests/unit_tests/tools/checkpoint/test_reverse_convert.py)
  — pure-logic coverage of every transform.
- [`test_reverse_convert_roundtrip.py`](../../../tests/unit_tests/tools/checkpoint/test_reverse_convert_roundtrip.py)
  — synthetic `torch_dist → fsdp → torch_dist` identity + a real-model load.

This harness is the **manual end-to-end GPU proof** on real Megatron-FSDP
checkpoints that complements those unit tests.
