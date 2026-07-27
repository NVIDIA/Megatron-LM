# Reverse checkpoint converter — end-to-end test suite

Opt-in, end-to-end validation for the offline **`fsdp_dtensor` → `torch_dist`**
reverse checkpoint converter
(`tools/checkpoint/checkpoint_inspector.py convert-fsdp-dtensor-to-torch-dist`).

The converter rewrites a Megatron-FSDP `fsdp_dtensor` checkpoint as a native
`torch_dist` checkpoint so a **classic (non-FSDP) N-D-parallel job can resume from
it** — weights *and* full distributed-optimizer state. This suite trains real tiny
models with Megatron-FSDP, converts their checkpoints, resumes a classic job from
the result, and **asserts** the conversion is correct.

> **How it runs.** This suite is a single controller that **spawns its own
> `torchrun` children** running the real `pretrain_gpt.py` — the training / resume
> stages need two different Megatron global-arg configs, which cannot coexist in one
> process. So run it as **plain pytest**, *not* under `torch.distributed.run`.

## Opt in

The suite is invisible to a default `pytest tests` run and to CI. Enable it with the
env var (robust) or the flag (convenience):

```bash
# from the repo root, inside the mcore dev container:
MCORE_CHECKPOINT_E2E=1 uv run pytest \
  tests/integration_tests/tools/checkpoint/fsdp_dtensor_to_torch_dist --run-e2e -q
```

Outputs (checkpoints + logs) land in a temp dir, or under `RESULTS_DIR` if set —
point it at scratch for the full matrix: `RESULTS_DIR=/scratch/ckpt_e2e`.

## What is tested — four checks (parametrized per family)

| Test module | Check | GPUs |
|---|---|---|
| `test_resume.py` | **Resume continuity** — first post-load `lm loss` ≈ FSDP ref (bf16 tol), LR exact, at each of two load points | 1 |
| `test_bitexact.py` | **Bit-exact diff** — load converted ckpt into a real classic model, re-save, strict per-tensor diff (weights + optimizer) | 1 |
| `test_reshard.py` | **Load-side reshard** — load the converted checkpoint under TP/PP/EP target layouts + first-post-load continuity | ≥2 |
| `test_source_sharding.py` | **Source-side sharding** — train a sharded FSDP source (DP2/TP2/EP2), convert, classic resume | ≥2 |

Losses are compared against **the same run's own FSDP reference** (never a
cross-machine golden), so the assertions are hardware-stable; LR is checked exact.

## Training schedule (derived, not hard-coded)

The suite trains to `TRAIN_ITERS` (default 100) saving every `SAVE_INTERVAL`
(default 20), and converts + validates two interior, mid-decay save points (default
60 and 80) where the LR is still moving. All of it is overridable per run:

| Env var | Default | Meaning |
|---|---|---|
| `MCORE_CHECKPOINT_E2E_TRAIN_ITERS` | 100 | total FSDP training iters |
| `MCORE_CHECKPOINT_E2E_SAVE_INTERVAL` | 20 | checkpoint save interval |
| `MCORE_CHECKPOINT_E2E_CONVERT_ITERS` | *derived* | comma-separated save iters to convert |
| `MCORE_CHECKPOINT_E2E_RESUME_EXTRA_ITERS` | 3 | iters the classic resume runs past the load point |

## Model coverage — 8 families

Each family (in `registry.py`) is a tiny-but-real architecture that gates one
converter transform: `dense`, `dense_swiglu`, `moe_grouped`, `moe_gated`, `mtp`,
`gdn_hybrid`, `moe_mla_mtp`, `dense_fp8`. The single training run per family is
shared across the resume, bit-exact and reshard checks (memoized session fixture).

## Known limitations (encoded as registry flags)

- **`dense_fp8` bit-exact** — `xfail`: the amax/scale `_extra_state` is dropped by
  design, so a few fp8 weight tensors re-quantize differently. Its resume check
  still passes under a looser (`loss_rtol`) tolerance.
- **EP>1 optimizer reshard** (`moe_grouped`/`moe_gated`) — strict `xfail`
  (`ChainedOptimizer` entry-count mismatch). The weights-only EP2 companion passes.
- **PP2 source sharding** — `skip`: Megatron-FSDP + pipeline-parallel *training*
  fails at model build, so no PP2 source can be produced.
- **`gdn_hybrid`** — `skip` if `flash-linear-attention` is not importable (it is
  pinned in the `dev` extra; never installed at test time).

## Layout / where things live

| File | Role |
|---|---|
| `registry.py` | `ModelFamily` / `ReshardCase` / `SourceShardCase` + `MODELS` — the single source of truth. |
| `config.py` | Arg vectors, deterministic env, FSDP-train / classic-load flags, parallel-flag maps, training schedule. |
| `harness.py` | Controller-side orchestration: launch torchrun, convert, parse metrics, assertions. No torch import. |
| `_bitexact_worker.py` | Per-family subprocess: real model build + load + resave + structured DCP diff → JSON verdict. |
| `_cases.py` | Parametrization builders (skip/xfail marks from registry flags). |
| `conftest.py` | Opt-in gate, `WORLD_SIZE>1` guard, memoizing per-family training fixtures. |
| `test_*.py` | The four checks. |

## Extending

- **Add a family:** one `ModelFamily(...)` entry in `registry.py`. Every check and
  all known-limitation handling picks it up automatically.
- **Add a check:** one `test_*.py` reusing `family_runs` + `harness`.
- **Add a layout:** one `ReshardCase` / `SourceShardCase` + a branch in
  `config.target_parallel_flags` / `config.source_parallel_flags`.

## Relationship to the unit tests

Pure-logic coverage of every converter transform stays in
[`tests/unit_tests/tools/checkpoint/test_reverse_convert.py`](../../../../unit_tests/tools/checkpoint/test_reverse_convert.py)
(CPU, always in CI). This suite is the **GPU end-to-end proof** on real
Megatron-FSDP checkpoints that complements it.
