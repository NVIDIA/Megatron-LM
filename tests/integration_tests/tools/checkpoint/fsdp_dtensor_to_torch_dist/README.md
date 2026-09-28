# Reverse checkpoint converter — end-to-end test suite

Opt-in, end-to-end validation for the offline **`fsdp_dtensor` → `torch_dist`**
reverse checkpoint converter
(`tools/checkpoint/checkpoint_inspector.py convert-fsdp-dtensor-to-torch-dist`).

The converter rewrites a Megatron-FSDP `fsdp_dtensor` checkpoint as a native
`torch_dist` checkpoint so a **classic (non-FSDP) N-D-parallel job can resume from
it** — weights *and* full distributed-optimizer state. This suite trains real tiny
models with Megatron-FSDP, converts their checkpoints, loads them into classic jobs,
and **asserts** the conversion is correct.

> **How it runs.** This suite is a single controller that **spawns its own
> `torchrun` children** running the real `pretrain_gpt.py` / `pretrain_hybrid.py` —
> the training and resume stages need different Megatron global-arg configs, which
> cannot coexist in one process. Run it as **plain pytest**, *not* under
> `torch.distributed.run`.

## Opt in

The suite is invisible to a default `pytest tests` run and to CI. Enable it with an
environment variable:

```bash
# from the repo root, inside the mcore dev container:
MCORE_CHECKPOINT_E2E=1 uv run pytest tests/integration_tests/tools/checkpoint/fsdp_dtensor_to_torch_dist -q
```

Outputs (checkpoints + logs) land in a temp dir, or under `RESULTS_DIR` if set.
Scope a run with `-k` (family name and/or test name).

## What is tested

Each family is trained once with Megatron-FSDP to iteration 100 (saving every 20,
dropout off); iterations 60 and 80 are converted and shared by every check.

| Test module | Check | GPUs |
|---|---|---|
| `test_bitexact.py` | **Bit-exact vs the source** — load the converted checkpoint into a real classic model + `DistributedOptimizer`, then compare every model tensor, fp32 master, `exp_avg` / `exp_avg_sq` and per-parameter group hyperparameter (`step`, `betas`, …) against the original `fsdp_dtensor` checkpoint with `torch.equal`, with coverage checked both ways | 1 |
| `test_resume.py` | **Resume continuity** — a classic job resumed from each converted checkpoint reproduces the FSDP run's `lm loss` (within `loss_rtol`) and LR (exactly) for 3 iterations | 1 |
| `test_multiprocess_convert.py` | **Multi-process convert** — `torchrun --nproc_per_node 2` conversion is identical (tensors + `common.pt`) to the single-process one | CPU |
| `test_reshard.py` | **Load-side reshard** — the converted checkpoint loads under TP2 / TP2+SP / PP2 / EP2 target layouts and continues the FSDP run (weights-only loads: first-iteration loss only) | 2 |
| `test_source_sharding.py` | **Source-side sharding** — an FSDP source trained on 2 GPUs (DP2 / TP2 / EP2) converts bit-exactly and resumes | 2 |

**Why both a bit-exact and a resume check.** The bit-exact check is the proof: the
oracle is the FSDP checkpoint itself (read without any converter code, keyed by the
model's own parameter names), and the comparand is what mcore's real loader put into
the classic job. The resume check proves the result *trains* like the source run
(scheduler, data position, optimizer step). It cannot replace the bit-exact check:
with dropout off the first resumed loss is exact, but an off-by-one Adam `step` only
shows as a ~1.6e-4 relative loss change two iterations later, and a one-ulp error in
an fp32 master is invisible.

## Model coverage

Each family in `registry.py` gates at least one converter transform:

| Family | Transform |
|---|---|
| `dense` | dense layer stacking |
| `dense_swiglu` | SwiGLU fc1 `_w`/`_v` merge |
| `moe_grouped` | grouped-GEMM expert restack |
| `moe_gated` | SequentialMLP `local_experts` restack, shared expert |
| `mtp` | Multi-Token Prediction keys |
| `gdn_hybrid` | Gated-DeltaNet `in_proj`/`conv1d` split, interleaved per-layer layout |
| `mamba_hybrid` | `HybridModel`: Mamba-2 `in_proj`/`conv1d` split, fp32 `router.expert_bias`, hybrid MTP |
| `moe_mla_mtp` | MLA + MTP over grouped experts |
| `dense_fp8` | FP8 `_extra_state` drop |

## Known limitations (encoded as registry flags)

- **`dense_fp8` bit-exact** — `xfail`: its GEMM weights are held re-quantized to
  fp8 with fresh scaling state (the amax/scale `_extra_state` is not in the
  `fsdp_dtensor` checkpoint), so those model tensors differ from the bf16 source;
  its fp32 masters and Adam moments are still bit-exact. Its resume check passes
  under a looser `loss_rtol`.
- **EP>1 optimizer reshard** (`moe_grouped`, `moe_gated`) — strict `xfail`
  (`ChainedOptimizer` entry-count mismatch). The weights-only EP2 companion passes
  (loss only: `--no-load-optim` also skips the LR-scheduler state).
- **PP2 source sharding** — `skip`: Megatron-FSDP + pipeline-parallel *training*
  fails at model build, so no PP2 source can be produced.
- **`mamba_hybrid` runs with `--eval-iters 0`**: Megatron-FSDP + Mamba crashes when
  entering evaluation (`MambaMixer.refresh_cache` mixes a DTensor into a plain
  tensor copy) — a training-side bug unrelated to checkpointing.
- **`gdn_hybrid`** — `skip` if `flash-linear-attention` is not importable.

## Layout

| File | Role |
|---|---|
| `registry.py` | `ModelFamily` / `ReshardCase` / `SourceShardCase` + `MODELS` — the single source of truth. |
| `config.py` | Arg vectors, deterministic env, FSDP-train / classic-load flags, parallel-flag maps, schedule. |
| `harness.py` | Controller-side orchestration: launch torchrun, convert, parse metrics, compare. |
| `_bitexact_worker.py` | Per-family subprocess: real classic model build + load, diff against the FSDP source → JSON verdict. |
| `_cases.py` | Parametrization builders (skip / xfail marks from registry flags). |
| `conftest.py` | Opt-in gate, `WORLD_SIZE>1` guard, memoizing per-family training fixtures. |
| `test_*.py` | The checks. |

## Extending

- **Add a family:** one `ModelFamily(...)` entry in `registry.py` (set `entrypoint`
  for a non-GPT model). Every check picks it up.
- **Add a layout:** one `ReshardCase` / `SourceShardCase` plus a branch in
  `config.target_parallel_flags` / `config.source_parallel_flags`.

## Relationship to the unit tests

Pure-logic coverage of every transform is in
[`test_reverse_convert.py`](../../../../unit_tests/tools/checkpoint/test_reverse_convert.py),
and [`test_reverse_convert_roundtrip.py`](../../../../unit_tests/tools/checkpoint/test_reverse_convert_roundtrip.py)
round-trips synthetic checkpoints through both converter CLIs; both run in CI. This
suite is the GPU end-to-end proof on real Megatron-FSDP checkpoints.
