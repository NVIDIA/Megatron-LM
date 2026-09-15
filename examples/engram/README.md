# Engram Hybrid: offline FineWeb comparison

This recipe compares `MoE-0.73A0.05B` and `Engram-0.73A0.05B-0.8` on identical
ordered GPT-2-tokenized FineWeb samples. It trains fresh Hybrid models; historical
GPT checkpoint conversion is outside its scope. Actual measurements belong in
the generated experiment artifacts, not in this recipe as unverified results.

## Fixed configuration

Both models use hidden size 512, eight logical blocks (one dense and seven MoE),
dense FFN 1344, expert/shared FFN 256, one shared expert, and routed top-8.
Attention uses 12 heads, `kv_channels=128`, one query group, and output gating.
RMSNorm, SwiGLU, RoPE, BF16, and one attention/MoE MTP branch are enabled.
The Hybrid pattern is `*-*E*E*E*E*E*E*E/*E`.

| Excluding embedding, LM head and MTP | MoE | Engram |
| --- | ---: | ---: |
| Routed experts | 256 | 208 |
| Expected backbone parameters | 730,309,120 | 730,417,360 |
| Expected activated parameters | 47,686,144 | 48,174,720 |
| Engram parameters | 0 | 132,400,848 |

The backbone difference is approximately 0.01482%. These are analytic expectations;
each run also writes local parameter inventories. Count MTP and embedding/head
separately when reporting full training-model size.

Engram applies before the second attention (main pattern position 2, memory ID 1).
It uses 2/3-grams, eight hash heads per order, width 40 per head, minimum rows
205700 per table, and a depthwise convolution of width 4. Distinct primes give
3,293,522 total rows, 131,740,880 table parameters and 659,968 fusion parameters.
The table uses `row_a2a` and RowSparseAdam with LR multiplier 5 and weight decay 0.
The MTP branch does not contain Engram.

Both arms train 36,754 updates with global batch 64 and sequence length 4096:
2,352,256 samples / 9,634,840,576 tokens. Training/model seed is 2026; Engram seed
is 0. Muon uses Adam coefficients (0.9, 0.95), weight decay 0.1 and clipping 1.0.

| Phase | Updates | LR | MTP weight |
| --- | ---: | --- | ---: |
| Warmup | 1–1,000 | Linear to 8e-4 | 0.3 |
| Stable | 1,001–33,079 | 8e-4 | 0.3 |
| Decay | 33,080–36,754 | Native cosine WSD toward 8e-5 | 0.15 |

`train.py` updates the actual MTP module configurations before the first forward
of each update, and for each microbatch/model chunk. Its phase derives from
completed training iterations after restore. It does not modify native scheduler
semantics. `recipe/update_lr` records LR before the update; native `learning-rate`
records scheduler output after the update, so boundary values can differ.

## Prepare local data

Use the existing GPT-2 tokenizer directory (50,257 vocabulary entries, padded to
50,304) and Megatron `.bin/.idx` data. No download or retokenization is performed.

```bash
python -m examples.engram.prepare_data \
  --source-config /path/to/original/per_split.json \
  --output /path/to/new/prepared-data
```

The preparation keeps source files unchanged and refuses an existing output
directory. It removes explicit training weights, letting the native builder use
one epoch per shard and blend by available sample counts. It shuffles original
holdout document IDs with seed 1234 and writes disjoint validation/test datasets
at the document boundary closest to half the original tokens. Source mappings,
counts and hashes accompany the new configuration.

The expected training inventory is 103 shards, 10,255,324,043 tokens, and
2,503,734 complete 4096-token samples. The common budget covers approximately
93.95%. The actual instantiated indices are audited before training: document and
shuffle permutations, blend bounds, unique sample positions, full label intervals,
and at least 90% unique valid main-label coverage. This recipe retains EOD losses
and has no padding token, which makes the full-sample loss mask exact.
MTP targets and overlapping next-token context are not counted as new source data.
The audit does not assert that source documents have been text-deduplicated.

## Acceptance and formal runs

```bash
cp examples/engram/local_data.sh.example examples/engram/local_data.sh
# Fill local paths and select the existing W&B project/group.

# Use a dedicated acceptance RUN_ROOT; retain the full schedule and data horizon.
EXIT_INTERVAL=130 SAVE_INTERVAL=128 EVAL_INTERVAL=64 \
  bash examples/engram/pretrain_moe_baseline.sh
EXIT_INTERVAL=130 SAVE_INTERVAL=128 EVAL_INTERVAL=64 \
  bash examples/engram/pretrain_moe_engram.sh

# After all acceptance gates pass, use a fresh formal RUN_ROOT, sequentially:
bash examples/engram/pretrain_moe_baseline.sh
bash examples/engram/pretrain_moe_engram.sh
```

Acceptance covers the requested first 128 updates, then continues to 130 to retain
a genuinely uninterrupted reference for checkpoint tests. The temporary 64-step
validation interval exercises two native evaluation iterations at steps 64 and
128; formal training retains interval 500. `recipe.json` records these execution
overrides separately from the common model/token budget.

Default topology is eight GPUs, TP1/PP1/CP1/EP8/ETP1, micro batch 8, global batch
64. If memory requires a lower micro batch, lower it equally in both arms; the
native gradient accumulation preserves the global batch. `MOE_TOKEN_DISPATCHER`
defaults to `flex` with HybridEP; use `alltoall` where that optional backend is
unavailable. Checkpoints are saved every 2,000 steps and at completion. Relaunching
the same arm resumes its own checkpoint and sample progress.

Periodic validation retains the existing 500-step interval and two evaluation
iterations. Its metric is main LM CE; MTP and MoE auxiliary losses remain separate.
After each formal run, evaluate each complete independent holdout:

```bash
FULL_EVAL_SPLIT=valid bash examples/engram/pretrain_moe_baseline.sh
FULL_EVAL_SPLIT=test bash examples/engram/pretrain_moe_baseline.sh
FULL_EVAL_SPLIT=valid bash examples/engram/pretrain_moe_engram.sh
FULL_EVAL_SPLIT=test bash examples/engram/pretrain_moe_engram.sh
```

Full evaluation loads model weights through the native evaluation-only driver.
It uses the native token-weighted loss reduction and pads the final DP batch with
zero-loss samples, avoiding repetition of real samples on shorter ranks.
The last partial sequence is retained. For this evaluation dataset only, an absent
padding token uses the negative sentinel -1, which native GPTDataset masks before
model input; the tokenizer and training masks are unchanged. Each holdout therefore
evaluates all shifted targets, exactly its token count minus the first token.
Full-holdout results are `final/valid_ce` and `final/test_ce`; they are distinct
from short periodic validation. TensorBoard places each result at the evaluated
checkpoint's step. W&B uses the custom horizontal axis `final/checkpoint_step` for
`final/*` and lets its internal history step advance normally. This preserves the
checkpoint coordinate when evaluating an older checkpoint in the same run, without
submitting an obsolete SDK history step that would discard the result.

## TensorBoard and W&B

Both loggers use Megatron's designated logging rank. Each arm has its own
checkpoint, TensorBoard, W&B and artifact directories and a persisted W&B run ID;
both arms share the experiment group and data cache. Existing credentials stay
outside the repository. W&B saves offline records when connectivity is unavailable;
its sync status must be reported accurately. Weights remain on the training server.

| Metric | Meaning |
| --- | --- |
| `lm loss` | Main training CE, excluding auxiliary losses |
| `lm loss validation` | Native short periodic validation CE |
| `final/valid_ce`, `final/test_ce` | Complete independent holdout CE; W&B horizontal axis is `final/checkpoint_step` |
| `final/checkpoint_step` | Step of the evaluated checkpoint, independent of W&B's internal history counter |
| Native MTP/MoE metrics | Auxiliary losses, separate from main CE |
| `recipe/mtp_weight` | Actual auxiliary coefficient for this update |
| `recipe/phase` | 0 warmup, 1 stable, 2 decay |
| `recipe/update_lr` | Ordinary optimizer LR used by this update |
| `recipe/engram_update_lr` | Sparse table LR used by this update |
| `learning-rate` | Native LR after scheduler advancement |
| `recipe/completed_*_before_update` | Data progress before this update |

The ordinary training provider returns only train and validation datasets. Native
short test evaluation would reuse the validation metric tag at the same final step;
test CE is recorded only by the independent full-test invocation instead.

Native logging retains gradient norm, skipped updates, timing, throughput, and GPU
memory. Compare TensorBoard/W&B values during acceptance, and audit restart histories
before reporting a complete run. Offline W&B restarts may create multiple local
segments even with the same run ID; keep
all segments and reconcile overlapping steps when syncing or exporting. A run ID
alone does not prove that offline logs have been merged without duplicates.

Every invocation archives its configuration under `artifacts/sessions/`. The initial
training `artifacts/recipe.json` is preserved when restoring or running full evaluation,
so a later evaluation command cannot overwrite the original training identity.

Estimate the required training window from each arm's measured steady-step time on
the intended hardware, then add startup, checkpoint and validation overhead. Record
the source commit and configuration with that measurement. Short-run throughput is
useful for resource planning; it does not establish comparative convergence or a
sustained speedup.

## Audit logs and checkpoint restoration

Use native `torch_dist` checkpoints with `--dist-ckpt-optim-fully-reshardable`.
This selects the upstream optimizer format needed for supported topology changes;
the default DP-only format is insufficient for that acceptance requirement.
The flag alone is not proof of compatibility: save/load and topology-change tests
must verify model parameters, optimizer state, RNG, scheduler, and sample progress.

```bash
python -m examples.engram.check_logs --run-dir /path/to/acceptance/arm \
  --end-step 130 --require-health-metrics --output /path/to/log-audit.json
python -m examples.engram.check_events --run-dir /path/to/acceptance/arm \
  --output /path/to/event-audit.json
python -m examples.engram.check_checkpoint \
  --checkpoint /path/to/acceptance/arm/checkpoints/iter_0000128 \
  --iteration 128 --output /path/to/checkpoint-audit.json
python -m examples.engram.prepare_resume \
  --reference-run-dir /path/to/acceptance/arm \
  --resume-run-dir /path/to/acceptance/arm-resume
```

Add `--engram` to these tools for the Engram arm. The log audit then also checks
the table update LR. Execute the two commands emitted by `prepare_resume` in order:
restore 128, train/save 129, then restore 129 and
train/save 130 in the same branch run directory and W&B identity. The branch has
a new ID, so it does not mix resumed measurements into the uninterrupted reference.
The log audit's `--require-experiment-metrics` checks auxiliary losses, batch size,
memory and timing. This option does not change logging or upload existing records.
Audit its logs with `--start-step 129 --end-step 130`. Compare final model tensors:

```bash
python -m examples.engram.check_checkpoint \
  --checkpoint /path/to/acceptance/arm/checkpoints/iter_0000130 \
  --compare /path/to/acceptance/arm-resume/checkpoints/iter_0000130 \
  --iteration 130 --output /path/to/resume-comparison.json
```

The log audit reads actual TensorBoard events and W&B journal history. It requires
one finite scalar per metric/step, matching values, continuous samples/tokens,
and the preceding post-scheduler LR equaling the next update's LR. A persisted
run-ID file by itself is insufficient. The checkpoint inspector checks native
iteration, sample progress, scheduler, optimizer and RNG metadata. Tensor comparison
reads global logical model tensors one at a time on CPU and reports exact equality
or explicit differences. It does not mislabel model equality as a complete optimizer
state comparison; the native optimizer/RNG resume tests remain required.

For the complete same-topology continuation gate, compare every native storage
entry, including Muon momentum, dense FP32 masters/Adam moments, sparse row
coordinates/moments/masters/step counters, RNG objects, scheduler and consumed
samples:

```bash
python -m examples.engram.check_training_state \
  --left /path/to/acceptance/arm/checkpoints/iter_0000130 \
  --right /path/to/acceptance/arm-resume/checkpoints/iter_0000130 \
  --output /path/to/full-state-comparison.json
```

This defaults to exact equality. Explicit `--atol`/`--rtol` tolerances apply only
to floating values; all differences remain in the report and integer row counters,
coordinates and RNG bytes always require exact equality. Native TP/PP changes
intentionally omit restoring incompatible per-rank RNG streams; a topology test
may explicitly exclude RNG but cannot claim exact RNG continuation across that
change. The same-topology gate must retain RNG comparison.

`artifacts/events.jsonl` records startup/resume, MTP coefficient changes, validation,
completed checkpoints and process exit with the actual event step. TensorBoard stores
the same records as text. Live W&B event metadata uses run configuration updates,
so observing a checkpoint from an earlier step cannot rewind scalar-history steps.
After native W&B finalization, the final save/exit use native checkpoint artifacts
and exit records; `check_events` verifies that those records actually exist in the
local journals. An event's declared sink alone does not pass the audit. Abrupt
process termination can prevent final journal flushing and must be reported as such.
