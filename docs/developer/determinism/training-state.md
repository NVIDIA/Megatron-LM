---
orphan: true
---

# Training-state replay and checkpoint resume

The state protocol compares independent fresh processes and uninterrupted versus
checkpoint-resumed training. It also runs a broken resume that deliberately
omits a state restore. Loss curves, first-moment comparisons and same-process
forward/backward tests do not establish this contract.

## Run the pilot

Use a clean checkout and an output directory outside the source tree:

```bash
# CPU training validates the harness; it is not MCore GPU evidence.
python -m tools.determinism.run_state_replay \
  --backend cpu --output /tmp/state-replay-cpu

# One node, one worker per GPU, TP=4. Requires the MCore GPU dependency stack.
python -m tools.determinism.run_state_replay \
  --backend mcore_gpt --world-size 4 --output /tmp/state-replay-gpu

# Actual pretrain_gpt loop: TP=2, DP=2, BF16 and distributed optimizer/checkpoints.
python -m tools.determinism.run_state_replay \
  --backend megatron_gpt --world-size 4 --output /tmp/state-replay-training
```

The output directory must be new. By default the protocol runs four steps,
saves a checkpoint at step 2, and performs four separate process launches:

1. An uninterrupted reference, retaining the checkpoint.
2. An independent run from the same seeds and data recipe.
3. A new process that loads the reference checkpoint and executes steps 3–4.
4. A resume that omits RNG restore, which must produce an observed mismatch.

Both GPU adapters require the early `megatron.determinism` API from
[MCore #7419](https://github.com/NVIDIA/Megatron-LM/pull/7419). Each worker configures
the shared policy before importing training or Core, whose optional GPU backends
can initialize CUDA during import. The training entrypoint validates the resolved
recipe again. The CPU harness does not require the GPU startup package.

For `cpu` and `mcore_gpt`, `--control optimizer`, `--control scheduler` and
`--control dataloader` select other deliberately omitted restores. The real
`megatron_gpt` training adapter currently supports the RNG control, using the
existing checkpoint loader's `no_load_rng` flag. No goldens are changed.
Fresh replay compares every step; resumed runs compare every post-checkpoint
step. Every declared rank is required, including ranks without logged loss.

The CPU pilot trains a small MLP with dropout. The GPU pilot trains a two-layer,
64-hidden-size MCore GPT using TP=1/2/4/8, FP32 parameters and gradients, Torch
AdamW, StepLR, dropout, and generated data. Its objective is mean squared local
logits, not language-model cross entropy. DP/PP/VPP/EP/FSDP, MCore's distributed
optimizer, mixed-precision master weights, FP8/FP4 and production data loaders
need additional adapters and validation. This pilot does not certify those
paths or the named Nemotron/DSV recipes.

The H100 and GB200 `determinism-state.yaml` recipes each select two per-step jobs: the
`mcore_gpt` pilot and the `megatron_gpt` training adapter, with eight and four
ranks respectively. They use the existing integration-test
selection path; actual scheduling still depends on the CI scope and protected
runner approval. Each also adds six stop-point jobs to nightly cadence (or an
explicit cadence bypass): both original GPU adapters plus TP=2/PP=2 training
with one or two virtual chunks, the precision-aware optimizer recipe below,
and the hybrid CPU/GPU Adam recipe below,
using steps 3 and 5 of a five-step schedule. The normal
PR selection retains its two existing jobs. A configured recipe is not GPU
execution evidence.

## Compare selected stop points

Per-step state capture synchronizes the device and copies state to the host.
To remove that diagnostic work between steps, launch independent protocols
that each capture only one selected boundary:

```bash
python -m tools.determinism.run_state_replay \
  --backend megatron_gpt --world-size 4 \
  --steps 5 --checkpoint-step 2 --stop-steps 3 5 \
  --output /tmp/state-stop-points
```

This runs eight independent worker groups: fresh reference, fresh repeat,
resume and omitted-restore control for step 3, then four new groups for step 5.
Every target must follow the checkpoint and be at most `--steps`. Each rank
must publish exactly one snapshot and completion step; earlier capture files,
missing ranks, mismatched capture declarations and failed workers are unverified.
The aggregate passes only when every requested target passes all three comparisons.
Each resume identifies its own target's reference checkpoint.

`--steps` remains the original training horizon. The real Megatron adapter uses
`--exit-interval` to stop at the target while preserving `--train-iters`, the
learning-rate schedule and dataset indexing. It observes the existing exit
decision and accepts only a successful target-step exit after training cleanup.
It keeps the original post-step callbacks and checkpoint save/load behavior.
The TP-only/CPU worker similarly skips diagnostic state collection, explicit
device synchronization and scalar extraction before the target.

The result is still a scoped diagnostic recipe. Normal logging, callbacks and
synchronous checkpoints remain, including checkpoint identity checks. These
runs do not prove an absence of all synchronization, validate unsupported
overlap modes or reproduce a production recipe's contention. Use the original
recipe for performance measurements and extend the adapter before making a
production acceptance claim.

## Real Megatron training adapter

`megatron_gpt` executes the repository's `pretrain_gpt.py`, including its real
language-model loss, forward/backward schedule, distributed Adam optimizer,
learning-rate scheduler, sampler, and synchronous `torch_dist` save/load path.
By default it uses TP=2, PP=CP=1 and DP=2/4 on four/eight GPUs, BF16 model weights and FP32
master parameters, dropout, and two 128-hidden-size transformer layers.
MockGPT uses 512 documents with maximum document length 64; those explicit
test-data dimensions are recorded alongside the full training arguments.

Select `--pipeline-size 2` with `--backend megatron_gpt` to exercise the existing
pipeline schedule with one layer per stage. On four GPUs this is TP=2/PP=2/DP=1;
on eight GPUs it is TP=2/PP=2/DP=2. Global batch size remains the world size, so
each step uses four microbatches and covers pipeline warmup, steady state and
cooldown. For example:

```bash
python -m tools.determinism.run_state_replay \
  --backend megatron_gpt --world-size 4 --pipeline-size 2 \
  --steps 5 --checkpoint-step 2 --stop-steps 3 5 \
  --output /tmp/state-pipeline-stop-points
```

Every rank records its actual TP/PP/DP/CP group sizes and coordinates. The
coordinator requires each TP/PP/DP coordinate exactly once, with the expected
data-loader owners. With two stages, both first and last stages construct data
on TP rank zero; their TP peers receive the broadcast. Capture also checks one
model chunk, the local layer count, and the embedding/output endpoint roles.
All stages must supply model state, gradients and local optimizer moments;
missing stage records cannot pass. The actual synchronous distributed checkpoint
is retained and verified for every rank, including both pipeline stages.

Add `--virtual-pipeline-size 2` to the PP=2 recipe to exercise four transformer
layers with two virtual chunks per physical stage, one layer per chunk:

```bash
python -m tools.determinism.run_state_replay \
  --backend megatron_gpt --world-size 4 \
  --pipeline-size 2 --virtual-pipeline-size 2 \
  --steps 5 --checkpoint-step 2 --stop-steps 3 5 \
  --output /tmp/state-virtual-pipeline-stop-points
```

Megatron requires P2P overlap for this two-stage interleaved schedule. The
adapter requires its ordinary overlap policy with output deallocation enabled,
non-batched P2P, and warmup/flush overlap disabled. The existing schedule waits
on sends before output deallocation, drains backward sends, and asserts its
receive-work queues are empty before returning. The observer retains that
schedule and captures after optimizer/scheduler updates; target capture also
synchronizes the device. Grad/parameter-gather overlap and deferred embedding
weight gradients remain unsupported.

Every virtual chunk has an explicit identity, global layer number, endpoint
role, communication policy and loader slot in provenance. Missing, reordered
or duplicated chunks cannot pass. With this layout only chunk zero on physical
stage zero owns embeddings/data, and chunk one on physical stage one owns the
output/loss data; only TP rank zero builds their loaders. Other local slots
explicitly record that they have no loader. Each live chunk iterator retains
its own data/index arrays, RNG and checked actual sample cursor. Model and
gradient state include both chunks; the optimizer adapter includes all local
parameter/moment shards. The reference's actual distributed checkpoint must
restore all of them.

Only the documented PP=1/2 recipes and the two-chunk PP=2 virtual recipe are
supported. Other pipeline sizes, virtual layouts and overlap policies require
additional state/boundary validation. Declaring more ranks does not enable them.

The diagnostic worker temporarily observes the training module's loader,
train, checkpoint and post-step callbacks. The callbacks retain their original
behavior and return values. Capture runs after optimizer/scheduler updates and
consumed-sample bookkeeping, before checkpoint save and the next zero-grad.
Successful entrypoint completion (or the validated stop-point exit) and every
required capture remain mandatory.

The optimizer adapter reads each chained optimizer's **inner** state dict and
local master-parameter groups. The distributed optimizer's outer `state_dict`
intentionally omits parameter-dependent moments and is insufficient. Both Adam
moments, group/step state, local master parameters/gradients, loss scale and
scaler state are captured without gathering shards. Every rank is required.

### Precision-aware Adam storage

Select `--optimizer-mode precision_aware_fp16` for a separate BF16 TP=2/PP=1
recipe with FP32 main gradients, scaled FP16 first/second moments, and the
native BF16-plus-`int16` master-parameter remainder representation:

```bash
python -m tools.determinism.run_state_replay \
  --backend megatron_gpt --world-size 4 --optimizer-mode precision_aware_fp16 \
  --steps 5 --checkpoint-step 2 --stop-steps 3 5 \
  --output /tmp/state-precision-aware-stop-points
```

TE's checkpoint `state_dict()` converts low-precision moments and does not
include the separate scaling map. This adapter reads the raw optimizer state
through the base PyTorch serializer, retaining FP16 moment bytes, both FP32
scales, `int16` remainder bytes, group/step state, optimizer policy, dtype-range
tensors and overflow state. It neither casts these states to FP32 nor changes
the native checkpoint save/load implementation.

Every local optimizer shard is bound to its actual model chunk, parameter name
and byte-sharing element range. Missing moments, scales or remainders, wrong
dtypes, copied or misidentified shards, unsupported hooks and unknown state
keys cannot pass. CPU controls independently perturb the second moment,
master remainder and scale bytes and require the unchanged state comparator
to detect each difference.

This named optimizer recipe does not enable other moment/master dtypes,
offloading, quantized model parameters, capturable optimizers or combinations
with PP/VPP. Those require their own complete storage and boundary adapters.

### Hybrid CPU/GPU Adam storage

The `hybrid_precision_aware_fp32` mode uses the actual
Megatron distributed optimizer with 50% CPU offload, Torch AdamW on CPU,
TE FusedAdam on GPU, pinned CPU copies and native D2H/H2D overlap. It uses
FP32 moments and masters and routes BF16 model shards through the
precision-aware distributed interface, as required by Megatron's CPU offload
CLI. This mode requires the native
correctness fixes in [#7449](https://github.com/NVIDIA/Megatron-LM/pull/7449),
in addition to the early startup API in #7419.

```bash
python -m tools.determinism.run_state_replay \
  --backend megatron_gpt --world-size 4 --optimizer-mode hybrid_precision_aware_fp32 \
  --steps 5 --checkpoint-step 2 --stop-steps 3 5 \
  --output /tmp/state-hybrid-stop-points
```

Capture includes every CPU/GPU child and outer state, both moments, CPU
per-parameter and GPU group counters, parameters and gradients, live master
and CPU copies, pinned gradient buffers, defaults, dispatch/storage policy,
native hook code/ownership and storage aliases. The model mapping uses actual
tensor storage and named shard ranges: hybrid parameter order can differ from
the distributed wrapper's earlier group positions. Copied or unowned shards,
stale parameter/gradient copies, unknown state and changed hooks are unverified.
Native D2H events must be drained and both distinct transfer streams complete
before the existing capture barrier; the adapter does not drain an incomplete
transfer to make a snapshot pass.

Two schema decisions are explicit in provenance. The outer optimizer's missing
`differentiable` option and the False value inserted by PyTorch restore both
mean False; True and non-boolean values are rejected. Child defaults remain
exact. Constructor-only CPU/GPU group containers retain validated parameter
membership; their stale option dictionaries are not live optimizer state.
The active child groups, all outer options and child defaults are captured in
full. No tensor or scalar numerical tolerance is introduced.

This named TP=2/PP=1 recipe requires actual partial CPU/GPU ownership on every
rank. Full offload, other backends, unpinned copies, low-precision moments,
quantized parameters, additional overlap and PP/VPP combinations remain
unverified until their own complete adapters and GPU protocols pass.

The loader adapter supports the single-pass MockGPT sampler with zero workers.
It records the actual index arrays, document lengths, cached masks/positions,
dedicated loader RNG and sampler configuration. It derives the absolute next
sample from the iterator's observed yields plus its initial sampler position,
then checks that against training's consumed-sample counter. A resumed iterator
has a different local yield count; its canonical next sample must match.
Worker prefetch, cyclic/external loaders and real-data content identity require
additional adapters. They are rejected, rather than represented as only a cursor.

The reference records every file in the completed distributed checkpoint
directory. Resume validates that same directory before and after Megatron's
load and checks the returned iteration. Comparison rechecks the real files;
changed, missing or additional shards invalidate the evidence. Checkpoint
hashes identify the loaded files; state comparisons still use raw bytes.

This is an additional GPU validation recipe, **not an executed GPU result**.
FP8/FP4, other precision-aware/offloaded optimizer configurations, other communication overlap policies,
broader PP/VPP/CP/EP/FSDP layouts, real datasets and production recipe stop-point validation
remain separate work. Unsupported state formats fail visibly.

## Capture contract

`tools.determinism.training_state.write_snapshot` accepts seven required
components: model parameters/buffers/extra state, gradients, optimizer,
precision, RNG, scheduler and data-loader state. Capture after the optimizer
and scheduler updates, before zeroing gradients. Synchronize pending device
and optimizer work first. Enumerate all model chunks, both ordinary gradients
and any `main_grad` buffers, and all state that can affect the next step.

The adapter owns completeness: calling `state_dict()` is not a universal
guarantee that every backend exposes its precision or optimizer state. Record
disabled features explicitly with a reason/configuration, not an empty object.
Include Python, NumPy, Torch CPU, each owned CUDA device and model-parallel RNG
streams as applicable. A data cursor alone is insufficient when a loader also
owns generator, sampler, worker or prefetch state.

The writer stores a JSON index and raw logical bytes. It preserves shape, dtype,
signed zero and NaN payloads; it excludes unused tensor-storage padding.
Plain dense tensors, NumPy numeric arrays/scalars, mappings with string/integer
keys, sequences, Python scalars and byte buffers are supported. Sparse,
quantized, DTensor/backend-specific tensor subclasses and opaque objects need
explicit adapters. Unsupported or empty required state is unverified, never
silently skipped. Model and gradient comparisons must contain nonempty arrays.

After successful training, check that source/runtime provenance did not change
and call `complete_rank` with every captured step. The reader requires these
completion records; a partially failed run cannot pass from its earlier files.
Independent process/run identities and matching source, software, hardware,
environment and recipe records are required. Dirty-source observations remain
available for debugging but do not pass the gate. Provenance is recorded by
the protocol, not cryptographic attestation.

The pilot saves each rank's checkpoint with `checkpoint_path` and
`record_checkpoint`. Resume verifies the actual file identity and records the
reference run, rank, step and checksum. Checkpoint checksums establish which
file was loaded; state equality is checked using the complete captured bytes,
not hashes. `record_checkpoint_directory` and `read_checkpoint_record` provide
the corresponding identity contract for the real distributed checkpoint;
the adapter continues to use Megatron's existing producer/restore path.

## Inspect results

`report.json` contains fresh/resume/control results, required and compared
snapshot counts, provenance, and the first differing step, rank and state path.
Byte differences also report byte offset and, for arrays, flat element index.
Command lines, worker logs, raw state and checkpoint files remain alongside it.
In stop-point mode the root report aggregates `stop-00000003/report.json`, etc.;
each subdirectory retains a complete four-launch protocol. Capture mode, full
training horizon, checkpoint and target step are included in provenance.

The protocol exits 0 only when both normal comparisons match and the
deliberately broken resume differs. Exit 1 means a comparison/control failed;
exit 2 means evidence is missing, incompatible, dirty or a worker failed.
An observed difference does not by itself identify a nondeterministic kernel;
incorrect restore, source/context drift and adapter omissions also need diagnosis.

To compare existing captures without running training:

```bash
python -m tools.determinism.training_state /tmp/run/reference /tmp/run/resume \
  --comparison resume --world-size 4 --steps 3 4 --output /tmp/comparison.json
```

The comparison command exits 0 for equal, 1 for different and 2 for unverified.
These results cover the declared adapter, ranks and steps. They are separate
from operator coverage, independent-reference correctness and performance.
Capture synchronizes and copies state to the host; use the original,
uninstrumented recipe for performance measurements. Per-step capture can also
change scheduling between steps. The stop-point mode removes earlier diagnostic
snapshots, but final production validation still needs the original recipe's
execution and contention conditions and complete adapters for its state.

The broken-resume control must change model, gradient, or optimizer state beyond
the omitted component. An RNG-only or scheduler-counter-only difference does not
pass the sensitivity gate. Each worker phase has a configurable `--phase-timeout`
(default 600 seconds); interruption terminates the worker process group. GPU
workers require the shared startup API from #7419; hybrid capture also requires
the optimizer state fixes from #7449. These are landing prerequisites.
