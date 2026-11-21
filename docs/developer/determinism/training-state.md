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
```

The output directory must be new. By default the protocol runs four steps,
saves a checkpoint at step 2, and performs four separate process launches:

1. An uninterrupted reference, retaining the checkpoint.
2. An independent run from the same seeds and data recipe.
3. A new process that loads the reference checkpoint and executes steps 3–4.
4. A resume that omits RNG restore, which must produce an observed mismatch.

`--control optimizer`, `--control scheduler` and `--control dataloader` select
other deliberately omitted restores.
Fresh replay compares every step; resumed runs compare every post-checkpoint
step. Every declared rank is required, including ranks without logged loss.

The CPU pilot trains a small MLP with dropout, Torch AdamW, StepLR and generated
data in one process. It validates the snapshot format, the comparator and the
four-launch protocol, including the omitted-restore controls. It does not
certify any MCore GPU model, parallel layout, distributed optimizer, mixed
precision, FP8/FP4 or production data loader; those need their own adapters and
validation. The CPU protocol needs no GPU, and the unit tests run it end to end.

## Compare selected stop points

Per-step state capture enumerates and copies all state after every step.
To remove that diagnostic work between steps, launch independent protocols
that each capture only one selected boundary:

```bash
python -m tools.determinism.run_state_replay \
  --backend cpu --steps 5 --checkpoint-step 2 --stop-steps 3 5 \
  --output /tmp/state-stop-points
```

This runs eight independent worker launches: fresh reference, fresh repeat,
resume and omitted-restore control for step 3, then four new launches for step 5.
Every target must follow the checkpoint and be at most `--steps`. Each rank
must publish exactly one snapshot and completion step; earlier capture files,
missing ranks, mismatched capture declarations and failed workers are unverified.
The aggregate passes only when every requested target passes all three comparisons.
Each resume identifies its own target's reference checkpoint.

`--steps` remains the original training horizon: the scheduler and data recipe
are unchanged, and the worker stops after the target step. Before the target,
the worker skips diagnostic state collection and scalar extraction; it still
saves and records the checkpoint.

The result is still a scoped diagnostic recipe. These runs do not reproduce a
production recipe's execution or contention. Use the original recipe for
performance measurements and add a complete adapter before making a production
acceptance claim.

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
the corresponding identity contract for directory checkpoints, such as
distributed checkpoints. An adapter must record the checkpoint that the
framework's existing save path wrote and verify it around the existing restore
path, rather than relabel this pilot's files.

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
  --comparison resume --world-size 1 --steps 3 4 --output /tmp/comparison.json
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
(default 600 seconds); interruption terminates the worker process group.
