<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. -->

# Energon data for heterogeneous MIMO

`examples/mimo/pretrain_mimo.py` defaults to mock data. Select
`--dataset-provider energon` to use native Energon datasets and task encoders
with the heterogeneous MIMO training loop. This bridge uses the repository's
existing Energon dependency; no dependency or source patch is required.

## Application configuration

`--data-path` takes one Energon dataset or recipe. Define blending and source
selection in the recipe. Select an importable application function with
`--energon-data-config MODULE:FUNCTION`. It receives the ordinary parsed training
arguments after runtime/tokenizer setup and returns `EnergonDataConfig`:

```python
from examples.mimo.training.energon_provider import EnergonDataConfig
from my_project.tasks import MyTaskEncoder, merge_encoder_batches


def make_config(args):
    return EnergonDataConfig(
        create_task_encoder=lambda *, split: MyTaskEncoder(split=split),
        checkpoint_identity="my-task-configuration",
        merge_encoder_batches=merge_encoder_batches,
        dataset_kwargs={"shuffle_buffer_size": 100, "max_samples_per_sequence": 1},
        loader_kwargs={"prefetch_factor": 2},
    )
```

Pass `--energon-data-config my_project.data:make_config --data-path /data/recipe.yaml`
alongside the existing model/topology arguments. The application supplies the
two task-specific functions in this example; this is not a built-in data recipe.

`create_task_encoder(*, split)` returns a fresh native TaskEncoder for every
language sample lane. The same encoder configuration runs on matching language
and encoder replicas. It owns cooking, media processing, tokenization, packing
callbacks, labels and loss masks. Do not use training rank or model role to change
sample selection. Follow Energon's restorable/stateless callback rules for randomness.
Native errors fail fast rather than dropping a sample on only one replica.

Native dataset/loader options are forwarded unchanged. The bridge reserves
path, task encoder, worker config and batch size to maintain lane alignment.
`shuffle_buffer_size` and `max_samples_per_sequence` default to None when omitted;
the bridge does not choose a packing strategy. `--num-workers` is native workers
per sample-lane loader; an encoder rank owns multiple such loaders when DP differs.
`--energon-logical-workers` optionally sets the global number of logical dataset
partitions across language sample lanes. The physical worker count is
`language_lanes * max(--num-workers, 1)`; the explicit logical count must be
positive, no larger than that count, and divide it evenly. For example, with four
language lanes and two physical workers per lane, `--energon-logical-workers 2`
keeps eight worker processes while assigning the dataset to two logical streams.
If omitted, the option is not passed to Energon and its native one logical
partition per physical worker behavior remains in effect. Encoder replicas
receive the same global logical count as language lanes; replica count does not
multiply it.

Normal pretrain tokenizer initialization is used. Applications may use the
existing tokenizer choices; prepared token IDs can use `NullTokenizer` with
`--vocab-size`. No image token spelling, template or tokenizer wrapper is imposed.
Configure the model provider's modality token IDs to match the TaskEncoder's
output; the bridge does not infer them from tokenizer spellings.

## Batch contract and encoder assembly

The TaskEncoder returns canonical MIMO dictionaries: `input_ids` has shape
`[micro_batch_size, sequence_length]`, with language labels/masks and optional
attention/packing metadata. On language ranks these fields pass through exactly;
only unused `modality_inputs` and native `__restore_key__` bookkeeping are removed.
The bridge never derives or shifts targets.

`merge_encoder_batches(batches, *, lane_ids, module_name)` receives ordered,
already-batched lane dictionaries, even when there is only one lane. It returns
the encoder's canonical forward dictionary. Preserve input rows in lane-major,
then sample-major order. `modality_inputs` must use the topology's `module_name`
as its outer key; inner encoder names/arguments follow the configured modality
submodule. The application owns tensor concatenation dimensions, packed offsets,
lengths, padding and missing modalities. Do not invoke TaskEncoder.batch here:
it expects individual samples inside native worker context.

The merger must be deterministic and stateless. It may materialize transferable
lazy media carried in `modality_inputs`, allowing language replicas to discard it without
decoding pixels/audio. It must not filter samples, advance a loader, use untracked
randomness or keep its own cursor. An entirely absent modality can omit
`modality_inputs` while preserving all input token rows.
As with all native worker outputs, deferred objects must be serializable and
usable in the receiving process; do not pass worker-owned locks or thread pools.

## Topology and checkpoints

Initial support is train-only (`--eval-iters 0`), disjoint grids, one encoder,
TP/PP/CP1 and no GTP rematerialization. Language-only mode is supported. Encoder
DP must divide the language sample-lane count. Each encoder reads its consecutive
group of language lanes, using identical native rank/world-size/seed/workers.
Micro-batch sizes above one are supported. Encoder feature prefetch and encoder
DDP overlap are rejected; language-model overlap is unaffected.

Unequal DP requires positive workers. Native zero-worker execution shares
process RNG/decorator seed state across lane replicas; it cannot guarantee
matching arbitrary randomized tasks. Equal-DP/language-only zero-worker use
still requires deterministic, natively restorable task callbacks.

Language lanes own the native state. Encoder replicas save none and restore
their corresponding language-lane files before iteration. `--dataloader-save`
defaults to model `--save`; `--energon-dataloader-load` defaults to model `--load`.
Both take checkpoint roots, not iteration directories or individual files:

```text
/checkpoints/run/iter_0000100/mp_rank_00/train_dataloader_dprank000.pt
```

State is saved at normal model checkpoint events. A separate dataloader load
root requires model `--load`, whose checkpoint determines the iteration.

Resume is fixed-configuration and strict. Missing state fails except when an
initial model-weight load explicitly starts a new run. The saved contract checks
native loader configuration, sample placement, seed, workers, microbatch size,
dependency version and application identity. Change `checkpoint_identity` when
application semantics hidden in callbacks change, and keep dataset artifacts
immutable. Cross-version, edited-recipe and changed-DP migrations are separate
capabilities; no consumed-samples fast-forward is substituted for saved state.
When `--energon-logical-workers` is explicitly selected, its value is also part
of the checkpoint contract, so changing it rejects strict restore. The default
omitted setting retains the existing contract shape. Constructor support is
checked only when the option is selected; dependencies without native
`WorkerConfig.logical_workers` support continue to serve default mode.
Exact replay with physical fan-out also requires native stride checkpoints to
preserve their pending discarded outputs. Constructor support alone does not
qualify that behavior; use a compatible checkpoint-safe dependency before
resuming a fanned-out stream.

The iterator owns all lane loaders and closes them on failure and training exit.
The existing dependency predates a public loader close API, so cleanup has a
small persistent-worker compatibility fallback; newer native close APIs are
used when available.

### Image metadata placement

Provide dynamic image sizes as CPU `imgs_sizes` tensors. The shared MIMO batch
mover retains this field on the host because vision encoders read its dimensions
in Python. Pixel tensors, token tensors and `PackedSeqParams` attention indices
still move to CUDA. This also applies when inputs are nested under a modality.
