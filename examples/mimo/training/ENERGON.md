<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. -->

# Energon data for heterogeneous MIMO

`examples/mimo/pretrain_mimo.py` defaults to mock data. Select
`--dataset-provider energon` to use native Energon datasets and task encoders
with the heterogeneous MIMO training loop.

Use the repository's uv development environment for prepared BinIdx recipes and
streaming checkpoints. `pyproject.toml` and `uv.lock` pin the required native
interfaces and the worker-fanout checkpoint fix at an exact source revision. Other installation
methods must provide a compatible Energon version with those interfaces.

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

Applications may optionally supply
`create_encoder_task_encoder(*, split)` in `EnergonDataConfig`. The bridge uses
it only on encoder replicas; existing applications keep using
`create_task_encoder` for both roles. This hook may change terminal stateless
batch materialization, such as preparing CPU image tensors in loader workers.
It must preserve sample selection, packing, native loader configuration and
checkpoint state: authoritative language checkpoints must restore both roles.
Use the same encoder class and callback identities, and validate actual replay,
not only matching configuration hashes. Mark a terminal callback skip-safe only
when omitting it cannot change the subsequent stream or required state.

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

The prepared RADIO application uses the optional encoder factory to read,
decode and transform images in native loader workers. Its terminal batch is
explicitly `stateless(skip_safe=True)`, using Energon's existing API to omit
heavy work for sibling-owned outputs during physical-worker fanout. Language
replicas retain lazy image plans and perform no pixel reads. Encoder consumers
assemble ordered ready tiles and packed attention metadata without repeating
image IO or transforms; direct lazy batches retain the existing merge path.
No sampling, image budget, label, geometry or checkpoint format changes are
introduced by this work placement. With `--num-workers 0`, the same callback
runs in the caller; positive workers enable parallel preparation. Prefetched
decoded tensors consume more memory than lazy plans, so worker/prefetch settings
still need sizing for the selected image policy.

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

### Explicit fixed-worker-count language-DP migration

`--energon-dp-resume` enables a separate checkpoint format for changing the
number of language sample lanes while keeping the global physical worker count
fixed. It requires `--micro-batch-size 1`, positive `--num-workers`, a positive
`--global-batch-size` divisible by `language_lanes * --num-workers`, and one
logical stream per physical worker (`L=P`). This migration does not support
`L<P`, zero-worker loaders, changed total worker count, or changed global batch
size. These checks apply to the actual language-lane count used to construct
each loader.

The mode stores a format-2 profile alongside the ordinary native worker state.
It binds application identity, dependency version, task and recipe semantics,
global batch size, the normalized native pipeline, and ordered per-worker
sampler partitions. Only known native dataset edges are interpreted; opaque
factory and cooker configuration remains part of the strict semantic profile.
The native config hash is rechecked before profile use. Migration is accepted
only when every source lane is present once, all worker progress agrees, each
lane is at `next_worker_id=0`, and the saved state is on a committed global
batch boundary. Native worker states are regrouped in rank-major order; model,
optimizer, RNG and scheduler checkpoint handling remains Megatron's normal
resume path.


Stock Megatron ignores saved model RNG when the total world size changes. This
command preserves native dataloader worker RNG and leaves that model-RNG policy
unchanged. Same-topology model RNG restoration uses the ordinary framework path.

To migrate, pass each source language-lane dataloader file explicitly and write
to a new output directory. For example:

```bash
python -m examples.mimo.training.energon_checkpoint \
  --input-dataloader-checkpoints \
  /checkpoints/run/iter_0000100/mp_rank_00/train_dataloader_dprank000.pt \
  /checkpoints/run/iter_0000100/mp_rank_00/train_dataloader_dprank001.pt \
  --new-language-lanes 1 \
  --output-directory /checkpoints/run/dataloader-migrations/iter_0000100/mp_rank_00
```

The command accepts individual dataloader-only files, never discovers files
from a checkpoint tree, and refuses an existing output directory. Resume using
`--energon-dataloader-load /checkpoints/run/dataloader-migrations` and
`--energon-dp-resume`, keeping model loading on the normal `--load` path. The
output iteration must match that model checkpoint. Writing multiple rank files
is not atomic: on an interrupted or failed write, preserve the partial directory
for diagnosis and retry into a new output root. Format-2 checkpoints always
require DP-resume mode, including same-layout restores, so their semantic and
global-batch profiles cannot be bypassed by disabling the flag. A migrated state must first be restored and saved
normally at the destination topology before it can be migrated again.

The iterator owns all lane loaders and closes them on failure and training exit.
The existing dependency predates a public loader close API, so cleanup has a
small persistent-worker compatibility fallback; newer native close APIs are
used when available.

### Image metadata placement

Provide dynamic image sizes as CPU `imgs_sizes` tensors. The shared MIMO batch
mover retains this field on the host because vision encoders read its dimensions
in Python. Pixel tensors, token tensors and `PackedSeqParams` attention indices
still move to CUDA. This also applies when inputs are nested under a modality.

## Prepared BinIdx causal training with RADIO

Use the prepared-token cooker and image-safe streaming task through the ordinary
application hook:

```text
--dataset-provider energon \
--energon-data-config examples.mimo.training.prepared_binidx:make_config \
--data-path /data/recipe.yaml \
--tokenizer-type NullTokenizer --vocab-size 262144 \
--energon-eos-token-id 2 --energon-pad-token-id 3 --image-token-id 18 \
--energon-image-token-budget 1024 --seq-length 8192 \
--patch-dim 16 \
--dynamic-resolution --disable-vision-class-token --pixel-shuffle
```

The encoder bounds IDs using the initialized runtime tokenizer vocabulary,
including model padding. The null tokenizer supplies only that vocabulary for
already prepared IDs; it does not process the documents. Token IDs and the
vocabulary size in this example are illustrative: pass the actual prepared corpus and
model IDs. No tokenizer, BOS replacement, EOS insertion or second target shift
runs in this application. The EOS/PAD flags are required only by this selected
factory. `--energon-image-token-budget` limits emitted visual embeddings per image
and must fit the sequence length; source image-end/break IDs remain ordinary text
positions with their original eligibility. Dynamic resizing uses existing RADIO
normalization and aspect-ratio planning, with optional 2x2 pixel shuffle and no
thumbnail. Decoded image dimensions must agree with the original width/height metadata;
realized emitted geometry must also agree with the planned visual spans. Both
checks run before model forward.

The recipe uses the prepared cooker tags and native auxiliary stores described
in [PREPARED_BINIDX.md](../../multimodal/data_loading/PREPARED_BINIDX.md). Optional
`indexed-token-mask` companions use the same source-bound target eligibility
protocol. The factory registers that protocol and uses native `NoCachePool`;
no custom filesystem, image-storage or sampling implementation is required.

Native packing stays isolated per source. Each batch has fixed `[B,S]` IDs,
already aligned labels/masks and row positions `0..S-1`. Attention is ordinary
causal attention across each packed row, including source document boundaries;
positions do not reset at fragments. Targetless completed packs reject as described
in the streaming task documentation. Other attention/BOS/fully-masked policies
are separate capabilities.

Language ranks consume IDs and targets without reading image pixels. Encoder
loader workers realize the selected plans; the merger assembles their ready
tiles in lane-major, then batch-row order and
supplies the actual RADIO wrapper's `x`, host `imgs_sizes` and packed-attention
metadata. Realization checks patch geometry against the visual spans. Native
checkpoint cursors remain authoritative; the application identity fingerprints
all selected token/objective/sequence/image policies hidden inside callbacks.
The application supports dynamic RADIO and native Pixtral with dropped class
tokens. It does not add video, token-budget dataset subsets or custom storage.


## Native Pixtral and prepared BinIdx inputs

The prepared-data application supports the MIMO native Pixtral tower as an
opt-in alternative to RADIO. Select `--vision-model-type pixtral-vit-1p6b`
for the 30-layer, 2048-wide tower with FFN6144, 16 heads, patch14 and native
2x2 spatial merging. This is a separate preset from `pixtral-vit-large`, whose
architecture is 48 layers/1664 hidden units. The new preset is qualified through
the MIMO prepared-data route. Conventional VLM entrypoints are not qualified
by this addition.

Use these encoder/data options with the normal MIMO language-model and topology
arguments. Encoder TP must be 1 for the optional pretrained initialization:

```bash
--dataset-provider energon \
--energon-data-config examples.mimo.training.prepared_binidx:make_config \
--vision-model-type pixtral-vit-1p6b \
--dynamic-resolution --disable-vision-class-token \
--patch-dim 14 --img-h 1540 --img-w 1540 \
--vision-projection-type mlp2 --freeze-vit \
--load-vision-from /path/to/vision_architecture_export \
--exit-on-missing-checkpoint \
--energon-image-token-budget 3025 --seq-length 8192
```

Do not enable RADIO pixel shuffle for native Pixtral. Its own merger reduces
patch count by four while retaining the 2048-wide output, so the projector reads
2048 features. Processed images retain their aspect ratio within patch-grid
rounding, align both axes to 28 pixels and stay within the configured height and
width. The budget counts emitted visual embeddings. A 1540x1540 image has 12100
premerge patches and 3025 emitted embeddings. Packed attention boundaries use
premerge counts, and `imgs_sizes` remains on CPU during model transfer. CLIP
normalization uses the existing public image-processing constants.

The full MIMO entrypoint derives omitted image/patch values from the selected
preset before stock validation. Explicit incompatible native patch sizes and
pixel shuffle are rejected. Existing RADIO selection, mock defaults and prepared
replay identities remain unchanged. Native Pixtral mock mode is not enabled by
this addition. Use the prepared Energon application or provide an application
that follows the same packed native-image geometry.

`--load-vision-from DIRECTORY` is an optional initial architecture-export load,
performed on encoder ranks before DDP and optimizer construction. It expects a
PyTorch distributed checkpoint with exactly every named encoder parameter under
`model.vision_model.`, matching shapes and BF16 source tensors. Runtime-only
Transformer Engine state is retained. Missing/extra weights, a different
architecture, nonfinite values, meta parameters and unsupported TP are errors.
The loader copies into existing parameter objects and verifies the result.

A full training `--load` takes precedence over the encoder export. Combining
both flags requires `--exit-on-missing-checkpoint`, because stock missing-tracker
fallback must not create a frozen random tower. A full `--pretrained-checkpoint`
also bypasses the separate encoder export and retains its stock strict loading
behavior. For a fresh run, omit both full-model checkpoint options. Keep native
checkpoint identities with their immutable data and image policies when resuming.

Freezing the encoder leaves the selected projector trainable unless separately
frozen. Matching this encoder does not by itself match a reference language
model, data blend, resize policy, optimizer, precision or learning-rate schedule.
Those choices must also be aligned for a parity comparison.
