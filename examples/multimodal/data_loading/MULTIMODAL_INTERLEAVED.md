# Multimodal interleaved pretraining

This opt-in path adds causal pretraining with image-safe streaming packing.
It currently supports text and images, not video or audio. The umbrella name
leaves room for future media types; it does not change the objective to SFT.
Existing supervised and mock paths remain available. Initial CPU and one-GPU
pipecleaning does not certify full-model training, scaling or learning parity.

## Ownership

The shared interleaved TaskEncoder owns causal encoding and image-safe split/pack
callbacks. Energon owns source mixing, worker sharding, carryover and replay.
Consumer bridges own batch formatting; providers own role/lane placement, state
delegation and lifecycle. Current model/communicator/forward-step contracts stay.
Mock, interleaved encoder batches and encoder-lane fanout share encoder-input
collation in `examples/mimo/training/batch.py`; mock generation remains separate.
Both interleaved providers share strict checkpoint restoration in the loader.

## Consumer adapters and future extensibility

The intended split is shared document processing followed by consumer-specific
batch adapters, not a complete encoding/packing pipeline for each modality mix.
The current adapters have different model contracts:

- [`VlmInterleavedTaskEncoder`](multimodal_interleaved_vlm_provider.py) and
  `build_interleaved_vlm_loaders` are the batch adapter and provider in the same
  module, not two independent pipelines. They target the conventional LLaVA
  consumer: compact image sentinels remain in the batch and the model expands
  them. The VLM name accurately describes today's vision-and-language scope.
- [`MimoInterleavedTaskEncoder`](../../mimo/data/energon_provider.py) targets the
  canonical MIMO batch contract, expands image slots in the batch, and separates
  language targets from encoder inputs. Although MIMO supports multiple modality
  encoders architecturally, this interleaved provider currently requires exactly
  one encoder grid. It accepts the resolved `radio`, `pixtral-vit`, or
  `pixtral-vit-large` image policy; it is not an AVLM provider today.

The shared core is extensible, not already modality-generic:
[`InterleavedDocument`](multimodal_interleaved_samples.py) currently contains
strings, prepared `TokenizedText` and `ImageMedia`; encoded records carry
`image_params` and image positions.
The cooker, token-cost calculation, protected split boundaries and materialization
also handle images specifically. Adding an AVLM wrapper alone cannot enable audio.

### Future audio/video extension checklist

- Extend the shared document schema and cooker with explicit typed media and
  metadata, retaining lazy decoding. Audio needs its own duration/sample-rate
  policy; video needs frame/timestamp and temporal-boundary policy. Do not encode
  either as an untyped image reference merely to pass the existing cooker.
- Generalize media plans, token costs and protected spans as each modality is
  implemented. Reuse the shared causal encoding, streaming packing and replay
  machinery instead of cloning it. Preserve EOS and next-token alignment, and
  define new target/loss-mask semantics explicitly: the current image-conditioning
  mask is not automatically correct for an audio-generation objective.
- Extend consumer materialization and batch translation. For MIMO, extend the
  example's single-encoder grid/CLI construction and validate modality routing
  and language/encoder lane placement for multiple encoder grids. Removing the
  provider's one-encoder guard alone is insufficient.
- Add a thin AVLM adapter only when a different consumer needs a distinct batch
  contract. Otherwise extend the existing consumer adapter. Consumer-based names
  such as LLaVA and MIMO may become clearer than VLM/AVLM as support grows; no such
  API rename is implemented here. Video support can still be vision-and-language
  support and does not by itself require an AVLM adapter.
- Keep existing text/image behavior covered while adding modality-specific token
  budgets, boundaries, loss masks, lazy role ownership and checkpoint replay
  tests. Include new policies in the resume contract; reject incompatible old
  state rather than promising an untested migration.

Existing conversation/SFT encoders are not drop-in replacements for this causal
path. Reuse appropriate media-processing primitives without importing their
chat-template, assistant-supervision or packing semantics wholesale. No audio or
video support is enabled by this design guidance.

## Selection and development compatibility

The conventional VLM entrypoint opts in with `--multimodal-interleaved`.
MIMO selects it with `--dataset-provider energon`. Related options use the
`--multimodal-interleaved-` prefix:

| Suffix | Meaning |
| --- | --- |
| `min-image-patches`, `max-image-patches` | Total input patches per image before native merging or pixel shuffle; patch size comes from the selected encoder. Defaults 4 and 4096; neither grid-side lengths nor final image embeddings. |
| `shuffle-buffer-documents` | Documents per worker per packing-group branch before encoding/packing; default 0 disables buffering. A buffer of 1 gives no shuffle. |
| `dataloader-checkpoint-root` | Root containing iteration and rank directories with the dataloader `.pt` files; defaults to the model `--load` root. It requires a model checkpoint load. |

MIMO uses `--seq-length` for the expanded model-token budget; the conventional VLM
adapter uses `--decoder-seq-length`. Image slots count toward that budget. MIMO's
`--image-seq-length` and `--image-token-id` configure mock data; the real provider
derives lengths from patch plans and the structural ID from the tokenizer.
`--mock-dataset-size` counts samples in each lane's split-local synthetic dataset.
Real Energon workers are selected with `--num-workers`; mock loaders are single-process.

Mock data validates encoder geometry and batch/control-flow contracts while
retaining its upstream target-only mask. Its final image input may predict text.
The real interleaved provider masks both image inputs and image targets; mock
losses are not a substitute for checking that real-data supervision contract.

### Encoder-dependent image geometry

MIMO resolves `--vision-model-type` before creating workers. The selected
registry metadata supplies patch size, normalization, image bounds, and native
reduction; explicit supported options are validated against that policy. This
does not construct an encoder or load weights, so the same policy is available
for dataloader-only checks. Interleaved image-policy construction requires a
complete encoder spec; an explicit `patch_size`, if supplied, must match it.
Missing patch metadata is an error, with no implicit patch16 fallback. Patch
size alone does not define the required reduction and normalization policy.

The conventional `VlmInterleavedTaskEncoder` remains RADIO-only. Real MIMO
interleaved data requires dynamic resolution for every supported encoder,
including RADIO: pass `--dynamic-resolution` for `radio` and `pixtral-vit`.
The RADIO fixed-resolution default is retained for existing model/mock callers,
not real interleaved training. `pixtral-vit` uses patch16 with no native merger.
`pixtral-vit-large` uses patch14 and a native 2x2 merger, deriving dynamic
resolution when omitted. Do not enable
`--pixel-shuffle` for native ViT. Encoder selection, architecture overrides and
complete encoder-option examples are in [the MIMO guide](../../mimo/README.md).

Both real interleaved providers reject fixed resolution, `--use-tiling`,
`--match-tiling-dynamic-resolution`, `--masked-tiling-dynamic-resolution`,
thumbnails, tile tags and tile-based aspect-ratio selection. Tile counts must
remain one. These options cannot silently select a different preprocessing path.
Each image produces one dynamic patch grid; RADIO's 224x224 model-construction
defaults do not force square resizing or cap its dynamic grid dimensions.

For a resized image of height h and width w, input-patch count is
`(h / patch_size) * (w / patch_size)`. Model image-token count additionally
divides each grid axis by the resolved spatial reduction. Both axes must be
integral and fit the image bounds; an invalid grid is rejected, not floored.
The existing resize strategy aligns the planned grid before collation.
At 224x224, RADIO patch16 with shuffle2 emits49tokens, whereas patch14 with
native merge2 emits64. Native merge retains feature width H; RADIO pixel
shuffle emits4H. The projector uses the corresponding encoder output width.

Patch budgets always count input patches: for example, a maximum of64 means
at most64 patch14 input cells, hence16 output tokens for native merge2.
Premerge packed-attention lengths remain separate from those output counts.
Mock and real MIMO data share these accounting rules and the existing loss
masks. Image conditioning slots, image targets, and padding remain masked;
ordinary text targets retain the causal objective.

Dataloader-only validation predicts image-token counts from metadata and checks
pixels, grids, packing, and masks. When a model is instantiated, the native
adapter also checks actual patch projection geometry, output count and feature
width. Pixel and packed-attention tensors move to CUDA; native `imgs_sizes`
metadata stays on CPU. The included native-model tests exercise the actual
encoder; predicted counts alone do not establish encoder or checkpoint
compatibility.

An unchanged effective RADIO policy keeps its existing loader identity. A new
encoder policy has an explicit identity including geometry, reduction, bounds
and pixel statistics. Ordinary and DP-changing restore reject changed policies;
switching encoders or image policy requires fresh loader state. Architecture-only
weight compatibility remains the model checkpoint loader's responsibility.

The conventional supervised-path `--packing-buffer-size`, `--shuffle-buffer-size`
and `--max-samples-per-sequence` do not control this path. Interleaved packing is
always streaming, uses its own document-shuffle option and currently sets 100 samples
per source slice. Its checkpoint restoration is strict regardless of the supervised
`--strict-dataloader-state-load` switch. Tokenizer prompt-format settings configure
the wrapper; this causal path does not render conversations or insert system messages.

The earlier development suffixes `min-patches`, `max-patches`, `shuffle-buffer-size`
and `dataloader-load` are replaced by the explicit names above, without aliases.
This CLI rename preserves defaults and serialized checkpoint-contract fields.

This replaces the development-only `raw_*` API and flags without compatibility
aliases. Module/class names affect Energon's resolved configuration and serialized
state, so old development loader checkpoints can be rejected. Start fresh loader
states after the rename; old artifacts are not migrated or overwritten. Recipe
keys, image-store protocols and the state-envelope format are unchanged.

## Runtime dependency

Build commands, input pins and validation scope are documented in
[the runtime recipe](../../../docker/n5-runtime-arm64-cuda13.4/README.md).
That recipe installs the locked Energon source with finite-source stream
exhaustion checkpoint support, without patching an installed wheel. The locked
VCS origin/version and saved state capability identify the dependency.
Old stream checkpoints are not migrated; the resolved pipeline fingerprint
and Energon's required state field reject incompatible restores.

## Neutral image-store recipes

Use `aux.images: document-images://images` for original local TARs plus existing
`.tar.msidx` document-offset sidecars. Use
`aux.images: compact-document-images://images.json` for compact HSG storage.
The JSON configuration has four explicit fields, resolved relative to that file:

```json
{"catalog":"catalog.sqlite","dataset_id":"YOUR_IMMUTABLE_DATASET_ID","offset_root":"document-offsets","logical_prefix":"YOUR_LOGICAL_PREFIX/"}
```

`catalog` names a SQLite file (at most 1 GiB) and its paired readiness receipt;
the full catalog checksum is verified before any image reads. `dataset_id` is
the immutable identity that both must declare. `offset_root` is the directory for
document-offset sidecars and the store's provenance identity. A reference carrying
`document_index` reads its existing `.tar.msidx` sidecar there. Absolute extents
omit `document_index` and do not read sidecars; `offset_root` can be `.` in that case.

The source TAR itself need not exist for compact reads. The logical prefix is an
exact catalog-coordinate mapping, not a built-in corpus or S3 alias. Each image
must resolve uniquely inside a verified segment. Audited exclusions fail rather
than silently dropping images. Full-coverage readiness is required; there is no
production subset bypass. Catalog/index checksums and physical sizes are checked;
payload full-SHA verification is the catalog preparer's prior responsibility.
This reader does not generate catalogs, indexes, manifests or production data.

Direct storage diagnostics may supply `validation_run_id` to `CompactImageCatalog`
for an existing bounded canary catalog. The receipt and `canary:` dataset identity
must match that run and explicitly forbid production use. Default admission still
requires full coverage; this opt-in is not exposed by the training configuration.
All catalog, index and range integrity checks remain active in validation mode.

### Independent text and image selections

Text representation and image storage are two recipe choices, not separate
packing modes or extra command-line switches:

| Text input | Local TAR/ranges | Compact ranges |
| --- | --- | --- |
| JSONL, default interleaved cooker | `aux.images: document-images://images` | `aux.images: compact-document-images://images.json` |
| BinIdx, `tags.cook: multimodal_interleaved_binidx` and `aux.metadata` | Same local image store | Same compact image store |

The native reader follows the text file type. Image-bearing BinIdx additionally needs the
explicit cooker tag and its co-indexed metadata; a mismatched or missing tag gives
an actionable input-schema error. Both cookers produce the same typed document.
Image bytes then pass through the same lazy decoder and image policy, followed by
the same masks, source-isolated streaming packer and consumer adapters.

Switch the image protocol only when keys, extents and namespaces describe the
same logical images and the compact catalog covers them. `key` is an opaque
store key, not necessarily a filename. Changing a protocol does not translate
coordinates or normalize legacy nested TARs. Changing an input recipe/backend
also changes its resume contract; old states are not migrated across that change.
Ordinary training configuration still requires a full-coverage compact catalog.

## Data contract

Interleaved JSONL may carry text, repo_texts, or ordered
parts/multimodal_document content.
Structural images require width/height and an aux.images key or explicit
document-relative extent metadata. Images remain lazy through selection; there
is no source-specific loader dependency. Raw JSONL is tokenized on the fly;
prepared BinIdx input can reuse existing text IDs as described below.

Text is encoded without a chat template. Adjacent text pieces are joined before
encoding; all literal special-token spellings remain ordinary text through HF's
explicit split_special_tokens mode. The interleaved path requires HF fast encoding;
GigaToken remains available on existing paths, but is rejected here. EOS is appended
once per original document if missing. Next-token targets are assigned before fragmentation, so a
fragment keeps its original final lookahead target. Image positions/targets and
padding are masked. Arbitrary suffixes do not receive artificial EOS/BOS tokens.

Both providers require `MultimodalTokenizer`, with `--tokenizer-prompt-format`
selected to match the vocabulary's wrapper configuration; causal loading does
not render that format's chat template. The existing `nemotron6-moe` default is
unchanged, but is not required by the interleaved provider. For an HF export
using `chatml`, declare an existing padding token in its tokenizer configuration
without adding vocabulary entries or changing IDs. Raw and prepared inputs must
still match the loaded vocabulary, structural image ID and EOS policy.

Streaming uses expanded image costs. Images are never split or silently dropped.
Unsplittable images/targetless packs fail explicitly. Same-document successive
fragments are legitimate progress; physical offsets need not advance under pread.

The state envelope also fingerprints Energon's resolved loader configuration,
capturing nested blend weights/tags without walking recipes or hashing corpus
payloads. Auxiliary configuration, catalogs and prepared manifests must remain
immutable: the resolved configuration is not complete dataset-content identity.
Both new providers use the existing canonical checkpoint-name helper with
explicit TP0/PP0/expert0 coordinates, as the current dataloader writer does; this
avoids querying uninitialized global expert groups in explicit-group MIMO.
Missing/null/malformed state fails closed except missing state during an explicit
iteration-zero finetune/pretrained-weight initialization. Malformed-present state
still fails in that case. No old-version state migration is implemented.

### Prepared BinIdx text with image annotations

The same causal path also accepts native `.bin`/`.idx` token documents. Select
the explicit cooker and co-indexed image metadata in the recipe:

```yaml
__module__: megatron.energon
__class__: Recipe
splits:
  train:
    path: tokens.bin
    tags:
      cook: multimodal_interleaved_binidx
      drop_bos_token_id: 1
    aux:
      metadata: indexed-jsonl://metadata.jsonl
      images: compact-document-images://images.json
```

`tokens.idx` must accompany `tokens.bin`. Each sequence exposed by the native
BinIdx reader is one document; this path does not reconstruct documents from
several pre-existing sequences. Prepare `metadata.jsonl` with Energon's standard
JSONL index. Each original document index selects the corresponding metadata row,
including when a recipe subset is used. For example:

```json
{"images":[{"position":965,"width":640,"height":480,"key":"YOUR_OPAQUE_IMAGE_STORE_KEY"}]}
```

Positions are zero-based insertion boundaries in the original text token array,
including any BOS. Insert each image before that text token without consuming or
replacing it. Positions must be nondecreasing and precede the final token; equal
positions preserve image order. The optional `drop_bos_token_id` verifies and
removes exactly that first token. An image at boundary zero stays at the start.
Omit that tag to retain the supplied BOS. Text-only metadata rows use
`{"images":[]}`. Image keys follow the selected auxiliary store's contract.
Absolute extent keys can be used with compact storage without a `.msidx` sidecar.

Prepared IDs must already use the configured tokenizer's vocabulary and end with
its EOS. The encoder rejects invalid/out-of-range IDs and structural image IDs
embedded in the text array; image placeholders come exclusively from metadata.
It does not re-tokenize text or translate vocabularies. Verify producer metadata
and the actual tokenizer artifact before using a prepared source: matching
vocabulary size alone does not establish compatibility. Keep those artifacts,
the token/index pair, companion metadata/index and image configuration immutable
across resume. The resume fingerprint cannot detect every in-place content edit.

Without a supervision companion this remains causal all-text supervision, with
the existing image/padding masks. Both consumer adapters share insertion, image
planning, labels, source-isolated streaming packing and checkpoint behavior.

### Optional prepared-token supervision masks

Configure `aux.token_loss_mask` with the provider's
`indexed-token-mask://<primary-prefix>.mask` protocol. The `.mask`, `.bin` and
`.idx` must have the same immutable prefix. This explicitly binds the auxiliary
to that primary; another prefix is rejected even if its document count matches.
The producer must prepare the companion for exactly that token artifact and
retain the recipe/artifacts together. Counts and lengths cannot establish
semantic identity or detect an incorrectly produced same-prefix companion.

A complete text-only recipe needs no image annotation JSONL:

```yaml
__module__: megatron.energon
__class__: Recipe
splits:
  train:
    path: code-v1.bin
    tags:
      cook: multimodal_interleaved_binidx
      text_only: true
      drop_bos_token_id: 3  # Omit to retain the source BOS; use the actual BOS ID.
    aux:
      token_loss_mask: indexed-token-mask://code-v1.mask
```

`text_only: true` explicitly declares the source has no structural images. It
rejects image auxiliaries. Image-bearing sources retain the annotation contract
above and may use the same optional mask:

```yaml
__module__: megatron.energon
__class__: Recipe
splits:
  train:
    path: illustrated-v1.bin
    tags:
      cook: multimodal_interleaved_binidx
    aux:
      metadata: indexed-jsonl://illustrated-v1-metadata.jsonl
      images: filesystem://image-root
      token_loss_mask: indexed-token-mask://illustrated-v1.mask
```

Document N of the mask accompanies original primary index N, including under
subset selection and shuffling, never a shuffled or packed-row index. Flags
describe source-token ownership before BOS removal, image insertion or next-token
label shifting: True/1 is eligible for target loss, False/0 is excluded. Run
lengths must sum to the original text-token count, including source BOS/EOS.
BOS removal slices ownership with IDs; inserted image slots consume no source
text ownership. EOS keeps its supplied ownership. The cooker carries ownership
in `TokenizedText.token_loss_mask`; the encoder constructs aligned labels from it.

The BinIdx cooker owns original-index lookup, ownership expansion, BOS/image
slicing and attachment to `TokenizedText`. The related `binidx_token_mask.py`
adapter owns IO binding, keys, reserved-key decoding and the recipe protocol,
beside the other storage adapters. The shared encoder consumes optional ownership
to construct target labels independently of the source format; raw JSONL supplies
no ownership and keeps its existing behavior. Existing image/padding exclusions
and consumer loss/MTP handling remain with their current owners.

The version-2 binary format uses little-endian integers: an 8-byte row per run
(positive int32 length, uint8 flag 0/1, three zero reserved bytes), followed by
`document_count + 1` signed int64 segment-row offsets, then the 40-byte footer
`<8sQQQQ`: magic `4d41534b49440001`, version 2, document count, total segments,
index byte offset. Index offset equals eight times total segments; index
endpoints are 0 and total segments. The footer ends at EOF. Construction reads
only the footer and endpoints. Each access validates its index pair and runs;
only the accessed document expands to booleans. Reads use private range handles
and share no mutable seek cursor across workers. A configured companion fails if
missing, unsupported or malformed, if its document count disagrees with the
primary, if bound to the wrong primary prefix, or if an accessed document's mask
length differs from its source-token count. It never falls back to full-text loss.

Labels are formed once, before cuts, using the **target** token's ownership.
For IDs `[BOS, a, separator, filename, body, EOS]` and ownership
`[1, 1, 0, 1, 1, 1]`, labels are
`[a, IGNORE, filename, body, EOS, IGNORE]`. All input IDs remain visible. Existing
image-input, image-target and padding exclusions override train eligibility. The
existing packer's fragment labels retain their lookahead targets; neither consumer adapter
shifts them again. MIMO converts ignored labels to its floating loss mask.
MTP consumes these existing labels and loss masks, advancing both for each
prediction head.

Targetless documents and packs remain errors. A long all-masked prompt can fill
an entire sequence before any eligible response token and trigger this limit,
even when its full document has trainable targets. Choose prepared documents and
sequence budgets that leave a target in each produced pack; the loader does not
drop fragments, unmask prompts, or change zero-target optimizer behavior.

Mask paths, primary binding and ownership semantics enter the existing resolved
restore fingerprint. The optional mask leaves unmasked recipe identity unchanged
at a fixed runtime; dependency-version changes still follow the existing restore
compatibility checks. Keep masks and all other auxiliaries immutable; changing
their contents in place is unsupported. The restore fingerprint does not hash
full auxiliary contents. Use a new versioned prefix/recipe when changing an
objective. Construction never hashes a GB mask or corpus tree.

#### Prepared conversation/SFT supervision

Conversation masks and code masks use exactly the same binary v2 consumer; each
producer chooses different spans. For SFT, externally render and tokenize the
turns in the intended format, assign ownership to every source token in the
system/user/assistant/tool spans according to the chosen supervision recipe, and
serialize the co-indexed immutable `.mask` alongside those exact `.bin`/`.idx`
artifacts. Ownership of EOS and every turn/control marker must be explicit.
System, user and tool masking is a recipe choice; assistant-only supervision is
one example, not a hardcoded policy. The cooker and encoder do not infer roles,
parse delimiters, render chat, decode/re-tokenize text or regenerate prepared IDs.

For an assistant-only example, let each name below denote one prepared token ID,
with no images or BOS removal. This recipe excludes the prompt and assistant turn
marker, and includes the answer and terminal EOS. `IGNORE` is the ignored-label
value, not an input token:

| Source input token | Ownership | Aligned next-token label |
| --- | --- | --- |
| `BOS` | 0 | `IGNORE` |
| `SYSTEM` | 0 | `IGNORE` |
| `USER` | 0 | `IGNORE` |
| `ASSISTANT_MARKER` | 0 | `answer1` |
| `answer1` | 1 | `answer2` |
| `answer2` | 1 | `EOS` |
| `EOS` | 1 | `IGNORE` |

All prompt/marker IDs remain inputs: the excluded assistant marker can predict
the first eligible answer token. The answer predicts eligible EOS; the final EOS
has no following target. Image-bearing conversations use the same ownership
alignment and image exclusions described above. Using an existing prepared
corpus for multimodal training or SFT requires verifying its vocabulary, EOS,
image annotations and intended ownership; this interface does not establish
raw-role/tokenizer integration or validate arbitrary existing artifacts.

#### Runtime and coverage

Use the Energon revision pinned by this branch's `pyproject.toml` and `uv.lock`
in the Python environment running the checks, including its child processes.
An older container's installed Energon does not acquire this reader just by
adding a source checkout to the parent's `PYTHONPATH`; some probes deliberately
set their own import path. Install the pinned dependency in the selected
environment or build the runtime from the updated lock.

The focused CPU checks include synthetic reader corruption cases, target/cut
alignment, both adapters, pending-fragment serialized replay and per-head MTP
label/loss-mask rolls for packed CP=1 (both supported layouts):

```bash
# From the pinned Megatron-Energon checkout:
python -m unittest discover -s tests -p test_token_mask.py
# From megatron-lm, using the same Python environment:
python -m tools.check_multimodal_interleaved
```

These CPU checks do not qualify distributed MTP or full GPU training. Validate
those separately with the intended model, data and worker topology.

## Data-parallel checkpoint redistribution

For the derivation of dense DP, expert DP and independent sample lanes, see
[the MIMO topology guide](../../mimo/README.md). World size means distributed
ranks; this conversion tool changes the number of language data lanes.

Both providers accept nonnegative `--num-workers`; zero selects synchronous,
fixed-topology loading. Training entrypoints inherit the shared parser's default
(currently 2); pass `--num-workers 0` explicitly for synchronous loading.
To prepare for DP changes, start with a
positive worker count. The number of independent streams is language DP lanes
times workers per lane. That total, global batch size, microbatch size (one),
seed, data, tokenizer and processing policies must remain unchanged during a
redistribution. For example, 2 lanes with 2 workers can become 1 lane with 4
workers or 4 lanes with 1 worker. A workers0 checkpoint cannot be converted, and
saved streams cannot be split to create additional workers.
All existing provider topology restrictions still apply, including MIMO's
language/encoder lane divisibility. A data checkpoint conversion does not change
model, optimizer or distributed-checkpoint compatibility.

Positive-worker runs configured with a global batch size save format2 metadata
alongside the native state. Normal restore still requires the complete contract.
For an explicit DP change, stop at a committed global-batch boundary with the
native next-worker cursor zero on every lane. Global batch size must be divisible
by total workers. Supply every authoritative language-lane data checkpoint from
the same iteration; encoder mirrors restore those same files independently.
An untouched process-worker loader has no native checkpoint; save after a
complete committed worker round rather than synthesizing an initial state.

Run the CPU conversion tool in the qualified container. Pass the individual
`train_dataloader_dprankNNN.pt` files for every authoritative language lane from
one iteration, not the containing directory, model weights or DCP shards. Write
to a fresh target directory in the usual checkpoint layout. For example, two
language lanes with two workers each become one lane with four workers:

```bash
python -m tools.redistribute_multimodal_interleaved \
  --input-dataloader-checkpoints \
    /original/iter_0000100/mp_rank_00/train_dataloader_dprank000.pt \
    /original/iter_0000100/mp_rank_00/train_dataloader_dprank001.pt \
  --new-num-data-lanes 1 \
  --output-dir /converted/iter_0000100/mp_rank_00
```

This creates `/converted/iter_0000100/mp_rank_00/train_dataloader_dprank000.pt`.
The destination worker count is derived as saved total workers divided by the
new number of data lanes. This number is not the destination job's `WORLD_SIZE`:
model parallel partitions and separate encoder ranks are accounted for elsewhere.
The development-only `--input-files` and `--new-world-size` names were replaced
by the explicit options above; they are not compatibility aliases.

Use `--multimodal-interleaved-dataloader-checkpoint-root /converted` with the destination
worker count and the same model checkpoint iteration. The tool writes canonical
rank filenames with the existing `dataloader_state_dict` wrapper. It refuses
existing output directories and never rewrites source checkpoints. Inputs are
trusted PyTorch data checkpoints; do not supply model weights or untrusted files.

The tool delegates worker-state regrouping to Energon. It protects semantic
configuration separately from ordered per-worker sampler partitions. The actual
destination loader checks both before restoring native state or yielding data.
A subsequent normal save records the destination's complete real contract;
restore and save there before another redistribution. Missing lanes, changed
policies, mismatched progress, unsupported cursors and workers0 states fail
explicitly. Finite tails may contain incomplete global batches; preserving their
remaining samples does not make them complete reproducible global batches.

Custom image stores return bytes; the interleaved encoder selects the same PIL/content-
guessing SampleDecoder policy as the existing SFT encoder. Range keys need no
filename extension, and only encoder-role realization decodes selected pixels.
Uncompressed TAR extents may stop after a complete member payload, with optional
zero padding. Additional members, truncated payloads and compressed ranges fail.

## Validation and remaining integration

Run the focused CPU worker and DP-resize checks in the pinned runtime:

```bash
python -m unittest tools.check_multimodal_interleaved_dp -v
```

They cover fixed-total-worker round trips, complete global batches and per-worker
suffixes, image masks/pixels, finite-document coverage, encoder mirrors/fanout,
worker shutdown and rejected checkpoint changes. These are dataloader checks;
they do not qualify model or optimizer rescaling.

The storage-only checks can run locally without loading Megatron, Torch, PIL or
Energon, or installing any dependencies:

```bash
python3 tests/standalone/test_document_image_ranges.py -v
```

These 30 checks cover storage-format/range behavior only. They do not qualify the
Energon adapters, tokenizer, split/replay, consumer providers or GPU training.
The training package's Python >=3.12 requirement is unchanged; the isolated
storage harness is also Python 3.9-safe.

Run the functional checks in the pinned validation runtime:

```bash
python -m tools.check_multimodal_interleaved
python -m examples.mimo.pretrain_mimo --help
```

Coverage includes real consumer/provider contracts, HF ordinary/default
tokenization, causal/image masks, lazy PNG/compact decoding, repeated/new-process
fixed-configuration continuation, finite-source final-pack replay, selected
numeric-buffer SFT and mock compatibility. This is not qualification of every
legacy SFT/media/distributed path or a production-tokenizer/corpus audit.
Dependency-isolation audits are maintained separately from this functional suite.

Build, saved-image verification and runtime qualification boundaries are
documented in [the runtime recipe](../../../docker/n5-runtime-arm64-cuda13.4/README.md).
No model-training, learning-parity or scaling result is claimed.
