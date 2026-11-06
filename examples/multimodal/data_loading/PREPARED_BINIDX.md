<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. -->

# Cook prepared BinIdx documents

`cook_multimodal_interleaved_binidx` joins native Energon `.bin/.idx` token
records with image annotations while retaining the prepared IDs. It returns an
`InterleavedDocument` containing `TokenizedText` and existing `ImageMedia` parts.
Image pixels remain lazy. The caller owns vocabulary/EOS policy, token-to-target
shifting, image transforms, packing and model batches.

This opt-in example uses the BinIdx, auxiliary FileStore and cooker APIs from
the public Energon stack 277. Use a compatible dependency before running it.
Registering this cooker on your application encoder does not change existing
conversation/SFT cookers or default training paths.

## Inputs

- `tokens.bin` and `tokens.idx`: native prepared integer-token records. Each
  record must be a nonempty, one-dimensional array of nonnegative IDs.
- `annotations/<index>.json`: one metadata object per original BinIdx record.
  The index is the primary `SourceInfo.index`, including after subset selection
  and shuffling; it is not the sample's displayed key or iteration position.
- An image FileStore whose keys match the metadata. The example below uses
  ordinary filesystem images and JSON files through native Energon stores.

For example, `annotations/7.json` can contain:

```json
{"images": [{"position": 2, "width": 640, "height": 480, "key": "screen.png"}]}
```

An empty-image record uses `{"images": []}`. Positions are boundaries in the
original token array: position zero precedes its first token. Positions must be
nondecreasing and less than the token count, leaving the final token as text.
Repeated positions retain the annotation's image order. Width and height are
positive integer source dimensions; they do not request resizing.

## Native recipe and encoder

Paths in this recipe are relative to the recipe file:

```yaml
__module__: megatron.energon
__class__: Recipe
splits:
  train:
    path: tokens.bin
    tags:
      cook: multimodal_interleaved_binidx
    aux:
      metadata: filesystem://annotations
      images: filesystem://images
```

Register the cooker explicitly and iterate unbatched documents:

```python
from megatron.energon import Cooker, DefaultTaskEncoder, WorkerConfig, get_loader, get_train_dataset
from megatron.energon.cache import NoCachePool
from megatron.energon.decoders.sample_decoder import SampleDecoder

from examples.multimodal.data_loading.cookers.multimodal_interleaved_binidx import (
    cook_multimodal_interleaved_binidx,
)


class PreparedEncoder(DefaultTaskEncoder):
    cookers = (
        Cooker(
            cook=cook_multimodal_interleaved_binidx,
            has_tags={"cook": "multimodal_interleaved_binidx"},
        ),
    )
    decoder = SampleDecoder(image_decode="pil")


dataset = get_train_dataset(
    "recipe.yaml",
    worker_config=WorkerConfig(rank=0, world_size=1, num_workers=0),
    task_encoder=PreparedEncoder(),
    batch_size=None,
    packing_buffer_size=None,
    shuffle_buffer_size=None,
    max_samples_per_sequence=None,
    repeat=False,
)
loader = get_loader(dataset, cache_pool=NoCachePool())
try:
    for document in loader:
        print(document.parts)  # Lazy images are materialized only through value.get().
finally:
    loader.close()
```

The cooker requires an explicit loader cache pool. `NoCachePool` defers lazy
reads without maintaining a shared file cache; applications can supply their
chosen native cache pool instead.

For text-only sources, set `text_only: true` in the source tags and omit both
image auxiliaries. The default retains every token, including a source BOS.
To remove exactly one verified leading token, set `drop_bos_token_id` to its
nonnegative integer ID. Image positions still refer to the original array;
images before the removed BOS remain at the document start. This is explicit
source preparation policy, not an automatic BOS/EOS conversion.

## Optional prepared-token eligibility

Add `token_loss_mask` only when the source has an eligibility companion. It must
share the primary's prefix (`tokens.bin` / `tokens.mask`) and original record
count. A row supplies one boolean flag per original token, including source
BOS/EOS, before BOS removal or image insertion. `True` marks an eligible target
token; `False` excludes that target without removing its ID from the context.
The cooker slices flags with the text spans and does not shift labels or infer
chat roles. With no companion, `TokenizedText.token_loss_mask` is `None`.

Register the native auxiliary protocol explicitly before loading the recipe,
and select the decoder that preserves its reserved mask keys as bytes:

```python
from megatron.energon import register_aux_filestore_protocol

from examples.multimodal.data_loading.binidx_token_mask import (
    BinIdxTokenMaskDecoder,
    IndexedTokenMaskReference,
)

register_aux_filestore_protocol(
    "indexed-token-mask", lambda path: IndexedTokenMaskReference(fs_path=path)
)


class SupervisedPreparedEncoder(PreparedEncoder):
    decoder = BinIdxTokenMaskDecoder(image_decode="pil")
```

Use this encoder with the previous loader example and add this entry under
`aux` in the recipe:

```yaml
token_loss_mask: indexed-token-mask://tokens.mask
```

The decoder delegates other files to ordinary Energon decoding. Metadata and
image companions retain their behavior and image pixels remain lazy. Masks
always use the primary's original record index after subsets/shuffling; source
prefix, record count and token count mismatches fail rather than silently
applying another document's flags. Changing companion contents requires a new
immutable source identity when checkpoint continuity matters.

### Indexed mask layout

The companion contains little-endian run rows, a record index and a footer:

- Each 8-byte run is a positive signed int32 length, a byte flag (`0` or `1`)
  and three zero reserved bytes.
- The signed int64 index has `record_count + 1` entries measured in runs.
- The 40-byte footer is `<8sQQQQ>`: signature `MASKID\x00\x01`, a reserved
  uint64, record count, run count and the byte offset of the index. The first
  uint64 is ignored for compatibility; it is not a version-validation gate.

Construction validates the layout and index endpoints with bounded reads.
Requested index pairs and run rows are checked on access. Expansion validates
the expected original token count before allocating the boolean list. This is
not a checksum or an eager scan of every row.

## Encode prepared documents for causal prediction

`PreparedDocumentTaskEncoder` uses the cooker above and produces unbatched
`EncodedPreparedDocument` samples. It retains source IDs, expands typed images
into visual positions, and shifts target eligibility once. It does not tokenize,
append BOS/EOS, truncate, split, pad or pack documents. Existing conversation
encoders and training defaults are unchanged.

Construct it with an existing image strategy configured for the selected model:

```python
from examples.multimodal.data_loading.prepared_task_encoder import PreparedDocumentTaskEncoder

encoder = PreparedDocumentTaskEncoder(
    image_strategy=visual_only_strategy,
    image_token_id=image_token_id,
    eos_token_id=eos_token_id,
    vocab_size=vocab_size,
    image_token_budget=4096,
)
```

The five inputs are explicit: the strategy supplies image geometry, the two IDs
are distinct vocabulary IDs, `vocab_size` bounds every prepared text ID, and
`image_token_budget` is a positive **per-image visual-position limit**. The
strategy's returned count is also checked against this limit. It does not bound
document length or share a budget among images. The optional `ignore_index`
defaults to `-100` and must be outside the vocabulary.

Use `encoder` as `task_encoder` in the previous native loader example with
`batch_size=None` and `packing_buffer_size=None`. Set the worker's
`global_error_handler` to `megatron.energon.errors.reraise_exception` to surface
invalid documents immediately. Keep an explicit cache pool. This encoder uses
`BinIdxTokenMaskDecoder`; register the auxiliary mask protocol above only when
the recipe selects a mask companion. Text-only recipes work with the same
encoder, without invoking the image strategy.

The image strategy must count **visual embeddings only**, matching the model's
patch size and any pooling/pixel shuffle. For example, an existing RADIO strategy
with 16-pixel patches and 2-by-2 pixel shuffle counts
`(width // 16) * (height // 16) // 4` on its aligned, resized dimensions. This
formula is specific to that configuration. Do not reuse a count that also adds
row separators, image break/end IDs or other serialization overhead. Structural
IDs already in the prepared text remain ordinary text targets with their source
eligibility; this encoder never inserts them. The structural `image_token_id`
itself is reserved for typed images and must not appear in text parts.

The output contains CPU int64 `input_ids` and `labels`, float32 `loss_mask`,
half-open `image_spans`, and corresponding lazy `image_params`. Source/restore
identity is preserved. Geometry planning uses metadata; pixels remain unread
until a consumer materializes a plan. Each span contains exactly the plan's
visual embedding count.

Target ownership belongs to the **next token**, including at image boundaries:

```text
input_ids:  [A, IMG, IMG, B, EOS]
labels:     [-100, -100, B, EOS, -100]
loss_mask:  [0, 0, 1, 1, 0]
```

Visual targets are excluded, while the final visual position predicts `B`.
With a source end marker between the image and `B`, the end-marker position
predicts `B` instead. Whether the marker itself is supervised follows its source
flag. An excluded text token stays in context and can still predict the next
eligible token; targets never jump over excluded positions. The final position
has no lookahead and is ignored. Missing terminal EOS, malformed IDs/masks/plans,
over-budget plans and completely targetless documents raise errors.

This is a document-encoding component, not a complete training application.
Consumers must preserve image spans when splitting sequences, assemble fixed
batches, materialize image plans and provide the model's positions, attention
policy and modality metadata. Do not shift these labels again. Distributed MTP,
image-aware packing and model integration require their own qualification.

## Stream image-safe fixed-length sequences

`PreparedStreamingTaskEncoder` extends the prepared-document encoder with
pull-based sequence splitting and packing. It reuses the already shifted IDs,
labels, loss mask and lazy image plans. It does not tokenize, expand images a
second time, shift labels again, or read pixels while selecting packs.

```python
from examples.multimodal.data_loading.prepared_streaming import (
    PreparedStreamingTaskEncoder,
)

encoder = PreparedStreamingTaskEncoder(
    image_strategy=visual_only_strategy,
    image_token_id=image_token_id,
    eos_token_id=eos_token_id,
    vocab_size=vocab_size,
    image_token_budget=4096,
    sequence_length=8192,
    pad_token_id=pad_token_id,
)
from megatron.energon import WorkerConfig
from megatron.energon.errors import reraise_exception

worker_config = WorkerConfig(
    rank=0, world_size=1, num_workers=0,
    global_error_handler=reraise_exception,
    restore_error_handler=reraise_exception,
)
dataset = get_train_dataset(
    "recipe.yaml",
    worker_config=worker_config,
    task_encoder=encoder,
    batch_size=None,
    packing_buffer_size="stream",
    shuffle_buffer_size=None,
    max_samples_per_sequence=None,
)
loader = get_loader(dataset, cache_pool=NoCachePool())
```

The sequence length is a positive built-in integer. The padding ID must be in
the vocabulary and differ from the structural image sentinel; it may equal EOS
when that is the model's padding policy. IDs, labels, masks and image plans stay
on CPU. Native streaming packing owns partial-sample buffering, restore keys and
checkpoint advancement.

The selector pulls only until a sequence is full, an image boundary blocks the
next window, or the source ends. Windows use half-open positions in the
expanded document. Cuts at an image start or end are valid; a cut inside the
visual span is rejected. If the next image does not fit the remaining positions,
the current sequence is padded and the complete image suffix is carried into
the next sequence. An image whose visual span exceeds `sequence_length` raises
an error; it is never resized, truncated or dropped by the packer. An exact
image-end cut retains the previous visual input's already-shifted label for the
following text token, even when that text appears in the next sequence.

Each completed output is a `PackedPreparedDocument`. Its `valid_length` excludes
padding, `fragment_offsets` mark each fragment's packed boundaries, and
`fragments` record `(source_key, source_start, source_stop)` windows in original
expanded positions. Image spans are rebased into the packed sequence, and the
same lazy image-plan objects remain attached. Source metadata is retained across
all fragments. Packing groups are isolated per loaded source so one packed
sequence does not combine records from different recipe sources.

Labels and masks are concatenated exactly as prepared; the packer does not make
targets across document boundaries. IDs pad with `pad_token_id`, labels with
`ignore_index`, and masks with zero. A completed pack with no eligible targets
raises `ValueError`. This can occur when a fully masked document fragment fills
a sequence by itself, even if a later fragment of that source document has
eligible targets. An ordinary unmasked document can also leave an EOS-only final
suffix: five positions split at length four leave a final EOS position with no
lookahead. If this suffix is the last source input, its targetless pack is rejected.
Use the fail-fast error handler shown above so errors cannot become skipped packs.
A pack combining that fragment with any eligible fragment is
valid. This explicit limitation prevents silently discarding input; the encoder
does not implement an allow-fully-masked policy.

Set `packing_buffer_size="stream"` to use this task encoder. `batch_size=None`
keeps outputs as individual fixed-length samples; model-specific batching,
attention/position policy, modality materialization and MIMO role wiring remain
consumer responsibilities. The unbatched `PreparedDocumentTaskEncoder` path
with packing disabled remains available unchanged.
