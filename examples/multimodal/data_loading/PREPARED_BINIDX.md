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
