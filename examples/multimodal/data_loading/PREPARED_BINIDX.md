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
