# Prepared Energon data for heterogeneous MIMO

`examples/mimo/pretrain_mimo.py` keeps synthetic data as its default. Select
`--dataset-provider energon` to read prepared BinIdx text and image auxiliaries
from one `--data-path` Energon recipe. Blend weights and source selection belong
in that recipe. Each source uses its original prepared token IDs and optional
target-ownership mask; no chat template or text tokenization is applied here.

The recipe selects the shared BinIdx cooker through
`cook: multimodal_interleaved_binidx`. Its co-indexed image metadata records
structural image positions and auxiliary-store keys. Sampling, per-source
streaming packing, partial documents and replay state belong to Energon and the
shared data layer, rather than a second MIMO reader or checkpoint engine.
Packing groups keep each output sequence within one dataset source.

Initial supported configuration:

- Existing RADIO tower, patch16, dynamic resolution, pixel shuffle enabled,
  and emitted class tokens disabled.
- Micro-batch size1; language TP/PP/CP and encoder TP1; no GTP rematerialization,
  feature prefetch or evaluation. Encoder and language grids are disjoint.
- Positive encoder DP dividing the language data-lane count. Each encoder
  consumes its contiguous ordered set of language lanes; identical seed and
  recipe preserve matching text/image packs. Only language lanes save state.
- `--tokenizer-type MultimodalTokenizer` with the matching HF export. This
  supplies vocabulary, EOS/pad and structural-image IDs for already-prepared
  data. `--tokenizer-prompt-format` selects compatible wrapper metadata rather
  than rendering conversations.

Image patch budgets are input patches before pixel shuffle, not emitted image
tokens. `--seq-length` includes expanded image tokens and padding. Existing
shifted text ownership is retained; image slots and padding have ignored labels.
Packed document boundaries reset position IDs and form the model's packed
attention metadata. Image pixels are materialized only on encoder ranks.
`--num-workers` is the number of Energon workers per language data lane; it does
not change the mock loaders, which remain single-process.

Loader checkpoints default to model `--save`/`--load` roots. The optional
`--multimodal-interleaved-dataloader-checkpoint-root` takes a checkpoint root,
not an iteration folder or `.pt` file. Fixed-configuration resume is strict;
missing/incompatible required state raises instead of silently restarting.

This provider needs shared prepared-data APIs and Energon's streaming packing,
auxiliary BinIdx stores and token-mask interfaces. An older installed Energon
release is not sufficient merely because it can import. The reusable data-layer
location and released dependency version must be settled before publication.
Native-ViT integration, raw JSONL encoding and expanded parallel configurations
are separate work. The existing supervised `examples/mimo/train.py` paths remain
unchanged.
