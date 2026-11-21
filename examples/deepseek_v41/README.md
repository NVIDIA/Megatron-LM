# DeepSeek V4.1 text training

The model composes each logical block from a `V` CSA2 attention layer and an `E` MoE layer. Use the V4.1 model builder with `--dsv4-version v4.1`, the physical `VE` pattern, untied embeddings and the usual Hybrid training entry point.

## Engram

The model uses the shared `deepseek_v41` Engram variant. With a released Engram configuration, pass `--engram-config PATH` and `--engram-tokenizer-map PATH` (or the V4.1 `--engram-token-map` spelling). The map can be the released 1-D PyTorch tensor or its versioned JSON conversion. Layer placement uses physical V layer numbers. Hash seeds still use the original zero-based logical block numbers. A generic artifact generated directly with physical layer seeds will fail validation.

To create a portable JSON artifact from an existing tensor map:

```sh
python -m tools.engram.convert_v41_tokenizer_map --config engram.json --tensor-map tokens.pt --output tokens-v41.json
```

The common Engram tables may be sharded independently of the expert-parallel group. V4.1 uses contiguous CP history and preserves document padding. Lookup runs outside checkpoint replay. The model-state loader accepts a gathered legacy per-layer table and validates any legacy hash buffers against the artifact. Loading a sharded legacy optimizer state requires conversion to the new per-head table layout.

## Pipeline and validation

Use static context parallelism with packed THD inputs. Typed pipeline plans carry the logical and padded sequence boundaries and the Engram token IDs to every physical or virtual stage. Lookup is completed before layer replay; raw tokens are absent from the layer replay schema. The model adapter supports explicit per-layer graph capture and one-layer full recompute. The public model and training PRs are independent drafts targeting main. Integration qualification uses the shared-state, stage, mHC, CSA2 and Engram prerequisites together.
