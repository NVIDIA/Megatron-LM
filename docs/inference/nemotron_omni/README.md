# Nemotron Omni on Megatron-Core inference — handoff

**Status: unvalidated draft. None of this code has ever been executed.**

The machine it was written on has no `torch` and no GPU, so nothing here has
been imported, unit-tested, or run against a checkpoint. It is lint-clean and
format-clean, and that is the entire extent of the verification. Treat every
file as a reviewed-by-eye first draft, not as working code. The first task for
whoever picks this up is to get it to *import*, then to get
`tests/unit_tests/inference/multimodal/` to pass, and only then to think about
numerics.

## The documents in this directory

| File | What it is | Who wrote it |
| --- | --- | --- |
| `mcore-inference-design.md` | The implementation design this branch follows. Section numbers referenced below all point here. | Written for this task |
| `vllm-reference.md` | Behavioural reference for the existing vLLM Nemotron Omni support. The source of truth for *what the model does*. | Pre-existing |
| `checkpoint-config.md` | The resolved checkpoint configuration — actual layer counts, hidden sizes, token ids. The source of truth for *numbers*. | Pre-existing |

Read the design doc's §0 (TL;DR) and §3 (chosen architecture) first; they are
about six pages together and everything else is reference material.

`checkpoint-config.md` opens by citing a doc called
`NEMOTRON_OMNI_MULTIMODAL_DESIGN.md`, which does not exist in this repo — it is
an external document that came with the config. Its §3.4 is cited again around
line 316. If you can find that document, it may answer some of the open
questions below.

## The one idea you need

The dynamic inference path is built around integer token ids: a request is a
list of ids, the model looks them up in an embedding table, and the KV cache is
addressed by token position. Multimodal embeddings do not fit that — they are
dense rows produced by an encoder, with no token id that maps to them.

The approach taken here is a **masked overwrite into fixed-address buffers**.
`DynamicInferenceContext` allocates two buffers up front,
`token_to_mm_embedding` of shape `[max_tokens, 1, hidden]` and a parallel bool
mask `token_to_is_mm`. Immediately after the embedding lookup, the model calls
`inference_context.apply_multimodal_embeddings(decoder_input)`, which is a
single `torch.where` selecting the encoder row wherever the mask is set.

Two properties make this work under CUDA graphs, and both are easy to break:

1. **The buffers are allocated once and never reallocated.** Graph replay reads
   the addresses captured at capture time, so a reallocation silently makes the
   graph read stale memory.
2. **The overwrite is unconditional.** It is deliberately *not* guarded by "does
   this step have multimodal tokens". A host-side branch there gets resolved
   once during capture and frozen for every replay, so a step captured without
   multimodal tokens would skip injection forever after. An all-`False` mask
   makes the op an identity instead, and the cost is one elementwise pass.

If you find yourself wanting to add an `if` around the injection for
performance, re-read §3.2 first.

The corresponding write side is
`DynamicInferenceContext.scatter_multimodal_embeddings`, which is chunk-aware:
under chunked prefill an image span can straddle a chunk boundary, so it
bisects the request's `mm_embed_positions` (kept on CPU precisely so this
bisect does not force a device sync) to find the rows belonging to the current
chunk.

## What is where

Engine-side plumbing, modality-agnostic:

- `megatron/core/inference/config.py` — the `enable_multimodal` flag. Off by
  default; when off, nothing is allocated and the embedding path is unchanged.
- `megatron/core/inference/contexts/dynamic_context.py` — the buffers, the
  scatter, the masked overwrite, mask clearing per step.
- `megatron/core/inference/contexts/base_context.py` — identity default so
  static contexts are unaffected.
- `megatron/core/inference/inference_request.py` — `mm_embeddings`,
  `mm_embed_positions`, `mm_content_hash` on the request.
- `megatron/core/inference/engines/dynamic_engine.py` — the optional
  `multimodal_encoder` hook and admission-time encoding.
- `megatron/core/models/hybrid/hybrid_model.py`,
  `megatron/core/models/gpt/gpt_model.py` — the one-line injection call.
- `megatron/core/inference/multimodal/types.py` — the
  `MultimodalEmbeddingProvider` interface. **The engine knows only this file.**
  Everything model-specific sits behind it, so a second multimodal model should
  need no engine change.

Nemotron Omni specifics, all under
`megatron/core/inference/multimodal/nemotron_omni/`: `config.py`,
`image_processor.py` (the dynamic-resolution tiler), `video_processor.py`,
`audio_processor.py` (log-mel front-end), `prompt.py` (placeholder expansion),
`encoder_stack.py`, `checkpoint.py` (HF weight mapping), `provider.py`,
`builder.py`.

Shared model code promoted out of examples: `megatron/core/models/vision/
pixel_shuffle.py`, `megatron/core/models/vision/evs.py`,
`megatron/core/models/audio/parakeet_model.py`.

## Things that will bite you

**The token-count contract is the whole game.** The host expands placeholder
spans into real token ids; the device produces encoder rows. If those two counts
disagree by even one, you get a shape mismatch far away from the cause. Design
doc §6.1 has the per-modality table. Port and test that before touching
anything else — most of the "copy bit-for-bit" behaviours in §5.4 exist only
because they change a token count.

**`class_token_len` is 10, not 8.** This checkpoint has 4 class tokens plus 6
registers. mcore's default is 8. Getting it wrong strips the wrong number of
tokens per image and degrades output *silently* — no crash. This is the kind of
bug that costs a week.

**There are two incompatible pixel shuffles in this tree** and they are
genuinely different operations, not two spellings of the same one. The one in
`llava_model.py`'s `h`/`w` branch folds 4 horizontally adjacent patches; the one
promoted to `megatron/core/models/vision/pixel_shuffle.py` folds a 2×2 spatial
block. This branch uses the 2×2 version because that is what vLLM does. See
open question Q1 — this is unresolved and a wrong choice produces
plausible-looking but degraded output.

**Mamba needs `mamba_num_heads=64` set explicitly.** Otherwise it gets derived
from `expand: 2` and you get the wrong head count.

**Encoders run at request admission, not per step.** This is deliberate (§3.1):
encoder output shape varies per request, so running a tower inside the step
would break graph capture and would re-run the tower once per prefill chunk.
The cost is no cross-request encoder cache in v1, where vLLM has one.

## Deliberate divergences from vLLM

Documented in full in §5. The ones that are choices rather than consequences:

- **Batched video encoding.** vLLM sets
  `requires_sequential_video_encoding = True` and encodes videos one at a time,
  purely because batched dynamic-resolution video is unimplemented there.
  mcore's `_apply_temporal_grouping` already handles mixed image/video items in
  one packed batch. Numerically identical, materially faster.
- **LayerScale.** vLLM skips it, and mcore's RADIO has no LayerScale at all, so
  vLLM parity holds for free. Both may differ from the HF reference, which does
  apply it. The mapper warns rather than silently dropping. See Q2.
- **Mamba SSM state forced to fp32.**
- **Prefix caching disabled for multimodal requests.** Identical prompts with
  different images would otherwise collide on placeholder token ids and produce
  a false cache hit. Content-hash chaining into the block hashes is the real
  fix and is deferred; `mm_content_hash` is already plumbed through the request
  for it.
- **PP is asserted to 1** for multimodal. Matches vLLM. Injection happens on the
  first pipeline stage only and is not forwarded.

## Known gaps

**The chat completions endpoint rejects multimodal requests with a 400.** This
is the largest functional gap. The endpoint tokenizes locally and ships token
ids over ZMQ, which cannot carry raw media, so there is currently no HTTP path
for a multimodal request. In-process `engine.add_request` works. Fixing this
needs a transport decision — shared memory, a side-channel upload, or encoding
before the ZMQ hop — and that decision was not made here.

**The Parakeet subsampling length formula was reconstructed from first
principles**, not verified against `transformers`, because the library was not
installable on the authoring machine. Check it early; an off-by-N here breaks
audio token counts.

**`transformers >= 5.5.3` is required** for `ParakeetEncoder`, and
`pyproject.toml` pins nothing. If the CI container ships 4.x this becomes a
dependency bump. Untested.

**Test coverage is one file** covering the token-count contract and chunk-aware
injection, against a fake tokenizer. §6.2 lists the seven tests that should
exist; five of them do not.

## Open questions, in priority order

These are in §7.2 with full context. Q1 is the one that blocks correctness.

1. **Q1 — Which pixel shuffle did the checkpoint train with?** Both are live in
   the tree for dynamic resolution. vLLM agrees with the 2×2 version, which is
   what this branch uses, but if Omni was trained through `LLaVAModel` that is
   wrong and vLLM has a latent bug. The config cannot answer this; you need the
   training recipe owner.
2. **Q2 — Are the checkpoint's `ls1`/`ls2` ≈ 1.0?** Needs a `state_dict` dump.
   Determines whether an HF-reference mismatch is expected or a bug.
3. **Q3 — `video_pruning_rate`: config's `0.7` or vLLM's `None` default?** This
   branch follows the config, which is a deliberate divergence.
4. **Q4 — Image token budget from engine `max_model_len` or the HF `262144`?**
   Following vLLM makes image *resolution* depend on a deployment flag.
5. **Q5 — What `transformers` version does the CI container ship?**
6. **Q6 — Is `sound_timestamps` a real feature?** mcore's `LLaVAModel` threads
   it with no vLLM counterpart.

## Suggested first week

1. Get it to import in the CI container. Expect real breakage; nothing has run.
2. Run and fix `tests/unit_tests/inference/multimodal/`.
3. Port the tiler-parity fixture from
   `vllm/transformers_utils/processors/nano_nemotron_vl.py:290-327` — it is a
   ready-made doctest that pins the token-count contract.
4. Load a real checkpoint through `checkpoint.py` and diff the resulting
   `state_dict` keys against the HF file. Key mismatches surface here cheaply.
5. Get Q1 answered before spending time on numerics.
6. Then chase the oracle cut points in §6.3, in order (a) → (d).
