# Nemotron Omni on Megatron-Core Inference — Design (DYNAMIC path)

> **Goal.** End-to-end Nemotron 3 Nano/Super Omni (text + image + video + audio)
> serving on `DynamicInferenceEngine`, with paged KV, chunked prefill, CUDA
> graphs, TP/EP, and the `inference_optimized` MoE + Mamba stack intact.
>
> **Companion doc.** [`vllm-reference.md`](vllm-reference.md)
> is the vLLM-side reference. This document does not repeat it; it maps that
> behaviour onto Megatron-Core and records every place the two must or will
> differ.
>
> **Reading order.** §1 (what already exists, incl. §1.4 the resolved
> checkpoint config) → §2 (the one hard problem) → §3 (chosen architecture) →
> §4 (file-by-file plan). §5 is the vLLM-divergence log. §7 records what is
> settled and what I still need answered.

---

## 0. TL;DR

The model is *almost entirely already in this repo*. What is missing is the
inference plumbing.

| | Status |
| --- | --- |
| RADIO vision tower (dynamic resolution, CPE, video tubelets, separate video embedder) | **Exists**, production, mcore-native: `megatron/core/models/vision/radio.py` |
| Nemotron-H hybrid LM (Mamba2 + attention + MoE), accepts precomputed embeddings | **Exists**: `megatron/core/models/hybrid/hybrid_model.py:415` |
| Pixel shuffle (dynamic-res variant that matches vLLM bit-for-bit) | **Exists**: `examples/mimo/model_providers/radio_encoder.py:121` |
| Vision projector | **Exists but is missing the leading `RMSNorm`** the checkpoint requires (§7.1): `megatron/core/models/vision/multimodal_projector.py` |
| RADIO + hybrid VLM, assembled and trained | **Exists**: `examples/mimo/model_providers/nemotron_moe_vlm.py`, and `megatron/core/models/multimodal/llava_model.py:223` |
| Parakeet audio tower | **Partial**: `megatron/core/models/huggingface/fastconformer_model.py` loads a *separate* HF/NeMo checkpoint, not the Omni checkpoint's `sound_encoder.*` weights |
| Host preprocessing (dynamic tiler, mel front-end, placeholder expansion) | **Missing** the Omni-specific budgeting; adjacent pieces exist in `examples/` |
| EVS video pruning | **Missing** |
| **Multimodal support in the dynamic inference path** | **Missing entirely** — this is the real work |

The only multimodal inference in the tree is static-batch, batch-size-1, and
architecturally unreusable: `VLMInferenceWrapper` + `VLMTextGenerationController`
+ `VLMInferenceRequest`, driven from `examples/multimodal/run_text_generation.py`
with `legacy=True`. It handles image-token expansion by fudging a scalar
`sequence_len_offset`
(`megatron/core/inference/model_inference_wrappers/multimodal/vlm_inference_wrapper.py:199-214`).
The dynamic path has no such scalar — it has per-token position/block tables and
per-request prefix-cache hashes — so **none of that technique transfers.**

Estimated work: ~2,600 new lines, ~120 modified lines across 11 existing files,
in 6 landable phases. The two riskiest items are the host-side token-count
contract (§6.1) and the pixel-shuffle ordering ambiguity (§7.2, Q1) — both are
correctness cliffs, not performance work.

The real checkpoint config is available and fully transcribed in §1.4, so no
model dimension is a guess. Three of its values are traps that produce silent
wrongness rather than a load error: `class_token_len=10` (mcore defaults to 8),
`mamba_num_heads=64` (must be set explicitly, or `expand: 2` derives 5376
instead of 4096), and `projector_hidden_size=20480` (4× the LM's own `d_ff`).

---

## 1. Starting position

### 1.1 mcore RADIO is a superset of vLLM RADIO

This is the single most important fact in the port. `RADIOViTModel`
(`megatron/core/models/vision/radio.py:32`) already implements everything the
vLLM tower does, and in one place rather than three:

| Concern | vLLM | mcore |
| --- | --- | --- |
| Patch embed from pre-flattened `3·P²` vectors | `ViTPatchLinear`, `radio.py:496` | `self.embedder` = `ColumnParallelLinear(3·T·P² → hidden)`, `radio.py:169-199` |
| Dynamic-resolution packed sequence, per-image pos-enc | `apply_pos_enc_dynamic`, `radio.py:265` | `forward` dynamic branch, `radio.py:381-396` |
| CLS/register tokens prepended **per image** inside the packed sequence | `cls_token_dynamic`, `radio.py:295` | `forward`, `radio.py:400-425` |
| Video tubelets, last-frame-repeat padding, separate video embedder | `forward_video`, `radio.py:216` | `_apply_temporal_grouping`, `radio.py:449` |
| CPE position-embedding interpolation | `_get_pos_embeddings`, `radio.py:434` | `_get_pos_embeddings`, `radio.py:596` |
| Varlen attention across packed items | `MaskMetadata(cu_seqlens, max_seqlen)` → `flash_attn_varlen_func` | `PackedSeqParams(qkv_format='thd')` → mcore `TransformerBlock` |
| TP sharding of attention/MLP | `QKVParallelLinear` + `RowParallelLinear` | standard mcore `TransformerBlock` with TE specs |

Three places mcore is strictly *more* capable, which we should keep:

- **Mixed image+video in one encoder call.** mcore's `_apply_temporal_grouping`
  handles `num_frames == 1` (image) and `num_frames > 1` (video) items in the
  same packed batch. vLLM cannot, which is exactly why it sets
  `requires_sequential_video_encoding = True` and encodes videos one at a time
  (`gpu_model_runner.py:3143-3171`). **We do not need that limitation.**
- **Pos-embed fast paths** for the int32-overflow case and aspect-ratio select
  (`radio.py:680-696`), absent in vLLM.
- **FP8 class-token padding hook** (`radio.py:716`).

**Implication:** vLLM is *not* the source of truth for the tower. mcore is —
vLLM's own comments say so (`radio.py:242` "order follows Megatron training";
`processors/nano_nemotron_vl.py:145-180` "mirrors Megatron-LM's
`image_processing.py`"). Where the two disagree numerically, suspect vLLM first
(see §5.1 on LayerScale).

### 1.2 The LM already accepts precomputed embeddings

`HybridModel.forward` (`megatron/core/models/hybrid/hybrid_model.py:415-429`)
takes `decoder_input`, and `:453-473` bypasses the embedding layer when it is
provided. The comment at `:468-470` says out loud that this is for VLM wrappers.
No model-side change is needed to *feed* embeddings — only to *inject* them
(§3.2).

### 1.3 What the dynamic path looks like today

Two facts define the problem:

**Token storage is a single flat `int64` buffer.** `DynamicInferenceContext`
keeps one coalesced pinned CPU `uint8` buffer with typed views
(`dynamic_context.py:1035-1047`), mirrored on GPU by `ContextGPUView`
(`gpu_view.py:127-175`), transferred in one `cudaMemcpyAsync` per step. Every
token-level field is an integer type. There is no float, hidden-size-wide buffer
anywhere. All addresses are fixed for CUDA-graph replay.

**The forward call is a hard-coded three-key dict.**

```810:828:megatron/core/inference/text_generation_controllers/text_generation_controller.py
    def _dynamic_step_forward_logits(self, input_ids: Tensor, position_ids: Tensor):
        context = self.inference_wrapped_model.inference_context
        if context.config.materialize_only_last_token_logits:
            logits_seq_len = context.num_last_token_logits
        else:
            logits_seq_len = context.padded_active_token_count

        with torch.inference_mode():
            logits = self.inference_wrapped_model.run_one_forward_step(
                {"tokens": input_ids, "position_ids": position_ids, "attention_mask": None}
            )
```

`input_ids` is `[1, num_tokens]` — batch dim is always 1, all requests packed
flat along the token axis (THD style). `AbstractModelInferenceWrapper._forward`
(`abstract_model_inference_wrapper.py:109-127`) never passes `decoder_input`.

### 1.4 The checkpoint, resolved

Real values from `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`, transcribed
in [`checkpoint-config.md`](checkpoint-config.md). Everything below is settled —
do **not** use the CI-shrunk fixtures in `vllm/tests/models/registry.py`.

**Vision tower is C-RADIOv4-H.** Direct mapping onto `RADIOViTModel`:

| HF field | Value | `RADIOViTModel` arg |
| --- | --- | --- |
| `args["model"] = "vit_huge_patch16_224"` | `(1280, 32, 16, 5120)` | `TransformerConfig`: `hidden_size=1280, num_layers=32, num_attention_heads=16, kv_channels=80, ffn_hidden_size=5120` — exactly what `examples/mimo/model_providers/radio_encoder.py:90-95` already builds |
| `patch_size` | 16 | `patch_dim=16` |
| `preferred_resolution` | `[768, 768]` | `img_h=img_w=768` |
| `args["cpe_max_size"]` | 2048 | `max_img_h=max_img_w=2048` → `max_num_rows=max_num_cols=128`, pos-embed grid 16384 (mcore's defaults already are 2048) |
| `args["teachers"]` (4 unique) + `cls_token_per_teacher` | `num_cls_tokens=4` | — |
| `args["register_multiple"] = 10` | `num_registers = 10 − (4 % 10) = 6` | — |
| **combined** | **`num_skip = 4 + 6 = 10`** | **`class_token_len=10`** |
| `video_temporal_patch_size` | 2 | `temporal_patch_dim=2` |
| `separate_video_embedder` | `true` | `separate_video_embedder=True` |
| `args["min_num_patches"]`, `args["max_num_patches"]` | 1024, 13312 | processor only (§4.2) ⇒ **256–3328 tokens per image** |
| `video_target_num_patches`, `video_maintain_aspect_ratio` | 1024, `true` | processor only ⇒ 256 tokens per 2-frame tubelet |
| defaults, pin explicitly | `qkv_bias=True`, `qk_normalization=False`, LayerNorm `eps=1e-6`, `hidden_act=gelu` | matches `radio_encoder.py:96-107` |

> **`class_token_len` must be 10, not the mcore default of 8.** mcore stores one
> combined `class_token` parameter `[class_token_len, hidden]`; vLLM splits it
> into CLS + registers but concatenates them identically, and the converter maps
> `patch_generator.*cls_token → class_token`. Getting this wrong is a silent
> off-by-10-tokens-per-image error.

**Projectors.** Both are `RMSNorm → Linear → ReLU² → Linear`, no bias:

- vision `mlp1`: `RMSNorm(5120, eps=1e-5) → 5120→20480 → ReLU² → 20480→2688`
- audio: `RMSNorm(1024, eps=1e-5) → 1024→4096 → ReLU² → 4096→2688`

`projector_hidden_size = 20480` is large — 4× the LM's own `d_ff`. Do not assume
it is small.

**Audio tower.** Parakeet Conformer: 24 layers, `d=1024`, 8 heads (head_dim 128),
FFN 4096, depthwise conv kernel 9, `convolution_bias=false`. 128 mel bins @
16 kHz, `subsampling_factor=8` ⇒ **80 ms per audio token, 12.5 tokens/second**.
Note `SoundConfig.feat_in` defaults to 80 and the checkpoint does not override
it — the real mel width is `num_mel_bins=128`. **Do not wire `feat_in` to
anything.**

**Language model.** Two findings that make the LM side nearly free:

1. **`hybrid_override_pattern` transfers verbatim.** The checkpoint's
   `"MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"` (52 chars = 23 `M`
   + 23 `E` + 6 `*`) uses exactly mcore's symbol set — `Symbols.MAMBA = "M"`,
   `Symbols.MOE = 'E'`, `Symbols.ATTENTION = "*"`
   (`megatron/core/models/hybrid/hybrid_layer_allocation.py:17-26`). No
   translation layer needed. Attention layers land at indices 5, 12, 19, 26, 33,
   42. There are **no** dense MLP layers, so `intermediate_size: 1856` is inert.
2. **`expand: 2` is stored but unused** — the real Mamba inner size is
   `mamba_num_heads × mamba_head_dim = 64 × 64 = 4096`, **not**
   `2 × 2688 = 5376`. mcore derives `mamba_num_heads` from
   `hidden_size * expand // mamba_head_dim` only when it is `None`
   (`transformer_config.py:1188-1190`), so set it explicitly:
   `mamba_num_heads=64, mamba_head_dim=64, mamba_state_dim=128,
   mamba_num_groups=8`. Leaving it unset silently builds a 5376-wide mixer that
   will not load.

Remaining LM shapes: `hidden_size=2688`, `vocab_size=131072` (untied),
GQA 32/2 heads with `head_dim=128`, RoPE θ=1e4 fully rotary, 128 routed experts
top-6 with `d_ff=1856` plus 1 shared expert `d_ff=3712`, sigmoid routing with
`norm_topk_prob` and `routed_scaling_factor=2.5`, `relu2` activation throughout.

**Quantization, settled empirically.** The NVFP4 variant's `quantized_layers`
(5986 entries) contains **zero** entries under `vision_model.*`,
`sound_encoder.*`, `sound_projection.*`, or `mlp1.*`. Only the LM is quantized
(NVFP4 routed experts, FP8 Mamba `in_proj`/`out_proj` and shared experts, FP8 KV
cache). Budget both towers and both projectors as bf16 and do not look for tower
scale tensors.

**Special tokens.** `<img>` / `</img>` / `<image>`(18) / `<video>`(131081) /
`<so_start>` / `<so_embedding>`(27) / `<so_end>`. vLLM hard-codes the *strings*
and resolves ids through the tokenizer; keep the strings identical.

---

## 2. The one hard problem

Everything else is porting. This is design.

A multimodal request's prompt contains placeholder spans (`<image>`×N,
`<so_embedding>`×N, per-tubelet video spans). Those positions must receive
**encoder-produced embedding rows** instead of embedding-table lookups. In the
dynamic path this collides with five subsystems at once:

1. **CUDA graphs.** The graphed region for hybrid models is owned by
   `HybridModel` at `inference_cuda_graph_scope=block`, and *includes the
   embedding layer* (`hybrid_model.py:379-413`). Any injection must be inside
   that capture, with fixed tensor addresses and a static grid — the
   `optimize-inference-siddharth` hard rules #1 and #3.
2. **Chunked prefill.** A chunk boundary can land in the middle of an image's
   token span (`dynamic_engine.py:1858-2031`). Injection must be indexed by
   `finished_chunk_token_count`, never by whole-prompt offset.
3. **Prefix caching.** `compute_block_hashes_batched`
   (`inference_request.py:91-129`) hashes **raw token ids**. Two different
   images expand to *identical* placeholder ids, so two different images produce
   the same block hash → **false cache hit → wrong output.** This is a
   correctness bug, not a perf issue.
4. **Token accounting.** `_add_request` derives `num_tokens_to_generate`,
   block counts, and cache keys from `len(request.prompt_tokens)`
   (`dynamic_engine.py:1100-1149`). Expansion must therefore happen *before*
   `_add_request`, not inside the step loop.
5. **Sequence parallelism.** Embeddings are SP-scattered after the embedding
   layer (`hybrid_model.py:468-473`); injection must happen on the full
   sequence, before that scatter.

### 2.1 Options considered

| Option | Verdict |
| --- | --- |
| **(A) Run the encoders inline in the LM forward**, as `LLaVAModel` does for training and as `VLMInferenceWrapper` does for static inference | **Rejected.** Encoder output shape varies per request → breaks CUDA-graph capture. Chunked prefill would re-run the tower once per chunk. The tower would sit inside the graphed block. |
| **(B) Pre-embed the whole prompt on the host**, pass a fully-formed `decoder_input` | **Rejected.** Requires the wrapper to call `model.embedding` itself, duplicating the SP-scatter and quantization-padding logic, and moves the embedding layer *out* of the block-scope graph — undoing a measured win (see `skills/optimize-inference-siddharth/references/cuda-graphs.md`). |
| **(C) Run encoders once at admission; scatter rows into a fixed-address GPU buffer; masked-overwrite inside the LM after the embedding layer** | **Chosen.** |

---

## 3. Chosen architecture

Three stages, mirroring vLLM's host/device split (which is a good split — the
host stage is tokenizer-bound and cacheable, the device stage is pure tensor
work), but with the encoder pulled *out* of the graphed step.

```
add_request(request_id, prompt, sampling_params, multimodal_data=…)
│
│  STAGE 1 — HOST (CPU, per request, cacheable)
├─ NemotronOmniProcessor(prompt_text, images, videos, audios)
│    ├─ images → dynamic tiler → pre-patchified [1, ΣPᵢ, 3·P²] + imgs_sizes + num_tokens_per_image
│    ├─ video  → resized frames + frames_indices + frame_duration_ms (INTEGER, §5.4)
│    ├─ audio  → 30 s clips → log-mel [clips, T, mel] + attention mask + audio_num_clips
│    └─ prompt text → EXPANDED token ids  +  mm_embed_positions (which positions get rows)
│
│  STAGE 2 — DEVICE, eager, outside the graphed step
├─ NemotronOmniEncoderStack.encode(...)
│    ├─ RADIOViTModel(dynamic)  → strip CLS/registers → pixel_shuffle_dyn → vision_projection
│    ├─ RADIOViTModel(temporal) → pixel_shuffle → projection → EVS → hybrid text+vision assembly
│    └─ ParakeetEncoder → ParakeetProjection → per-clip trim → concat per item
│    ⇒ mm_embeddings [Σ N_mm, hidden]  (bf16, GPU)
│
└─ DynamicInferenceRequest(prompt_tokens=<expanded>,
                           mm_embeddings=…, mm_embed_positions=…)
   → _add_request()   ← unchanged accounting, now over the EXPANDED length

PER STEP (unchanged control flow)
│
├─ context.add_request(req, prefill_chunk_length)
│    token_to_input_ids[a:b] = this_round_tokens                    (existing)
│    NEW: rows whose position ∈ [a,b) → token_to_mm_embedding[…]
│    NEW: token_to_is_mm[…] = True
│
└─ HybridModel.forward(..., inference_context=ctx)
     decoder_input = self.embedding(input_ids, position_ids)        (existing)
     NEW: decoder_input = ctx.apply_multimodal_embeddings(decoder_input)
          → torch.where(is_mm, mm_buf, decoder_input)               ← graph-safe
     ...SP scatter, quantization padding zero, decoder...           (existing)
```

### 3.1 Why admission-time encoding

Running the towers at `add_request` keeps the per-step host critical section
(`bookkeeping` → `detokenization` → `coordinator_communication`) untouched, and
keeps the graphed region pure. The cost is that a large media payload adds
encoder latency to the admitting call, which for a batch of long videos is
seconds.

**v1:** encode at admission, synchronously. Simple, correct, easy to validate
against vLLM.

**v2 (follow-up, not in scope):** move Stage 2 to a dedicated CUDA stream with
a content-hash encoder cache, mirroring
`gpu_model_runner.py:3203-3211`, and let the scheduler admit a request whose
encoder work has not finished yet. Design the request fields (§4.3) so this is
a scheduler change only, not a data-model change.

### 3.2 Why a masked overwrite is graph-safe

Two preallocated, fixed-address GPU buffers:

```python
# megatron/core/inference/contexts/dynamic_context.py (new, allocated once)
self.token_to_mm_embedding  # [max_tokens, 1, hidden]  params_dtype
self.token_to_is_mm         # [max_tokens, 1, 1]       bool
```

Injection is then

```python
decoder_input = torch.where(self.token_to_is_mm[:n], self.token_to_mm_embedding[:n], decoder_input)
```

Fixed shapes for a given graph bucket, fixed addresses, values change between
replays. This is precisely the pattern hard rule #1 prescribes: "`fill_` it into
a preallocated, fixed-address GPU tensor and read it inside the kernel." Padding
rows have `is_mm == False`, so §5 hard rule #5 ("padding must not do real work")
is satisfied without a zeroing pass.

Note the orientation: mcore's embedding returns `[s, b, h]` with `b == 1`, so
the buffers are `[max_tokens, 1, hidden]`, not `[max_tokens, hidden]`.

**Memory cost.** `max_tokens × hidden × 2 bytes`. This is the *per-step* token
budget, not `max_sequence_length` — e.g. 8192 tokens × 4096 hidden × 2 = 64 MiB.
Gate the allocation on a config flag so text-only deployments pay nothing.

### 3.3 Placement of the injection call

Immediately after `decoder_input = self.embedding(...)` in
`hybrid_model.py:456`, and **before** both the quantization padding zero
(`:460-466`) and the SP scatter (`:468-473`). Order matters:

- before SP scatter, because the mm buffer is indexed by global token position;
- before the padding zero, so the padding zero still wins on padded rows.

---

## 4. File-by-file change plan

Six phases. Each is independently landable and testable. Phase 0–2 give
image-only single-request inference; that is the right first milestone.

### Phase 0 — Embedding injection into the dynamic path (no multimodal yet)

The enabling change. Testable on its own with a synthetic embedding buffer.

| # | File | Change |
| --- | --- | --- |
| 0.1 | `megatron/core/inference/config.py` | Add `enable_multimodal: bool = False` and `multimodal_hidden_size: Optional[int]`. Gates the buffer allocation. |
| 0.2 | `megatron/core/inference/contexts/dynamic_context.py` | Allocate `token_to_mm_embedding` `[max_tokens, 1, hidden]` and `token_to_is_mm` `[max_tokens, 1, 1]` in `__init__`. **Keep them out of the coalesced `_cpu_bookkeeping_buf`** (`:1123-1129`) — embeddings are produced on GPU and never need the pinned H2D mirror; folding them in would dominate the single per-step `cudaMemcpyAsync` (`:2562`). |
| 0.3 | same | Add `apply_multimodal_embeddings(decoder_input)`, and `has_multimodal_embeddings()` (cheap host-side bool, refreshed once per step from a counter — **not** a `.item()`). |
| 0.4 | same, `reset_tensors` (`:2642`) | `token_to_is_mm.fill_(False)`. Do **not** zero `token_to_mm_embedding` — nothing reads it where the mask is False (hard rule #5). |
| 0.5 | `megatron/core/models/hybrid/hybrid_model.py:456` | Insert the injection between the embedding call and the SP scatter (§3.3). |
| 0.6 | `megatron/core/models/gpt/gpt_model.py:331-345` | Same insertion, for symmetry / non-hybrid Omni variants. Optional for v1. |

**Risk:** the CUDA-graph capture must see the injection. Verify
`num_cuda_graphs` still reports the expected bucket count and that a mm step
does not silently fall back to eager (see `cuda-graphs.md`).

### Phase 1 — Request-level multimodal payload

| # | File | Change |
| --- | --- | --- |
| 1.1 | `megatron/core/inference/inference_request.py:359-427` | Add to `DynamicInferenceRequest`: `mm_embeddings: Optional[Tensor]` `[N_mm, hidden]`, `mm_embed_positions: Optional[Tensor]` `[N_mm]` (int32, positions within the expanded prompt), `mm_content_hash: Optional[int]`. |
| 1.2 | same, `serialize` (`:172-198`) | `serialize()` converts every tensor attribute via `.cpu().tolist()`. Explicitly exclude `mm_embeddings` — serializing a few-MB bf16 tensor per request over the coordinator IPC would wreck the host path. |
| 1.3 | same, `DynamicInferenceRequestRecord.merge` (`:714-772`) | Field list is explicit; new fields drop silently otherwise. |
| 1.4 | same, `__post_init__` → `_compute_block_hashes` (`:392-415`) | **Correctness-critical.** Either (a) fold `mm_content_hash` into the parent digest chain of `compute_block_hashes_batched`, or (b) set `enable_prefix_caching = False` whenever `mm_embeddings is not None`. **v1: (b).** It is two lines and cannot be wrong. (a) is the right long-term answer and is a clean follow-up. |
| 1.5 | `megatron/core/inference/contexts/dynamic_context.py:3182-3196` | In `add_request`, after the existing `token_to_input_ids` write, scatter mm rows for positions in `[prefix_skip_tokens, prefill_chunk_length)` into `token_to_mm_embedding[active_token_count + (pos - a)]` and set `token_to_is_mm`. Index off `req.finished_chunk_token_count` so chunked prefill is handled by construction. |
| 1.6 | `megatron/core/inference/engines/dynamic_engine.py:1166-1223` | Widen `add_request` with `multimodal_data: Optional[MultimodalData] = None`. Run Stage 1 + Stage 2 and set the expanded `prompt_tokens` **before** calling `_add_request` (§2, item 4). |

### Phase 2 — Vision path

| # | File | Change |
| --- | --- | --- |
| 2.1 | `megatron/core/inference/multimodal/nemotron_omni/config.py` **(new)** | Parse the composite HF `config.json` into `NemotronOmniConfig` + `RadioVisionConfig` + `SoundConfig`. Mirror vLLM's field reads (`nano_nemotron_vl.py:923-1012`) but resolve the RADIO dims from the `vit_huge_patch16_224` → `(1280, 32, 16, 5120)` table (`vllm/transformers_utils/configs/radio.py:12-20`) into a `TransformerConfig`, reusing `examples/mimo/model_providers/radio_encoder.py:86` `radio_vision_config` as the template. |
| 2.2 | `.../nemotron_omni/image_processor.py` **(new, ~350 lines)** | Port `DynamicResolutionImageTiler` (`vllm/transformers_utils/processors/nano_nemotron_vl.py:251-569`) verbatim-in-behaviour: the `×4` token→patch budget conversion, the `[min_num_patches, max_num_patches]` clip, the 10-round proportional scale-down, `round(dim/patch + 0.5)` (**not** `math.ceil`), `factor = min(…, 1.0)` never-upscale, round-to-multiple-of-2 preferring up, and `stack()` producing `[1, ΣPᵢ, 3·P²]`. Resize is bicubic `antialias=True, align_corners=False`, fp32, then `(x/255 − mean)/std`. |
| 2.3 | `.../nemotron_omni/encoder_stack.py` **(new)** | `NemotronOmniEncoderStack`: owns `RADIOViTModel`, the vision projector, and (Phase 4) the audio tower. `encode_images()` = RADIO(dynamic) → strip `class_token_len` per image → `_pixel_shuffle_dynamic_res` → projector → split by `num_tokens_per_image`. Build `PackedSeqParams(qkv_format='thd')` from `imgs_sizes`; note `RADIOViTModel.forward` **mutates `packed_seq_params` in place** when adding class tokens (`radio.py:411-423`), so pass a fresh object per call. |
| 2.4 | `megatron/core/models/vision/pixel_shuffle.py` **(new, ~30 lines)** | Promote `_pixel_shuffle_dynamic_res` out of `examples/mimo/model_providers/radio_encoder.py:121` into core, unchanged. It is the version that matches vLLM (verified — see §5.2). Have both `examples/mimo` and the new encoder stack import it. |
| 2.5 | `megatron/core/inference/multimodal/nemotron_omni/prompt.py` **(new)** | Placeholder expansion + the token-count contract of §6.1. Tokenize each span component independently (`<img>`, `<image>`×k, `</img>`) — never the joined string. |

### Phase 3 — Video path

| # | File | Change |
| --- | --- | --- |
| 3.1 | `.../nemotron_omni/video_processor.py` **(new)** | Frame resize to a common target from `video_target_num_patches`, aspect-preserving mode snapping both sides to multiples of 2. Cross-check against `examples/multimodal/image_processing.py:47` `find_closest_area_weighted_aspect_ratio` — vLLM says its version mirrors that file, so that is the oracle, not vLLM. `frame_duration_ms = int(1000.0 / fps)`, integer, non-negotiable (§5.4). |
| 3.2 | `megatron/core/models/vision/evs.py` **(new, ~80 lines)** | Direct port of `compute_retained_tokens_count` + `compute_retention_mask` (`vllm/multimodal/evs.py:16-92`). Load-bearing details: the literal `255` sentinel (not `+inf`) for frame 0, `stable=True` descending argsort, `spatial_merge_size=1`, applied **after** the projector on LM-dim embeddings. Skip the M-RoPE half of that file — Nemotron-H does not use M-RoPE. |
| 3.3 | `.../nemotron_omni/encoder_stack.py` | `encode_videos()`: one `RADIOViTModel` call with `num_frames=[…]` and `temporal_patch_dim=T`. **We can batch videos and images together** — mcore's `_apply_temporal_grouping` supports mixed items, so vLLM's `requires_sequential_video_encoding` workaround is unnecessary (§5.3). |
| 3.4 | same | `_create_final_video_embeddings` equivalent: re-derive the per-tubelet separator token ids from the *actual* retained counts, embed them through the LM embedding table, and scatter vision rows into the `<image>` positions. This makes the video encoder output depend on the tokenizer and the LM embedding table — an unusual coupling, faithfully reproduced (§5.5). |

### Phase 4 — Audio path

| # | File | Change |
| --- | --- | --- |
| 4.1 | `.../nemotron_omni/audio_processor.py` **(new, ~220 lines)** | Port `ParakeetExtractor` (`vllm/model_executor/models/parakeet.py:138-335`): mono by channel mean; 30 s clip split with a ≥0.1 s tail; right-pad to batch max; pre-emphasis `x[t] − 0.97·x[t−1]` with valid-length masking; STFT `n_fft=512, hop=160, win=400`, Hann `periodic=False`, `pad_mode="constant"`; slaney mel filterbank; `log(mel + 2⁻²⁴)`; per-clip normalization over valid frames with **sample** variance `/(n−1)`. All fp32. Cross-check the mel filterbank against the vendored `megatron/core/models/audio/nemo_audio_preprocessing.py:111` `_create_mel_filterbank` rather than pulling in `transformers.audio_utils`. |
| 4.2 | `megatron/core/models/audio/parakeet_model.py` **(new)** | `ProjectedParakeet` equivalent: HF `transformers.ParakeetEncoder` + `ParakeetProjection` (`RMSNorm → Linear → ReLU² → Linear`). Replicated, unquantized, `llm_dtype` in, bf16 out before projection. **Do not** reuse `ParakeetHuggingFaceModel` (`megatron/core/models/huggingface/fastconformer_model.py:28`) as-is: it loads a *separate* `hf://`/`nemo://` checkpoint via `AutoModel.from_pretrained`, whereas we must load `sound_encoder.*` out of the Omni checkpoint. Reuse its dtype/sampling-rate helpers only. |
| 4.3 | same | Expose `_get_subsampling_output_length`. It is called from three places (host token count, device trim, dummy-input sizing) and must agree with the host extractor exactly. |
| 4.4 | `.../nemotron_omni/encoder_stack.py` | `encode_audio()`: encode all clips, trim each to `_get_subsampling_output_length(mask.sum(1))`, concatenate clips belonging to one audio item. Assert the result length equals the host's `audio_token_count` (§6.1). |

**Dependency risk.** `transformers` is unpinned in `pyproject.toml:76,87`, and
`ParakeetEncoder` / `ParakeetEncoderConfig` require **transformers ≥ 5.5.3**
(vLLM's own pin). Verify the CI container ships that; if not, this becomes a
`mcore-build-and-dependency` task or forces a native conformer reimplementation
(≈600 lines). See §7.2, Q5. Fixed shapes from §1.4: 24 layers, `d=1024`, 8 heads,
FFN 4096, conv kernel 9, no conv bias, 128 mel bins, `subsampling_factor=8`
(80 ms per audio token). Ignore `SoundConfig.feat_in`, which is a stale default
of 80 and is not overridden by the checkpoint.

### Phase 5 — Checkpoint loading

| # | File | Change |
| --- | --- | --- |
| 5.1 | `tools/checkpoint/` or a new `.../nemotron_omni/checkpoint.py` | HF → mcore mapping. No single path exists today: `megatron/core/export/` is TRT-LLM only, and `tools/checkpoint/loader_llava.py` reads RADIO but sources vision weights from `torch.hub` (`examples/multimodal/model_converter/radio_converter.py`), not HF safetensors. |
| 5.2 | same | Four prefix classes, mirroring `nano_nemotron_vl.py:1509-1568`: `language_model.backbone.*` → hybrid LM; `mlp1.*` → vision projector; `vision_model.radio_model.*` → `RADIOViTModel`; `sound_encoder.*` / `sound_projection.*` → audio tower. Skip `input_conditioner.*` (normalization lives in the processor) and `summary_idxs`. Reuse the QKV head-interleave reindex from `radio_converter.py:42-95`. |
| 5.3 | same | **Do not** skip `ls1`/`ls2` the way vLLM does (§5.1). Load them if present; assert-warn if they are not ~1.0. |
| 5.4 | same | Mamba SSM state dtype: force **fp32**. vLLM's own config hook for this is keyed to the non-Omni architecture string and therefore never fires for Omni checkpoints (`vllm/model_executor/models/config.py:857`) — a vLLM bug we must not copy. |

### Phase 6 — Serving surface and tests

| # | File | Change |
| --- | --- | --- |
| 6.1 | `.../dynamic_text_gen_server/endpoints/chat_completions.py:258-270` | Today OpenAI multimodal content blocks are **flattened to text and images discarded**. Route `image_url` / `input_audio` / video parts into `multimodal_data` instead. |
| 6.2 | `megatron/core/inference/text_generation_controllers/text_generation_controller.py` | No change expected — the controller stays modality-agnostic because injection is context-mediated. Confirm `_dynamic_step_context_init` (`:717`) needs nothing. |
| 6.3 | `tests/unit_tests/inference/multimodal/` **(new)** | See §6.2. |
| 6.4 | `tests/functional_tests/` | Add an Omni recipe via `skills/add-inference-functional-tests`. |

### Files touched, at a glance

**Modified (11):** `inference/config.py`, `inference/contexts/dynamic_context.py`,
`inference/contexts/gpu_view.py` (only if buffers are folded in — recommended
not), `inference/inference_request.py`, `inference/engines/dynamic_engine.py`,
`models/hybrid/hybrid_model.py`, `models/gpt/gpt_model.py`,
`.../endpoints/chat_completions.py`, `examples/mimo/model_providers/radio_encoder.py`
(import moved), `pyproject.toml` (maybe), `tools/checkpoint/…`.

**New (~10):** `megatron/core/inference/multimodal/nemotron_omni/{config,image_processor,video_processor,audio_processor,prompt,encoder_stack,checkpoint}.py`,
`megatron/core/models/vision/{pixel_shuffle,evs}.py`,
`megatron/core/models/audio/parakeet_model.py`.

Notably **not** touched: `RADIOViTModel`, `HybridStack`, the MoE stack, the KV
block allocator, the attention metadata, the sampling path.

---

## 5. Divergences from the vLLM implementation

The task asked for this explicitly. Divergences split into three kinds:
**vLLM bugs we should not copy**, **vLLM limitations we can beat**, and
**forced differences** from mcore's architecture.

### 5.1 Both vLLM and mcore discard LayerScale — the risk is against HF, not vLLM

`RadioModel.load_weights` explicitly `continue`s on any `ls1`/`ls2` key
(`radio.py:780-782`); `InternVisionEncoderLayer.__init__` then initializes them
to `initializer_factor * ones` = 1.0 and multiplies by them in forward. So in
vLLM **layer scale is identity**. The skip branch only exists because the
checkpoint *does* carry those tensors.

`megatron/core/models/vision/radio.py` has no LayerScale at all — grep for
`ls1`/`ls2`/`layer_scale` returns nothing. So **mcore already matches vLLM here
for free**, and cut point (b) parity is unaffected. The exposure is that *both*
may differ from the HF reference implementation, which does apply them.

**Action:** dump `blocks.*.ls{1,2}` from the checkpoint and check ≈1.0. If they
are non-unit, both engines are dropping a real transform, and mcore's radio-g
converter (`examples/multimodal/model_converter/radio_converter.py`, which maps
`blocks.N.ls{1,2}.grandma → decoder.layers.N.ls{1,2}`) shows the shape of the
fix. Track as a correctness bug against HF, not as a port bug.

### 5.2 Pixel shuffle: mcore has two mutually incompatible versions

Verified by hand-deriving the index maps:

| Implementation | Behaviour |
| --- | --- |
| `examples/mimo/model_providers/radio_encoder.py:121` `_pixel_shuffle_dynamic_res` | Folds a **2×2 spatial neighbourhood** into channels; 4c layout `[h-parity, w-parity, channel]`. **Identical to vLLM's `pixel_shuffle_dynamic_res`.** |
| `megatron/core/models/multimodal/llava_model.py:1342-1347` — the `h`/`w`-provided branch | `x.reshape(n, patches // 4, c*4)`. A **plain reshape**, grouping 4 *horizontally adjacent* patches (a 1×4 strip). **Not equivalent.** |
| `llava_model.py:1348-1366` — the square branch | Verified identical to both mimo's and vLLM's. Only the local variable naming differs (`n, w, h, c = x.size()` on a `(n, d1, d2, c)` tensor). |

`LLaVAModel`'s packed dynamic-res path calls the non-equivalent branch
(`llava_model.py:1112`), while the MIMO provider deliberately does not — its
docstring says *"Element ordering intentionally differs from core `pixel_shuffle`
(e2e-validated); do not swap to match it."*

**Action:** use the mimo/vLLM semantics (Phase 2.4), and raise the `LLaVAModel`
discrepancy with the training owners (§7.2, Q1). One of the two training paths is
producing checkpoints the other cannot serve.

### 5.3 We can beat vLLM on video encoding

vLLM declares `requires_sequential_video_encoding = True`
(`nano_nemotron_vl.py:904`) and encodes videos one at a time
(`gpu_model_runner.py:3143-3171`) purely because batched dynamic-resolution
video is unimplemented there. mcore's `_apply_temporal_grouping`
(`radio.py:449-550`) already handles mixed image/video items in one packed
batch. **Deliberate divergence: batch them.** Numerically identical (the tubelet
grouping and last-frame padding are per-item), materially faster.

Similarly, vLLM has **no batch-level encoder data parallelism** for this model
(`supports_encoder_tp_data` is not set, so `--mm-encoder-tp-mode data` is
downgraded with a warning). Nothing in the math forbids sharding the media batch
across ranks in mcore; the replicated audio tower in particular would benefit.
Out of scope for v1, worth noting as headroom.

### 5.4 Behaviours that must be copied bit-for-bit

Each of these has a specific failure mode; none are stylistic.

| Behaviour | Why | Failure if wrong |
| --- | --- | --- |
| `frame_duration_ms = int(1000.0 / fps)`, integer | Timestamps are re-rendered inside the model in bf16 context; float fps made host and device disagree | Different timestamp strings → different separator token counts → shape mismatch |
| Tokenize span components **independently** | `separator`, `<img>`, `<image>`×k, `</img>` concatenated, never joined-then-tokenized | Token merging across boundaries changes the count |
| EVS: `255` sentinel, `stable=True` argsort, `kept = max(tokens_per_frame, int(total·(1−q)))`, pruning after the projector | Reproducibility | Non-deterministic retention set |
| Tubelet padding by **last-frame repetition**, einops order `(tubelets frames) spatial feat -> tubelets spatial (frames feat)` | Matches training | Garbage video features |
| Dynamic tiler: `round(dim/P + 0.5)`, `factor = min(…, 1.0)`, round grids to multiples of 2 preferring up, 10-round budget loop | Token-count parity | Shape mismatch |
| Audio: 30 s clips + ≥0.1 s tail, per-clip normalization with **sample** variance, token count floored at 1 | Token-count parity | Off-by-N audio tokens |
| Image token budget depends on **prompt length**: `max_model_len − text_prompt_length − 4` | Image resolution varies with prompt | Different resolution than the checkpoint expects |
| CLS/register stripping: `num_skip` tokens **per item** | mcore's `class_token_len` must equal vLLM's `num_cls_tokens + num_registers`; for this checkpoint that is `4 + 6 = 10`, **not** mcore's default of 8 (§1.4) | Off-by-10 per image, silently |
| Encoders unquantized even for FP8/NVFP4 checkpoints | vLLM never passes `quant_config` to the towers | Failed scale loads |

### 5.5 Forced architectural differences

| vLLM | mcore | Consequence |
| --- | --- | --- |
| `MaskMetadata(cu_seqlens, max_seqlen)` threaded 5 levels into `MMEncoderAttention` → `flash_attn_varlen_func` | `PackedSeqParams(qkv_format='thd')` into `TransformerBlock` | Equivalent varlen attention. Watch the in-place `packed_seq_params` mutation at `radio.py:411-423` |
| `embed_multimodal` called by the runner per step, with a content-hash encoder cache | Encoders run at admission (§3.1) | Different latency profile; no cross-request reuse in v1 |
| Video embeddings returned as the *full* replacement span (separators embedded via the LM table) | Same, faithfully reproduced | Encoder output couples to tokenizer + embedding table |
| `is_embed` mask machinery in the multimodal registry | `mm_embed_positions` on the request + `token_to_is_mm` in the context | Same semantics, simpler |
| No pipeline parallelism (`SupportsPP` absent) | PP is available, but `_allocate_recv_buffer` (`abstract_model_inference_wrapper.py:148-158`) divides `seq_len` by `tp_size` for SP | PP+multimodal needs explicit validation; consider gating it off in v1 |
| Encoder runs eagerly; `SupportsEncoderCudaGraph` absent | Same — encoders stay outside the graph | Matches |
| `torch.compile` on the host resize/mel ops and per ViT layer | Not replicated | Perf only, no semantic change |

---

## 6. Correctness contract and validation

### 6.1 The token-count contract — port this first

Before any kernel work, assert that host-side placeholder counts equal
device-side embedding rows.

| Modality | Expanded span | Count | Which positions get rows |
| --- | --- | --- | --- |
| Image | `<img>` + `<image>`×F + `</img>` | `F = grid_h·grid_w / 4` (dynamic path) | only `<image>` positions |
| Audio | `<so_start>` + `<so_embedding>`×N + `<so_end>` | `N = Σ` subsampled clip lengths, ≥1 | only `<so_embedding>` positions |
| Video | per tubelet: separator + `<img>` + `<image>`×k + `</img>` | `Σk = frames·tokens_per_frame`, or EVS `kept` | **all positions in the span** |

The video row is the subtle one, and the reason for Phase 3.4.

Add a debug assertion in `add_request` that
`mm_embeddings.shape[0] == mm_embed_positions.numel()` and that every position is
within the expanded prompt. Keep it behind a flag; it is a host-side check but
runs once per request, not per step.

### 6.2 Test matrix

Unit tests (`tests/unit_tests/inference/multimodal/`), following the reference-
implementation pattern from `skills/add-inference-unit-tests`:

1. **Tiler parity** — the doctest in
   `vllm/transformers_utils/processors/nano_nemotron_vl.py:290-327` is a ready-made
   fixture: `patch=16, ds=0.5, max_model_len=16384, target=8192` → `(2880, 2880)`
   → 8100 tokens post-shuffle.
2. **Token-count parity** across: single image; multi-image under a tight
   dynamic budget; 1-frame video; `T` not dividing the frame count; EVS on/off;
   single- and multi-clip audio.
3. **Pixel-shuffle equivalence** — assert the promoted core function matches an
   explicit 2×2-fold reference on non-square grids, and *differs* from
   `llava_model.pixel_shuffle(h=,w=)` (a regression guard so the two are never
   silently swapped).
4. **EVS gating** — first frame fully retained; `stable=True` tie-breaking;
   re-derive `kept` independently in the test so a changed bound fails loudly.
5. **Injection graph-safety** — capture a CUDA graph over a mm step, replay with
   different embedding *values* at the same addresses, assert the output changes;
   assert padded rows are untouched by prefilling the buffer with a sentinel.
6. **Chunked-prefill boundary** — force a chunk boundary inside an image span,
   assert `decoder_input` equals the unchunked result.
7. **Prefix-cache safety** — two requests, identical prompts, *different*
   images: assert no false cache hit (i.e. that Phase 1.4 fires).

### 6.3 Oracle comparison

Run vLLM with a fixed seed and dump four cut points; compare against mcore.

| Cut point | Tolerance |
| --- | --- |
| (a) processor outputs: `pixel_values_flat`, `input_audio_features`, expanded token ids | **exact** (integers and fp32 math) |
| (b) raw tower output before pixel-shuffle/projection | bf16 tolerance — **expect a delta if LayerScale ≠ 1** (§5.1) |
| (c) per-item embeddings from `embed_multimodal` | bf16 tolerance |
| (d) final `inputs_embeds` / `decoder_input` | bf16 tolerance |

Getting (a) and the token counts exact is what prevents the long tail of
shape-mismatch bugs. Everything else is a numerics conversation.

---

## 7. Decisions taken, and what is still open

### 7.1 Settled

| Decision | Resolution |
| --- | --- |
| **v1 scope** | Image-first. Phases 0–2 land images; video and audio follow. |
| **Prefix caching** | Multimodal requests bypass the prefix cache in v1 (Phase 1.4b). Content-hash chaining is deferred. |
| **Pipeline parallelism** | Assert `PP == 1` for multimodal requests. Matches vLLM, avoids the `_allocate_recv_buffer` SP-division interaction (`abstract_model_inference_wrapper.py:148-158`) and the dead PP paths in `vlm_inference_wrapper.py:50,53`. |
| **Reference config** | Available — [`checkpoint-config.md`](checkpoint-config.md), transcribed in §1.4. Phase 2.1 is unblocked. The CI-shrunk fixtures at `vllm/tests/models/registry.py:1182-1206` are not to be used. |
| **`video_maintain_aspect_ratio`** | Implement against `true` (the checkpoint's value), not the fixture's `false`. |
| **Vision projector norm** | Confirmed: `mlp1` starts with `RMSNorm(5120, eps=1e-5)`. mcore's `MultimodalProjector` with `linear_fc1=ColumnParallelLinear` (`nemotron_moe_vlm.py:85-94`) has no norm, so Phase 2.3 must either swap in `TELayerNormColumnParallelLinear` or add an explicit leading `RMSNorm`. Same for the audio projector at `RMSNorm(1024, eps=1e-5)`. |
| **Tower quantization** | Confirmed unquantized: the NVFP4 variant's 5986 `quantized_layers` entries contain nothing under `vision_model.*`, `sound_encoder.*`, `sound_projection.*`, or `mlp1.*`. Budget both towers and both projectors as bf16. |
| **LayerScale parity** | mcore's RADIO has no LayerScale either, so vLLM parity holds for free (§5.1). Only HF-reference parity is at risk. |
| **LM config** | `hybrid_override_pattern` transfers verbatim; `mamba_num_heads=64` must be set explicitly to override the `expand: 2` derivation (§1.4). |

### 7.2 Still open

**Q1 — Which pixel shuffle did the Nemotron Omni checkpoint train with?**
`examples/mimo/.../radio_encoder.py:121` (2×2 spatial fold, matches vLLM) and
`llava_model.py:1342-1347` (plain reshape, 1×4 strip) are genuinely different
operations, and both are live in the tree for dynamic resolution. vLLM agrees
with the MIMO one, so I plan to use it — but if Omni was trained through
`LLaVAModel`, that is wrong and vLLM has a latent bug. **Who owns the Omni
training recipe, and which path did it run?** This is now the top blocking
question: the config cannot answer it, and a wrong choice produces plausible-
looking but degraded output rather than a crash.

**Q2 — Are the checkpoint's `ls1`/`ls2` ≈ 1.0?**
Needs a `state_dict` key/value dump, not the config. If non-unit, vLLM *and*
mcore are both dropping a real transform (§5.1). Determines whether an
HF-reference mismatch at cut point (b) is expected.

**Q3 — `video_pruning_rate`: follow the config or the CLI?**
The checkpoint asks for `0.7` and the HF reference honours it; vLLM reads it
**only** from `--video-pruning-rate`, defaulting to `None`, so out of the box
vLLM does no EVS pruning at all. Matching vLLM's default means ignoring an
explicit checkpoint request. My inclination is to default to the config's `0.7`
and let a CLI flag override, which is a deliberate divergence — **confirm?**

**Q4 — Image token budget: engine `max_model_len` or the HF `262144`?**
vLLM computes the per-image patch budget from
`max_model_len − text_prompt_length − 4`, so image *resolution* changes with the
serving flag. The HF processor config carries a fixed `max_model_len: 16384`.
Following vLLM makes output depend on a deployment flag; following HF pins it.
Recommend pinning to 16384 for reproducibility, with vLLM-compatible behaviour
behind a flag — **confirm?**

**Q5 — Which `transformers` version does the CI container ship?**
`ParakeetEncoder` needs ≥ 5.5.3, and `pyproject.toml:76,87` pins nothing. If the
container is on 4.x, Phase 4 either becomes a dependency bump or a ~600-line
native conformer. Related: vLLM tolerates a v4↔v5 `convolution_bias=False`
difference where conv bias params are unregistered but may exist in the
checkpoint (`parakeet.py:112-131`); I will need the same tolerance. Not blocking
until Phase 4.

**Q6 — Is `sound_timestamps` a real feature?**
mcore's `LLaVAModel` threads a `sound_timestamps` argument
(`llava_model.py:946, 1257`) with **no vLLM counterpart**. Nemotron Omni feature
that vLLM is missing, or dead code from another model? Affects Phase 4's
interface only.

---

## 8. Sequencing

| Phase | Deliverable | Gate |
| --- | --- | --- |
| 0 | Embedding injection, synthetic buffer | Graph-safety test (6.2.5) passes; text-only perf unchanged |
| 1 | Request payload + context scatter | Chunked-prefill test (6.2.6) and prefix-cache test (6.2.7) pass |
| 2 | **Image-only Omni inference works** | Token-count parity (6.2.2) + oracle cut points (a)/(d) for images |
| 3 | Video + EVS | Oracle parity for `T`-not-dividing and EVS on/off |
| 4 | Audio | Oracle parity for multi-clip audio |
| 5 | HF checkpoint load (config mapping per §1.4) | Real checkpoint produces sane generations |
| 6 | Server surface + functional test recipe | `run-inference-functional-tests` green |

Phase 2 is the milestone that proves the architecture. Everything after it is
additive.

---

## 9. Cross-references

- Inference optimization rules this design obeys (fixed addresses, no per-step
  host sync, padding does no work, graph scope):
  `skills/optimize-inference-siddharth/SKILL.md` and
  `references/cuda-graphs.md`
- Test authoring: `skills/add-inference-unit-tests`,
  `skills/add-inference-functional-tests`
- The resolved checkpoint config: [`checkpoint-config.md`](checkpoint-config.md)
- Dependency / container changes (§7.2, Q5):
  `.claude/skills/mcore-build-and-dependency/SKILL.md`
- vLLM behavioural reference: [`vllm-reference.md`](vllm-reference.md)
