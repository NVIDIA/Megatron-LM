# Nemotron 3 Nano/Super Omni — Multimodal Preprocessing & Encoder Design

> **Purpose.** This is an implementation-oriented reference for the vLLM
> implementation of NVIDIA Nemotron Omni (audio + video + image + text), written
> so that it can be used to port the model to another inference backend
> (specifically **Megatron-Core inference**). Every non-obvious behaviour is
> traced to a file and line range in this repository so the port can be verified
> against the source of truth.
>
> **Audience.** An engineer/agent that has this repository checked out and needs
> to reproduce Nemotron Omni's input pipeline and encoders exactly (bit-for-bit
> where practical, token-count-for-token-count always).
>
> Line numbers refer to the state of the tree at the time of writing. If a range
> looks off by a few lines, search for the named symbol instead.

---

## 1. Model identity and registration

Nemotron Omni is not a separate model class in vLLM. Three HF architecture
strings all resolve to the same implementation class:

```text
vllm/model_executor/models/registry.py:513-515
    "NemotronH_Nano_VL_V2":             ("nano_nemotron_vl", "NemotronH_Nano_VL_V2"),
    "NemotronH_Nano_Omni_Reasoning_V3": ("nano_nemotron_vl", "NemotronH_Nano_VL_V2"),
    "NemotronH_Super_Omni_Reasoning_V3":("nano_nemotron_vl", "NemotronH_Nano_VL_V2"),
```

The implementing class is `NemotronH_Nano_VL_V2`
(`vllm/model_executor/models/nano_nemotron_vl.py:901`). "Omni" vs "VL" is decided
purely by **whether the checkpoint config contains a `sound_config` block**; if
it does, an audio tower is constructed (`nano_nemotron_vl.py:979-990`). There is
no other behavioural difference keyed off the architecture name.

Reference checkpoints (from `tests/models/registry.py:1182-1206`):

| Architecture | Example HF id |
| --- | --- |
| `NemotronH_Nano_Omni_Reasoning_V3` | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` |
| `NemotronH_Super_Omni_Reasoning_V3` | (same family, larger; not available online in tests) |

Interfaces implemented (`nano_nemotron_vl.py:901-905`):

```python
class NemotronH_Nano_VL_V2(
    nn.Module, HasInnerState, IsHybrid, SupportsMultiModal, SupportsMultiModalPruning
):
    requires_sequential_video_encoding = True
```

Notable **absences**, which are real constraints in vLLM today:

- No `SupportsPP` → **no pipeline parallelism**.
- No `supports_encoder_tp_data = True` → **no batch-level encoder data
  parallelism** (see §7.4).
- No `SupportsEncoderCudaGraph` → encoder is always run eagerly (the CUDA-graph
  manager in the runner skips it).
- `SupportsMultiModalPruning` is declared, but `recompute_mrope_positions` is
  **not** implemented — it is never called because Nemotron-H does not use
  M-RoPE (the runner guards with `self.uses_mrope`,
  `vllm/v1/worker/gpu_model_runner.py:3304`). The marker's only practical effect
  for this model is enabling per-video sequential encoding
  (`gpu_model_runner.py:3143-3171`).

---

## 2. File index (start here)

| Concern | Path |
| --- | --- |
| Model + encoders + embedding assembly | `vllm/model_executor/models/nano_nemotron_vl.py` |
| Host-side processor (images/video/audio → tensors + prompt text) | `vllm/transformers_utils/processors/nano_nemotron_vl.py` |
| RADIO vision tower | `vllm/model_executor/models/radio.py` |
| Shared ViT blocks (attention/MLP, TP layers) | `vllm/model_executor/models/intern_vit.py` |
| Parakeet audio tower + mel front-end | `vllm/model_executor/models/parakeet.py` |
| RADIO config synthesis | `vllm/transformers_utils/configs/radio.py` |
| Parakeet config + extractor config | `vllm/transformers_utils/configs/parakeet.py` |
| Nemotron-H LM config | `vllm/transformers_utils/configs/nemotron_h.py` |
| Nemotron-H LM (Mamba2 + attention + MoE) | `vllm/model_executor/models/nemotron_h.py` |
| EVS video token pruning | `vllm/multimodal/evs.py` |
| ViT TP/DP mode helper | `vllm/model_executor/models/vision.py:142-159` |
| Runtime config hooks (mamba cache dtype) | `vllm/model_executor/models/config.py:605-641, 857` |
| Encoder execution / caching / merge | `vllm/v1/worker/gpu_model_runner.py:3123-3213, 3260-3330` |
| Multimodal contribution guide (framework concepts) | `docs/contributing/model/multimodal.md` |
| Encoder TP/DP documentation | `docs/configuration/optimization.md:255-275` |
| Unit test (weight-skipping behaviour) | `tests/models/multimodal/test_nano_nemotron_vl.py` |

---

## 3. Config surface

The HF `config.json` is a composite. vLLM reads it as follows.

### 3.1 Top-level (vision/adapter-related)

Consumed in `nano_nemotron_vl.py:923-1012` and
`processors/nano_nemotron_vl.py:581-623`:

| Field | Use |
| --- | --- |
| `force_image_size` | Base tile size in pixels (`image_size`) |
| `patch_size` | ViT patch size (pixels) |
| `downsample_ratio` | Pixel-shuffle factor; must be `0.5` (asserted for the dynamic tiler, `processors/...:277-283`) |
| `use_thumbnail` | Whether the fixed-tile path appends a thumbnail tile |
| `norm_mean`, `norm_std` | RGB normalization, applied **in the processor**, not in the model |
| `vit_hidden_size` | RADIO hidden size, used to size `mlp1` |
| `projector_hidden_size` | `mlp1` intermediate size |
| `template`, `image_tag_type`, `ps_version` | Prompt/format metadata; `ps_version != "v1"` is the supported layout |
| `vision_config` | RADIO settings (see below) |
| `text_config` | Nemotron-H LM config |
| `sound_config` | Parakeet audio settings; **presence enables audio** |

### 3.2 `vision_config` → `RadioConfig`

Synthesized in `nano_nemotron_vl.py:1570-1600`; the config class is
`vllm/transformers_utils/configs/radio.py:23-110`. Important:

- `vision_config.args["model"]` is a timm name; hidden size / depth / heads /
  FFN size come from a **hard-coded table**
  (`configs/radio.py:12-17`). Nemotron Omni uses `vit_huge_patch16_224` →
  `(hidden=1280, layers=32, heads=16, ffn=5120)`.
- `preferred_resolution[0]` → `image_size` (default 224 if absent).
- `video_temporal_patch_size` (call it **T**) and `separate_video_embedder` are
  read from the *top level of* `vision_config`, not from `args`
  (`nano_nemotron_vl.py:1580-1587`).
- `vision_config.args` also carries `min_num_patches` / `max_num_patches`, and
  **their presence is the switch for dynamic-resolution images**
  (`processors/nano_nemotron_vl.py:621-623`). Nemotron Omni checkpoints set
  them, so Omni uses the dynamic-resolution image path.
- `cpe_max_size` controls the size of the learned position-embedding grid that
  gets interpolated/windowed for arbitrary input sizes.
- `teachers` (RADIO distillation teachers) only determines the **number of CLS
  tokens** (`num_cls_tokens = len(set(t["name"] for t in teachers))` when
  `cls_token_per_teacher`) and which summaries are selected. Summaries are
  computed but **discarded** by this model.

### 3.3 `sound_config` → `ParakeetConfig` / `ExtractorConfig`

`vllm/transformers_utils/configs/parakeet.py`:

- `ParakeetConfig.from_hf_config` (lines 25-37) forwards the whole sound config
  into HF's `ParakeetEncoderConfig`, forcing `scale_input=False`,
  `attention_bias=False`, and `max_position_embeddings = max_model_len + 1`,
  and attaching `llm_hidden_size`.
- `ExtractorConfig.from_hf_config` (lines 57-72) pulls the mel front-end
  parameters: `num_mel_bins → feature_size`, `sampling_rate`,
  `subsampling_factor`, `subsampling_conv_kernel_size`,
  `subsampling_conv_stride`, plus optional overrides for `hop_length` (160),
  `win_length` (400), `preemphasis` (0.97), `n_fft` (512), `padding_value` (0).
  Clip windowing defaults: `clip_duration_s = 30`, `clip_min_duration_s = 0.1`.
- Projection sizing: `projection_hidden_size`, `projection_bias`,
  `projection_eps` (default 1e-5).

### 3.4 Runtime config hook (Mamba state cache dtype)

`vllm/model_executor/models/config.py:635-641` defines `NemotronHNanoVLV2Config`
which forces the **SSM state cache to the dtype declared in `text_config`, or
`float32` by default** (`config.py:605-625`). It is registered only for the
`"NemotronH_Nano_VL_V2"` architecture string (`config.py:857`), and
`MODELS_CONFIG_MAP` is keyed by architecture
(`vllm/config/vllm.py:2040-2050`), so **the Omni architecture names do not get
this hook**. Port implication: do not infer the SSM state dtype from vLLM's
runtime behaviour on Omni checkpoints — **use fp32 Mamba SSM state**, which is
the documented safe default for Nemotron-H ("Only `float32` is known to have no
accuracy issues by default", `config.py:606-607`).

---

## 4. Two-stage architecture

```
                 HOST (CPU, per request, cacheable)                    DEVICE (per forward)
 ┌───────────────────────────────────────────────────────┐   ┌──────────────────────────────────────┐
 │ NanoNemotronVLProcessor.__call__                      │   │ NemotronH_Nano_VL_V2.embed_multimodal│
 │   images → tiles/patch-vectors + norm                 │   │   images → RADIO → pixel-shuffle→mlp1│
 │   video  → resized frames + frame metadata            │   │   video  → RADIO(tubelets) → EVS →   │
 │   audio  → log-mel features + attention mask          │──►│            hybrid text+vision embeds │
 │   text   → prompt with expanded placeholder spans     │   │   audio  → Parakeet → projection     │
 │ + token-count bookkeeping (must match encoder output) │   │                                      │
 └───────────────────────────────────────────────────────┘   └──────────────────┬───────────────────┘
                                                                               │  per-item tensors
                                                            ┌──────────────────▼───────────────────┐
                                                            │ scatter into inputs_embeds at         │
                                                            │ placeholder positions, then           │
                                                            │ Nemotron-H LM forward (text-only)     │
                                                            └───────────────────────────────────────┘
```

The **contract between the two stages** is the token count: the host stage
expands each media placeholder into exactly as many token ids as the device
stage will produce embedding rows for. Any mismatch is a hard failure. This is
the single most important thing to get right in a port (§6).

Encoder outputs are cached per media item by content hash
(`gpu_model_runner.py:3203-3211`), so the encoder runs once per unique item even
across requests/chunks.

---

## 5. Host stage: preprocessing

Entry point: `NanoNemotronVLProcessor.__call__`
(`processors/nano_nemotron_vl.py:1009-1074`). Order matters: images, then video,
then audio, then tokenize (lines 1031-1047). Each `_preprocess_*` both returns
tensors **and rewrites the prompt text**, expanding placeholders in place.

Special tokens (`processors/nano_nemotron_vl.py:36-41`):

```python
IMG_START = "<img>";        IMG_END = "</img>";        IMG_CONTEXT = "<image>"
AUDIO_START = "<so_start>"; AUDIO_END = "<so_end>";    AUDIO_CONTEXT = "<so_embedding>"
```

Video uses `<video>` as the user-facing placeholder and `IMG_CONTEXT` as the
per-token embedding placeholder (there is no dedicated video token;
`NanoNemotronVLProcessingInfo.get_video_token` returns `IMG_CONTEXT`,
`nano_nemotron_vl.py:223-224`).

### 5.1 Images — fixed-tile path (InternVL-style)

Used when `vision_config.args` has **no** `min_num_patches`.
`dynamic_preprocess` (`processors/nano_nemotron_vl.py:93-142`):

1. Choose a tile grid from InternVL target ratios for `max_num_tiles`
   (default `DEFAULT_NUM_TILES = 12`, line 45) via
   `get_internvl_target_ratios` / `calculate_internvl_targets` (imported from
   `processors/internvl.py`).
2. Bicubic resize (antialias, `align_corners=False`) to `(target_h, target_w)`,
   fused with `x/255` and `(x-mean)/std` in one compiled op
   (`_bicubic_resize_and_normalize`, lines 59-82; `@torch.compile(dynamic=True)`).
3. Cut into `image_size × image_size` tiles by reshape/permute (lines 124-130).
4. If `use_thumbnail` and more than one tile, append a whole-image thumbnail
   resized to `image_size²` (lines 132-140).

Output: `pixel_values_flat` `[sum_tiles, 3, image_size, image_size]` and
`image_num_patches` (tiles per image, including thumbnail)
(`processors/...:702-716`).

Tokens per image: `num_patches * num_image_token`, where

```
num_image_token = (image_size / patch_size)² * downsample_ratio²
                 (processors/nano_nemotron_vl.py:600-602)
```

i.e. 256 for `image_size=512, patch_size=16, downsample_ratio=0.5`.

### 5.2 Images — dynamic-resolution path (what Omni actually uses)

Class `DynamicResolutionImageTiler` (`processors/nano_nemotron_vl.py:251-569`).
Each image becomes **one** variable-size "tile" whose patch grid is budgeted
against the remaining context length:

1. Token budget: `max_model_len - text_prompt_length - 4`
   (`max_num_tokens_available`, lines 329-330). `text_prompt_length` is the
   token length of the prompt with `<image>` markers removed
   (`processors/...:686-689`).
2. `compute_params` (lines 463-548) converts the post-shuffle token budget into
   a patch budget (`×4` because pixel-shuffle merges 2×2 patches → 1 token),
   clips per-image budget to `[min_num_patches, max_num_patches]`, then runs up
   to 10 rounds of proportional scale-down until the total fits.
3. `process_media` (lines 376-461) computes the target patch grid from the
   original aspect ratio (`round(dim/patch + 0.5)`, scaled by
   `factor = min(sqrt(budget/patches), 1.0)` — i.e. **never upscales**), bumps
   it up to `min_num_patches` if needed, and rounds each side to a multiple of 2
   (pixel-shuffle divisor), preferring to round **up** if it still fits, else
   down (never below 2).
4. `apply_params` (lines 357-374) resizes + normalizes to
   `(grid_h*patch, grid_w*patch)`.
5. `stack` (lines 550-569) **pre-patchifies on the host**: each image
   `[3, H, W]` is rearranged to `[(H/P)*(W/P), 3*P*P]` and all images are
   concatenated into one sequence, then unsqueezed to `[1, total_patches, 3*P*P]`.

Consequences for a port:

- In this path the model receives **flattened patch vectors**, not pixels; the
  ViT's patch embedder is a plain `Linear(3*P*P → hidden)` applied to them
  (`radio.py:496-509`, used at `radio.py:201-206`).
- Per-image token count is `(grid_h*grid_w)/4` (`_get_num_embeddings`, lines
  285-288) and is carried explicitly as `num_tokens_per_image`.
- `imgs_sizes` (per-image `(H, W)` in pixels) is carried so the tower can
  rebuild per-image boundaries for position embeddings, CLS insertion, and
  varlen attention.

### 5.3 Video

`_preprocess_video` (`processors/nano_nemotron_vl.py:883-979`).

**Frame resize.** `video_to_pixel_values` (lines 217-248) resizes every frame to
a common target derived from `vision_config.video_target_num_patches` (or
`video_target_img_size`, converted at lines 798-811). Two modes
(`get_video_target_size_and_feature_size`, lines 183-214):

- `video_maintain_aspect_ratio=True` → distribute the patch-area budget by
  aspect ratio, snap both sides to multiples of 2
  (`_compute_aspect_preserving_size`, lines 145-180). The comment notes this
  mirrors Megatron-LM's `image_processing.py` — **use the Megatron original as
  the cross-check when porting**.
- else → square grid, `side = floor(sqrt(budget)/2)*2`.

`feature_size` (tokens per frame *before* temporal grouping) is
`(target_h/patch * ds) * (target_w/patch * ds)`.

**Temporal grouping.** With `T = video_temporal_patch_size > 1`, frames are
grouped into tubelets; the number of token-bearing units is
`num_tubelets = ceil(num_frames / T)` (line 942). Tokens per tubelet equals
tokens per frame (the temporal dimension is folded into the embedder's input
dimension, not into the token count).

**Metadata carried to the device stage** (lines 919-924):
`pixel_values_flat_video` `[sum_frames, 3, H, W]`, `video_num_patches`
(frames per video), `frames_indices` (original frame indices, one per frame),
`frame_duration_ms`.

`frame_duration_ms = int(1000.0 / fps)` — an **integer** on purpose
(lines 899-907): timestamps are recomputed inside the model in bf16 context, and
float fps caused timestamp strings to differ between host and device, changing
the tokenized separator length. Reproduce this integer rounding exactly.

**Frame separator text.** `get_video_repl`
(`processors/nano_nemotron_vl.py:1095-1201`) builds the replacement token ids:

- `T == 1`, timestamps known:
  `"\n"(if i>0) + f"Frame {i+1} sampled at {ts:.2f} seconds: "`
- `T > 1`: one separator per tubelet, joining the frames in the group with
  `" and "`, first frame capitalized `Frame`, subsequent lowercase `frame`,
  then `": "`, with a leading `"\n"` for groups after the first, e.g.
  `"\nFrame 3 sampled at 1.00 seconds and frame 4 sampled at 1.50 seconds: "`
- no timestamps: `"\n"(if i>0) + f"Frame {i+1}: "`

Then, per tubelet: `separator_tokens + <img> + <image>*tokens + </img>`
(lines 1194-1199). Separators are batch-tokenized and each component is
tokenized **independently** so that token counts never change due to merging
across boundaries — this is a correctness requirement, not an optimization.

`ts = frame_index * frame_duration_ms / 1000.0` (lines 48-56); note the index
used is the **original** frame index from `frames_indices`, so timestamps
survive frame subsampling.

**EVS budget (host side).** If `--video-pruning-rate q > 0`
(`vllm/config/multimodal.py:191-195`), the *total* token count for the video is
`compute_retained_tokens_count(tokens_per_frame, num_tubelets, q)`
(`vllm/multimodal/evs.py:16-35`):

```
total  = tokens_per_frame * num_frames
kept   = max(tokens_per_frame, int(total * (1 - q)))   # first frame always survives
```

The host then assigns *all* kept tokens to the first tubelet and 0 to the rest
(`processors/...:944-957`) — only the total matters, because the device stage
rebuilds the actual distribution (§6.3).

### 5.4 Audio

Front-end: `ParakeetExtractor` (`vllm/model_executor/models/parakeet.py:138-335`),
instantiated by the processor when `sound_config` exists
(`processors/nano_nemotron_vl.py:813-816`) and invoked at line 1006. Note that
this class lives in the *model* file but runs on the **host** (default
`device="cpu"`, line 290) — it is preprocessing, not part of the model graph.

Pipeline (`__call__`, lines 286-330):

1. Force mono by channel mean (warns) — lines 297-304.
2. **Clip splitting** (`_clip_sizes`, lines 253-259; `split_audio_into_clips`,
   271-284): full 30 s clips (`clip_duration_s * sampling_rate` samples) plus a
   remainder clip padded up to at least `clip_min_duration_s = 0.1 s`. Total
   audio is zero-padded to the sum of clip sizes. Clips become the batch
   dimension of the encoder.
3. Right-pad all clips to the batch max with `padding_value` (`_pad_raw_speech`,
   lines 238-251).
4. **Pre-emphasis** with per-clip valid-length masking: `x[t] - 0.97*x[t-1]`,
   first sample untouched, positions beyond the valid length zeroed
   (lines 198-214).
5. **STFT** `n_fft=512, hop=160, win=400`, Hann window `periodic=False`,
   `pad_mode="constant"` (lines 170-187).
6. **Mel filterbank** `mel_filter_bank(..., norm="slaney", mel_scale="slaney")`
   over `n_fft//2+1` bins → `num_mel_bins`; power spectrum
   (`real² + imag²`), then `log(mel + 2^-24)` and transpose to
   `[batch, time, mel]` (lines 154-196).
7. **Per-clip normalization** over valid frames only, using
   `features_length = (samples + n_fft//2*2 - n_fft) // hop_length`, with
   sample variance (`/(n-1)`) and `(x-mean)/(std+1e-5)`, then re-masked
   (lines 216-236).

Outputs: `input_audio_features [num_clips, time, mel]`,
`feature_attention_mask [num_clips, time]`, `audio_num_clips` (clips per audio
item).

**Token count** (`audio_token_count`, lines 261-269): for each clip,
`num_frames = clip_samples // hop_length`, then HF's
`ParakeetEncoder._get_subsampling_output_length(num_frames)`; sum over clips,
floored at 1. Replacement text:
`<so_start> + <so_embedding> * n + <so_end>` (lines 1086-1093).

### 5.5 Token / placeholder accounting contract

This table is the porting contract. `is_embed` marks which positions of the
expanded span receive encoder embeddings.

| Modality | Expanded span | Token count | `is_embed` |
| --- | --- | --- | --- |
| Image | `<img>` + `<image>`×F + `</img>` | F = `num_patches × num_image_token` (fixed-tile) or `grid_h*grid_w/4` (dynamic) | only `<image>` positions (`select_text`, `processors/...:1076-1084`) |
| Audio | `<so_start>` + `<so_embedding>`×N + `<so_end>` | N = Σ subsampled clip lengths, ≥1 | only `<so_embedding>` positions (`processors/...:1086-1093`) |
| Video | per tubelet: separator text + `<img>` + `<image>`×k + `</img>` | Σk = frames×tokens_per_frame, or EVS `kept` | **all positions in the span** (`PromptUpdateDetails.from_seq`, `processors/...:1201`) |

The video case is the subtle one. `from_seq` leaves `is_embed=None`, which the
framework interprets as "assign embeddings to every position"
(`vllm/multimodal/processing/processor.py:212-226`). Therefore the model must
emit embeddings for the separator and marker tokens too — it does so by
embedding those token ids through the LM embedding table and interleaving
(§6.4). A port that instead keeps text tokens as text must make sure the same
final `inputs_embeds` results; the vLLM behaviour is equivalent as long as the
same token ids are embedded with the same embedding table.

Duplicated logic warning: video/audio/image replacement counts are computed in
**two** places — inside the HF-processor emulation
(`processors/...:944-977`, `981-1007`) and again in the framework-level prompt
update callbacks (`nano_nemotron_vl.py:432-596`). Both must agree.

### 5.6 `use_audio_in_video`

`NanoNemotronVLMultiModalProcessor.apply`
(`nano_nemotron_vl.py:658-760`) implements "take the audio track from the
video": it either consumes audio items already supplied upstream, or extracts
them from `original_video_bytes` with `load_audio_pyav`
(`_extract_audio_from_videos`, lines 598-656), then injects one
`<so_embedding>` marker immediately after each `<video>` marker **that actually
had an audio stream** (lines 714-726). Videos without audio are skipped, so
video↔audio pairing is preserved. Requires the video loader to keep raw bytes.

### 5.7 Memory profiling / dummy inputs

`NanoNemotronVLDummyInputsBuilder` (`nano_nemotron_vl.py:763-893`) synthesizes
worst-case media for startup profiling: largest image size, `num_frames` from
`--media-io-kwargs`, and dummy audio length
`min(10 min, ParakeetExtractor.audio_length(sound_config, tokens_per_audio))`
(lines 874-889; `MAX_AUDIO_LEN_S` at line 98). `ParakeetExtractor.audio_length`
(`parakeet.py:332-335`) inverts the token count:
`tokens * subsampling_factor * hop_length`. A Megatron port needs an equivalent
upper-bound estimator to size activation memory, since the encoders' activation
peak (especially long video) dominates.

---

## 6. Device stage: encoders and embedding assembly

Dispatch: `embed_multimodal` (`nano_nemotron_vl.py:1430-1462`) iterates
modalities in `kwargs` insertion order and concatenates per-item tensors.
Parsing/validation of the kwargs into typed schemas happens in
`_parse_and_validate_multimodal_inputs` (lines 1403-1428); the schemas
themselves (with documented dimensions) are at lines 101-196 and are a compact
spec of the encoder input formats.

The runner calls `embed_multimodal` **once per modality group per step**
(`gpu_model_runner.py:3126-3199`), with the exception that video items are
encoded **one at a time** when EVS or `requires_sequential_video_encoding` is
set (lines 3143-3171) — a memory-safety hack because video batching isn't
supported for the dynamic-resolution/conv3d path yet.

### 6.1 RADIO vision tower

Construction: `get_vit_model_from_radio_config` (`nano_nemotron_vl.py:1570-1600`)
builds `RadioModel` (`radio.py:695-742`) → `RadioInternVisionModel`
(`radio.py:571-692`) → `RadioVisionEncoder` (32 × `RadioVisionEncoderLayer`).
Note it is constructed **without `quant_config`** (line 1600) — the vision tower
is always unquantized (§7.5) — and cast to the LM dtype
(`nano_nemotron_vl.py:955-957`).

Forward path for a batch of tiles/frames (`radio.py:654-692`):

1. **Patchify + embed.**
   - Fixed-size input: `Im2Patches` rearranges `[N,3,H,W]` →
     `[N, (H/P)(W/P), 3P²]` (`radio.py:472-493`), then
     `embedder: Linear(3P² → hidden)` (`ViTPatchLinear`, lines 496-509).
   - Dynamic-resolution input: patches arrive pre-flattened from the host, so
     only `embedder` runs (`radio.py:201-206`).
   - Video with `T > 1`: `forward_video` (`radio.py:216-263`) pads the frame
     count up to a multiple of T **by repeating the last frame**, groups
     `(tubelets frames) spatial feat → tubelets spatial (frames feat)`, and
     applies a **separate** `video_embedder: Linear(3·T·P² → hidden)`
     (constructed at `radio.py:166-178`; `separate_video_embedder=False` is
     explicitly unsupported). The einops grouping order is stated to follow
     Megatron training — match it exactly.
2. **Position embeddings.** Learned grid `pos_embed [1, num_rows*num_cols, hidden]`
   sized by `cpe_max_size`, then window-selected and/or bilinearly interpolated
   to the actual patch grid (`_get_pos_embeddings`, `radio.py:434-469`). In CPE
   mode it first interpolates to a square `max(input_dims)` grid, then windows.
   Dynamic resolution applies this per image inside the packed sequence
   (`apply_pos_enc_dynamic`, lines 265-293).
3. **CLS + register tokens** prepended per image (`ClsToken`, `radio.py:60-106`;
   packed variant `cls_token_dynamic`, lines 295-312). Count:
   `num_cls_tokens` (from unique teachers) + `num_registers`
   (`register_multiple - num_tokens % register_multiple`).
4. **32 transformer blocks** (`radio.py:537-553`, base at
   `intern_vit.py:292-349`): pre-norm residual with per-branch layer scale,
   `x = x + attn(norm1(x))*ls1`, `x = x + mlp(norm2(x))*ls2`. MLP is
   `Linear → act (gelu) → Linear` (`intern_vit.py:250-284`). Optional QK-norm
   (`RadioConfig.qk_normalization`, default False).
   **`ls1`/`ls2` are initialized to `initializer_factor * ones` (= 1.0) and the
   checkpoint's layer-scale tensors are deliberately skipped when loading**
   (`radio.py:780-782`), i.e. layer scale is identity in vLLM. Verify against
   the checkpoint before replicating; if the checkpoint has meaningful
   layerscale, vLLM would be ignoring it.
5. **Varlen attention.** When several differently-sized images (or several
   tubelets) share one packed sequence, `MaskMetadata(cu_seqlens, max_seqlen)`
   is built (`radio.py:627-652`) and threaded into the attention op so items
   don't attend across boundaries (`radio.py:518-534`). `max_seqlen` is kept on
   **CPU** to avoid a device sync. For the video path, all tubelets have equal
   length and the batch is flattened to `[1, total, hidden]` then unflattened
   after the encoder (`radio.py:666-690`).
6. **Strip CLS/registers, split per image.** `_extract_final`
   (`radio.py:793-822`) drops the first `num_skip` tokens of each item and
   returns `(summary, features)`. The model uses only `features`
   (`nano_nemotron_vl.py:1056`, `1086-1088`).

### 6.2 Pixel shuffle and projector (`mlp1`)

`extract_feature` (`nano_nemotron_vl.py:1062-1103`):

- Micro-batches at most `128 - (128 % T)` frames per ViT call to cap activation
  memory (lines 1067-1077). Chunk boundaries must not split a tubelet.
- Casts ViT output to **bf16** unconditionally (line 1089).
- `pixel_shuffle` (lines 1014-1031): reshape to `[N, H/P, W/P, C]`, fold each
  2×2 spatial neighbourhood into the channel dim → `[N, h/2, w/2, 4C]`. The
  `ps_version == "v1"` branch exists only to warn about a transposed legacy
  layout; the supported path permutes `(0,1,3,2,4,5)`.
- Dynamic resolution variant `pixel_shuffle_dynamic_res` (lines 1033-1050)
  splits the packed sequence by per-image patch counts first.
- `mlp1` (constructed at lines 964-978): `RMSNorm(4·vit_hidden, eps=1e-5) →
  Linear(4·vit_hidden → projector_hidden, no bias) → ReLU² → Linear(projector_hidden
  → llm_hidden, no bias)`. `ReLUSquaredActivation` = `relu(x)²`.

Per-image splitting into a tuple of tensors: `_process_image_input`
(lines 1146-1163, split by `num_patches * feature_size`) and
`_process_image_input_dynamic` (lines 1132-1144, split by
`num_tokens_per_image`).

### 6.3 EVS (Efficient Video Sampling) pruning

`compute_retention_mask` (`vllm/multimodal/evs.py:38-92`), called from
`_process_video_input` (`nano_nemotron_vl.py:1198-1215`):

1. Reshape embeddings to `[T_units, rows, cols, hidden]` where
   `rows = frame_h*ds/patch`, `cols = frame_w*ds/patch` (post-shuffle grid,
   `nano_nemotron_vl.py:1179-1184`), `spatial_merge_size=1`.
2. Dissimilarity per token = `1 - cosine_similarity(frame_t, frame_{t-1})`.
3. Frame 0 gets sentinel dissimilarity `255` so it is always fully retained.
4. `argsort(descending=True, stable=True)` over the flattened dissimilarity;
   keep the top `compute_retained_tokens_count(...)` indices; build a boolean
   mask.
5. The mask is applied, and the **actual retained tokens per frame** is
   `mask.reshape(T,rows,cols).sum((1,2))` (`nano_nemotron_vl.py:1210-1214`).

The stable sort and the `255` sentinel are load-bearing for reproducibility.
Note the pruning is applied **after** the projector, on LM-dimension embeddings.

### 6.4 Video embedding assembly (hybrid text + vision)

`_create_final_video_embeddings` (`nano_nemotron_vl.py:1287-1342`):

1. Re-run `NanoNemotronVLProcessor.get_video_repl` **on device-side counts**,
   with the real per-frame retained counts, pre-tokenized marker ids, and the
   same timestamps. This yields the exact token-id sequence for the video span.
2. `is_video_embed = isin(repl_token_ids, img_context_token_ids)`.
3. `text_embeddings = LM.embed_input_ids(repl_token_ids)` — the separator and
   `<img>`/`</img>` tokens are embedded with the LM embedding table.
4. `_merge_multimodal_embeddings` scatters the vision rows into the
   `is_video_embed` positions.

Result: one dense tensor per video covering the entire span, matching the token
count allocated on the host (§5.5). Because host and device compute the span
independently, the tokenizer, timestamp formatting, and integer
`frame_duration_ms` must be identical on both sides.

For the temporal path, per-video extraction happens in
`_extract_video_embeddings_temporal` (lines 1232-1254) — one `extract_feature`
call per video with `num_frames=nf`, which selects the conv3d/tubelet route and
the no-mask flash-attention path.

### 6.5 Audio encoder

`_process_audio_input` (`nano_nemotron_vl.py:1256-1285`):

1. Move features to the tower's device, cast to `llm_dtype`.
2. `ProjectedParakeet.forward` (`parakeet.py:66-75`): HF `ParakeetEncoder`
   (Conformer: subsampling conv front-end + conformer blocks) → cast to bf16 →
   `ParakeetProjection` = `RMSNorm → Linear → ReLU² → Linear(→ llm_hidden)`
   (`parakeet.py:27-45`).
3. Trim padding: valid output length per clip via
   `encoder._get_subsampling_output_length(mask.sum(dim=1))`, then slice
   `sound_embeds[clip, :valid_len]`.
4. Concatenate the clips belonging to the same audio item
   (`audio_num_clips`) into one tensor per item.

Step 3/4 is exactly the inverse of the host-side clip splitting, and its result
must equal `audio_token_count` from §5.4.

### 6.6 Merging into `inputs_embeds` and the LM

The generic framework path applies:
`SupportsMultiModal.get_input_embeddings` (`interfaces.py:386-415`) embeds text
ids then calls `_merge_multimodal_embeddings` with the `is_multimodal` mask
built from placeholder ranges (`gpu_model_runner.py:3260-3330`). The model's own
`forward` (`nano_nemotron_vl.py:1464-1483`) is then a pure pass-through to the
Nemotron-H LM with `inputs_embeds`; there is no cross-attention and no
vision-specific rope.

LM side (for context): `NemotronHForCausalLM`
(`vllm/model_executor/models/nemotron_h.py`) is a hybrid Mamba2 + attention
stack whose layer order comes from `hybrid_override_pattern` in `text_config`,
with MoE layers (`NemotronHMoE`, `nemotron_h.py:126-236`). Mamba state shapes
and dtypes are delegated from the multimodal wrapper
(`nano_nemotron_vl.py:1612-1626`).

---

## 7. Parallelism, replication, and "frozen"-ness

### 7.1 Summary table

| Component | Module / file | Under TP=N | Mechanism |
| --- | --- | --- | --- |
| RADIO attention (QKV, out-proj) | `intern_vit.py:186-216` via `radio.py:518` | **Sharded** (head-parallel) | `QKVParallelLinear` + `RowParallelLinear` (all-reduce at out-proj) |
| RADIO MLP (fc1/fc2) | `intern_vit.py:250-284` | **Sharded** | `ColumnParallelLinear` + `RowParallelLinear` |
| RADIO patch embedder / pos-embed / CLS / registers / layer norms | `radio.py:109-509` | Replicated | plain `nn.Linear` / `nn.Parameter` |
| Vision projector `mlp1` | `nano_nemotron_vl.py:964-978` | Replicated | plain `nn.Linear` + `RMSNorm` |
| Parakeet audio encoder | `parakeet.py:61-62` (HF module) | Replicated | plain HF `nn.Linear`/conv |
| Parakeet projection | `parakeet.py:27-45` | Replicated | plain `nn.Linear` |
| Nemotron-H LM (attention, Mamba2, dense MLP) | `nemotron_h.py` | Sharded (TP) | vLLM parallel layers |
| Nemotron-H MoE experts | `nemotron_h.py:212-236` | Sharded (TP and/or **EP**) | `FusedMoE`, `enable_eplb` |

Activation-wise, **all TP ranks execute the whole encoder graph on the same
inputs**; TP splits weights and reduces partial results, it does not split the
media batch.

### 7.2 How the ViT TP decision is made

`InternParallelAttention.__init__` (`intern_vit.py:169-183`):

```python
use_data_parallel = is_vit_use_data_parallel()
tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()
use_data_parallel = use_data_parallel or (num_heads + num_dummy_heads) % tp_size != 0
self.tp_size = 1 if use_data_parallel else tp_size
```

and the linears receive `disable_tp=use_data_parallel`
(`intern_vit.py:193, 215, 268, 276`). `is_vit_use_data_parallel`
(`vision.py:142-159`) returns True when `mm_encoder_tp_mode == "data"`.

Practical consequence for Nemotron Omni: RADIO `vit_huge` has **16 heads**, so
TP ∈ {1,2,4,8,16} shards cleanly; any TP size that does not divide 16 silently
falls back to a fully replicated vision encoder (with a warning).

If QK-norm were enabled, `_apply_qk_norm` (`intern_vit.py:225-235`) adds an
all-gather + re-split around the norms; with the default
`qk_normalization=False` this does not occur.

### 7.3 Communication points per ViT block (TP > 1)

1. `qkv` — column-parallel, no comm.
2. attention — local heads only.
3. `proj` — row-parallel → **all-reduce**.
4. `fc1` — column-parallel, no comm.
5. `fc2` — row-parallel → **all-reduce**.

So 2 all-reduces per block × 32 blocks. Everything after the tower
(pixel-shuffle, `mlp1`, EVS, embedding assembly) is replicated dense compute on
identical inputs, hence identical outputs on all ranks — no comm needed.

### 7.4 Encoder data parallelism is *not* available

Batch-level encoder DP (each rank encodes a different subset of items with
replicated weights) requires the model to opt in with
`supports_encoder_tp_data = True` (`interfaces.py:119-123`,
`docs/configuration/optimization.md:255-275`). `NemotronH_Nano_VL_V2` does not,
so `--mm-encoder-tp-mode data` is downgraded with a warning
(`vllm/config/model.py:705-714`). A Megatron-Core port is free to shard the
media batch across ranks instead; nothing in the math forbids it, and the
replicated audio tower in particular would benefit.

### 7.5 Quantization

`get_vit_model_from_radio_config` (`nano_nemotron_vl.py:1570-1600`) never passes
`quant_config` to `RadioModel`, and `ProjectedParakeet` builds raw HF modules.
Therefore, for FP8 / NVFP4 Nemotron Omni checkpoints, **only the language model
is quantized; both encoders and both projectors run in the LM's bf16 dtype**.
Plan the port's memory budget accordingly, and don't attempt to load
quantization scales for tower weights.

### 7.6 Expert parallelism

EP (and EPLB) applies exclusively to `FusedMoE` inside the Nemotron-H LM
(`nemotron_h.py:135-236`: `get_ep_group()`, `enable_eplb`,
`num_redundant_experts`, `is_sequence_parallel`). Neither encoder contains MoE
layers, so `--enable-expert-parallel` has no effect on media encoding.

### 7.7 "Frozen"? — inference-only semantics

- vLLM is inference-only: no autograd, no optimizer, `torch.inference_mode` at
  the worker level. Encoder weights are loaded once and never updated. In the
  training sense, everything is frozen.
- Training-only code paths are inert: e.g. `pos_dropout` in the patch generator
  is gated on `self.training` (`radio.py:398-405`) and never fires.
- Some tensors are **buffers, not parameters**: `summary_idxs`
  (`radio.py:720-727`, skipped at load), the mel filterbank and Hann window
  (`@cache`d in `ParakeetExtractor`, `parakeet.py:149-168`), and the Parakeet
  feature-extractor buffers, which are explicitly **not** loaded into the model
  because the host-side extractor owns that computation
  (`parakeet.py:88-90`).
- LoRA: `get_mm_mapping` (`nano_nemotron_vl.py:1485-1493`) declares
  `language_model` / connector (`mlp1`, `sound_encoder.projection`) /
  tower (`vision_model`, `sound_encoder.encoder`) groups, which is how adapters
  are scoped. Towers are not LoRA targets by default.

### 7.8 Conditional construction (skipping towers)

`_mark_language_model` / `_mark_tower_model`
(`interfaces.py:221-298`) are context managers used in `__init__`
(`nano_nemotron_vl.py:945-990`) to tag submodules. Effects:

- All modalities limited to 0 (`--limit-mm-per-prompt`) → towers are not
  materialized, and `load_weights` skips their tensors
  (`nano_nemotron_vl.py:1501-1568`; unit-tested in
  `tests/models/multimodal/test_nano_nemotron_vl.py`).
- `--mm-encoder-only` → the LM is skipped (disaggregated encoder serving).

A port should preserve the ability to build tower-only or LM-only instances if
it wants encoder disaggregation.

---

## 8. Weight loading and checkpoint mapping

`NemotronH_Nano_VL_V2.load_weights` (`nano_nemotron_vl.py:1501-1568`) plus
`WeightsMapper` (lines 907-911). Checkpoint → runtime mapping:

| Checkpoint prefix | Destination | Notes |
| --- | --- | --- |
| `language_model.backbone.*` | `language_model.model.*` | via `hf_to_vllm_mapper`; then the leading component is stripped before handing to `NemotronHForCausalLM.load_weights` (line 1535) |
| `mlp1.*` | `mlp1.*` | loaded directly with `default_weight_loader` (lines 1561-1565) |
| `vision_model.radio_model.*` | RADIO | prefix `vision_model.` stripped (line 1545), then remapped in `RadioModel.load_weights` |
| `…radio_model.model.blocks.{i}.*` | `model.encoder.layers.{i}.*` | `radio.py:773-783` |
| `…radio_model.model.patch_generator.*` | `model.patch_generator.*` | `radio.py:769-770` |
| `…blocks.{i}.ls1/ls2` | **skipped** | `radio.py:780-782` — layer scale stays at 1.0 |
| `…radio_model.input_conditioner.*` | **skipped** | normalization happens in the processor (`radio.py:763-766`) |
| `…radio_model.summary_idxs` | **skipped** | buffer rebuilt from config (`radio.py:761-762`) |
| `sound_encoder.*` | `sound_encoder.encoder.*` | `parakeet.py:91-92` |
| `sound_projection.*` | `sound_encoder.projection.*` | `parakeet.py:93-94` |
| `sound_encoder.encoder.feature_extractor.*` | **skipped** | host-side extractor owns it (`parakeet.py:88-90`) |

Additional details worth copying:

- Conv bias tolerance: with `convolution_bias=False`, transformers v5 does not
  register conv bias params, but checkpoints may still contain them; those
  specific names are skipped (`parakeet.py:112-131`).
- Load ordering: LM weights are streamed lazily through a generator while the
  smaller tower/adapter tensors are `detach().clone()`d into lists and loaded
  afterwards, to stay safe with reusable-buffer weight streamers
  (`nano_nemotron_vl.py:1521-1558`). Relevant only if your loader reuses
  buffers.
- Packed QKV: RADIO declares `packed_modules_mapping = {"qkv": ["qkv"]}`
  (`radio.py:572-574, 696-698`) because the checkpoint already stores a fused
  QKV tensor; the TP loader shards it by head.

---

## 9. Numerics and dtype map

| Stage | dtype |
| --- | --- |
| Host image/video resize + normalize | fp32 compute, cast to `config.dtype` at the end (`processors/...:59-82`) |
| Host mel front-end | fp32 throughout (`parakeet.py:170-236`) |
| RADIO tower weights/activations | LM dtype (bf16) — `.to(llm_dtype)` at `nano_nemotron_vl.py:955` |
| ViT output before pixel shuffle | hard cast to **bf16** (`nano_nemotron_vl.py:1057, 1089`) |
| `mlp1` | bf16 (`nano_nemotron_vl.py:978`) |
| Parakeet encoder | `llm_dtype` in, output hard cast to bf16 before projection (`parakeet.py:72-74`) |
| Position-embedding interpolation | fp32 internally, cast back (`radio.py:451-465`) |
| Mamba SSM state cache | fp32 recommended (`config.py:605-625`) |

`torch.compile` is used in three places that a port should be aware of (they
change nothing semantically but explain graph-break-avoiding code shapes):
host-side resize/normalize and mel ops (`processors/...:59`, `parakeet.py:189,
198, 216`), and the ViT encoder layer via
`@support_torch_compile(..., enable_if=should_torch_compile_mm_encoder,
is_encoder=True)` (`intern_vit.py:287-291`), gated on
`compilation_config.compile_mm_encoder`.

---

## 10. Porting checklist and known pitfalls

Ordered roughly by how likely each one is to bite.

1. **Token-count parity first.** Before touching kernels, port §5.5 and assert
   that host-side placeholder counts equal device-side embedding rows for:
   single image, multi-image (dynamic budget), 1-frame video, T-not-dividing
   frame count, EVS on/off, single/multi-clip audio.
2. **Integer `frame_duration_ms`.** `int(1000.0 / fps)`. Using float fps
   produces different timestamp strings → different separator token counts →
   shape mismatch (`processors/...:899-907`).
3. **Tokenize span components independently.** Separator, `<img>`,
   `<image>`×k, `</img>` are tokenized separately and concatenated
   (`processors/...:1191-1199`). Tokenizing the joined string can merge tokens
   across boundaries.
4. **Video span is fully "embedded".** All positions, including separators, get
   embeddings from the model (§5.5/§6.4). Either replicate the hybrid assembly
   or make sure your equivalent produces identical `inputs_embeds`.
5. **EVS details:** first frame always fully retained via the `255` sentinel;
   `kept = max(tokens_per_frame, int(total*(1-q)))`; `stable=True` argsort;
   pruning applied *after* the projector on LM-dim embeddings; host allocates
   the total, device decides the distribution.
6. **Tubelet padding by last-frame repetition**, not zeros, and the exact einops
   grouping order `(tubelets frames) spatial feat -> tubelets spatial (frames feat)`
   (`radio.py:234-252`).
7. **Dynamic-resolution images are pre-patchified on the host** and packed into
   one sequence with per-image `cu_seqlens`; the ViT embedder is a Linear over
   `3·P²` vectors. Don't assume a conv patch embed
   (`processors/...:550-569`, `radio.py:201-206`).
8. **Never upscale** in the dynamic tiler (`factor = min(..., 1.0)`), round patch
   grids to multiples of 2, respect `min_num_patches`/`max_num_patches`, and the
   10-iteration budget loop (`processors/...:376-548`).
9. **Audio clipping:** 30 s clips + ≥0.1 s tail, zero-padded to the clip sum;
   per-clip normalization over valid frames with sample variance; token count
   floored at 1 (`parakeet.py:253-330`).
10. **CLS/register stripping.** `num_skip = num_cls_tokens + num_registers`
    tokens are removed *per item* after the tower (`radio.py:793-822`), and
    register count derives from `register_multiple`.
11. **Layer scale is identity in vLLM** (`radio.py:780-782`). Check whether the
    checkpoint actually contains meaningful `ls1/ls2`; if it does, vLLM and a
    faithful Megatron port would disagree, and the vLLM side is the suspect.
12. **`input_conditioner` lives in the processor.** Normalization stats come
    from `config.norm_mean/norm_std`, not from checkpoint weights
    (`radio.py:763-766`).
13. **Micro-batching of 128 frames** in `extract_feature` is memory hygiene, but
    the batch size must stay a multiple of T (`nano_nemotron_vl.py:1067-1077`).
14. **Encoders are unquantized** even for FP8/NVFP4 checkpoints (§7.5).
15. **Mamba SSM state in fp32** (§3.4) — and be aware vLLM's own config hook is
    keyed to the non-Omni architecture name.
16. Known limitations recorded in this tree, useful as expectation-setting:
    `requires_sequential_video_encoding` exists because batched dynamic-res
    video is unsupported (`nano_nemotron_vl.py:904-905`,
    `gpu_model_runner.py:3131-3150`), and the test registry disables
    `video_maintain_aspect_ratio` due to a mixed-resolution processor bug
    (`tests/models/registry.py:1194-1199`).

---

## 11. Suggested Megatron-Core inference decomposition

A mapping that keeps the vLLM boundaries (they are good boundaries — the host
stage is CPU/tokenizer-bound and cacheable, the device stage is pure tensor
work):

| Responsibility | vLLM home | Suggested Megatron-Core home |
| --- | --- | --- |
| Media decode + resize/normalize + clip split + mel | `processors/nano_nemotron_vl.py`, `parakeet.py:138-335` | request-preprocessing / dataloader-side module, reusing Megatron-LM's `image_processing.py` where it already exists |
| Placeholder expansion + token accounting | `processors/...` + `nano_nemotron_vl.py:432-596` | prompt-builder utility; must be shared with whatever computes the final embedding span |
| Vision tower | `radio.py` + `intern_vit.py` | Megatron ViT/`TransformerBlock` with column/row-parallel linears; varlen attention via `cu_seqlens` |
| Audio tower | `parakeet.py:48-75` | Conformer encoder; start replicated, consider batch sharding later |
| Projectors | `nano_nemotron_vl.py:964-978`, `parakeet.py:27-45` | replicated MLPs |
| EVS pruning | `vllm/multimodal/evs.py` | direct port; pure tensor code, no comm |
| Embedding scatter | `interfaces.py:386-415`, `models/utils.py::_merge_multimodal_embeddings` | inference-engine input-embedding assembly |
| LM | `nemotron_h.py` | existing Megatron Nemotron-H hybrid + MoE |

Validation strategy: run vLLM as the oracle with a fixed seed and dump
intermediate tensors at four cut points — (a) processor outputs
(`pixel_values_flat*`, `input_audio_features`, expanded token ids), (b) raw tower
outputs before pixel-shuffle/projection, (c) per-item embeddings returned by
`embed_multimodal`, (d) final `inputs_embeds`. Compare (a) exactly (integers and
fp32 math), (b)-(d) with a bf16-appropriate tolerance. Getting (a) and the token
counts identical is what prevents the long tail of shape-mismatch bugs.

---

## 12. Reference material

- `docs/contributing/model/multimodal.md` — framework concepts: processing info,
  dummy inputs, prompt updates, field configs. Read this to understand *why* the
  host stage is split the way it is.
- `docs/contributing/model/basic.md:120-148` — hybrid Mamba/attention model
  requirements (`IsHybrid`, state shape/dtype hooks, `MODELS_CONFIG_MAP`).
- `docs/configuration/optimization.md:230-280` — encoder TP `weights` vs `data`
  modes and which models support batch-level DP.
- `tests/models/registry.py:1182-1206` — the canonical test configuration for
  Omni, including the `hf_overrides` needed to shrink the model for CI.
- `tests/models/multimodal/test_nano_nemotron_vl.py` — behaviour of
  `load_weights` in text-only / image-only modes.
- Upstream: NVIDIA model card for `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-*`
  and the vLLM blog post on Nemotron 3 Nano Omni for the serving-flag surface
  (`--reasoning-parser nemotron_v3`, `--tool-call-parser qwen3_coder`,
  `--video-pruning-rate`, `--media-io-kwargs`).
