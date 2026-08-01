# Nemotron 3 Nano Omni — Checkpoint Configuration Reference

> Companion to `NEMOTRON_OMNI_MULTIMODAL_DESIGN.md`. That document describes the
> *pipeline*; this one pins down the *actual numbers* from the released
> checkpoint and states, field by field, which ones vLLM reads and where.

## 0. Is there a config file in this repo?

**No.** vLLM does not ship model configs — it reads `config.json` from the
checkpoint at load time (with `--trust-remote-code`, because Omni's config class
lives in the checkpoint repo). The only Omni-related config material inside this
tree is the CI shrink-down override in `tests/models/registry.py:1182-1206`,
which is *not* the real config.

The config transcribed below was fetched from the canonical checkpoint:

```bash
curl -sL https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/raw/main/config.json
curl -sL https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/raw/main/configuration.py
```

Everything in this file is verbatim from the BF16 checkpoint unless marked
otherwise. FP8 / NVFP4 variants differ only by an added `quantization_config`
(§8).

### Files in the checkpoint repo

Config/code (all needed with `--trust-remote-code`):

| File | Role |
| --- | --- |
| `config.json` | the composite config (§2) |
| `configuration.py` | `NemotronH_Nano_Omni_Reasoning_V3_Config` (§1) |
| `configuration_nemotron_h.py`, `configuration_radio.py` | sub-configs |
| `modeling.py`, `modeling_nemotron_h.py`, `audio_model.py` | HF reference implementation |
| `processing.py`, `processing_utils.py`, `image_processing.py`, `video_processing.py`, `video_io.py`, `evs.py` | HF reference preprocessing — **the parity oracle for a port** |
| `preprocessor_config.json` | HF processor defaults (§6) |
| `generation_config.json` | sampling defaults (§7) |
| `chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` | prompt/tokenizer |
| 17 × `model-*.safetensors` + `model.safetensors.index.json` | weights |

`image_processing.py` / `video_processing.py` / `evs.py` in the checkpoint are
the same logic vLLM reimplements in
`vllm/transformers_utils/processors/nano_nemotron_vl.py` and
`vllm/multimodal/evs.py`. When porting, diff against both.

---

## 1. Config class shape (and the `llm_config` → `text_config` alias)

`configuration.py` defines `NemotronH_Nano_Omni_Reasoning_V3_Config`
(`is_composition = True`) with three sub-configs: `vision_config` (`RADIOConfig`),
`llm_config` (`NemotronHConfig`), `sound_config` (`SoundConfig`, optional).

The important detail for vLLM: the on-disk key is **`llm_config`**, but vLLM's
`NemotronH_Nano_VL_V2` reads **`config.text_config`**
(`vllm/model_executor/models/nano_nemotron_vl.py:948, 961, 1141, 1614`). The
checkpoint bridges this with an explicit property alias:

```python
# configuration.py:118-123 (verbatim, comment included)
# vLLM's `NemotronH_Nano_VL_V2` implementation reads the language-model sub-config as
# `config.text_config`. Our HF config stores it as `config.llm_config`; expose an alias so the
# same config object loads under both loaders without having to duplicate the dict on disk.
@property
def text_config(self):
    return self.llm_config
```

**Port implication:** if your backend parses `config.json` directly rather than
through the remote config class, read `llm_config`. `text_config` does not exist
on disk. Likewise `sound_config` is `None` when absent, and that absence is the
only switch that disables the audio tower.

Also set by the config class at construction time
(`configuration.py:114-116`): `vision_config.use_flash_attn` is derived from
`attn_implementation` (default `"flash_attention_2"`), and
`llm_config._attn_implementation` is propagated. vLLM ignores both and uses its
own attention backends.

---

## 2. Top-level fields

| Field | Value | Read by vLLM? | Where |
| --- | --- | --- | --- |
| `architectures` | `["NemotronH_Nano_Omni_Reasoning_V3"]` | yes | `models/registry.py:514` |
| `model_type` | `"NemotronH_Nano_Omni_Reasoning_V3"` | yes (HF dispatch) | — |
| `auto_map` | `configuration.…_Config`, `modeling.…` | yes (`trust_remote_code`) | — |
| `torch_dtype` | `bfloat16` | yes | also becomes the processor output dtype (`processors/nano_nemotron_vl.py:619`) |
| `max_sequence_length` | `131072` | **no** | vLLM uses `--max-model-len` / `llm_config.max_position_embeddings` |
| `force_image_size` | `512` | yes | `nano_nemotron_vl.py:928`, `processors/…:596` |
| `patch_size` | `16` | yes | `nano_nemotron_vl.py:929`, `processors/…:597` |
| `downsample_ratio` | `0.5` | yes | pixel-shuffle factor; the dynamic tiler asserts exactly `0.5` (`processors/…:277-283`) |
| `use_thumbnail` | `true` | yes, but **inert for Omni** | only the fixed-tile path uses it; Omni takes the dynamic-resolution path (§3.2) |
| `ps_version` | `"v2"` | yes | `nano_nemotron_vl.py:936`; `"v1"` would trigger the legacy transposed-shuffle warning |
| `template` | `"n5h_5p5_nanov2"` | stored, unused | `nano_nemotron_vl.py:931` |
| `image_tag_type` | `"internvl"` | stored, unused | `nano_nemotron_vl.py:937` |
| `vit_hidden_size` | `1280` | yes | sizes `mlp1` input (`nano_nemotron_vl.py:960`) |
| `projector_hidden_size` | `20480` | yes | `mlp1` hidden width (`nano_nemotron_vl.py:961`) |
| `norm_mean` | `[0.48145466, 0.4578275, 0.40821073]` | yes | OpenAI-CLIP stats; applied **in the processor** |
| `norm_std` | `[0.26862954, 0.26130258, 0.27577711]` | yes | ″ |
| `video_pruning_rate` | `0.7` | **no** | see §9.1 — vLLM only honours the CLI flag |
| `video_temporal_patch_size` | `2` | no (top-level copy) | vLLM reads the `vision_config` copy (also `2`) |
| `eos_token_id` | `11` | yes | `generation_config.json` widens this to `[2, 11]` |
| `img_context_token` / `_id` | `"<image>"` / `18` | token **string** yes, id **no** | vLLM re-tokenizes the strings (`nano_nemotron_vl.py:996-1004`) |
| `video_context_token` / `_id` | `"<video>"` / `131081` | string yes, id no | `<video>` is only a user-facing marker; per-token placeholder is `<image>` |
| `img_start_token` / `img_end_token` | `"<img>"` / `"</img>"` | yes (hard-coded constants in vLLM) | `processors/…:36-38` |
| `sound_context_token` / `_id` | `"<so_embedding>"` / `27` | string yes, id no | `processors/…:41`; `<so_start>`/`<so_end>` are vLLM constants |

Note the asymmetry: vLLM hard-codes the special-token *strings* and resolves ids
through the tokenizer, so a checkpoint that renamed these tokens would break
vLLM even though the ids are in the config. Keep the strings identical.

---

## 3. `vision_config` — C-RADIOv4-H

Metadata: `version: "c-radio_v4-h"`, `architectures: ["RADIOModel"]`,
`auto_map` pointing at `nvidia/C-RADIOv4-H--hf_model.RADIOConfig`.

### 3.1 Fields vLLM reads

| Field | Value | Use |
| --- | --- | --- |
| `args["model"]` | `"vit_huge_patch16_224"` | selects `(hidden=1280, layers=32, heads=16, ffn=5120)` from the hard-coded table `vllm/transformers_utils/configs/radio.py:12-17` |
| `patch_size` | `16` | ViT patch size (`nano_nemotron_vl.py:1578`) |
| `preferred_resolution` | `[768, 768]` | → `RadioConfig.image_size = 768` (`nano_nemotron_vl.py:1576-1577`) |
| `args["cpe_max_size"]` | `2048` | learned pos-embed grid = `2048/16 = 128 × 128 = 16384` entries |
| `args["register_multiple"]` | `10` | register-token count |
| `args["cls_token_per_teacher"]` | `true` | one CLS token per unique teacher |
| `args["teachers"]` | 4 entries: `clip`, `siglip`, `dino_v2`, `sam` | → `num_cls_tokens = 4`; `use_summary` true for the first three, false for `sam` → `summary_idxs = [0,1,2]` (summaries are computed then **discarded**, `nano_nemotron_vl.py:1056`) |
| `args["min_num_patches"]` | `1024` | **presence enables dynamic-resolution images** (`processors/…:621-623`) |
| `args["max_num_patches"]` | `13312` | per-image patch ceiling |
| `video_target_num_patches` | `1024` | per-frame patch budget (`processors/…:798`) |
| `video_maintain_aspect_ratio` | `true` | aspect-preserving frame resize (`processors/…:791-793`) |
| `video_temporal_patch_size` | `2` | **T** — tubelet depth (`nano_nemotron_vl.py:1582-1584`) |
| `separate_video_embedder` | `true` | dedicated `Linear(3·T·P² → 1280)` video embedder; `false` is unsupported (`radio.py:166-178`) |

Derived token geometry:

- CLS + registers stripped per item: `num_registers = 10 - (4 % 10) = 6`, so
  **`num_skip = 4 + 6 = 10`** tokens dropped after the tower
  (`radio.py:73-91, 319-332, 797-801`).
- Image tokens per item = `patches / 4` (pixel-shuffle), so the budget
  `[1024, 13312]` patches ⇒ **`[256, 3328]` tokens per image**.
- 13312 patches ≈ a 115×115 grid ≈ 1840×1840 px, comfortably inside the
  128×128 pos-embed grid — the two limits are consistent by design.
- Video: 1024 patches/frame ⇒ ~32×32 grid ⇒ **256 tokens per tubelet**
  (= 2 frames), i.e. 128 tokens/frame effective.

### 3.2 Not read by vLLM

`max_resolution: 2048`, `vitdet_window_size: null`, `video_prompt_version: 2`,
`feature_normalizer_config`, `inter_feature_normalizer_config`,
`adaptor_configs`, `adaptor_names`, and the duplicated top-level
`min_num_patches`/`max_num_patches` (vLLM reads the copies inside `args`).

`args` has **156 keys, of which vLLM uses 7** (listed above). The other 149 are
timm training arguments serialized into the checkpoint — optimizer/schedule
(`lr_base`, `sched`, `decay_epochs`), augmentation (`mixup`, `cutmix`, `hflip`,
`color_jitter`), distillation bookkeeping (`fd_loss_fn`, `crd_loss`,
`stream_teachers`), logging (`wandb_*`, `log_interval`), distributed
(`world_size: 256`, `rank`, `fsdp`), and unused arch knobs (`mlp_hidden_size`,
`mlp_version`, `vitdet_version`, `spectral_reparam`). Ignore all of them; the
full list is in Appendix A.

Values that come from `RadioConfig` **defaults**, not the checkpoint
(`configs/radio.py:62-82`) — worth pinning explicitly in a port:
`qkv_bias=True`, `qk_normalization=False`, `norm_type="layer_norm"`,
`layer_norm_eps=1e-6`, `hidden_act="gelu"`, `initializer_factor=1.0`
(→ layer scale = 1.0, and checkpoint `ls1`/`ls2` are skipped at load,
`radio.py:780-782`).

---

## 4. `sound_config` — Parakeet (this block is what makes it "Omni")

```json
{
  "model_type": "parakeet",
  "hidden_size": 1024,
  "num_attention_heads": 8,
  "num_hidden_layers": 24,
  "intermediate_size": 4096,
  "conv_kernel_size": 9,
  "convolution_bias": false,
  "subsampling_conv_channels": 256,
  "subsampling_conv_kernel_size": 3,
  "subsampling_conv_stride": 2,
  "subsampling_factor": 8,
  "num_mel_bins": 128,
  "projection_hidden_size": 4096,
  "projection_bias": false,
  "sampling_rate": 16000
}
```

| Group | Values |
| --- | --- |
| Conformer encoder | 24 layers, `d=1024`, 8 heads (head_dim 128), FFN 4096, depthwise conv kernel 9, **no conv bias** |
| Subsampling front-end | 3 stacked convs, 256 channels, kernel 3, stride 2 → `subsampling_factor = 8` |
| Mel front-end | 128 mel bins @ 16 kHz; `hop_length=160`, `win_length=400`, `n_fft=512`, `preemphasis=0.97` come from `ExtractorConfig` defaults (`configs/parakeet.py:40-72`) since the checkpoint doesn't override them |
| Projection | `RMSNorm(1024, eps=1e-5) → Linear(1024→4096) → ReLU² → Linear(4096→2688)`, no bias (`parakeet.py:27-45`) |

Token rate: `hop_length=160` @ 16 kHz = **10 ms per mel frame**;
`subsampling_factor=8` ⇒ **80 ms per audio token ⇒ 12.5 tokens/second**. A full
30 s clip (`clip_duration_s=30`, an `ExtractorConfig` default) yields ~375
tokens. Exact counts must come from `ParakeetExtractor.audio_token_count`
(`parakeet.py:261-269`), which calls HF's `_get_subsampling_output_length`.

Two traps:

- `convolution_bias: false` combined with transformers v5 means the conv bias
  params are never registered, yet the checkpoint may still contain those
  tensors; vLLM tolerates this by name-skipping (`parakeet.py:112-131`).
- The remote `SoundConfig` declares `feat_in: int = 80` as a default
  (`configuration.py:34`) and the checkpoint does **not** set it, so `feat_in`
  stays 80 while the real mel width is `num_mel_bins = 128`. vLLM correctly uses
  `num_mel_bins` (`configs/parakeet.py:66`). Do not wire `feat_in` to anything.
  Similarly `SoundConfig`'s defaults `conv_kernel_size=31`,
  `projection_hidden_size=20480`, `projection_bias=True` are all overridden by
  the checkpoint — never rely on the class defaults.

---

## 5. `llm_config` — Nemotron-H hybrid MoE (the "30B-A3B")

```json
{
  "model_type": "nemotron_h",
  "hidden_size": 2688,
  "num_hidden_layers": 52,
  "vocab_size": 131072,
  "tie_word_embeddings": false,
  "hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
  "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
  "rope_theta": 10000, "partial_rotary_factor": 1.0,
  "max_position_embeddings": 262144, "sliding_window": null,
  "mamba_num_heads": 64, "mamba_head_dim": 64, "n_groups": 8,
  "ssm_state_size": 128, "conv_kernel": 4, "expand": 2, "chunk_size": 128,
  "use_conv_bias": true, "mamba_proj_bias": false, "mamba_hidden_act": "silu",
  "time_step_min": 0.001, "time_step_max": 0.1, "time_step_floor": 0.0001,
  "time_step_limit": [0.0, 1e30],
  "n_routed_experts": 128, "num_experts_per_tok": 6, "moe_intermediate_size": 1856,
  "n_shared_experts": 1, "moe_shared_expert_intermediate_size": 3712,
  "norm_topk_prob": true, "routed_scaling_factor": 2.5,
  "n_group": 1, "topk_group": 1,
  "intermediate_size": 1856, "mlp_hidden_act": "relu2", "mlp_bias": false,
  "norm_eps": 1e-05, "layer_norm_epsilon": 1e-05,
  "residual_in_fp32": false, "rescale_prenorm_residual": true,
  "attention_bias": false, "attention_dropout": 0.0, "hidden_dropout": 0.0,
  "use_mamba_kernels": true, "use_cache": true, "use_bias": false,
  "bos_token_id": 1, "eos_token_id": 11, "pad_token_id": 0,
  "initializer_range": 0.02, "num_logits_to_keep": 1
}
```

### 5.1 Layer composition

`hybrid_override_pattern` is one character per layer
(`vllm/model_executor/models/nemotron_h.py:531-536, 574-596`):

| Char | Layer type |
| --- | --- |
| `M` | `NemotronHMambaDecoderLayer` (Mamba-2) |
| `*` | `NemotronHAttentionDecoderLayer` |
| `E` | `NemotronHMoEDecoderLayer` |
| `-` | `NemotronHMLPDecoderLayer` (dense MLP) |

For this checkpoint the pattern has length **52** and decomposes as:

- **23 × `M`** (Mamba-2)
- **23 × `E`** (MoE)
- **6 × `*`** (attention), at layer indices **5, 12, 19, 26, 33, 42**
- **0 × `-`** — there are no dense MLP layers, so `intermediate_size: 1856` is
  effectively unused (it only feeds `NemotronHMLP` defaults).

`num_hidden_layers: 52` agrees with `len(pattern)`; vLLM builds layers from the
pattern length, not from `num_hidden_layers` (`nemotron_h.py:594-596`).

### 5.2 Derived shapes

| Quantity | Value |
| --- | --- |
| Attention | Q `32×128 = 4096`, KV `2×128 = 256` (GQA 16:1), RoPE θ=10000, fully rotary |
| Mamba-2 inner | `mamba_num_heads × mamba_head_dim = 64 × 64 = 4096` (`nemotron_h.py:377, 780`); 8 groups, state size 128, conv width 4, chunk 128. **`expand: 2` is stored but unused** — vLLM never derives the inner size from it, so do not compute `2 × 2688 = 5376` |
| MoE | 128 routed experts, top-6, `d_ff = 1856` each; 1 shared expert with `d_ff = 3712`; sigmoid routing with `e_score_correction_bias`, `norm_topk_prob=true`, output scaled by 2.5 (`nemotron_h.py:150-236`) |
| Activation | `relu2` (ReLU²) everywhere in MLP/MoE |
| Embedding | 131072 × 2688, untied LM head |
| Context | `max_position_embeddings: 262144` (checkpoint card advertises 131072 usable) |

### 5.3 Key names: read the JSON literally

`NemotronHConfig.__init__` takes the Mamba knobs under `mamba_*` aliases and
re-exports them under different attribute names
(`mamba_n_groups → n_groups`, `mamba_d_conv → conv_kernel`,
`mamba_expand → expand`, `mamba_dt_min → time_step_min`,
`mamba_conv_bias → use_conv_bias`, `mamba_chunk_size → chunk_size`;
`configs/nemotron_h.py:242-255`). The checkpoint stores the **post-mapping**
names, so those keys arrive as generic `**kwargs`. Because
`super().__init__(**kwargs)` runs *after* the aliased assignments
(`configs/nemotron_h.py:268-274`), the JSON values overwrite the alias defaults
and win. Net effect: the literal JSON keys are the effective values — but a
checkpoint that set, say, `n_groups` to something other than the
`mamba_n_groups` default would depend on that ordering to take effect. Read the
JSON keys as-is and don't re-derive them from the `mamba_*` aliases.

### 5.4 SSM state cache dtype

`residual_in_fp32: false`, and there is **no `mamba_ssm_cache_dtype` key**. vLLM's
Nemotron-H default is `float32` when unset
(`vllm/model_executor/models/config.py:605-625`), but the hook that applies it is
registered only for the `"NemotronH_Nano_VL_V2"` architecture string
(`config.py:857`), so it does **not** fire for the Omni architecture (see
`NEMOTRON_OMNI_MULTIMODAL_DESIGN.md` §3.4). Use **fp32 SSM state** in a port.

---

## 6. `preprocessor_config.json`

```json
{
  "image_processor_type": "NemotronH_Nano_Omni_Reasoning_V3ImageProcessor",
  "auto_map": {
    "AutoImageProcessor": "image_processing.NemotronH_Nano_Omni_Reasoning_V3ImageProcessor",
    "AutoVideoProcessor": "video_processing.NemotronH_Nano_Omni_Reasoning_V3VideoProcessor",
    "AutoProcessor": "processing.NemotronH_Nano_Omni_Reasoning_V3Processor"
  },
  "patch_size": 16,
  "downsample_ratio": 0.5,
  "norm_mean": [0.48145466, 0.4578275, 0.40821073],
  "norm_std": [0.26862954, 0.26130258, 0.27577711],
  "min_num_patches": 1024,
  "max_num_patches": 13312,
  "max_model_len": 16384
}
```

vLLM does **not** use this file for these values — it builds its own processor
from `config.json` plus the engine's `max_model_len`
(`nano_nemotron_vl.py:200-209`). Note `max_model_len: 16384` here is only the HF
processor's default token budget for dynamic-resolution tiling; under vLLM the
image token budget scales with whatever `--max-model-len` you serve with, so
**image resolution is a function of the served context length** (see
`DynamicResolutionImageTiler.max_num_tokens_available`,
`processors/nano_nemotron_vl.py:329-330`). Two deployments with different
`--max-model-len` will tile the same image differently. A port must decide
deliberately whether to follow vLLM (engine `max_model_len`) or HF (fixed
16384); they disagree.

---

## 7. `generation_config.json`

```json
{
  "bos_token_id": 1, "eos_token_id": [2, 11], "pad_token_id": 0,
  "do_sample": true, "temperature": 0.6, "top_p": 0.95,
  "max_new_tokens": 16384,
  "reasoning_budget": 16384, "reasoning_grace": 512,
  "repetition_penalty": 1.0
}
```

`reasoning_budget` / `reasoning_grace` are Nemotron-V3 reasoning-parser
concepts, not standard HF fields; they pair with
`--reasoning-parser nemotron_v3` (`vllm/reasoning/nemotron_v3_engine_reasoning_parser.py`,
`vllm/parser/nemotron_v3.py`). Note `eos_token_id` is a **list** here versus the
scalar `11` in `config.json`.

---

## 8. Quantized variants (FP8 / NVFP4)

Diffing the NVFP4 checkpoint's `config.json` against BF16: the **only**
difference is an added top-level `quantization_config`. Every architectural
field is identical.

```json
"quantization_config": {
  "quant_method": "modelopt",
  "quant_algo": "MIXED_PRECISION",
  "kv_cache_scheme": {"dynamic": false, "num_bits": 8, "type": "float"},
  "producer": {"name": "modelopt", "version": "0.43.0rc2.dev…"},
  "config_groups": { … },
  "quantized_layers": { … },
  "ignore": []
}
```

Empirically, from `quantized_layers` (5986 entries):

| Fact | Value |
| --- | --- |
| Modules quantized | **`language_model.*` only** — zero entries under `vision_model.*`, `sound_encoder.*`, `sound_projection.*`, `mlp1.*` |
| NVFP4 (group_size 16) | 5888 layers — the routed MoE experts (`…mixer.experts.N.{up,down}_proj`) |
| FP8 | 98 layers — Mamba `in_proj`/`out_proj` and MoE `shared_experts` |
| KV cache | FP8 (hence `--kv-cache-dtype fp8` in NVIDIA's serve command) |

This is direct confirmation of the design-doc claim (§7.5): **the vision and
audio towers and both projectors stay bf16 regardless of variant.** Budget
tower memory as bf16 and don't look for tower scale tensors.

---

## 9. Config-level gotchas

### 9.1 `video_pruning_rate: 0.7` in the config is ignored by vLLM

The checkpoint requests 70 % video-token pruning, and the HF reference
implementation honours it. vLLM reads the pruning rate **only** from engine
config (`--video-pruning-rate`):

```python
# vllm/model_executor/models/nano_nemotron_vl.py:226-227
def get_video_pruning_rate(self) -> float | None:
    return self.ctx.get_mm_config().video_pruning_rate
```

`MultiModalConfig.video_pruning_rate` defaults to `None`
(`vllm/config/multimodal.py:191-195`). So serving without the flag runs video
**unpruned**, which is both slower and off-distribution relative to how the
model was trained/tuned. NVIDIA's published command passes
`--video-pruning-rate 0.5`. A port should decide explicitly: honour the config
value, the CLI, or CLI-overrides-config — and document it.

Concretely, with `video_target_num_patches: 1024`, `T=2`, 256 sampled frames:
128 tubelets × 256 tokens = **32768 video tokens unpruned**, → 16384 at q=0.5,
→ 9830 at q=0.7. On a 131072 context this is the difference between a workable
and an unusable video prompt.

### 9.2 Two different "image sizes"

`force_image_size: 512` (top level) and `preferred_resolution: [768, 768]`
(vision_config) are both live but feed different things:

- `512` → the processor's `image_size` and `num_image_token = (512/16)²·0.25 =
  256` (`processors/…:596-602`). Because Omni takes the dynamic-resolution path,
  `num_image_token` is **not** used for images; it survives only as a fallback
  for video token counting and dummy-data sizing.
- `768` → `RadioConfig.image_size`, i.e. the ViT's nominal `input_dims` used when
  no explicit size is given (`nano_nemotron_vl.py:1576-1577`).

Neither bounds the actual image resolution — that comes from
`min_num_patches`/`max_num_patches` and the context budget. Don't "simplify" by
collapsing these.

### 9.3 `use_thumbnail: true` is inert

The dynamic tiler is constructed without `use_thumbnail` and asserts it is False
(`processors/…:269, 610-618`); thumbnails only exist in the fixed-tile
(`dynamic_preprocess`) path, which Omni does not take. Do not add a thumbnail
tile.

### 9.4 Duplicated fields that must stay consistent

`video_temporal_patch_size` appears at the top level (HF modeling reads it) and
in `vision_config` (vLLM reads it); both are `2`.
`min_num_patches`/`max_num_patches` appear both in `vision_config` and in
`vision_config.args`; vLLM reads the `args` copies. If a future checkpoint
desynchronizes these, HF and vLLM will silently disagree.

### 9.5 CI overrides are not real values

`tests/models/registry.py:1188-1202` overrides `min_num_patches: 1`,
`max_num_patches: 12`, `num_hidden_layers: 2`,
`hybrid_override_pattern: "M*"`, and forces
`video_maintain_aspect_ratio: false` (with a TODO noting a known mixed-resolution
processor bug when it is `true` — the real checkpoint sets `true`). Never treat
those as checkpoint values.

---

## 10. Quick reference card

```text
Total / active params ....... ~30B / ~3B (A3B)
LM .......................... Nemotron-H hybrid, 52 layers = 23 Mamba2 + 23 MoE + 6 attention
                              d=2688, vocab=131072, GQA 32/2 heads (head_dim 128), RoPE θ=1e4
                              MoE: 128 experts, top-6, d_ff=1856, +1 shared expert d_ff=3712
                              Mamba2: 64 heads × 64, 8 groups, state 128, conv 4, chunk 128
Vision tower ................ C-RADIOv4-H = ViT-H/16: d=1280, 32 layers, 16 heads, ffn=5120
                              4 CLS + 6 register tokens (num_skip=10), pos-embed grid 128×128
                              image tokens/item ∈ [256, 3328]; video 256 tokens per 2-frame tubelet
Vision projector (mlp1) ..... RMSNorm(5120) → 5120→20480 → ReLU² → 20480→2688 (no bias)
Audio tower ................. Parakeet Conformer: d=1024, 24 layers, 8 heads, ffn=4096, conv k=9
                              128 mel bins @16kHz, 8× subsampling ⇒ 12.5 tokens/s (80 ms/token)
Audio projector ............. RMSNorm(1024) → 1024→4096 → ReLU² → 4096→2688 (no bias)
Special tokens .............. <img> </img> <image>(18) | <video>(131081) | <so_start> <so_embedding>(27) <so_end>
Precision ................... bf16 everywhere; FP8/NVFP4 variants quantize the LM only
Serving flags ............... --trust-remote-code --reasoning-parser nemotron_v3
                              --tool-call-parser qwen3_coder --video-pruning-rate 0.5
                              --media-io-kwargs '{"video":{"fps":2,"num_frames":256}}'
                              --kv-cache-dtype fp8 (quantized variants only)
```

---

## Appendix A — `vision_config.args` keys ignored by vLLM (149 of 156)

Kept for completeness so a port can confirm nothing was missed. None of these
affect inference:

```text
aa, amp, amp_dtype, amp_impl, aug_repeats, aug_splits, bn_eps, bn_momentum,
cache_dir, channels_last, checkpoint_hist, chk_keep_forever, class_map,
clip_grad, clip_mode, coco_annotations_file, coco_image_dir, color_jitter,
cooldown_epochs, crd_loss, crd_loss_weight, crop_pct, cutmix, cutmix_minmax,
dataset_download, debug_full_knn, decay_epochs, decay_milestones, decay_rate,
depchain, dist_bn, dist_norm_weight, distributed, drop, drop_block,
drop_connect, drop_path, dtype, epoch_repeats, eval, eval_metric, eval_teacher,
eval_teacher_only, eval_throughput, fast_norm, fd_loss_fn,
feature_normalization, feature_summarizer, feature_upscale_factor,
force_new_wandb_id, force_spectral_reparam, freeze_bn, fsdp, fuser, gp,
grad_accum_steps, grad_checkpointing, head_init_bias, head_init_scale,
head_warmup, head_weight_decay, hflip, img_size, in_chans, initial_checkpoint,
input_size, interpolation, layer_decay, local_rank, log_interval, log_mlflow,
log_wandb, loss_auto_balance, lr_base, lr_base_scale, lr_base_size,
lr_cycle_decay, lr_cycle_limit, lr_cycle_mul, lr_k_decay, lr_noise,
lr_noise_pct, lr_noise_std, mean, mesa, min_lr, mixup, mixup_mode,
mixup_off_epoch, mixup_prob, mixup_switch_prob, mlp_hidden_size, mlp_num_inner,
mlp_version, model_kwargs, model_norm, momentum, no_aug, no_ddp_bb,
no_prefetcher, no_resume_opt, num_classes, opt_betas, opt_eps, patience_epochs,
pin_mem, prefetcher, pretrained, rank, ratio, recount, recovery_interval,
remode, reprob, reset_loss_state, resplit, save_images, scale, sched, seed,
smoothing, spectral_heads, spectral_reparam, split_bn, start_epoch, std,
stream_teachers, sync_bn, synchronize_step, torchcompile, torchscript,
train_interpolation, train_split, tta, use_coco, use_multi_epochs_loader,
val_ema_only, val_split, vflip, vitdet_version, wandb_entity, wandb_job_type,
wandb_name, wandb_project, warmup_lr, warmup_prefix, worker_seeding, workers,
world_size
```

Two that look meaningful but are not: `args["dtype"] = "bfloat16"` /
`args["amp_dtype"]` are timm training settings — the runtime dtype comes from
the top-level `torch_dtype` and the LM's dtype. And `args["mlp_hidden_size"]
= 1520` / `mlp_version = "v2"` describe RADIO's *distillation heads*, not the
ViT FFN (which is 5120, from the timm name table).

## Appendix B — `teachers` block verbatim

Only `name` and `use_summary` matter to vLLM (CLS-token count and summary
selection); the rest documents how C-RADIOv4-H was distilled.

```json
[
  {"name": "clip",    "model": "ViT-H-14-378-quickgelu",   "type": "open_clip", "pretrained": "dfn5b", "input_size": 378,  "feature_distillation": true, "fd_normalize": false, "use_summary": true},
  {"name": "siglip",  "model": "ViT-SO400M-14-SigLIP-384", "type": "open_clip", "pretrained": "webli", "input_size": 378,  "feature_distillation": true, "fd_normalize": false, "use_summary": true},
  {"name": "dino_v2", "model": "dinov2_vitg14_reg",        "type": "dino_v2",                          "input_size": 378,  "feature_distillation": true, "fd_normalize": false, "use_summary": true},
  {"name": "sam",     "model": "vit-h",                    "type": "sam",                              "input_size": 1024, "feature_distillation": true, "fd_normalize": false, "use_summary": false}
]
```
