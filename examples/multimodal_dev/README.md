# multimodal_dev — Standalone Multimodal Training

Standalone, model-agnostic training entry point for multimodal
vision-language models built on Megatron-Core (FSDP + EP).

## Directory Structure

```
multimodal_dev/
├── pretrain_multimodal.py   # Training entry point (model-agnostic)
├── forward_step.py          # Forward step, TP broadcast, loss computation
├── arguments.py             # Multimodal CLI arguments
├── data/
│   ├── mock.py              # Fixed-length mock data for end-to-end testing
│   ├── mock_varlen/         # Deterministic multimodal varlen mock (packed-document paradigm)
│   │   ├── distributions.py    # Generator-agnostic numeric helpers (numpy/math only)
│   │   ├── packed_document.py  # Pure-numpy document -> window plan kernel + context-scaled default
│   │   └── qwen35_vl.py        # Torch dataset, provider, config resolver, geometry adapter
│   └── cord_v2.py           # CORD-V2 receipt-OCR data provider
├── models/
│   ├── __init__.py          # MODEL_REGISTRY — central model registry
│   ├── base.py              # MultimodalModel base class (vision encoder + GPTModel)
│   └── qwen35_vl/           # Qwen3.5-VL architecture
│       ├── factory.py       # Factory functions for pretrain entry point
│       ├── model.py         # Qwen35VLModel (MRoPE, vision encoder wiring)
│       ├── configuration.py # TransformerConfig builders and constants
│       ├── specs.py         # Layer spec builders (hybrid attention, ViT)
│       ├── mrope.py         # 3D MRoPE position ID computation
│       └── vision_encoder.py# ViT encoder (patch embed, merger, RoPE)
└── scripts/                 # Launch scripts (torchrun, Slurm)
```

## Quick Start

```bash
torchrun --nproc_per_node=8 multimodal_dev/pretrain_multimodal.py \
    --model-arch qwen35_vl \
    --dataset-provider mock \
    ... # other Megatron args (--num-layers, --hidden-size, etc.)
```

## Variable-Length Mock Data (packed_document)

`mock_varlen` generates deterministic variable-length packed THD windows
of WHOLE documents. A document is a complete logical TRAINING SAMPLE
that must fit one physical window and is never split (sample-atomic
packing); its FINAL logical length is drawn first, from a
document-count-weighted mixture, and the images live INSIDE that length:

    L_target ~ component.length            (truncated lognormal, max <= S)
    k        ~ image counts feasible under THIS L_target (renormalized)
    text     = L_target - sum_j(1 + V_j)   (>= 1 by construction)

so the realized document-length distribution IS the configured one, and
`logical_document_length == L_target <= S` — a document may legally
reach S and occupy a whole window, and can never exceed it. The stream
is `image_atoms + text + EOD` with the pure-text mock's exact shift
semantics: `input = stream[:-1]`, `labels = stream[1:]` inside the
document; the terminal EOD is the last input position's TARGET, never an
input position; supervision is full-sequence (every text/EOD target;
image-placeholder targets are -100; `loss_mask == (labels != -100)`).
Image placement is fixed at the document prefix (interleaved placement
is an explicit non-goal). Fixed-shape single-image data is served by
`--dataset-provider mock` instead.

The DATA side runs zero-config at any `--seq-length` in [4096, 131072]
(the verified 128K RUNNING configuration additionally requires
`--recompute-vision --recompute-vision-whole-tower`; the qualification recipe's explicit
65,536 raw-patch guard — `--max-vision-patches-per-microbatch` is unset by
default — bounds the vision payload only and does not by itself guarantee
memory runnability; MTP is rejected at this entry only when combined with an
effective sequence-parallel layout, i.e. `--sequence-parallel` at TP>1): with no
dataset config at all, the **context-scaled practical default** applies
(`context_scaled_default` in `data/mock_varlen/packed_document.py` — the ONLY
default; a reference-shaped practical default, structural/derived, not
fitted). Set the sequence length and the distribution follows: a fixed
"short" component (weight 95, lengths bulked around 1-2K) plus a "long"
component (weight 5) whose support runs continuously from 2K to S — no
mid-range vacuum, meaningful mass near the top of the context. Outside
the domain the resolver raises.

```bash
torchrun --nproc_per_node=8 examples/multimodal_dev/pretrain_multimodal.py \
    --model-arch qwen35_vl \
    --dataset-provider mock_varlen \
    --seq-length 4096 \
    --micro-batch-size 1 \
    --use-vanilla-collate-fn \
    --use-packed-sequence \
    --pad-packed-seq-alignment max \
    --max-seqlen-per-dp-cp-rank 4096 \
    ... # other Megatron model and training arguments
```

To deviate, pass a COMPLETE profile JSON (all four top-level keys):
partial configs are never merged onto the scaled default — a partial
override of an S-dependent baseline is ambiguous, so it fails loudly and
the error carries the fully resolved default JSON for the current S;
copy it, edit (even for a plan_seed-only tweak), and pass it back.

Each dataset item has exactly these fields (T = the window's logical
token count, S = `--seq-length` = the physical window capacity):

| Field | Per-sample shape | Meaning |
|-------|------------------|---------|
| `input_ids` | `[T]`, `T <= S` | Whole documents only; NO physical padding in the dataset (the tail to S is the runtime packer's dummy THD sequence) |
| `labels` | `[T]` | Full-sequence shifted targets; `-100` where the target is an image placeholder; each document's last position targets the EOD |
| `loss_mask` | `[T]` | Float mask, `== (labels != -100)` by construction |
| `pixel_values` | `[sum(P_j), D]` | Ordered flattened raw patches for all images |
| `image_grid_thw` | `[N, 3]` | Ordered `(T, H, W)` patch grids (document -> in-document order) |
| `seq_lens` | `[num_documents]` | Whole per-document input lengths (== each document's L_target), `sum == T` |

Configuration:

- `--multimodal-varlen-mock-dataset-config-json`: the single, optional
  generator config (inline JSON or a JSON-file path). Omitted (or `{}`),
  the context-scaled default for the final seq_length applies; an
  explicit config must be COMPLETE. Distinct from the core
  `--varlen-mock-dataset-config-json`, which belongs to the text-side
  `--use-varlen-dataset` datasets. The four top-level keys:
  - `components`: non-empty list of mixture components, each with
    exactly `name` (non-empty, unique — statistics, errors, and
    snapshots address components by name), `weight` (document-count
    proportion, finite > 0; disable a component by deleting it),
    `length` (`min`/`max`/`mean` (post-truncation)/`sigma`, 0 = constant
    — the FINAL logical sample length including image atoms, excluding
    the EOD, raw patches, and all physical padding; `max` may reach S
    and must not exceed the aligned window budget), and
    `images_per_document` (strict integer categorical; counts must be
    unique — a duplicated count is rejected as a likely typo; zero weights
    disable an entry). Startup validates per component that at least
    one positive-weight count fits at `length.min` and EVERY
    positive-weight count fits at `length.max` (a count that can never
    be drawn is a dead entry and fails loudly). Per document, the count
    is drawn from the categorical RESTRICTED to the counts feasible
    under the drawn L_target (renormalized, reported as
    count-conditioning); if the drawn geometries overshoot L_target the
    LARGEST atom is substituted with the smallest drawable bucket
    (ties -> earliest) until they fit. Images are NEVER dropped.
  - `image_sizes`: processed `[height, width]` resolution buckets (each
    divisible by `patch_size * spatial_merge_size`) with optional
    categorical `weights`; per-image sizes are drawn from this set.
  - `plan_pool_windows`: bounds the pre-built plan corpus independently
    of the virtual dataset length. Default `"auto"` =
    `max(2048, ceil(2**26 / seq_length))`. An explicit integer is an
    upper bound, clamped by the split's sample count.
  - `plan_seed`: seeds the window LAYOUT (default 1234), independently
    of `--seed`; training seeds vary token/pixel content only.

  Window-level statistics — documents per window, image counts,
  vision-token share, realized padding, component composition — are
  emergent and measured, not configured. The provider's deterministic
  startup scan walks the FULL plan pool at construction, logs the
  resolved profile plus the pool
  maxima (max raw patches per window/image, max images per window, max
  logical tokens, padding, count-/geometry-conditioning fractions) as
  the launch artifact, and fails BEFORE the DataLoader when the pool's
  heaviest window exceeds `--max-vision-patches-per-microbatch`. The
  scan applies to explicit configs exactly as to the default. Plan
  construction (auto pool, bucket geometry, kernel config) goes through
  ONE shared helper, `build_packed_document_plan`, so no consumer
  re-derives any of them.

  Default provenance: every number in the default (mixture weights,
  length parameters, image rows, and the 12-bucket `image_sizes` table,
  which a sentinel test pins row by row) is
  labeled structural/derived and documented in the comment block above
  `context_scaled_default` in `data/mock_varlen/packed_document.py` — that
  block is the single provenance authority.
- `--max-vision-patches-per-microbatch` / `--max-vision-patches-per-image`:
  hard packer-level fail-fast guards checked identically on every TP
  rank on the broadcasted batch (so all ranks raise together);
  violations raise with the actual payload, the limit, and the offending
  geometry instead of surfacing as an opaque CUDA OOM. The per-image
  guard defaults to the largest drawable (weight > 0) bucket's raw-patch
  count; the per-microbatch guard stays explicit because it states a
  hardware memory envelope. The startup pool scan makes over-budget
  plans a startup failure.
- `--total-seq-length`: ignored by this provider — `--seq-length` is the
  sole capacity authority (the legacy knob belongs to the fixed-shape
  providers). The provider bounds `--seq-length` to [1, 2097152] even
  for explicit profiles: plan construction cost scales with
  `seq_length * plan_pool_windows`.

Packing is original-order next_fit with lookahead (prefix-stable: the
plan's first N windows never change when the pool grows). Bin capacity
is PHYSICAL: each document costs `align_up(L, A)` where A is the
CP/SP-derived per-segment alignment and the capacity is
`max_seqlen_per_dp_cp_rank * CP` — both provider-derived runtime
parameters, never data-profile keys. The dataset emits the logical
window (sum(seq_lens) = T <= S, whole documents as segments, no
fabricated padding text); the physical tail to S is represented by the
runtime packer's ordinary dummy THD sequence, so the provider requires
`--pad-packed-seq-alignment max` and `--max-seqlen-per-dp-cp-rank` (with
`max_seqlen_per_dp_cp_rank * CP == seq_length`). The dummy THD tail
itself is an implementation invariant, not a tunable: it is the core
default, and both the provider and the runtime packer fail fast if it
is disabled (including via the auto-generated
`--no-pad-packed-seq-by-appending-dummy-seq` switch). Padding is accounted
three-way (physical = logical + segment-alignment padding + window-tail
padding); the pool-aggregate fraction (~14-22% under the default mixture,
the price of sample atomicity under a heavy-tailed length distribution) is
reported in the startup artifact, never gated.

The packed-THD packer splices each sample's `seq_lens` segments into
`cu_seqlens` with independent per-segment CP/SP alignment padding (real
padding tokens after every internal segment, so the tensor layout matches
`cu_seqlens_padded`); the padded BSHD layout rejects multi-segment samples.
`micro_batch_size` must be 1 — one item already is a full-capacity
window — and `--use-packed-sequence` plus the identity collator
(`--use-vanilla-collate-fn`) are required (both enforced by the
provider). Do **not** combine with `--use-varlen-dataset` or
`--sequence-packing-scheduler`: the core packing scheduler transports
token-axis tensors and length metadata only (its per-stage field
retention keeps a fixed six-key set), so ragged vision payloads —
`pixel_values` rows keyed by raw patches and `image_grid_thw` rows keyed
by images — would be silently dropped rather than rerouted alongside
their placeholder tokens, and its balancing cost model does not see the
vision payload at all. This dataset therefore packs at plan time and
ships complete windows. Packed THD + HybridEP flex dispatch
requires `--moe-hybridep-pad-variable-tokens`. An image-free microbatch
still runs the vision tower once on a minimal zero-weighted dummy image
so every rank produces vision grads for bucketed grad synchronization.

Generation is deterministic and access-order independent per index: the
window layout is a pure function of the resolved profile and `plan_seed`,
so the full pool can be inspected at startup without materializing tokens
or pixels.

## Checkpoint Conversion (HF → Megatron-FSDP DTensor)

Convert a HuggingFace release to a Megatron-FSDP DTensor checkpoint via
[Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) before
pretraining from pretrained weights.

### Setup

Clone Bridge and pin its `3rdparty/Megatron-LM` submodule to this branch:

```bash
git clone --recurse-submodules https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge/3rdparty/Megatron-LM
git remote add wplf https://github.com/wplf/Megatron-LM.git
git fetch wplf feat/qwen35-vl-example
git checkout feat/qwen35-vl-example
cd ../..
```

### Convert

Single 8×GPU node, EP=8 / TP=CP=1; substitute any Qwen3.5 variant for
`--hf-model`:

```bash
PYTHONPATH=./src:./3rdparty/Megatron-LM/ \
  torchrun --nproc_per_node=8 \
  examples/conversion/mfsdp/convert_checkpoints_fsdp.py import \
  --hf-model Qwen/Qwen3.5-35B-A3B \
  --megatron-path ${WORKSPACE}/models/Qwen/Qwen3.5-35B-A3B-fsdp \
  --ckpt-format fsdp_dtensor \
  --ep 8
```

HF weights are auto-fetched on first run via `huggingface_hub`. Adjust
`--tp` / `--cp` / `--ep` to match the training topology (must satisfy
`WORLD_SIZE % (TP*CP*EP) == 0`).

### Output

```
${WORKSPACE}/models/Qwen/Qwen3.5-35B-A3B-fsdp/
├── iter_0000000/
│   ├── __0_0.distcp .. __7_0.distcp   # FSDP DTensor shards, one per rank (~18 GB each for 35B-A3B)
│   ├── .metadata
│   ├── run_config.yaml
│   └── train_state.pt
├── latest_checkpointed_iteration.txt
└── latest_train_state.pt
```

### Bridge dependency

Requires
[NVIDIA-NeMo/Megatron-Bridge#3987](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/3987)
(skip tokenizer save). Without that fix the checkpoint is still written
correctly but the script exits non-zero after save with
`AttributeError: 'TokenizerConfig' object has no attribute 'make_vocab_size_divisible_by'`
against this branch's `megatron.core.tokenizers.utils.build_tokenizer`.

## Architecture

`pretrain_multimodal.py` is **model-agnostic**. All model-specific logic
is delegated to factory functions registered in `MODEL_REGISTRY`
(`models/__init__.py`). The entry point handles only generic concerns:

- Building `language_config` from Megatron CLI args
- Constructing `vision_config` via the registry
- Applying vision recompute and dtype propagation
- Routing to model and dataset factories

The `forward_step` is also model-agnostic — it uses the model's
`compute_position_ids()` method polymorphically and passes a standard
batch dict.

## Adding a New Model Architecture

Adding a new model (e.g. `llava_next`) requires **no changes** to
`pretrain_multimodal.py` or `forward_step.py`. Follow these steps:

### Step 1 — Create the model package

```
multimodal_dev/models/llava_next/
├── __init__.py
├── factory.py          # Required: factory functions
├── configuration.py    # Vision/language TransformerConfig builders
├── model.py            # Model class (subclass MultimodalModel)
├── specs.py            # Layer spec builders
└── vision_encoder.py   # Vision encoder (if custom)
```

### Step 2 — Implement factory functions

Create `factory.py` with up to three functions:

```python
# models/llava_next/factory.py

def post_language_config(language_config, args):
    """(Optional) Mutate language_config with model-specific fields."""
    # e.g. language_config.some_field = value
    pass

def set_vision_flops_metadata(args, language_config, vision_config):
    """(Optional) Set vision FLOPs metadata on args."""
    args.count_vision_model_flops = True
    args.vision_flops_variant = "llava_next"
    # ... set dimension fields for FLOPs calculation

def build_model(args, language_config, vision_config, **kwargs):
    """(Required) Build and return the complete model instance."""
    from .model import LlavaNextModel
    from .specs import get_llava_next_language_spec

    language_spec = get_llava_next_language_spec(
        config=language_config,
        vp_stage=kwargs.get("vp_stage", None),
        pp_rank=None,
    )
    return LlavaNextModel(
        language_config=language_config,
        language_spec=language_spec,
        vision_config=vision_config,
        # ... model-specific args
    )
```

### Step 3 — Register in `MODEL_REGISTRY`

Add an entry in `models/__init__.py`:

```python
from multimodal_dev.models.llava_next.configuration import (
    get_llava_next_vision_config,
)
from multimodal_dev.models.llava_next.factory import (
    build_model as _build_llava_next_model,
    post_language_config as _llava_next_post_language_config,
    set_vision_flops_metadata as _llava_next_vision_flops,
)

MODEL_REGISTRY["llava_next"] = {
    "model_factory_fn": _build_llava_next_model,           # required
    "vision_config_fn": get_llava_next_vision_config,      # required
    "post_language_config_fn": _llava_next_post_language_config,  # optional
    "vision_flops_fn": _llava_next_vision_flops,           # optional
    "dataset_providers": {                                  # optional
        "mock": "multimodal_dev.data.llava_mock.train_valid_test_datasets_provider",
    },
}
```

### Step 4 — (Optional) Add a dataset provider

Create a dataset module under `data/` if the model needs custom data
preprocessing. The provider function signature is:

```python
def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Return (train_dataset, val_dataset, test_dataset)."""
    ...
```

Register it in the `dataset_providers` dict of the registry entry.
Providers can be either direct callables or dotted import path strings
(resolved lazily at runtime).

### Step 5 — Launch

```bash
torchrun --nproc_per_node=8 multimodal_dev/pretrain_multimodal.py \
    --model-arch llava_next \
    --dataset-provider mock \
    ...
```

## Registry Entry Reference

| Field | Required | Signature |
|-------|----------|-----------|
| `model_factory_fn` | Yes | `(args, language_config, vision_config, **kwargs) -> MegatronModule` |
| `vision_config_fn` | Yes | `(num_layers_override=None) -> TransformerConfig` |
| `post_language_config_fn` | No | `(language_config, args) -> None` |
| `vision_flops_fn` | No | `(args, language_config, vision_config) -> None` |
| `dataset_providers` | No | `Dict[str, str \| callable]` |
