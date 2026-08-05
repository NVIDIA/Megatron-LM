<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Migrate from GPTModel to HybridModel

This guide describes how to replace a Megatron Core `GPTModel` with a
`HybridModel`, convert an existing distributed checkpoint, and start or resume
training with the converted weights. The conversion stays in Megatron's
distributed-checkpoint format; it does not use Hugging Face as an intermediate
format.

## 1. What Is HybridModel?

A standard `GPTModel` decoder layer contains both a self-attention sublayer and
an MLP or MoE sublayer under one layer index. `HybridModel` instead builds an
ordered stack in which every position represents one layer family. New
programmatic model providers should describe that order with direct layer
specifications, as shown below. The `--hybrid-layer-pattern` syntax remains
available for CLI workflows and checkpoint migration:

Legacy patterns retain their existing runtime contract: the same string parser
and PP/VPP selector are used, layer classes receive the shared model config,
public layer-type lists remain symbol-valued, and legacy MTP, metrics,
checkpoint keys, and dynamic-inference behavior are unchanged. Direct specs
use a separate adapter so new per-occurrence behavior does not leak into that
path.

| Symbol | Layer family |
|--------|--------------|
| `M` | Mamba-2 state-space layer |
| `G` | Gated Delta Network (GDN) layer |
| `*` | Self-attention layer |
| `D` | DeepSeek Sparse Attention (DSA) layer |
| `-` | Dense MLP layer |
| `E` | Mixture-of-Experts (MoE) layer |

One pattern symbol is one HybridModel layer. Consequently, one GPT transformer
block becomes two HybridModel layers when preserving the original architecture:

| Source architecture | Equivalent HybridModel pattern | Hybrid layer count |
|---------------------|--------------------------------|--------------------|
| Two dense GPT blocks | `*-*-` | 4 |
| Two all-layer MoE GPT blocks | `*E*E` | 4 |

For example, source GPT layer 0 is split between Hybrid layers 0 and 1:
its attention parameters move to the first `*`, and its MLP parameters move to
the first `-` or `E`. Source GPT layer 1 maps to the next pair, and so on.

The pattern can also describe execution layout. A `|` marks a pipeline segment
boundary, and `/` introduces a repeated Multi-Token Prediction (MTP) pattern.
For example, `*-*-|*-*-` places four GPT-equivalent blocks across two pipeline
segments. Separators do not count as layers.

HybridModel provides the following benefits:

- Different layer families can be composed in one model without forcing every
  decoder block to have the same structure.
- Attention and dense or expert MLP layers can be placed and configured
  independently.
- Mamba, GDN, standard or DeepSeek attention, dense MLP, and MoE layers can use
  one pattern-driven model interface.
- Patterns that include Mamba can replace some quadratic attention layers with
  subquadratic sequence mixing and fixed-size recurrent inference state.
- Pipeline and virtual-pipeline segmentation can be expressed with the model
  pattern instead of a separate layer layout.
- The same model abstraction can describe a pure transformer (`*-` repeated),
  a pure Mamba model, or a heterogeneous architecture.

These capabilities do not imply an automatic throughput or quality improvement.
An architecture-preserving `*-` or `*E` migration should be validated for
numerical equivalence, and a pattern that adds another layer family should be
treated as a new architecture and benchmarked independently.

### Prefer direct layer specifications for new providers

`HybridModelConfig.layer_specs` is the preferred API for Python model
providers. Each `HybridLayerSpec` pairs an existing layer `ModuleSpec` with the
`TransformerConfig` used for that occurrence. This permits, for example,
different Mamba state dimensions or different MoE expert widths and top-k
values at individual positions without introducing another layer class.

The pattern is recursive: nested Python lists and list multiplication are
expanded in order, and aliases can be reused. `PipelineSplit()` remains a
structural marker rather than a layer. The following abbreviated example uses
the layer modules from the standard Hybrid stack:

```python
from copy import deepcopy

from megatron.core.models.hybrid.hybrid_architecture import HybridLayerSpec, PipelineSplit
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.training.models.hybrid import HybridModelConfig

modules = hybrid_stack_spec.submodules
# base_config has num_layers=10, pipeline_model_parallel_size=2, and
# virtual_pipeline_model_parallel_size=2.

mamba_config = deepcopy(base_config)
mamba_config.mamba_state_dim = 128

moe_4_config = deepcopy(base_config)
moe_4_config.moe_ffn_hidden_size = 1280
moe_4_config.moe_router_topk = 4

moe_8_config = deepcopy(base_config)
moe_8_config.moe_ffn_hidden_size = 2688
moe_8_config.moe_router_topk = 8

M = HybridLayerSpec(module_spec=modules.mamba_layer, config=mamba_config)
A = HybridLayerSpec(module_spec=modules.attention_layer, config=deepcopy(base_config))
E4 = HybridLayerSpec(module_spec=modules.moe_layer, config=moe_4_config)
E8 = HybridLayerSpec(module_spec=modules.moe_layer, config=moe_8_config)

# Four chunks for PP=2 and VPP=2. Nested lists are flattened in order.
layers = [
    [M, E4] * 2,
    PipelineSplit(),
    [M, A, E4],
    PipelineSplit(),
    [],
    PipelineSplit(),
    [[M, E8], A],
]

model_config = HybridModelConfig(
    transformer=base_config,
    vocab_size=vocab_size,
    layer_specs=layers,
    # One prediction depth; base_config.mtp_num_layers controls repetition.
    mtp_layer_specs=[A, E8],
)
```

The number of non-marker leaves in `layer_specs` must equal `num_layers`.
Configure exactly one architecture source: `layer_specs` for the direct API or
`hybrid_layer_pattern` for the legacy string API. Direct MTP layers belong in
the separate `mtp_layer_specs` field; do not put `PipelineSplit()` in that
field. `mtp_layer_specs` describes one prediction depth and
`mtp_num_layers` controls how many depths are built on the final logical
pipeline stage. That final stage must also contain at least one decoder layer;
standalone or separately split direct MTP placement is not supported.

The Python architecture definition is required when resuming a direct-spec
model. Distributed checkpoints preserve global layer keys and can be resharded
under another compatible PP/VPP split, but they do not serialize the
`ModuleSpec` tree or an architecture manifest.

For complete architecture-only definitions, see
[`nemotron_3_5_nano_30b_a3b.py`](../../examples/nemotron3/nemotron_3_5_nano_30b_a3b.py),
which deliberately reuses one config per layer family because Nano is not
heterogeneous, and
[`nemotron_labs_3_puzzle_75b_a9b.py`](../../examples/nemotron3/nemotron_labs_3_puzzle_75b_a9b.py),
which preserves Puzzle's occurrence-specific MoE widths and top-k values.

## 2. How to Convert a Checkpoint

There are two ways to bring `GPTModel` weights into a `HybridModel` run. Both
stay in Megatron's distributed-checkpoint format and can reshard across a
different tensor, pipeline, expert, or FSDP layout on the following load.

- **Option A — translate at load time (no separate step).** Start the hybrid
  run directly against the GPT checkpoint. The hybrid model retargets its own
  checkpoint state dict at the GPT checkpoint's keys during loading, so no
  second copy is written to disk. This supports both `torch_dist` checkpoints
  and Megatron-FSDP `fsdp_dtensor` checkpoints, including their optimizer
  state. This path also supports patterns that contain layer families with no
  GPT counterpart, such as Mamba (`M`) positions, which keep their fresh
  initialization.
- **Option B — convert offline to a new checkpoint.** Use
  [`tools/checkpoint/gpt_hybrid_conversion.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/tools/checkpoint/gpt_hybrid_conversion.py)
  to write a standalone `HybridModel` checkpoint whose keys already match the
  hybrid layout. Use this when you need a persisted hybrid checkpoint, an
  architecture-preserving `*-` or `*E` copy, or a target you can inspect before
  training. This path only supports `*-` and `*E` layouts.

### Option A: Translate at load time

Load-time translation is handled by
[`megatron/core/dist_checkpointing/gpt_checkpoint_interop.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/dist_checkpointing/gpt_checkpoint_interop.py).
It triggers automatically when a non-hybrid (GPT) checkpoint is loaded into a
`HybridModel` run: for `torch_dist`, the run's model and optimizer sharded
state dicts are rewritten into the GPT checkpoint's homogeneous-layer format;
for `fsdp_dtensor`, their explicit parameter-name mappings are rewritten onto
the GPT keys before Torch DCP planning. The checkpoint is read directly, and
the weights and optimizer state are resharded to the current
TP/PP/EP/ETP/FSDP layout. No conversion tool is run, and the GPT checkpoint on
disk is never modified.

The reverse mismatch is an error: loading a checkpoint that was saved by a
hybrid run into a non-hybrid run raises a `RuntimeError` that directs you to the
hybrid training entrypoint.

#### Select the checkpoint semantics

When the hybrid run loads a GPT checkpoint, it must set
`--hybrid-layer-pattern` so checkpoint layers can be paired with hybrid layer
positions. Point either `--load` or `--pretrained-checkpoint` at the GPT
checkpoint root (see [Section 3](#3-how-to-train-a-model)).

`--finetune` is optional and retains its normal checkpoint-loading meaning; the
GPT-to-Hybrid translation does not select it on the user's behalf:

- Without `--finetune`, a direct `--load` resumes iteration, optimizer,
  scheduler, RNG, and rerun state according to the normal checkpoint and
  parallel-layout compatibility rules.
- With `--finetune`, iteration, scheduler, RNG, and rerun state restart fresh.
  Model weights still load, and translated optimizer state loads unless
  `--no-load-optim` is set.
- The existing `--pretrained-checkpoint` fallback uses finetuning semantics
  when the `--load` directory contains no checkpoint. Use `--load` directly
  when full resume semantics are desired.

By default the GPT run's **optimizer state is also translated and loaded** —
Adam moments and fp32 master params for the attention and MLP layers carry over,
enabling architecture-preserving continued training. Pass `--no-load-optim` to
skip this and start every layer's optimizer state fresh.

For `torch_dist`, loading optimizer state requires the GPT checkpoint to use a
model-space distributed-optimizer format (`fully_reshardable` or
`fully_sharded_model_space`, i.e. saved with
`--dist-ckpt-optim-fully-reshardable`). The bucket-space formats key optimizer
state by a flat buffer layout that the extra hybrid layers reshuffle, so the
run raises an error directing you to re-save the checkpoint or pass
`--no-load-optim`.

For Megatron FSDP, save and load with `--ckpt-format fsdp_dtensor`. The loader
retargets the explicit DTensor model keys and the model-parameter names used by
the distributed optimizer, so model weights and optimizer state can be
resharded across a different FSDP, TP, EP, or ETP layout during the automatic
GPT-to-Hybrid load. This path is for Megatron FSDP; Torch FSDP2's `torch_dcp`
format is not supported by this automatic translation.

Layers without a GPT counterpart (for example Mamba `M` positions) have no
optimizer state in the checkpoint; their moments start fresh, and the run prints
a warning naming how many layers are affected.

#### Supported patterns and key mapping

The main pattern (the part before any `/` MTP suffix, with `|` pipeline
separators ignored) may contain:

| Symbol | Source of weights |
|--------|-------------------|
| `*` | GPT `self_attention` sub-module of the paired layer |
| `-` or `E` | GPT `mlp` sub-module of the paired layer (MoE tensors also live under `mlp.*`) |
| `M` | No GPT source; the Mamba layer keeps its fresh initialization |

Parameters are paired by occurrence: the *i*-th `*` position takes GPT layer
*i*'s attention, and the *i*-th `-`/`E` position takes GPT layer *i*'s MLP.
`decoder.final_norm` is loaded from GPT's `decoder.final_layernorm`, and
embedding and output weights are copied unchanged.

Because each GPT layer supplies exactly one attention and one MLP sub-module,
the loader rejects a pattern that:

- contains MTP layers (a `/...` suffix), which have no GPT source weights;
- uses a layer type it cannot translate, such as GDN (`G`) or DeepSeek Sparse
  Attention (`D`), whose weight layouts differ from GPT attention;
- mixes dense (`-`) and MoE (`E`) MLP positions in one pattern; or
- has an unequal or zero number of `*` and MLP positions.

The checkpoint's `num_layers` must equal the number of `*` positions in the
pattern; a mismatch is rejected.

```{warning}
Optimizer translation covers the Adam moments and fp32 master params. Pass
`--no-load-optim` for a weights-only load. Use `--finetune` when iteration,
scheduler, RNG, and rerun state should restart instead of resume.
```

### Option B: Convert offline with `gpt_hybrid_conversion.py`

#### Choose an architecture-preserving pattern

For a source checkpoint with *N* GPT layers:

- Use `*-` repeated *N* times for a dense GPT model.
- Use `*E` repeated *N* times for a GPT model whose MLP in every layer is MoE.

The converter maps parameters by occurrence, not merely by numeric layer index:

| Source parameter | Target parameter |
|------------------|------------------|
| Attention from GPT layer *i* | The *i*-th `*` layer |
| MLP or MoE from GPT layer *i* | The *i*-th `-` or `E` layer |
| Embedding and output weights | Copied without changing their model role |
| `decoder.final_layernorm` | Renamed to `decoder.final_norm` |

#### Check the prerequisites

The source must use one of these distributed-checkpoint formats:

- `torch_dist`
- `fsdp_dtensor`

Prefer a top-level checkpoint root containing
`latest_checkpointed_iteration.txt` as `--load-dir`. If `--load-dir` points
directly to a directory containing `metadata.json`, the converter writes a flat
target without a tracker file; the standard training entry point expects a
checkpoint root and tracker.

Run the converter from the repository root in a Megatron environment. A plain
`python` process is sufficient; `torchrun` and a GPU are not required. The tool
gathers full logical tensors on CPU, so the host must have enough memory for the
unsharded source and target model state dicts. Always use a target directory
that is different from the source directory.

```{warning}
This is a weights conversion, not a resumable full training-state conversion.
Sharded optimizer, RNG, rerun, and Transformer Engine `_extra_state` tensors are
not converted. Some non-tensor entries can remain in `common.pt`, but they do
not constitute a converted optimizer or RNG state. Start the converted model
with a fresh optimizer and RNG state.
```

#### Run the conversion

The following example converts a four-layer dense GPT model. Its equivalent
HybridModel has the eight-layer pattern `*-*-*-*-`:

```bash
uv run python tools/checkpoint/gpt_hybrid_conversion.py \
    --direction gpt-to-hybrid \
    --load-dir /path/to/gpt-checkpoints \
    --save-dir /path/to/hybrid-checkpoints \
    --hybrid-layer-pattern '*-*-*-*-' \
    --reset-iterations
```

Always quote the pattern because `*` and `|` have special meaning to a shell.
`--input-format auto` and `--output-format auto` are the defaults: the tool
detects the source backend and writes the same backend. `--reset-iterations`
resets the checkpoint iteration, consumed-sample counters, and cached
`train_iters` and `train_samples`; omit it when the new run must retain that
schedule metadata.

The number of `*` positions and the number of `-` or `E` positions must both
equal the source GPT layer count. The pattern validator rejects GDN, DSA, and
mixed dense/MoE layouts. When cached training arguments are present, the tool
also rejects interleaved MoE, experimental or linear attention, heterogeneous
block specifications, Multi-Latent Attention, and MTP checkpoints. That
source-feature validation is incomplete when `common.pt` has no cached `args`
or an older checkpoint lacks a field, so verify those features manually.

The conversion recognizes standard attention and MLP/MoE state-dict keys only.
Other layer-local tensors are omitted. The documented `hybrid_stack_spec` also
uses Transformer Engine's fused layernorm/linear layout; a local or otherwise
non-TE source layout requires a compatible custom Hybrid stack and key
conversion. Always perform the strict-load check described below.

Do not append an MTP `/...` suffix during conversion. The converter only maps
the main pattern before the first `/`, so it does not create MTP parameters.

When the source path is a checkpoint root with
`latest_checkpointed_iteration.txt`, the output contains an iteration directory
and a matching tracker file. The saved full-shape tensors can be resharded by a
later Megatron load for a different tensor, pipeline, expert, or FSDP layout.

## 3. How to Train a Model

### Update the training command

Start with the command that trained the GPT model and make these changes:

1. Replace `pretrain_gpt.py` with `pretrain_hybrid.py`.
2. Remove `--num-layers` and add the same ordered main-layer symbols used for
   conversion. Pipeline `|` separators may be added or moved. The command-line
   parser derives `num_layers` from the pattern.
3. Select the HybridModel stack specification with
   `--spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_stack_spec`.
4. Point a checkpoint input at the pretrained weights and write new training
   checkpoints to a separate directory:
   - With **Option A (load-time translation)**, point
     `--load` directly at the *GPT* checkpoint for resume semantics, or use
     `--pretrained-checkpoint` for finetuning semantics. The optimizer state is
     loaded by default; add `--no-load-optim` only if you want a fresh
     optimizer. No offline conversion is needed, and `--finetune` is not
     required by the translation.
   - With **Option B (offline conversion)**, point `--pretrained-checkpoint` at
     the converted *hybrid* checkpoint. Set `--ckpt-format` to the converter's
     `torch_dist` or `fsdp_dtensor` output format.

A minimal Option A migration — loading the GPT checkpoint and its optimizer
state directly for architecture-preserving continued training — looks like this:

```diff
- torchrun --nproc_per_node=8 pretrain_gpt.py \
-     --num-layers 4 \
-     --load /path/to/gpt-checkpoints \
-     --save /path/to/gpt-checkpoints
+ torchrun --nproc_per_node=8 pretrain_hybrid.py \
+     --hybrid-layer-pattern '*-*-*-*-' \
+     --spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_stack_spec \
+     --load /path/to/gpt-checkpoints \  # first launch; switch to the save directory later
+     --save /path/to/new-training-checkpoints
```

For Option B, point `--pretrained-checkpoint` at the converted hybrid
checkpoint instead.

Keep the existing architecture, optimizer, precision, data, and basic
TP/DP/EP/CP arguments unless this guide identifies a required change. Review
pattern-driven pipeline layout and GPT-specific dataset features separately.

```{warning}
`pretrain_hybrid.py` does not select `GPTFIMDataset` when `--fim-data` is set.
A GPT training workflow that uses fill-in-the-middle data needs a custom dataset
path or equivalent Hybrid entry-point support before migration.
```

With an empty `--load` directory, `--pretrained-checkpoint` loads the pretrained
weights with finetuning semantics: iteration starts at zero and RNG state is not
restored. For Option A the optimizer state is still warm-started unless
`--no-load-optim` is set (Option B always starts with a fresh optimizer). After
the job writes a checkpoint to `--load`, later launches resume the new
HybridModel training state normally.

### Train from scratch

To initialize every layer from scratch, use the same `pretrain_hybrid.py`,
`--hybrid-layer-pattern`, and `--spec` arguments, but omit
`--pretrained-checkpoint`. Point `--load` and `--save` at the new run directory
if later launches should resume it. Unlike checkpoint conversion, training from
scratch can use compatible layer families supported by the selected HybridModel
stack specification and can include an MTP suffix such as `M*M*/MM/MM`.
Pattern constraints still apply; for example, standard attention `*` and DSA
`D` cannot be used in the same model.

### Account for expanded layer indices

Any setting, mapping, or callback indexed by decoder layer must use HybridModel
indices. For the pattern `*E*E`, source layer 0 attention is Hybrid layer 0 and
its MoE is Hybrid layer 1; source layer 1 attention is Hybrid layer 2 and its
MoE is Hybrid layer 3. Expand attention-only lists with inactive entries for
the intervening MLP or MoE positions.

### Configure pipeline parallelism

For direct `layer_specs`, insert `PipelineSplit()` between logical model
chunks. Chunks use VPP-major, PP-minor order:

```text
chunk index = vp_stage * pipeline_model_parallel_size + pp_rank
```

For example, four chunks with PP=2 and VPP=2 map to `(vp=0, pp=0)`,
`(vp=0, pp=1)`, `(vp=1, pp=0)`, and `(vp=1, pp=1)`, in that order. The number
of chunks must be divisible by the PP size. Their quotient determines VPP; an
explicit `virtual_pipeline_model_parallel_size` must agree. Empty chunks are
allowed, and direct PP configurations with more than one rank require explicit
splits.

Split-defined layouts are mutually exclusive with
`pipeline_model_parallel_layout`, first- or last-stage layer-count overrides,
embedding/loss pipeline accounting, and the conventional per-virtual-stage
controls. The ordinary decoder input embedding is built only for
`(vp=0, pp=0)`, and the output and loss are built only for the final logical
chunk. When MTP is enabled, the existing MTP implementation also keeps its
synchronized embedding replica on that final chunk. `HybridModelBuilder`
supports both PP and VPP for this direct path.

For the legacy string API, add `|` separators without changing the ordered
layer symbols. For example, the converted pattern `*-*-*-*-` can be trained
with two pipeline segments as `*-*-|*-*-`. The number of pipe-delimited
segments must be divisible by `--pipeline-model-parallel-size`, and the same
VPP-major ordering applies.

Legacy validation and ownership rules are intentionally unchanged. In
particular, existing CLI recipes should keep following the combinations that
their current `--hybrid-layer-pattern` workflow accepts. When a legacy pattern
contains `|`, first- and last-stage layer-count overrides remain invalid;
express those segment sizes with additional pipe-delimited symbols instead.

### Update custom providers and conversion mappings

Custom providers and conversion mappings also need to account for these API and
state-dict differences:

- Build or register `HybridModel` instead of `GPTModel`.
- Supply a `hybrid_stack_spec` instead of a GPT transformer-layer spec.
- Prefer `layer_specs` with per-occurrence configs for new programmatic
  providers; use `hybrid_layer_pattern` only for legacy compatibility.
- Set programmatic `num_layers` to the number of direct layer leaves or main
  pattern symbols; unlike the CLI path, a custom provider might not derive it.
- Map attention and MLP/MoE parameters to their separate Hybrid layer indices.
- Use `decoder.final_norm` in HybridModel mappings instead of
  `decoder.final_layernorm`.
- Expand per-layer settings such as attention-window schedules to the full
  HybridModel layer order.

### Validate before scaling up

Before starting a long run:

- Load the converted checkpoint strictly and confirm that no model keys or
  tensor shapes are missing or unexpected.
- For a `*-` or `*E` migration, compare logits on a fixed batch against the
  source GPT model within the expected precision tolerance.
- Run a few training iterations and inspect the loss, gradient norms, and
  parameter counts by layer.
- Save and reload one new checkpoint to confirm that the new optimizer and RNG
  state resume correctly.
