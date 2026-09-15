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
ordered stack in which every position represents one layer family. The order is
described by `--hybrid-layer-pattern`:

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

## 2. How to Convert a Checkpoint

Use
[`tools/checkpoint/gpt_hybrid_conversion.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/tools/checkpoint/gpt_hybrid_conversion.py)
to convert a `GPTModel` checkpoint directly to `HybridModel` state-dict keys.

### Choose an architecture-preserving pattern

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

### Check the prerequisites

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

### Run the conversion

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
4. Load the converted weights with a fresh optimizer and write new training
   checkpoints to a separate directory. Set `--ckpt-format` to the converter's
   `torch_dist` or `fsdp_dtensor` output format.

A minimal migration of the model and checkpoint arguments looks like this:

```diff
- torchrun --nproc_per_node=8 pretrain_gpt.py \
-     --num-layers 4 \
-     --load /path/to/gpt-checkpoints \
-     --save /path/to/gpt-checkpoints
+ torchrun --nproc_per_node=8 pretrain_hybrid.py \
+     --hybrid-layer-pattern '*-*-*-*-' \
+     --pretrained-checkpoint /path/to/hybrid-checkpoints \  # first-time only
+     --load /path/to/new-training-checkpoints \
+     --save /path/to/new-training-checkpoints
```

Keep the existing architecture, optimizer, precision, data, and basic
TP/DP/EP/CP arguments unless this guide identifies a required change. Review
pattern-driven pipeline layout and GPT-specific dataset features separately.

```{warning}
`pretrain_hybrid.py` does not select `GPTFIMDataset` when `--fim-data` is set.
A GPT training workflow that uses fill-in-the-middle data needs a custom dataset
path or equivalent Hybrid entry-point support before migration.
```

With an empty `--load` directory, `--pretrained-checkpoint` loads the converted
weights with finetuning semantics: iteration starts at zero, and optimizer and
RNG state are not restored. After the job writes a checkpoint to `--load`, later
launches resume the new HybridModel training state normally.

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

For pipeline parallelism, add `|` separators without changing the ordered layer
symbols. For example, the converted pattern `*-*-*-*-` can be trained with two
pipeline segments as `*-*-|*-*-`. The number of pipe-delimited segments must be
divisible by `--pipeline-model-parallel-size`.

The pattern replaces conventional pipeline layout controls. Remove
`--num-layers-per-virtual-pipeline-stage`,
`--num-virtual-stages-per-pipeline-rank`, `--pipeline-model-parallel-layout`,
`--account-for-embedding-in-pipeline-split`, and
`--account-for-loss-in-pipeline-split`. When the pattern contains `|`, also
remove `--decoder-first-pipeline-num-layers` and
`--decoder-last-pipeline-num-layers`. Express virtual-pipeline segmentation
with additional pipe-delimited segments instead.

The declarative `HybridModelBuilder` currently rejects virtual pipeline
parallelism. Pipe-defined virtual stages are supported by the
`pretrain_hybrid.py` CLI builder, but custom builder users must avoid VPP or use
a path that explicitly supports it.

### Update custom providers and conversion mappings

Custom providers and conversion mappings also need to account for these API and
state-dict differences:

- Build or register `HybridModel` instead of `GPTModel`.
- Supply a `hybrid_stack_spec` instead of a GPT transformer-layer spec.
- Set programmatic `num_layers` to the number of layer symbols in the main
  pattern; unlike the CLI path, a custom provider might not derive it.
- Map attention and MLP/MoE parameters to their separate Hybrid layer indices.
- Use `decoder.final_norm` in HybridModel mappings instead of
  `decoder.final_layernorm`.
- Expand per-layer settings such as attention-window schedules to the full
  HybridModel pattern.

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
