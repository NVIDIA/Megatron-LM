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

Custom layer classes placed in a `HybridStack` must accept
`layer(hidden_states, LayerAuxInputs)` and return `(hidden_states, aux_outputs)`, where
`aux_outputs` is a `LayerAuxOutputs` object. Complete residual layers have `aux_outputs.bias=None`.
The binding decorators have three roles:

- `@bind_layer_interface` binds inputs and normalizes outputs for a complete layer.
- `@bind_layer_input_stage` binds inputs for a split stage, preserving its intermediate output.
- `@bind_layer_output_stage` normalizes a completion stage's output when passed
  `aux_inputs=LayerAuxInputs(...)`; legacy calls keep their natural output.

A `MegatronModule` can keep an ordinary explicit-argument `forward` and opt into
the shared interface:

```python
from megatron.core.transformer.layer_interface import bind_layer_interface
from megatron.core.transformer.module import MegatronModule

class CustomLayer(MegatronModule):
    @bind_layer_interface
    def forward(self, hidden_states, attention_mask=None, inference_context=None):
        # Implement the layer using the named arguments as usual.
        update = self.compute(hidden_states, attention_mask, inference_context)
        return hidden_states + update
```

The decorator prepares the field mapping once, using the forward signature.
Callers pass `layer(hidden_states, aux_inputs)` for every layer; layer implementations
need neither a `forward_layer` method nor manual container unpacking. Only named
fields are bound. A `**kwargs` parameter does not receive all container fields.
Decorated implementations may return a tensor when no auxiliary outputs remain. The helper
supplies `LayerAuxOutputs()`, so callers always unpack the same pair. Existing core
`(hidden_states, bias)` results are normalized to `LayerAuxOutputs(bias=bias)` for
shared calls. Explicit-argument core calls retain their tensor/bias contract.
Module calls accept the container positionally or through `inputs=` or `aux_inputs=`.
The container cannot be mixed with legacy arguments.

Forwarding layers can receive the existing container by declaring `aux_inputs`:

```python
@bind_layer_interface
def forward(self, hidden_states, aux_inputs: LayerAuxInputs):
    return self.inner_layer(hidden_states, aux_inputs)
```

The binder passes that object unchanged as a keyword argument, so hooks and the
`bind_layer_inputs` and `bind_layer_outputs` hooks run once. Shortcut and the hybrid mHC wrapper use
this form, as do HybridStack and MambaLayer. TransformerLayer also accepts explicit
arguments for non-hybrid callers. It carries Hybrid's existing container to attention
and MLP cores without packing legacy calls.
HybridModel constructs one base input
snapshot and composition layers carry it to their children. Scheduling and MTP state
changes derive snapshots; checkpoint replay and graph boundaries reconstruct their inputs.
Shortcut replaces its child CP/recompute state, while mHC clears its wrapper-owned
manager before invoking a complete inner layer.

GPT, ViT, RADIO, BERT, and T5 keep their explicit model-facing arguments.
TransformerBlock forwards them to TransformerLayer without constructing auxiliary-input
containers. `_forward_attention` and `_forward_mlp` retain their legacy execution and
argument lists, with optional `aux_inputs` plumbing for Hybrid calls. GPTModel's MTP call uses
explicit arguments, packed at the MultiTokenPredictionBlock boundary. Hybrid MTP
continues to receive the existing container from HybridModel.

`LayerInterface` defines two overridable binding methods, inherited through
`MegatronModule`:

- `bind_layer_inputs(hidden_states, aux_inputs, **kwargs)` returns `(args, kwargs)`
  for the ordinary forward. It owns validation, normalization, argument renaming,
  and omission, but does not execute the layer.
- `bind_layer_outputs(result)` returns `(hidden_states, LayerAuxOutputs)` and owns
  interpretation of the layer's natural output.

The common execution path is:

```python
args, kwargs = self.bind_layer_inputs(hidden_states, aux_inputs)
result = self._call_with_bound_args(*args, **kwargs)
return self.bind_layer_outputs(result)
```

`_call_with_bound_args` retains normal PyTorch hooks and CUDA-graph dispatch. Direct
`.forward(...)` calls use the same binding hooks around the raw implementation,
retaining PyTorch's usual bypass of module hooks and graph dispatch. Layers do not
receive execution callbacks. Explicit core calls bypass the two binding hooks.
Native container forwards also accept legacy flat keyword calls: their input hook
packs those fields, while the call retains the layer's natural output format.
Direct `.forward(..., aux_inputs=...)` calls retain that natural format.

For example, input adaptation can delegate to the default signature-based binding:

```python
def bind_layer_inputs(self, hidden_states, aux_inputs, **kwargs):
    if aux_inputs.residual_stream_recompute_context is not None:
        raise TypeError("This layer does not support residual-stream recomputation.")
    args, bound = super().bind_layer_inputs(hidden_states, aux_inputs, **kwargs)
    if bound.get("context") is None:
        bound.pop("context", None)
    return args, bound
```

Forward `**kwargs` when delegating: the optional `forward` argument selects the
method signature for a direct call or split stage. Normal module calls bind against
`self.forward`. The default output hook accepts a tensor, a tensor/bias pair, or an
already normalized output pair. Layers never return input containers. For example,
a wrapper returning an uncontracted mHC stream can name that auxiliary output:

```python
def bind_layer_outputs(self, result):
    hidden_states, stream = result
    return hidden_states, LayerAuxOutputs(mhc_multistream=stream)
```

TransformerLayer uses the default input hook to expose the existing arguments and
carry Hybrid's container. Its local and TE graph hooks remove the container after
binding. TE replay drops `None` keywords centrally after argument preparation, so
required attention masks can be synthesized first. TransformerLayer's TE argument
preparation also removes non-hash token IDs to match the captured signature.

The container and layer forwards both use `packed_sequence_cp_metadata`, so this
field binds directly by name. A forwarding method with `*args, **kwargs` must expose
a concrete signature, for example by placing `@copy_signature(the_forwarded_method)`
beneath `@bind_layer_interface`. Subclasses inherit the binding when they inherit
`forward`; an overridden `forward` should declare its own binding. Native
`forward(x, aux_inputs)` methods continue to work without a decorator and must return
the output pair themselves.

All shared calls use `LayerOutput = tuple[Tensor, LayerAuxOutputs]`:

- `bias`: deferred output-projection bias, for dropout/residual fusion.
- `mhc_multistream`: the pre-contraction decoder stream consumed by MTP.

Only `bias` is additive. Cross-attention context remains an unchanged input;
the mHC stream is an auxiliary output consumed separately. MoE combines its routed and shared expert
contributions internally; auxiliary losses and recurrent caches remain internal.

```python
hidden_states, aux_outputs = layer(hidden_states, aux_inputs)
```

The model constructs the shared inputs. Layers consume them without mutating or
returning them. Inputs are immutable: `aux_inputs.update(...)` derives a new snapshot
only when a field changes by identity. Scheduling callers use it for scoped invocation
settings; MTP uses it to advance its own depth state.

Transformer layers retain their natural `(output, context)` result for existing
execution paths; the output hook exposes only the computed hidden states. HybridStack's
output binder wraps its natural MTP stream in `LayerAuxOutputs(mhc_multistream=...)`.
Wrapper forwards produce outputs without carrying or restoring child input containers.

CrossAttention's `bind_layer_inputs` override maps `context` to
`key_value_states` and `context_mask` to `attention_mask` before hooks and execution.
It inherits the ordinary attention forward and its shared binding. Its projection bias occupies `aux_outputs.bias`.

Residual `write` operations still consume the internal projection pair
`(output, bias)`; they do not consume auxiliary state. A shared layer result can
be passed to a residual write as `(hidden_states, aux_outputs.bias)`.

`ShortcutMoEBlock` also accepts `layer(hidden_states, aux_inputs)` and returns
`(hidden_states, LayerAuxOutputs())`. Composite layers receive per-invocation scheduling state
through `LayerAuxInputs.execution_context`: a `LayerExecutionContext` containing the
quantization context factory, CP layout state, and physical-layer residual replay
plan. HybridStack supplies this context; direct calls to shortcut blocks
must supply it too. The block selects the contexts for both layers of its pair.
Shortcut scheduling arguments are supplied only through this context. Its
implementation derives call-local layout and recompute inputs for its compute layer;
those inputs do not escape the call.

Transformer and Mamba pre-core stages accept `(hidden_states, aux_inputs)` through
`@bind_layer_input_stage`. It calls the same `bind_layer_inputs` hook, selecting the
stage's signature through the `forward` argument. Intermediate state stays opaque:
pass the entire result to `forward_post_core_attn(*state, aux_inputs=aux_inputs)`
for a bound output. Legacy calls can pass `*state` alone. No separate stage-specific binding
hook is needed.

Unused attention/rotary/padding arguments have been removed from Mamba's pre-core
signature; shared callers can continue supplying them in `LayerAuxInputs`, and the
binding selects only the arguments the stage consumes.

MambaLayer passes its existing auxiliary inputs directly to the mixer. Default binding
ignores packed-sequence CP metadata for MambaMixer and GDN, whose forward signatures
do not consume it; GDP declares the field and receives it. The explicit split-stage
mixer call still forwards the metadata only when `chunkwise_context_parallel` is enabled.

Completed Transformer and Mamba split stages return `(hidden_states, aux_outputs)`
when passed `aux_inputs`. Their `forward_post_core_attn` methods use
`@bind_layer_output_stage`, which calls the same `bind_layer_outputs(result)` hook.
Without `aux_inputs`, they return their natural legacy output.
Shortcut consumes the completed output pair directly. Core projection stages retain
their internal `(output, bias)` contract for residual fusion.

TransformerBlock invokes layers with explicit arguments, including the optional mHC
manager. Cross-attention reads context from its inputs without replacing it.
Mamba carries the same input object to its mixer without inspecting the mixer configuration.

Stack and recompute callers pass available CP metadata in `LayerAuxInputs` without
inspecting layer types. Mamba's mixer call consumes it only in chunkwise mode.
Mamba's pre-core stage applies the same chunkwise-only policy. Transformer pre-core
stages do not declare CP metadata and ignore it in shared calls. Shortcut orchestration
does not select metadata for individual layer types.

MTP's `prepare_cp_layout` accepts hidden states and the existing `LayerAuxInputs`,
with labels, loss mask, and CP batch supplied separately. It returns
`(hidden_states, labels, loss_mask, aux_inputs)`, updating auxiliary fields only when
layout conversion is required. HybridModel passes the returned inputs directly to MTP.

MTP's `_proj_and_transformer_layer` and `_checkpointed_forward` accept
`(hidden_states, aux_inputs, *, decoder_input=...)` and forward the shared inputs to
the nested layer. The MTP block explicitly advances token IDs, positions, masks, or
precomputed embeddings before calling each depth as `layer(hidden_states, aux_inputs)`.
The block retains the embedding module for subsequent depths, but clears `embedding`
from the inputs passed to each depth so the module never enters an inner layer's graph inputs.
Each depth processes the prepared inputs and returns only its hidden states. Computed
embeddings stay local to the depth; precomputed embeddings advance without carrying
depth-local padding fills forward. The single-position inference entry point still
constructs its input snapshot at that boundary.

The stack-level `checkpointed_forward` helper accepts Hybrid's auxiliary inputs or
TransformerBlock's explicit keyword arguments and returns hidden states and extracted
intermediate tensors. Checkpoint primitives receive input tensors explicitly; replay
preserves the caller's input representation while non-tensor metadata stays in the closure.
Layers never return updated input fields through the checkpoint.

Transformer's raw attention/MLP helpers and `_run_mlp` carry `LayerAuxInputs` through
to their cores. Chunked MLP execution replaces only the mask and token-ID slices.
The `_forward_attention` and `_forward_mlp` entry points used by graph and pipeline
schedulers retain their explicit tensor arguments. Only Hybrid calls supply a container;
legacy calls continue invoking cores with explicit arguments. Checkpoint callbacks
reconstruct the core's inputs from the existing checkpoint arguments instead of retaining
unrelated Hybrid tensors. Local/TE graphs and split-stage overlap retain their explicit
execution interfaces.

This is transitional plumbing: a future PR will have HybridModel call the underlying
layers' bound forwards directly, removing the TransformerLayer adapter.

Mamba's mixer and Hybrid's eager self-attention and MLP/MoE calls use shared inputs.
Dense MLP bindings ignore routing fields; MoE bindings select the fields they consume.
Chunked MLP execution slices token IDs and padding masks alongside hidden states.
Checkpoint and residual-fusion boundaries extract `aux_outputs.bias` and retain tensor
outputs. TE partial MoE capture keeps its existing flat intermediate-output contract.

`MegatronModule` binds shared inputs before module hooks and CUDA-graph selection.
Hooks see the ordinary forward arguments and the forward's result; output binding
happens after execution or graph replay. TransformerLayer's forward returns
`(hidden_states, LayerAuxOutputs())` when `aux_inputs` is present, including an explicit
`aux_inputs=` call, and its legacy `(hidden_states, context)` otherwise. The outer
binding preserves an already-bound result. Graph replay hooks make the same distinction
for completed calls. The split-stage completion returns only hidden states, which
`@bind_layer_output_stage` wraps in the standard output format when passed `aux_inputs`;
TransformerLayer does not override `bind_layer_outputs`. HyperConnectionTransformerLayer
extracts its per-call recompute manager in `__call__` before binding and forwards an
auxiliary-input snapshot with that field cleared. It does not override
`bind_layer_inputs` or `forward`.
Direct `.forward(...)` calls also accept
the container, but bypass module hooks and graph selection as usual. Graphable
modules use `_call_with_bound_args` only to dispatch already-bound calls to the local
or TE graph hooks, or ordinary execution. TransformerLayer has no dispatch override;
graph-specific adaptation belongs in those hooks. HybridStack consumes and clears its
CP layout dictionary and plan before invoking layers; MTP keeps its embedding module
outside depth inputs.
Mamba carries the prepared auxiliary inputs directly into local graphs without a
dispatch override. Its TE capture hook constructs auxiliary inputs from TE's flat
tensor signature; replay exposes only consumed execution state, retaining the
non-tensor-input validation. Transformer retains its local graph input filtering.
TE capture uses explicit tensor arguments and preserves its flat
intermediate outputs. TE replay returns named outputs for completed packed calls and
the natural hidden-state/context pair for legacy calls. The shared binding does not
inspect graph dispatch or interpret capture intermediates. EP-overlap replay retains its scheduler-specific
intermediate outputs. The vision graph wrapper is unchanged.

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
