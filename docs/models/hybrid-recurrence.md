# Depth recurrence in HybridModel

Depth recurrence executes regions of the decoder several times using the same
parameters. For example, a four-layer decoder with the pattern `M-:*:-` executes
physical layers `0,1,2,2,2,3` when the loop count is three. The embedding, prefix,
suffix, final normalization, and language-model head each execute once.

## Configuration

Mark recurrent regions with colons in `--hybrid-layer-pattern`:

```bash
--hybrid-layer-pattern 'G-:G-*G-:G-'
--n-loops-distribution '1:0.25,2:0.25,4:0.25,8:0.25'
--spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_stack_spec
```

Use the existing layer symbols: `G` for Gated DeltaNet, `*` for attention, and `-`
for a dense MLP, among others. The general pattern is
`prefix:body1:body2:...:suffix`. Every interior section is a separate recurrent
region and must contain at least one layer. Regions may have different lengths;
either outer section may be empty. For example, `:*-:` repeats the entire decoder,
while `G-:G-:G-*G-:G-` repeats two differently sized regions. Empty interior sections,
such as `G-:G-::G-`, are invalid.

All regions use the same sampled loop count `k` for a training minibatch. They execute
in order: `prefix`, `body1` repeated `k` times, `body2` repeated `k` times, then
`suffix`. This is not a loop over the concatenation of the bodies. Each body has
its own parameters, shared across that body's iterations, not with other bodies.

Colons are not layers and do not contribute to `num_layers`, layer numbering,
parameter counts, or checkpoint tensor shapes.

`--n-loops-distribution` is a comma-separated list of `count:weight` pairs.
Counts must be distinct positive integers. Weights must be finite and positive;
they are automatically normalized to sum to one. For example, `1:1,2:3` becomes
probabilities `0.25` and `0.75`, and `1:25,2:25,4:25,8:25` gives equal probability
to all four counts. Already normalized probabilities are accepted as well.
A fixed count uses a single pair, e.g., `4:1.0`. Omitting the distribution runs
one loop. A distribution without loop boundaries is an error.

The loop count is selected with Python's standard-library RNG:
`random.Random(step).choices(counts, weights=probabilities, k=1)[0]`. Each call
creates a local RNG seeded with the logical training minibatch step. Every rank,
every gradient-accumulation microbatch in that step, and every recurrent region
uses the same count. Sampling performs no communication or device synchronization
and does not read, advance, or reseed the global Python or PyTorch RNGs. Microbatch
IDs and local forward counters do not affect sampling.

The step alone determines the draw; no separate seed or persistent sampler state
is needed. Restarting or retrying a step with the same policy reproduces its count.
Changing the distribution changes how that step's draw maps to a count.

`pretrain_hybrid.py`, ModelOpt's `finetune.py`, and `pretrain_hybrid_flex.py` supply
`recurrence_step` from the checkpointed `TrainState.iteration`.
Direct Core callers must pass a non-negative integer `recurrence_step` when training
with a stochastic distribution, unless they explicitly override `n_loops`. Keep the
step constant across all microbatches making up one minibatch, and advance it only
when advancing to the next training step. Fixed distributions and evaluation need
no step. For example:

```python
for input_ids, position_ids, attention_mask, labels in minibatch:
    loss = model(
        input_ids, position_ids, attention_mask,
        labels=labels, recurrence_step=step,
    )
    loss.mean().backward()
```

Use the same step and distribution on all participating ranks. Supplying an explicit
step rather than deriving it from forward-call order also permits future pipeline
stages to reproduce the same count without communicating the sampled value.

`--inference-n-loops` sets a fixed positive loop count for both inference and
evaluation (`model.eval()`). It does not affect training sampling. Its default is
`None` (omit the flag), which uses the largest count in the shared training
distribution without sampling. The same count applies to every recurrent region.
For example, `--inference-n-loops 2` evaluates at two loops even when the training
distribution includes larger counts.

The precedence for evaluation and inference is:

1. Explicit forward argument `n_loops`.
2. Configured `inference_n_loops`, if not `None`.
3. Largest training-distribution count.

Direct Core callers can set `HybridModel(..., inference_n_loops=2)` or use a
per-forward override, e.g.:

```python
model.eval()
with torch.no_grad():
    logits = model(input_ids, position_ids, attention_mask, n_loops=2)
```

The override applies to every region for this forward only and does not modify
the training policy. Use the same override on every participating rank and throughout
a training minibatch's accumulation microbatches. Selecting an inference loop count
does not enable cached decoding; inference contexts remain unsupported.

## State and gradients

The first region starts from the prefix output (or the embeddings when the prefix
is empty). Each subsequent region starts from the preceding region's final output.
With the ordinary `hybrid_stack_spec`, each iteration replaces the state with the
current body's output:

```text
s = prefix(embeddings)
for body in recurrent_regions:
    repeat k times:
        s = body(s)
logits = head(final_norm(suffix(s)))
```

Backpropagation traverses every iteration. Contributions to a shared parameter
accumulate into that parameter's gradient. At one loop the computation is the
ordinary hybrid decoder with the colons removed when using the ordinary stack
spec. It imposes no input re-injection or learned update rule. Layer behavior and
optional gating remain defined by the chosen static module specification. Neither
variant truncates gradients or introduces an auxiliary loss.

## Adaptive-delta gate

Choose a static gated stack spec to use the adaptive-delta rule of ELR/Koi §3.1:

```bash
--spec megatron.core.models.hybrid.hybrid_layer_specs adaptive_delta_hybrid_stack_spec
--hybrid-layer-pattern 'G-:G-*G-:G-'
--n-loops-distribution '1:1.0'
```

For a GDP backbone, select `adaptive_delta_gated_delta_product_stack_spec` instead;
it places GDP in the `M` slot, so use an appropriate `M`-containing layer pattern.
The `G` slot still denotes Gated DeltaNet. Custom specs can explicitly compose `HybridStackSubmodules.recurrence_gate` with
`ModuleSpec(module=AdaptiveDeltaGate)`. The existing ordinary stack specs are
unchanged. MTP uses ordinary, ungated submodules in the supplied gated specs.

Each region has an independent gate and retains its own original entry tensor `h`:

```text
s = h
repeat k times:
    u = s + h
    r = body(u)
    beta = sigmoid(W_beta @ u + b_beta) * softplus(gamma)
    s = s + beta * (r - s)
```

W_beta, b_beta, and gamma are shared across all passes of that region, but not
across regions. All three start at zero, giving an initial rate of `log(2)/2`
(approximately 0.35). The projection uses the model's parameter dtype; the nonlinear
rate and delta update are computed in FP32, and gamma stays FP32 under the standard
Float16Module wrapper. Gate parameters are replicated across TP ranks; with sequence
parallelism they participate in the replicated-parameter TP gradient reduction.

This gate is not bounded to `[0, 1]`, does not impose nGPT/unit-L2 normalization,
and does not detach any earlier loop. Gated specs currently require ordinary
residual width (no wide residuals or mHC). Even one gated pass differs from the
ordinary decoder: its body receives `2*h`, and its output is mixed with `h`.
One-loop adaptation training is therefore expected after conversion.

## Loop-conditioned modulation

Choose `loop_conditioned_adaptive_delta_hybrid_stack_spec` for the adaptive-delta
gate together with ELR §3.3 conditioning. The GDP variant is
`loop_conditioned_adaptive_delta_gated_delta_product_stack_spec`, with GDP in the
`M` slot. Matching `_inference_stack_spec` variants support full-sequence,
uncached evaluation with the inference-optimized implementation.

```bash
--spec megatron.core.models.hybrid.hybrid_layer_specs loop_conditioned_adaptive_delta_hybrid_stack_spec
--hybrid-layer-pattern 'G-:G-*G-:G-'
--n-loops-distribution '1:1,2:1,4:1'
--loop-conditioning-dim 256
--loop-conditioning-rank 256
--loop-conditioning-max-loops 4
```

For pass `i` of `k`, using one-based indices, a shared two-layer SiLU MLP embeds
progress `i/k` and budget `k/K`. Each scalar has a sinusoidal embedding of
`loop_conditioning_dim` channels. `K` is `loop_conditioning_max_loops`, a fixed
reference scale rather than an execution limit. Counts greater than `K` are
allowed. Keep this reference fixed when changing the training distribution or
evaluation depth; it is stored and validated in checkpoints. The embedding uses
exponentially spaced frequencies from 1 to approximately 1/10,000 cycles per unit;
the paper does not prescribe a frequency convention.

The prefix is unconditioned. Every recurrent and suffix layer has its own linear
modulation heads and low-rank projections, shared across that layer's passes. For
a pre-normalized residual branch, the update is:

```text
c_i = MLP(concat(sinusoidal(i/k), sinusoidal(k/K)))
z = (1 + lambda(c_i)) * Norm(x) + mu(c_i)
y = z + A @ (g(c_i) * (B @ z))
x_out = x + (1 + omega(c_i)) * f(y)
```

`Norm` retains the backbone's normalization and learned weights. `B` projects from
hidden width to `loop_conditioning_rank`; `A` projects back. Scaling applies to
the complete branch output, including any projection bias, before residual
addition. Attention, MLP, MoE, and Mamba/GDP branches use the same rule. The supplied
recipes separate fused normalization/projection modules to expose the normalized
input. The final stack norm and vocabulary projection are unchanged.

Each region also has independent entry and exit heads. With `h` denoting that
region's original input and `u_i = s_(i-1) + h`:

```text
z_1 = u_1
z_i = (1 + lambda_entry(c_i)) * RMSNorm(u_i)
      + mu_entry(c_i) + eta(c_i) * h                 # i > 1
r_i = body(z_i, c_i)
beta_i = sigmoid(W_beta @ u_i + b_beta) * softplus(gamma)
s_i = s_(i-1) + (pi_i + g_exit(c_i)) * beta_i * (r_i - s_(i-1))
pi_1 = 1; pi_i = 0 for i > 1
```

The adaptive gate still sees the unmodulated `u_i`. Progress resets to `1/k` at
each region, all regions share the same encoder and budget, and each body finishes
all its passes before the next starts. The suffix uses `c_k`, corresponding to
progress 1. MTP remains unconditioned and executes once.

All modulation heads start at zero; the encoder and low-rank bases start with
random weights. Consequently, the first pass reproduces the gate-only update,
and later passes have zero exit factors. With dropout disabled, this initially
reproduces the adapted one-loop function at every requested depth in exact
arithmetic. Floating-point kernel changes can introduce rounding differences.
Backpropagation includes every pass; later exit heads can learn immediately, and
zero-gradient paths keep all conditioning parameters participating even at one
loop. Conditioning tensors are call-local, not cached module state.

Conditioning parameters are replicated across tensor-parallel ranks. Like
Megatron's duplicated linear layers, CUDA initialization uses a shared RNG stream;
CPU initialization uses the standard shared CPU seed. No parameter broadcast is
needed. With sequence parallelism, the existing replicated-parameter gradient
reduction applies. Frequencies remain FP32 under the standard Float16Module wrapper.
At hidden width 3072, conditioning dimension/rank 256, one recurrent region, and
24 recurrent plus 14 output layers, conditioning adds 155,627,520 parameters,
matching the paper's rounded 155.6M figure.

## Restarting with a different policy

Recurrence changes the execution schedule, not module registration. Each physical
layer remains registered exactly once under `decoder.layers.<index>`. Without
conditioning, backbone layer keys and tensor shapes match the corresponding
pattern without colons. A non-recurrent checkpoint can therefore seed ordinary,
ungated recurrent training without converting weights or optimizer layout. Adding
gates or conditioning introduces new parameters and requires an explicit import.

Specify the desired distribution on **every restart**, including when using
`--use-checkpoint-args`. The distribution is a runtime policy, not a model weight,
and is deliberately not restored by the legacy checkpoint-argument loader. For
example, start with `--n-loops-distribution '1:1.0'`, then resume the same architecture
with `--n-loops-distribution '1:0.5,4:0.5'`. Without an explicit policy the default
remains one loop. Retain normal optimizer, scheduler, training-step, and RNG loading
when continuing training rather than using a weights-only finetune. Restoring the
training step is sufficient to reproduce loop sampling; normal RNG restoration
remains necessary for dropout and other stochastic model operations.

`--inference-n-loops` is also a runtime override, not an architectural change. The
legacy checkpoint-argument loader does not restore it. Supply the desired setting
for each run, or omit it to use the largest training loop count.

### Importing an ordinary checkpoint into a gated model

For the first import, keep the same physical layers and add recurrence boundaries,
choose the appropriate gated spec, and explicitly initialize the new parameters:

```bash
--load /path/to/ordinary/checkpoints
--use-checkpoint-args
--spec megatron.core.models.hybrid.hybrid_layer_specs adaptive_delta_hybrid_stack_spec
--hybrid-layer-pattern 'G-:G-*G-:G-'
--n-loops-distribution '1:1.0'
--init-recurrence-gates
--finetune
--save /path/to/gated-warmup/checkpoints
```

This supports persistent `torch` and `torch_dist` checkpoints. Existing backbone
keys/shapes remain unchanged, and only the new gate tensors are initialized to
zero. Missing backbone weights, partial gate checkpoints, or incompatible shapes
remain errors; the importer does not relax the model load to `strict=False`.
With `--use-checkpoint-args`, the explicitly requested gated spec takes precedence
over the source checkpoint's ordinary spec during this import.

Adding gate parameters changes the optimizer layout. This first import therefore
requires a fresh optimizer, explicitly selected with `--finetune` or
`--no-load-optim`; it does not attempt to remap old optimizer slots. `--finetune`
also resets iteration, scheduler, RNG, and data progress. `--no-load-optim` retains
the standard non-finetune progress/RNG behavior, but the standard loader also skips
the old LR scheduler in that mode: supply the intended warmup schedule. GPT-to-Hybrid
optimizer restoration must be explicitly disabled with `--no-load-optim`.
Remove `--ckpt-assume-constant-structure` for this first import so the new save plan
cannot reuse the ungated checkpoint's layout. Old optimizer main-parameter imports
(`--load-main-params-from-ckpt`) are not supported during this conversion.

After saving the gated warmup checkpoint, **remove `--init-recurrence-gates`** and
use a normal resume to retain learned gate weights and optimizer state. Leaving the
flag enabled on a gated checkpoint raises an error rather than overwriting trained
gates. Keep the trained recurrence-region boundaries and gated architecture on
resume; changing boundaries requires an explicit migration.

Direct Core callers importing ordinary state dictionaries can use
`initialize_recurrence_gate_state(model, state_dict)` from
`megatron.core.models.hybrid.recurrence_gate`, followed by
`model.load_state_dict(state_dict, strict=True)` and a new optimizer. Distributed
callers must first validate source keys with `validate_ungated_checkpoint_keys`,
then supply load-only metadata `initialize_recurrence_gates=True` when constructing
the target sharded model state. Never use that metadata when saving.

### Importing a gate-only checkpoint into a conditioned model

Select the matching conditioned recipe and initialize only the new conditioning
modules:

```bash
--load /path/to/gated-warmup/checkpoints
--use-checkpoint-args
--spec megatron.core.models.hybrid.hybrid_layer_specs loop_conditioned_adaptive_delta_hybrid_stack_spec
--hybrid-layer-pattern 'G-:G-*G-:G-'
--n-loops-distribution '1:1,2:1,4:1'
--loop-conditioning-dim 256
--loop-conditioning-rank 256
--loop-conditioning-max-loops 4
--init-loop-conditioning
--finetune
--save /path/to/conditioned/checkpoints
```

Persistent `torch` and `torch_dist` imports load the backbone and learned gates
strictly. The ordinary importer remaps fused normalization weights into their
separate modules; distributed tensor storage keys stay compatible. New norm
auxiliary state is initialized, not backbone weights. Missing or incompatible
backbone/gate tensors remain errors. With `--use-checkpoint-args`, the explicitly
selected conditioned spec and conditioning dimensions take precedence for this
import.

Adding conditioning changes parameter registration and optimizer layout. The
same fresh-optimizer requirements, scheduler/progress caveats, and restrictions
on constant-structure/main-parameter imports as the gate import apply. Use
`--finetune` or `--no-load-optim`. Add `--init-recurrence-gates` only if the source
also lacks gates. A checkpoint that already contains conditioning rejects
`--init-loop-conditioning`, protecting learned parameters from reinitialization.

After saving, remove initialization flags and resume normally, preserving both
gates and conditioning optimizer state. With `--use-checkpoint-args`, normal resume
restores conditioning dimension, rank, and reference budget. A conditioned
checkpoint cannot be silently loaded into an unconditioned recipe.

Direct Core callers can apply `initialize_loop_conditioning_state(model, state)`
from `megatron.core.models.hybrid.loop_conditioning` before a strict ordinary load
and construct a fresh optimizer. Distributed callers first validate source keys
with `validate_unconditioned_checkpoint_keys`, then construct the target state with
load-only metadata `initialize_loop_conditioning=True`. Never use this metadata
when saving.

## Initial scope

The implementation supports multiple recurrent regions of different lengths with
one shared loop-count policy, eager execution, full backpropagation, and Megatron
DDP. Separate per-region loop counts or distributions (including pipe-delimited
distribution syntax) are not exposed yet. Parsing stores physical ranges separately
from layer configs, and execution uses indices rather than replicated modules,
leaving room for later schedules without changing weight ownership.

The following combinations fail explicitly rather than silently executing the
wrong graph:

- Pipeline/virtual-pipeline parallelism, pipeline `|` boundaries, and partial model stages.
- Context parallelism and dynamic context parallelism.
- Activation recomputation, activation offloading, and paged stashing.
- CUDA graphs, FP8/FP4 quantization, weight rematerialization, and delayed weight gradients.
- Shortcut MoE, expert communication overlap, and FSDP wrappers.
- MoE routing replay inside a recurrent region. MoE layers in the prefix or suffix
  still execute once and can use routing replay.
- Cached inference: attention KV caches and state-space caches need distinct state
  per logical loop, which is not implemented. Full-sequence evaluation without an
  inference context is supported.

Conditioned recipes additionally reject fused residual updates and cross-attention.
MTP patterns after `/` retain their existing semantics and are not repeated.
Unmarked HybridModel configurations retain their existing behavior. Throughput
estimates based on unique layer counts do not include recurrent execution work.

## Local tests

```bash
uv run python -m torch.distributed.run --standalone --nproc-per-node 1 -m pytest -q \
  tests/unit_tests/models/hybrid/test_recurrence.py \
  tests/unit_tests/models/hybrid/test_recurrence_gate.py \
  tests/unit_tests/models/hybrid/test_loop_conditioning.py \
  tests/unit_tests/ssm/test_hybrid_layer_allocation.py
```

GDP backward kernels may require executable temporary storage for JIT compilation.
If `/tmp` is mounted `noexec`, set `TMPDIR` to an executable scratch directory.

Tests cover parsing, distribution validation, stateless step replay without changing
global RNG state or using collectives, CLI-to-builder and training-step propagation,
and inference/evaluation loop-count precedence. They verify ordinary and distributed
checkpoints saved without recurrence boundaries load into models
with boundaries. The ordinary checkpoint test also restores AdamW optimizer state,
the LR scheduler, and training progress, reproduces a one-loop optimizer update, and
then trains at greater depth. Both unroll comparisons are retained: one explicitly
calls shared layer instances, while the other builds a separate layer instance for
every logical execution. The independent-instance reference compares outputs and
input gradients, and sums the copies' parameter gradients to compare with the
recurrent model's shared gradients. It reuses the recurrent model across changing
loop counts and sequence lengths to expose stale state. These checks cover FP32
and BF16, attention/MLP and GDN/MLP bodies, and multiple recurrent regions.

Other single-/multi-region tests verify one shared count across accumulation
microbatches and regions, with each body finishing its loops before the next starts.
Gate tests independently check the formula/initialization, gradients against
independent layer and gate copies, gated DDP accumulation, and FP32 parameter
retention. They cover explicit ordinary/distributed ungated imports, strict failure
cases, ordinary optimizer continuation after a gated save, and both training
checkpoint-loader formats. The existing ungated unroll tests are retained unchanged.
Conditioning tests check the embedding/entry/exit/branch equations and gradients,
zero-head imports at multiple depths, independent per-pass model copies, interleaved
forward graphs, scope cleanup after exceptions, DDP accumulation, GDP and MoE
backward participation, ordinary/distributed imports and resumes, optimizer
continuation, reference-budget validation, and the Koi parameter count.
