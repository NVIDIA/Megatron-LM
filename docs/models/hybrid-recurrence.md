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

`pretrain_hybrid.py` supplies the checkpointed `TrainState.iteration` automatically.
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
Each iteration replaces the state with the current body's output:

```text
s = prefix(embeddings)
for body in recurrent_regions:
    repeat k times:
        s = body(s)
logits = head(final_norm(suffix(s)))
```

Backpropagation traverses every iteration. Contributions to a shared parameter
accumulate into that parameter's gradient. At one loop the computation is the
ordinary hybrid decoder with the colons removed. There is no input re-injection,
extra normalization, learned state adapter, gradient truncation, loop conditioning,
or auxiliary loss imposed by this implementation. Layer behavior remains defined
by the chosen static module specification.

## Restarting with a different policy

Recurrence changes the execution schedule, not module registration. Each physical
layer remains registered exactly once under `decoder.layers.<index>`. Ordinary
and distributed state dictionaries have the same layer keys and tensor shapes as
the corresponding pattern without colons. A non-recurrent checkpoint can therefore
seed recurrent training without converting its weights or optimizer layout.

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
- Cached inference: attention KV caches and state-space caches need distinct state
  per logical loop, which is not implemented. Full-sequence evaluation without an
  inference context is supported.

MTP patterns after `/` retain their existing semantics and are not repeated.
Unmarked HybridModel configurations retain their existing behavior. Throughput
estimates based on unique layer counts do not include recurrent execution work.

## Local tests

```bash
uv run python -m torch.distributed.run --standalone --nproc-per-node 1 -m pytest -q \
  tests/unit_tests/models/hybrid/test_recurrence.py \
  tests/unit_tests/ssm/test_hybrid_layer_allocation.py
```

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
