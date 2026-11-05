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

All regions use the same sampled loop count `k` for a forward pass. They execute
in order: `prefix`, `body1` repeated `k` times, `body2` repeated `k` times, then
`suffix`. This is not a loop over the concatenation of the bodies. Each body has
its own parameters, shared across that body's iterations, not with other bodies.

Colons are not layers and do not contribute to `num_layers`, layer numbering,
parameter counts, or checkpoint tensor shapes.

`--n-loops-distribution` is a comma-separated list of `count:probability` pairs.
Counts must be distinct positive integers. Probabilities must be finite, positive,
and sum to one (within an absolute tolerance of `1e-6`). A fixed count uses a
single pair, e.g., `4:1.0`. Omitting the distribution runs one loop. A distribution
without loop boundaries is an error.

Each training forward samples one count for the whole microbatch and applies it
to every recurrent region. The count is broadcast across the explicit
`pg_collection.tp_dp_cp` process group so all tensor, data, and expert parallel
ranks execute the same depth. Stochastic recurrence
requires that group to be supplied; the standard model initialization supplies it.
Fixed counts do not require a sampling group. Sampling uses the ordinary PyTorch
CPU random-number generator, which Megatron saves with its training RNG state.
Restoring that state restores the sampling sequence; changing the distribution
changes how subsequent draws map to counts.

Evaluation (`model.eval()`) uses the largest count in the distribution without
sampling. For full-sequence evaluation at a particular depth, use the forward
argument `n_loops`, e.g.:

```python
model.eval()
with torch.no_grad():
    logits = model(input_ids, position_ids, attention_mask, n_loops=2)
```

The override applies to every region for this forward only and does not modify
the training policy. Use the same override on every participating rank.

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
remains one loop. Retain normal optimizer, scheduler, and RNG loading when continuing
training rather than using a weights-only finetune.

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

Tests cover parsing, distribution validation, RNG replay and synchronization,
CLI-to-builder propagation, unchanged checkpoint structure, one-loop equivalence,
and single-/multi-region forward and gradient equivalence to explicit unrolling.
Multi-region tests also verify that all bodies use the same sampled count and
that each body completes its loops before the next body starts.
