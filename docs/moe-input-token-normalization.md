# MoE auxiliary normalization and target masking

With `--calculate-per-token-loss`, the training finalizer divides all summed
gradients by the optimizer step's global scored-token count, `T`. Router
regularizers normally operate on real input tokens, whose global count `N`
can differ from `T`. Dividing their summed gradients by `T` incorrectly changes
the auxiliary coefficient by `N/T` as SFT prompt masking changes.

## Default behavior

The standard GPT and Hybrid text training loops now correct this automatically:
the auxiliary backward seed is multiplied by `max(T,1)/max(N,1)` before the
shared finalizer divides by `max(T,1)`. The main-loss gradient is unchanged.
There is no opt-in normalization flag. Padding is excluded; prompt tokens
remain part of the routing regularizer.

For example, suppose two DP replicas each process 200 input tokens, but score
60 and 140 targets. If their auxiliary mean gradients are `g0` and `g1`, the
correct input-token mean is `(g0 + g1)/2`. Previously the per-token path gave
`g0 + g1`. Changing the target mask alone must not change this coefficient.

The count helper looks ahead one optimizer step on the logical output stage,
broadcasts counts across TP, reduces across DP, and broadcasts across PP.
Counts precede CP slicing, avoiding double counting CP replicas. Only CPU batch
references are buffered; shallow batch copies keep frontend CUDA assignments
out of those buffered records. No GPU activation graphs are retained. Interleaved
pipelines retain each virtual stage's original iterator except for the buffered
output-stage iterator. A rerun reads the rewound original iterator.

One step-wide ratio is essential when microbatches have different target
fractions. The ratio's device address stays stable between steps so captured
graphs do not replay a stale ratio. AMP loss scaling is composed with the
correction. An explicit `moe_grad_scale_func` remains a **full override**: callers
using it must implement their intended auxiliary normalization themselves.

An all-target-masked step still has an input-token routing regularizer; the main
loss contributes zero. A step with no real input tokens contributes zero.

## Optional target-only regularization

Set `--moe-aux-loss-masking` to restrict load-balancing and router z losses to
the positions selected by the main loss mask. The equivalent Core configuration
field is `TransformerConfig.moe_aux_loss_masking`.

The mask is binary and follows **prediction positions**: at position `i`, its
entry says whether the main loss scores the label predicted from hidden state
`i`. Do not shift the mask again based on the input token's role. Core model
callers supply the existing `loss_mask` argument; low-level router callers pass
a boolean `moe_aux_loss_mask` (`True` means included). Actual padding is always
excluded as well.

This affects auxiliary statistics and gradients only. Prompt tokens continue
to execute routing and expert computation. Expert-bias updates and routing
diagnostics continue to use all real tokens, and model logits are unchanged.
Masked and unmasked router gradients are tested against explicit selected-token
references.

- With `--calculate-per-token-loss`, auxiliary means use the scored-token
  domain across the optimizer step. No separate input-token correction is needed.
- Without it, existing equal-microbatch/rank weighting is retained. An entirely
  masked microbatch contributes zero; it is not removed from the microbatch
  divisor. Sequence-level auxiliary loss omits empty sequence columns from its
  within-microbatch sequence average.
- For `seq_aux_loss`, per-token mode weights nonempty sequence means by their
  selected-token counts; legacy mode averages those sequence means equally.
  These are the router's existing physical sequence columns, **not a new
  definition of per-conversation statistics for packed data**. Packed/SP/CP
  grouping limitations remain a separate upstream concern (issue #8014).
- Global auxiliary loss accumulates selected-token expert counts with their
  corresponding token counts, rather than averaging counts equally across
  microbatches. Empty microbatches therefore do not dilute its running distribution.
- Masked z loss uses a token-weighted TP/CP mean, including uneven or empty shards.

The logging tracker still follows Megatron's existing accumulation/reporting
policy. These backward-normalization changes do not introduce a separate
token-weighted validation metric for every auxiliary diagnostic.

## Integration and limitations

Ordinary PP and virtual/interleaved PP carry the metadata on every stage;
fixed TP/SP and CP use the same transformations as the main loss mask.
Full activation recomputation retains each forward's mask in its own closure.
This removes the previous PP=1 integration restriction.

SFT input-token counting currently identifies padding by a distinct PAD token
ID. A PAD/EOD alias is ambiguous and is rejected. Custom data frontends must
provide equivalent counts or install their own full `moe_grad_scale_func`;
they do not silently retain the incorrect normalization.

Unsupported combinations fail explicitly:

- Input-token correction: dynamic sequence-packing scheduler, GTP weight
  rematerialization, and MTP. These require counts from their actual runtime
  token plans, potentially separately for each router/depth.
- Target masking: sequence-packing scheduler, MTP, CUDA graphs, overlapped MoE
  schedule plans, training MLP chunking, and hybrid shortcut/mHC wrappers.
  These need additional mask plumbing and/or a defined per-chunk reduction.

Dynamic CP is currently rejected through its sequence-packing scheduler. The router's existing runtime CP
group support is useful, but the training scheduler must supply counts from
its actual variable token allocation. Upstream HybridModel itself currently
rejects dynamic CP. This is broader integration work, rather than a change to
the normalization formula. Static packed sequences with fixed CP are a
different path and are supported by the ordinary text frontend.

The input-token ratio has a CUDA-graph replay regression test; a complete
full-model graph-capture matrix has not been validated. The new target mask is
explicitly rejected with CUDA graphs until it is registered as a graph input.

## Validation

Run the focused distributed tests in the supported development environment:

```bash
uv run python -m torch.distributed.run --standalone --nproc-per-node=4 -m pytest -q \
  tests/unit_tests/training/test_moe_loss_normalization.py \
  tests/unit_tests/transformer/moe/test_aux_loss_masking.py \
  tests/unit_tests/determinism/kernels/test_aux_mask_replay.py
```

They cover independent gradient references, uneven DP/microbatch/CP counts,
empty target sets and CP shards, all three auxiliary balancing losses and z
loss, output-stage PP/VP count transport, AMP/full override behavior, graph
ratio replay, and bit-exact model forward/backward replay with retained masks.
Short full-training runs additionally exercise packed SFT, PP, TP/SP, CP, EP,
virtual pipelines, activation recomputation, and HybridModel integration.
These are correctness and execution checks, not convergence benchmarks.
