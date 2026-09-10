<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

# Balanced DSA routing with partial CUDA graphs and EP overlap

## Contract

This follow-up to #6058 supports fixed-CP, fixed-capacity `dp_balanced` THD inputs
with Transformer Engine partial CUDA graphs and EP 1F1B overlap. Dynamic pack
composition and physical microbatch counts may change within the captured
packing bound. Delayed weight gradients and full-iteration graphs are outside
this change. The first implementation admits PP1; a subsequent change adds
non-interleaved PP and VPP.

## Capture and input ownership

Model chunks retain their identity when the capture order expands into individual
layer callables. One indexing function maps a logical chunk, callable and
microbatch to TE's sample/graph index. Capture, route canonicalization and graph
installation use the same mapping. Expanded capture units are local bookkeeping,
not replacements for the model topology.

Each decoder chunk owns two typed route tensors per captured microbatch slot.
All DSA callables in that chunk/slot share them; distinct live slots cannot alias.
The schedule preprocess stages the current source plan once, on the compute
stream, and retains the staged PackedSeqParams on the invocation. Backward must
use that invocation's owners, never a later forward's mutable microbatch state.
Capture capacity includes both the packing upper bound and the overlap schedule's
liveness floor. PP1 overlap has two live forwards whenever multiple microbatches
are available. Existing pointer, schema and stale-route checks remain active.

Capture cleanup also zeros the indexer loss tracker in place. TE warmup executes
synthetic forwards that accumulate logging values, which must not reach the first
training replay's metrics. Cleanup preserves the captured tracker storage and
reduction groups because replay updates those addresses without running the
Python logging-registration code again. This cleanup does not change the
indexer loss or its backward scale.

## Model scheduling

GPTModel and HybridModel expose `build_schedule_plan` returning an
AbstractSchedulePlan. Hybrid scheduling shares the ordinary forward's embedding,
decoder boundary and output processing, and preserves checkpoint parameter names.
DSv4 attention, dense MLP and MoE layers expose their actual compute and
communication boundaries, including mHC aggregation and residual gradients.
Capture grouping cannot cross a boundary at which the overlap scheduler yields
to another microbatch. Eager layers retain their positions when expanding the
capture order: dropping a trailing dense layer changes the pairing of forward
and backward graphs and invalidates TE's shared-memory-pool lifetimes. Process
groups come from the model's collection.

Eager MoE routing normalizes the model's `[batch, sequence]` padding mask to
the router's local `[sequence, batch]` layout. Sequence-parallel chunks scatter
only unscattered masks, using the attention TP group supplied by the model.
Captured router outputs already include this normalization and must not repeat
it on replay. This preserves padding-dependent router losses and expert counts
without changing graph input tensors. Hybrid overlap rejects FP4 until its
separate quantization contexts are supported.

## Validation

Focused tests cover logical-to-capture indexing, changing pack counts, distinct
live route owners, exactly two metadata copies and cleanup. GPU validation uses
real DSA and EP collectives, comparing eager/PCG with overlap disabled/enabled
from identical inputs and parameters, including indexer and mHC gradients. PP
coverage includes short schedules, warmup, steady state, cooldown and evaluation.
Performance evidence must confirm graph replay and communication overlap.

The PP1 validation uses four GB200 GPUs, CP2/EP2, BF16, fixed 512-token local
capacity and variable THD packs. GPT attention-only capture completes training.
Hybrid `CEHEW-` with mHC and attention/router/preprocess capture completes 35
iterations with changing physical microbatch counts and evaluation after
iterations 10, 20 and 30. Separate real-EP2 tests compare every parameter's
gradient with ordinary Hybrid forward for mHC disabled and enabled. The route
arena and slot regression includes real TE graphs and 30 changing route replays.
