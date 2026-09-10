# Balanced DSv4 context-parallel indexer

The ratio-4 CSA indexer distributes each padded sequence over twice the CP
size. Rank r scores its head chunk and the mirrored tail chunk, then returns
the selected compressed-key IDs to contiguous query ownership. Every padded
sequence must be divisible by 2 * CP. Eager ineligible packs retain the ordinary
CP path; graph-dynamic packs must satisfy the fixed-capacity route contract.

## Batch metadata

Training entrypoints call prepare_packed_seq_params once after constructing
the microbatch's PackedSeqParams. This shared training helper finalizes generic
CP partition routes and conditionally prepares the balanced indexer layout.
The feature-specific dependency belongs in this helper, rather than in the
GPT or Hybrid entrypoint.

Middle pipeline stages pass their physical capacity explicitly when they
receive raw sequence boundaries. Dynamic packed graphs otherwise derive their
capacity from max_seqlen_per_dp_cp_rank * context_parallel_size. Host reads and
route construction happen before capture.

The graph input ABI remains two tensor owners: aligned int32 layout metadata
and an int64 route. Replay refreshes their values for the current pack without
changing tensor shapes or communication split sizes.

## Projection and scoring precision

Selection Q normally uses the existing TE projection, including the effective
per-layer training/evaluation override. MXFP8, current scaling, and block
scaling therefore retain their configured precision.

Delayed scaling needs a distinct selection path because extra head/tail
forwards would update the same Linear's amax history. Selection uses a
stateless nonquantized projection with the shared, ready parameter; FP8-only
weights are dequantized through the existing utility. The ordinary local Q
projection still executes once in no-grad checkpoint forwards to record the
metadata needed by recompute. The selection helper never rewrites TE's cached
FP8 state or contributes parameter gradients.

Both balanced calls use the existing compact scorer and configured
dsa_indexer_precision. Compact K padding is built once because the two halves
have identical segment lengths; their RoPE positions and causal offsets differ.
The stateless projection waits for parameter publication before reading shared
weights or bias, and normalizes TE's empty disabled-bias tensor to `None`.

Head and tail retain separate warmed-up workspace slots on their CSA module.
Candidate/output scratch follows the existing capture-pool allocation model.
Existing fused row-limit guards remain in force.

## Returning sparse-loss predictions

The balanced helper returns (topk_indices, logical_layout, compact_predict).
Sparse-loss training consumes the compact prediction just as ordinary CP does,
instead of recomputing a BF16 prediction for keys selected by an MXFP8 scorer.

When present, FP32 softmax values are bit-reinterpreted as int32 and appended
to the index columns of each row. The existing combine route permutes both
parts together and restores contiguous tensors on the source rank. This adds
payload bytes but no collectives: the dynamic route still uses two dispatch
and two combine all-to-alls. Calls without a compact prediction keep the
index-only payload.

## Validation

Coverage includes eligible multi-sequence CP2/CP4 versus CP1 output, indexer
loss and full gradients; actual selection precision and delayed amax across
recompute; and isolated dynamic-pack graph replay for BF16 and MXFP8.

The E2E comparison uses the standard Hybrid entrypoint with mixed-length THD
packs, MXFP8, and TE attention partial graphs. Its two arms differ only in
dsa_cp_balance_indexer. Full recompute is covered at module level because dev
does not support full recompute inside TE partial graphs.
