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

## Automatic layout selection

Omitting `--dsa-cp-balance-indexer` keeps selection contiguous. The bare flag
uses zigzag for every representable pack. Passing `--dsa-cp-balance-indexer auto`
selects once per packed microbatch. All ranks derive the same decision from
global padded sequence boundaries, the runtime CP size, the physical row
capacity and the model configuration. Each CSA indexer reads
`PackedSeqParams.csa_indexer_use_zigzag`. Its query dispatch and inverse
top-k exchange use the same decision. The model keeps contiguous CP token
ownership. Direct eager Core callers can resolve the decision on first use.

For a one-based query position `p` within a document, the ratio-4 indexer
can score `floor(p / 4)` completed compressed keys. The host model sums those
counts over each rank's contiguous intervals and its mirrored zigzag chunks.
Padding beyond the last supplied boundary moves with the query rows and
contributes zero keys. The model converts the difference between the two
peak rank counts to time using two dot-product FLOPs per head dimension.
It estimates routing time from the busiest rank's remote rows, local copies
of the full shard, and a fixed overhead. The row payload includes low-rank Q,
head weights, int32 top-k IDs and, when configured for sparse-loss training,
FP32 predictions. Zigzag must save more time than this routing estimate.
Contiguous decisions skip building the zigzag route.

| Configuration field | Initial value | Cost represented |
| --- | ---: | --- |
| `dsa_cp_indexer_auto_compute_tflops` | 500 TFLOP/s | Effective scoring throughput per rank |
| `dsa_cp_indexer_auto_exchange_gbps` | 200 GB/s | Exposed dispatch and return traffic after overlap |
| `dsa_cp_indexer_auto_copy_gbps` | 2000 GB/s | Effective throughput for local permutation and packing |
| `dsa_cp_indexer_auto_overhead_us` | 20 µs | Extra launches, amortized preparation and decision margin per use |

These coefficients are starting estimates. Calibrate them for the model,
hardware and CP topology before drawing throughput conclusions.
`MCORE_DSA_CP_BAL_DEBUG=1` logs the choice and pack eligibility on CP rank zero.

Auto requires attention to run eagerly, including when other modules use
CUDA graphs. Capturing attention fixes its collective sequence across replays.
The current fused scorer accepts up to 32,768 query rows per call. For a
representable pack with larger contiguous shards, auto uses the two fused halves
when each fits. An oversized or unrepresentable zigzag layout falls back to
contiguous scoring, using the unfused implementation when required for BF16.
MXFP8 requires a legal fused route for nonempty scoring.

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
Batch-entrypoint tests exercise the shared metadata helper with normalized
configuration and verify explicit CP-group propagation, attention-eager graph
scopes, legacy Hybrid THD construction, and unpacked SBHD behavior.

The E2E comparison uses the standard Hybrid entrypoint with mixed-length THD
packs, MXFP8, and TE attention partial graphs. Its two arms differ only in
dsa_cp_balance_indexer. Full recompute is covered at module level because dev
does not support full recompute inside TE partial graphs.
