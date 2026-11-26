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
capacity, the model configuration and a shared hardware estimate. Each CSA indexer reads
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

### Hardware estimates

The solver discovers hardware on the first imbalanced pack for which both
fused layouts fit. Each member of the runtime NCCL CP group reports its
assigned CUDA device's rates and placement in one small tensor all-gather.
Every member derives the same costs and caches them for the lifetime of the
group. Subsequent microbatches reuse those costs. Enabling `auto` requires
all members of a CP group to participate in this first resolution.

The GPU name selects nominal dense BF16 throughput and memory and link
bandwidth for H20, H100, H200, B200, B300, GB200 and GB300 GPUs. When
available, the CUDA memory clock and bus width refine the HBM estimate.
The cost model uses half the nominal compute rate for BF16, twice that for
MXFP8, and one quarter of HBM bandwidth for local packing and restoration.
On mixed devices it uses the fastest scoring rate and the slowest copy and
link rates to price the expected saving conservatively. The nominal rates
use per-GPU, dense and one-way specifications. See the
[HGX reference platform](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html),
[Hopper GPU specifications](https://www.nvidia.com/en-us/data-center/h100/) and
[Blackwell GPU specifications](https://www.nvidia.com/en-us/data-center/hgx/).

| CP placement | Effective exchange rate | Additional time per routed use |
| --- | ---: | ---: |
| Shared NVLink domain | Half the slowest one-way NVLink rate | 20 µs |
| Other same-host group | 32 GB/s | 30 µs |
| Other cross-host group | 25 GB/s | 50 µs |

When available, NVML's fabric UUID and clique ID identify a shared NVLink
domain across hosts. Without fabric telemetry, the solver assumes a local
NVLink domain only for supported switch-connected boards on the same host.
An unrecognized GPU or a group without NCCL keeps contiguous selection when
both fused layouts fit. The row-limit rules below still apply.
The bandwidth discounts and latency terms are solver heuristics. Actual
NCCL routing, overlap and launch latency depend on the deployment.
`MCORE_DSA_CP_BAL_DEBUG=1` logs the hardware estimate once per group and
the layout choice per pack on CP rank zero.

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
