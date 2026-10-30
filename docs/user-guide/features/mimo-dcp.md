# MIMO with dynamic context parallelism

MIMO can encode media independently on colocated or disjoint encoder ranks,
then route projected features into packs selected by the existing
sequence-packing scheduler. The encoder and decoder need not share a data
partition. An image is encoded on one producer; its projected rows can be
consumed by several context-parallel ranks.

The execution contract uses TP=PP=1, no GTP or expert GTP weight sharding,
packed THD input, per-token loss, and non-overlapped DDP.
Each decoder round partitions the decoder domain into disjoint CP packs.
The transport domain also includes encoder ranks; colocated execution uses the
same ranks for both roles. Pipeline parallelism and rerun are not supported.

## Public components

| API | Responsibility |
| --- | --- |
| `MimoModel.encode_modalities()` | Run the registered encoders/projectors and retain their graphs. |
| `planning.MediaTask`, `assign_media_to_producers()` | Assign whole media tasks using caller-provided costs and compute producer row offsets. |
| `planning.build_round_plans()` | Associate ordered media references with scheduled decoder packs. |
| `comm.pack_bridge.PackFeatureBridge` | Gather features to each pack leader using P2P, broadcast within that CP group, and return gradients along the saved reverse route. |
| `packing.prepare_mimo_step()` | Reuse the scheduler and THD pack builder, place positions at final physical offsets, and select explicit runtime groups. |
| `runtime.MimoStepRuntime` | Own one source step's encoder graph, pending transfer and FP32 gradient accumulation. |
| `runtime.LazyMimoStepRuntime` | Encode independent media chunks on first use and release each graph after its last consumer. |

Use module-level imports for planning, packing and runtime. Importing the base
model does not load the packing/data-scheduler implementation.

## Source ownership and metadata

Every transport rank currently receives the same ordered CPU mapping of source
samples and media descriptors. Only decoder ranks call `prepare_mimo_step()`
and materialize token tensors and THD packs on GPU. This replicated-source
contract is separate from encoder execution: each producer encodes only its
assigned media. CPU media preprocessing and materialization remain the caller's
responsibility; the current source adapter repeats them across both roles.
Rank-local datasets such as Energon need an acquisition/materialization adapter
before calling these APIs. The text-only reroute schema cannot carry arbitrary
media tensors or descriptors.

Samples contain `tokens`, `labels`, `loss_mask`, `position_ids`,
`original_seq_len`, `padded_seq_len`, and ordered `media_ids`. Stable sample and
media identities may be any hashable values. Scheduler-local indices do not
replace these identities. Token fields have the physical sample length;
positions can be `[S]` or `[axes, S]` and retain their original values.

Logical boundaries, physical offsets, padding masks and supervision masks have
different meanings. Preparation sanitizes padding, applies the existing packing
alignment, then places true positions at the final physical offsets. It returns
complete packs. **Combine text and modality embeddings before one unified CP
partition**; do not feed an already CP-local text batch back through MIMO.
The partition preserves the model's linear/attention layouts and their masks.

Pass pre-created `cp_groups` indexed by CP size for the current rank, using
`ProcessGroupCollection.dynamic_cp` for dynamic CP or the configured static CP
group. Include an explicit singleton when CP1 is enabled. Preparation validates
actual ranks against the feature plan. TP1 uses the same groups for TP x CP
unless matching groups are provided explicitly. Preparation never creates
groups or changes configuration.

`MimoStepPlan` reports source sample count, supervised token count, source length
statistics, and the actual `rounds`. The data adapter owns input-bin accounting
and resume progress. Neither source global-batch size nor consumed input bins
may be inferred from the number of decoder rounds.

## Explicit forward/backward lifetime

The following shows the colocated host lifecycle:

```python
import torch

from megatron.core.models.mimo.packing import prepare_mimo_step
from megatron.core.models.mimo.runtime import LazyMimoStepRuntime

# Construct MimoModel(..., external_feature_transport=True) with the encoder's
# own process groups. The caller supplies media processing and source acquisition.
plan = prepare_mimo_step(
    samples,
    feature_slices,
    scheduler=scheduler,
    domain_ranks=domain_ranks,
    cp_groups=cp_groups,
    config=model.config,
    device=device,
)
runtime = LazyMimoStepRuntime(transport_group)
runtime.begin_lazy_step(
    [item.feature_plans for item in plan.rounds],
    feature_slices,
    # Materialize only these media IDs, returning projected rows in their order.
    lambda media_ids: model.encode_modalities(prepare_media(media_ids))[modality_name],
    torch.empty((0, hidden_size), device=device, dtype=feature_dtype),
    forward_only=False,
)

for item in plan.rounds:
    transfer = runtime.route(item.feature_plans, item.cp_group)
    output, loss_mask = wrapped_model(
        **item.model_inputs,
        modality_embeddings={modality_name: transfer.features},
    )
    # The host computes its ordinary loss and completes decoder backward here.
    host_decoder_backward(output, loss_mask)
    runtime.complete_round(transfer)

runtime.finish_backward()
host_finalize_model_grads()
```

Media sharing a producer and first/last consumer rounds form one encoder chunk.
The runtime invokes its callback at first use, accumulates feature gradients in
FP32, and runs encoder backward immediately after the final consumer. Media
reused across rounds retain their own graph until then. The packing plan and
producer assignments do not change. Each callback must create an independent
graph and preserve media boundaries; chunking assumes media can be encoded
independently. Exact slice aliases share computation; partially overlapping
media slices are rejected.

Shared media can make chunk backward order differ from forward order. The
encoder must accumulate parameter gradients in completion order; first-forward
flags must not overwrite earlier contributions. Quantized encoder weight caches
and gradient-overwrite flags require host-specific handling and validation.

Empty producers supply `[0, hidden]` in the common device/dtype. They still call
the same rounds. For evaluation, pass `forward_only=True`, complete each round
after forward, then call `finish_forward()`. Execution mode is explicit;
`features.grad is None` is a valid zero contribution, not a completion signal.
The runtime rejects a second pending route, stale/duplicate completion handles,
and finishing before all declared rounds have completed.

Ordinary P2P does not connect autograd graphs. The transport returns detached
decoder leaves and explicitly applies the transpose of its forward route:
CP contributions are **summed**, repeated producer slices are added, and the
runtime accumulates contributions in FP32 before each chunk's encoder backward.
There is no extra CP/DP/token averaging at the feature boundary. The host performs normal
parameter-gradient reduction and loss normalization afterward.

`MimoStepRuntime.begin_step(features, num_rounds, forward_only=...)` remains
available for callers that already encoded a whole source step. It retains that
graph and runs one encoder backward at `finish_backward()`.

## Megatron training adapter

`examples/mimo/training/packed_step.py` provides `PackedMimoStep`. Its
`prepare_source` callback returns samples, feature slices and a callable mapping
ordered media IDs to local encoder inputs. The adapter uses lazy execution for
this contract; returning an input dictionary retains whole-step encoding.
The example CLI names this choice
`--mimo-dcp-encoder-schedule {whole-step,per-round}` (default: `whole-step`).
Model/data adapters map the selected schedule to the corresponding input contract.
Both modes plan the complete source batch before execution. `per-round` describes
encoder execution granularity; it does not itself imply overlap between GPU pools.
The callback owns dataset/model-specific policies. Call `install()`
explicitly to register the packing adapter and gradient-finalization wrapper.
The adapter defaults to the caller's process-group collection for runtime CP
groups; an explicit `cp_groups` mapping remains supported. The public core
runtime itself never installs hooks. External source iterators use
`--dataloader-type external`; the native scheduler still selects packing and
CP sizes before the data adapter prepares model inputs.

The adapter uses the existing native PP1 schedule: the next forward callback
returns the previous round's visual gradients and completes any finished encoder
chunks. The finalizer completes the last round before native gradient
finalization. Parameter gradients accumulate across all chunks, with one normal
optimizer update per source step. Evaluation
cleans up after the final forward rather than relying on a loss closure.
`examples/mimo/data/source_batch.py` supplies a deterministic replicated-source
iterator whose resume position is derived from consumed source samples.

Save checkpoints only at a complete source-step boundary. Pending transfers and
encoder graphs are transient and are not checkpoint state. Native MTP and MoE
auxiliary objectives retain their existing semantics; changing pack composition
does not promise identical auxiliary losses.

## Disaggregated native training

`examples/mimo/pretrain_mimo.py --mimo-packed` uses disjoint encoder and decoder
pools. The model provider supplies source-batch loading, media preparation and
optional weight initialization through `MimoProvider` callbacks. The packed entry
builds the configured native tokenizer and saves its assets with checkpoints;
use `NullTokenizer` for synthetic pretokenized fixtures.

Let **E** be the encoder ranks, **D** the decoder ranks, and **T = E ∪ D** the
transport domain. The encoder/projector runs on E with independent DP; only D
owns the decoder, its EP groups and its runtime CP groups. Construct the core
bridge/runtime with explicit `producer_ranks=E` and `decoder_ranks=D`;
omitting these arguments retains colocated behavior. Packs cover D exactly.
Producer-only transfers have no local decoder plan, features or CP operations.

For example, launch **20 GPU processes** with E on ranks 0–3 and D on ranks 4–19.
Add the following to the model provider, dataset, tokenizer, optimizer and
recomputation flags, using `pretrain_mimo.py` as the entry:

```bash
--mimo-packed --mimo-encoder-tp 1 --mimo-encoder-dp 4 \
--mimo-llm-offset 4 --mimo-llm-tp 1 --mimo-llm-pp 1 \
--mimo-llm-dp 2 --mimo-llm-cp 8 --mimo-llm-ep 16 --mimo-llm-expt-tp 1 \
--bf16 --use-distributed-optimizer --optimizer adam \
--calculate-per-token-loss --dataloader-type external \
--dynamic-context-parallel --sequence-packing-scheduler default_dynamic_cp \
--max-seqlen-per-dp-cp-rank 8192 --rerun-mode disabled
```

CP8 is the decoder's base layout; dynamic CP schedules across all 16 decoder
ranks. The 8192-token limit constrains each decoder rank, not the whole sample
or the encoder. Use `dp_balanced` without `--dynamic-context-parallel` for a
static-CP packing configuration. Native MoE auxiliary losses and MTP remain
enabled through their usual model flags and keep their existing semantics.

The supported disaggregated path requires BF16, encoder TP=CP=PP=1, decoder
TP=PP=1, distributed Adam, and no GTP rematerialization. DDP/parameter-gather
overlap, MoE communication overlap, encoder prefetch, and rerun are disabled.
Projectors stay on E. The encoder retains one source step's activation graph
until all decoder rounds return their gradients, then executes backward once.
Each pack's complete features are gathered to its leader and replicated within
that CP group; this is not selective transmission to individual token owners.

The adapter preserves native DDP, optimizer and checkpoint handling. Encoder
and decoder parameter gradients use SUM reductions over their own domains,
then both divide by the same global decoder supervision-token count. There is
no active-encoder averaging. Save/resume is supported at idle source-step
boundaries; in-flight transfers and activation graphs are not checkpointed.

## Validation

Run distributed tests using the repository's standard torch distributed runner.
The model, partition, transport, runtime and packed-step tests run on four ranks.
`test_multisample_multiimage_cp8.py` additionally requires **16 colocated ranks**
or **20 disjoint ranks (E4/D16)**, covering
multi-image samples sharing CP8 alongside CP4 packs, with producers outside the
consumer group and image rows crossing CP ranks. Smaller allocations skip that
case and do not constitute its validation.

The runtime test additionally uses eight ranks for disjoint E4/D4, static and
dynamic CP, multi-image packs, empty producers and text-only steps. The native
fixture at `tests/functional_tests/test_cases/mimo/disagg_dcp/run.py` exercises
the complete `pretrain_mimo.py` loop with a small ViT, MoE/MTP decoder and native
checkpoint restoration. These are validation targets, not a claim that every
listed configuration has passed end-to-end testing.

The functional runner at
`tests/functional_tests/test_cases/mimo/colocated_dcp/run.py` exercises a small
real ViT and MoE decoder with MTP, native DDP, distributed Adam, and distributed
checkpoint restoration. Run it with four ranks and `--layout cp1`,
`--layout static_cp2`, or `--layout dcp`; `--steps` is the final source-step
index, including after `--load`. It uses the native core schedule through the
example adapter, rather than the complete `pretrain()` application loop. Its
results qualify that generic model and execution contract, not a complete
production model, dataset adapter, or performance improvement.
