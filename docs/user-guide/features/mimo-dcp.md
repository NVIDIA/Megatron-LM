# Colocated MIMO with dynamic context parallelism

MIMO can encode media independently across the decoder's data/context-parallel
domain, then route projected features into packs selected by the existing
sequence-packing scheduler. The encoder and decoder need not share a data
partition. An image is encoded on one producer; its projected rows can be
consumed by several context-parallel ranks.

The initial execution contract is colocated, TP=PP=1, no GTP or expert GTP
weight sharding, packed THD input, per-token loss, and non-overlapped DDP.
Each decoder round partitions the entire transport domain into disjoint CP
packs. Pipeline parallelism, rerun, and disaggregated execution are outside
this interface's supported scope.

## Public components

| API | Responsibility |
| --- | --- |
| `MimoModel.encode_modalities()` | Run the registered encoders/projectors and retain their graphs. |
| `planning.MediaTask`, `assign_media_to_producers()` | Assign whole media tasks using caller-provided costs and compute producer row offsets. |
| `planning.build_round_plans()` | Associate ordered media references with scheduled decoder packs. |
| `comm.pack_bridge.PackFeatureBridge` | Gather features to each pack leader using P2P, broadcast within that CP group, and return gradients along the saved reverse route. |
| `packing.prepare_mimo_step()` | Reuse the scheduler and THD pack builder, place positions at final physical offsets, and select explicit runtime groups. |
| `runtime.MimoStepRuntime` | Own one source step's encoder graph, pending transfer and FP32 gradient accumulation. |

Use module-level imports for planning, packing and runtime. Importing the base
model does not load the packing/data-scheduler implementation.

## Source ownership and metadata

Every transport rank currently receives the same ordered mapping of source
samples and media descriptors. Source token tensors must be available on each
rank before preparation. This replicated-source contract is separate from
encoder execution: only assigned media are decoded/processed/encoded locally.
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

```python
from megatron.core.models.mimo.packing import prepare_mimo_step
from megatron.core.models.mimo.runtime import MimoStepRuntime

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
runtime = MimoStepRuntime(transport_group)
features = model.encode_modalities(encoder_inputs)[modality_name]
runtime.begin_step(features, len(plan.rounds), forward_only=False)

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

Empty producers supply `[0, hidden]` in the common device/dtype. They still call
the same rounds. For evaluation, pass `forward_only=True`, complete each round
after forward, then call `finish_forward()`. Execution mode is explicit;
`features.grad is None` is a valid zero contribution, not a completion signal.
The runtime rejects a second pending route, stale/duplicate completion handles,
and finishing before all declared rounds have completed.

Ordinary P2P does not connect autograd graphs. The transport returns detached
decoder leaves and explicitly applies the transpose of its forward route:
CP contributions are **summed**, repeated producer slices are added, and the
runtime accumulates rounds in FP32 before one encoder backward. There is no
extra CP/DP/token averaging at the feature boundary. The host performs normal
parameter-gradient reduction and loss normalization afterward.

## Megatron training adapter

`examples/mimo/training/packed_step.py` provides `PackedMimoStep`. Its
`prepare_source` callback returns samples, feature slices and local encoder
inputs; the callback owns dataset/model-specific policies. Call `install()`
explicitly to register the packing adapter and gradient-finalization wrapper.
The adapter defaults to the caller's process-group collection for runtime CP
groups; an explicit `cp_groups` mapping remains supported. The public core
runtime itself never installs hooks. External source iterators use
`--dataloader-type external`; the native scheduler still selects packing and
CP sizes before the data adapter prepares model inputs.

The adapter uses the existing native PP1 schedule: the next forward callback
completes the previous decoder backward, and the finalizer completes the last
round before encoder backward and native gradient finalization. Evaluation
cleans up after the final forward rather than relying on a loss closure.
`examples/mimo/data/source_batch.py` supplies a deterministic replicated-source
iterator whose resume position is derived from consumed source samples.

Save checkpoints only at a complete source-step boundary. Pending transfers and
encoder graphs are transient and are not checkpoint state. Native MTP and MoE
auxiliary objectives retain their existing semantics; changing pack composition
does not promise identical auxiliary losses.

## Validation

Run distributed tests using the repository's standard torch distributed runner.
The model, partition, transport, runtime and packed-step tests run on four ranks.
`test_multisample_multiimage_cp8.py` additionally requires **16 ranks**, covering
multi-image samples sharing CP8 alongside CP4 packs, with producers outside the
consumer group and image rows crossing CP ranks. Smaller allocations skip that
case and do not constitute its validation.

The functional runner at
`tests/functional_tests/test_cases/mimo/colocated_dcp/run.py` exercises a small
real ViT and MoE decoder with MTP, native DDP, distributed Adam, and distributed
checkpoint restoration. Run it with four ranks and `--layout cp1`,
`--layout static_cp2`, or `--layout dcp`; `--steps` is the final source-step
index, including after `--load`. It uses the native core schedule through the
example adapter, rather than the complete `pretrain()` application loop. Its
results qualify that generic model and execution contract, not a complete
production model, dataset adapter, or performance improvement.
