<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Combined 1F1B: Expert-Parallel All-to-All Overlap

Combined 1F1B, also called **1F1B A2A overlap** or **EP A2A overlap**, is a
fine-grained training schedule for Mixture-of-Experts (MoE) models. It overlaps
expert-parallel (EP) token-dispatch and token-combine All-to-All communication
from one microbatch with independent forward or backward computation from an
adjacent microbatch.

Enable the schedule with `--overlap-moe-expert-parallel-comm`.
`--delay-wgrad-compute` is a separate, optional optimization that exposes
weight-gradient GEMMs as additional work with which to hide communication.

This guide describes the current `main` implementation, its configuration and
compatibility rules, and the evidence needed to decide whether to use it.

## Motivation

An MoE layer routes tokens to experts that may reside on other EP ranks. Its
critical path contains two communication phases:

1. **Dispatch A2A** sends routed tokens to their experts.
2. **Combine A2A** sends expert outputs back to the originating ranks.

For fine-grained MoE models with large or cross-node EP domains, those
collectives can occupy a substantial fraction of iteration time. A faster
dispatcher reduces the duration of each collective, whereas Combined 1F1B hides
the remaining duration behind independent computation. These optimizations are
complementary; neither changes the dependency chain within one microbatch:

```text
attention/router -> dispatch A2A -> experts -> combine A2A -> next layer
```

Two pairings are possible:

- **FWD-FWD / BWD-BWD** overlaps like phases from adjacent microbatches, but
  doubles peak activation memory and is asymmetric because forward compute is
  typically only about half as long as backward compute.
- **FWD-BWD** pairs the forward of microbatch `i+1` with the backward of
  microbatch `i`. It does not add the same activation-memory penalty and gives
  a longer compute window for A2A. The first forward and final backward remain
  exposed fill/drain phases.

Megatron Core implements the second form. It retains the existing 1F1B pipeline
structure and splits backward expert computation into activation-gradient and
weight-gradient work when delayed wgrad is enabled. The design and ablations are
described in
[Scalable Training of Mixture-of-Experts Models with Megatron Core](https://arxiv.org/html/2603.07685#S4.SS2.SSS3).

The optimization changes execution order, not the model's mathematical
function. Its useful region is therefore determined by two measurements: how
much EP communication is exposed before overlap, and how much independent
forward/backward compute exists to cover it.

## Schedule overview

### Pipeline-parallel size 1

Without pipeline parallelism, the first forward and the final backward are the
fill and drain phases. Every middle phase combines the forward of the next
microbatch with the backward of the previous one:

| Phase | Forward work | Backward work |
| --- | --- | --- |
| 0 | microbatch 0 | — |
| 1 | microbatch 1 | microbatch 0 |
| 2 | microbatch 2 | microbatch 1 |
| ... | ... | ... |
| N | — | microbatch N-1 |

With only one microbatch there is no steady-state pair and therefore no
inter-microbatch overlap opportunity.

### Pipeline-parallel size greater than 1

Pipeline parallelism uses the interleaved 1F1B schedule. The overlap schedule
moves one additional forward microbatch into warmup. That makes the forward and
backward work paired in every steady-state 1F1B step independent, including on
the last physical pipeline stage.

Consequently, `pipeline_model_parallel_size > 1` requires virtual pipeline
parallelism (VPP). Configure it with
`--num-layers-per-virtual-pipeline-stage` or a pipeline layout that creates more
than one virtual model chunk. The schedule adds one warmup microbatch but keeps
the steady-state bubble structure of the underlying interleaved schedule.

Choose a global batch size that leaves steady-state microbatches after pipeline
warmup. If all microbatches are consumed by warmup, training is correct but the
combined steady state provides little or no benefit.

## Fine-grained execution

The schedule does not call a transformer layer as one opaque operation. A model
implements `build_schedule_plan()` and returns an `AbstractSchedulePlan`. For the
MCore `GPTModel`, each transformer or MTP layer is decomposed into these nodes:

| Node | Work | CUDA stream |
| --- | --- | --- |
| `pre_dispatch_computation` | Attention, pre-MLP normalization, routing, and local dispatch preprocessing | Compute |
| `moe_dispatch` | Token dispatch A2A | Communication |
| `mlp` | Routed-expert computation | Compute |
| `moe_combine` | Token combine A2A and local postprocessing | Communication |
| `mtp_post_process` | Optional MTP postprocessing and depth-output assembly | Compute |

Dense layers use no-op dispatch/combine nodes, so mixed dense/MoE decoder stacks
preserve a uniform scheduling interface.

A CUDA event shared by the schedule plan orders dependent nodes across the
compute and communication streams. Independent kernels may then run
concurrently. For one paired forward/backward layer, the issue order is:

```text
B.combine
F.pre_dispatch_computation
B.expert dgrad
F.dispatch
B.expert delayed wgrad
B.dispatch
F.expert
F.combine
B.pre_dispatch_computation [and delayed wgrad]
```

The dependency event makes `F.dispatch` wait for `F.pre_dispatch_computation`, `B.expert` wait
for `B.combine`, and so on, while allowing independent work on the other
stream to proceed. At model-chunk scope, forward layers run from first to last
while backward layers run from last to first. For PP/VPP, pipeline send/receive
callbacks are inserted so forward P2P can overlap attention backward and the
last attention wgrad can overlap backward P2P.

For example, paired four-layer chunks execute as:

```text
pre_forward / pre_backward
F.layer[0] + B.layer[3]
F.layer[1] + B.layer[2]
F.layer[2] + B.layer[1]
F.layer[3] + B.layer[0]
post_forward / post_backward
```

Within an MTP layer, MTP postprocessing runs after forward combine and before
backward combine on the compute stream. Multiple MTP depths are chained by the
schedule plan; see the MTP constraints below.

### Delayed weight-gradient computation

`--delay-wgrad-compute` asks Transformer Engine linear layers to separate
activation-gradient (`dgrad`) and weight-gradient (`wgrad`) work. The schedule
invokes the deferred `backward_dw()` callables at points where they help cover EP
or pipeline communication.

The flag is optional for correctness. Enable it when profiling shows that forward
computation alone is too short to cover backward A2A communication. It introduces
Transformer Engine version and feature-combination requirements, so establish a
working overlap baseline before adding it.

### Tensor lifetime management

Fine-grained scheduling makes ownership more subtle: the Python producer may
drop a tensor while kernels on another stream are still consuming its storage.
The current implementation uses two safeguards:

1. Every tensor input is recorded on its consumer stream with `record_stream`.
   For nodes with `free_input=True`, a `ScheduleNodeInputWrapper` keeps only
   metadata and the eventual input gradient after handing the real tensor to the
   forward callable. The schedule does not resize that input's storage to zero.
2. Nodes retain autograd gradient edges rather than the output tensors
   themselves. Autograd-saved tensors still own any storage required for backward;
   storage is reclaimed when its last owner releases it and recorded stream work
   completes.

`free_input` is selected per node rather than enabled globally:

| Node | `free_input` selection | Why |
| --- | --- | --- |
| Dense-layer nodes | `False` | Backward needs the original inputs. |
| `moe_combine` | `True` | The node can retain a storage-free gradient boundary. |
| `moe_dispatch` | `True` only outside DeepEP/DeepEPv2/HybridEP/NCCL-EP and when `moe_preprocess` is not CUDA-graphed | Those Flex backends retain dispatch inputs for backward; CUDA Graph preprocess inputs are fixed buffers. |
| `mlp` | `True` for FP8/FP4; otherwise only when the input is not passed unchanged into GroupedGEMM | The BF16/FP16 identity-postprocess cases—standard A2A with one local expert, HybridEP, and NCCL-EP—must retain the original input. |
| Other nodes | `False` | Their backward path still owns or saves the input. |

These policies are implemented in `should_free_input()` in
`megatron/core/models/common/utils.py` and `ScheduleNode` in
`megatron/core/pipeline_parallel/utils.py`.

## Requirements

### Hard requirements

These conditions are enforced for correctness:

| Area | Required setting | Reason |
| --- | --- | --- |
| Model path | An MCore model implementing `build_schedule_plan()` and a training `forward_step` that accepts `return_schedule_plan=True` | Combined 1F1B asks the model for fine-grained nodes instead of running a normal opaque forward. `GPTModel.build_schedule_plan()` is the production implementation. |
| PyTorch | 2.6 or newer | Older PyTorch versions have a known hang in this overlap path. |
| EP | `--expert-model-parallel-size > 1` | With EP=1 there is no expert-parallel A2A to overlap. |
| Dispatcher | `--moe-token-dispatcher-type alltoall` or `flex` | The `allgather` dispatcher does not expose the required dispatch/combine node interface. |
| Base dtype | `--bf16` or `--fp16` | FP8/FP4 can be enabled on top, but the model still requires a supported base dtype. |
| Training mode | `forward_only=False` | The schedule pairs forward with backward and is not selected for inference/evaluation-only execution. |
| PP > 1 | Enable VPP with `--num-layers-per-virtual-pipeline-stage` or an equivalent pipeline layout | The multi-stage implementation extends the interleaved 1F1B schedule and therefore needs more than one virtual model chunk. |
| MTP | Non-negative `mtp_num_layers`, with MoE MTP layers | Multiple depths are supported with the restrictions listed below; the MTP callable rejects sequence packing and cross-attention. |

`--delay-wgrad-compute` additionally requires
`--transformer-impl transformer_engine` and TE 2.3 or newer. It is not a
requirement for Combined 1F1B itself.

### Conditions for useful overlap

These are performance conditions, not startup requirements:

- At least two microbatches are needed to form a forward/backward pair. With
  PP/VPP, enough microbatches must remain after warmup to create a steady state.
- `CUDA_DEVICE_MAX_CONNECTIONS > 1` lets compute and communication streams make
  progress concurrently. A useful starting value is 32; profile it on the target hardware.
- A trace should show exposed dispatch/combine communication before enabling the
  schedule. Already-hidden A2A leaves little work for Combined 1F1B to improve.

## Minimal configuration

Add these flags to a complete `pretrain_gpt.py` launch, keeping the model, data,
optimizer, and distributed-launch settings appropriate for the target system:

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=32

torchrun ... pretrain_gpt.py \
  --transformer-impl transformer_engine \
  --bf16 \
  --disable-bias-linear \
  --num-experts 128 \
  --expert-model-parallel-size 8 \
  --expert-tensor-parallel-size 1 \
  --moe-token-dispatcher-type alltoall \
  --overlap-moe-expert-parallel-comm \
  --micro-batch-size 1 \
  --global-batch-size 256 \
  ...
```

The example disables linear bias to avoid the default bias-dropout fusion
combination rejected by TE grouped experts. Biased grouped experts are supported
with `bias_dropout_fusion=False`; disabling bias is not an overlap requirement.
The deprecated `--use-mcore-models` flag is accepted but ignored and is not needed.

After this baseline is correct and profiled, add `--delay-wgrad-compute` as a
separate A/B experiment if backward A2A is still exposed.

For PP/VPP, add settings such as:

```bash
--pipeline-model-parallel-size 4 \
--num-layers-per-virtual-pipeline-stage 1
```

The layer count, pipeline layout, and number of microbatches must still satisfy
the normal PP/VPP divisibility and warmup requirements.

## Configuration reference

### Core controls

| Flag | Default | Meaning | When to enable |
| --- | --- | --- | --- |
| `--overlap-moe-expert-parallel-comm` | Off | Selects the combined 1F1B schedule and overlaps EP communication with independent adjacent-microbatch computation. | Enable after profiling shows exposed EP dispatch/combine A2A. |
| `--delay-wgrad-compute` | Off | Splits TE linear dgrad and wgrad so wgrad can be scheduled separately. Requires the overlap flag. | Enable when A2A remains exposed because the available forward/dgrad compute is too short. |
| `--high-priority-a2a-comm-stream` | Off | Creates the schedule's communication stream at high CUDA priority. | Test when A2A launch or progress is delayed by competing compute. It can also steal resources from compute, so retain only with measured benefit. |
| `--ep-overlap-early-attn-memory-release` | Off | Runs attention backward earlier, before forward MLP, to release attention activations sooner. | Use to reduce an overlap-specific peak-memory regression. It can expose dispatch/combine communication and reduce throughput. |

`--overlap-dispatch-backward-with-experts-wgrad` is a different optimization.
It overlaps dispatch backward with expert wgrad and is mutually exclusive with
combined 1F1B and `--delay-wgrad-compute`.

### Parallelism and batch shape

| Setting | Effect and guidance |
| --- | --- |
| `--expert-model-parallel-size` | Must be greater than 1. Larger or cross-node EP domains often create more A2A latency to hide, but may also demand a faster dispatcher backend. |
| `--pipeline-model-parallel-size` | PP=1 uses the non-pipeline combined schedule. PP>1 uses the interleaved schedule and requires VPP. |
| `--num-layers-per-virtual-pipeline-stage` | Creates VPP chunks. Smaller chunks reduce the interleaved pipeline bubble but increase P2P frequency and scheduling overhead. Tune rather than minimizing blindly. |
| `--micro-batch-size` / `--global-batch-size` | Determine the number of microbatches. More microbatches improve fill/drain amortization and provide more overlap opportunities; they may increase iteration latency and interact with optimizer semantics. |
| `CUDA_DEVICE_MAX_CONNECTIONS` | Values greater than 1 allow the compute and communication streams to make progress concurrently; a starting value is 32. On Hopper and earlier, TP/CP overlap often prefers 1, so the best value must be profiled. |

On Blackwell and newer architectures the previous TP/CP requirement for
`CUDA_DEVICE_MAX_CONNECTIONS=1` no longer applies. On Hopper and earlier, combining
TP or CP with EP A2A overlap creates a tuning conflict: use 1 when prioritizing
TP/CP launch ordering or a larger value such as 32 when prioritizing EP overlap.

### CUDA Graphs

Combined 1F1B supports two different CUDA Graph strategies. They have different
capture boundaries and should not be configured as if they were interchangeable.

#### Partial CUDA Graphs

Partial capture covers only static per-layer regions and leaves
`moe_dispatch`, expert compute, and `moe_combine` visible to the two-stream
scheduler. A partial-capture configuration is:

```bash
--cuda-graph-impl transformer_engine \
--cuda-graph-modules attn moe_router moe_preprocess \
--cuda-graph-warmup-steps 2
```

The module list must be nonempty and must not contain `moe` or `mlp` for
Combined 1F1B. `moe_preprocess` requires `moe_router`; because its graph
inputs are fixed buffers, the dispatch node keeps those inputs instead of using
`free_input=True`.

Transformer Engine captures callables in execution order. Combined 1F1B
therefore expands the normal chunk-level order into a layer-level order:
positive entries are forward layers, negative entries are backward layers, and
half-step entries such as `-3.5` are delayed-wgrad graphs. This preserves the
cross-chunk layer interleaving used at runtime.

Version-dependent combinations are:

- Delayed wgrad with TE partial graphs requires TE 2.10 or newer.
- Delayed wgrad plus `--overlap-grad-reduce` requires TE 2.8 or newer.
- Delayed wgrad without `--gradient-accumulation-fusion` requires TE 2.7 or
  newer.
- With `--overlap-grad-reduce`, capturing attention wgrad or a
  non-overlapped shared-expert wgrad requires
  TE 2.12 or newer and `--gradient-accumulation-fusion`. Attention bias is not
  supported in that combination.

#### Full-iteration CUDA Graphs

Full-iteration capture wraps the forward/backward training schedule, excluding
the optimizer step, while retaining the fine-grained compute and communication
streams inside the graph:

```bash
--cuda-graph-impl full_iteration \
--no-check-for-nan-in-loss-and-grad
```

`--cuda-graph-modules` must be empty in this mode. Dynamic dropless MoE routing
requires a synchronization-free, graph-safe path with static buffer budgets.
See [Paged Stash](paged_stash.md) for HybridEP capacity settings and the supported
TE op-fuser or device-initiated GroupedTensor expert paths. Enabling the graph
flag alone does not make an arbitrary dispatcher graph-safe.

Combined 1F1B full activation recomputation is currently eager-only, even though
the generic CUDA Graph configuration supports other recomputation combinations.

#### CUDA Graph cautions

- Partial and full-iteration capture are alternative modes; do not supply
  per-layer modules with `--cuda-graph-impl full_iteration`.
- Use a graph-safe Transformer Engine RNG tracker and at least one warmup step.
  For example, use `--te-rng-tracker` and two warmup steps.
- Scheduler-produced THD partial TE graphs with PP>1 reject both Combined 1F1B
  and delayed wgrad. Check the sequence-packing scheduler constraints separately.
- Fixed graph input buffers cannot be resized or released. Revisit
  `free_input` whenever a callable is moved into or out of graph scope.
- Full-iteration MoE capture requires static shapes throughout the iteration;
  paged stash is the documented path, not a general guarantee for every
  dispatcher backend.

### Activation recomputation

Selective recomputation is supported, but `moe` must not appear in
`--recompute-modules` because the overlap schedule already decomposes the MoE
layer. Full recomputation is implemented as a backward-time replay of layer
segments rather than through the standard checkpoint primitive.

For `--recompute-granularity full`:

- Set both `--recompute-method` and `--recompute-num-layers` explicitly.
- Use `--cuda-graph-impl none`; neither partial nor full-iteration graphs are
  supported with Combined 1F1B full recomputation.
- Disable fine-grained activation offloading and paged stash for this combination.
- `uniform` groups decoder layers by `recompute_num_layers`.
- `block` recomputes the first `recompute_num_layers` decoder layers, one segment
  per layer.
- With MTP, only one depth is supported under full recomputation. For `uniform`,
  `--recompute-num-layers` must be 1.
- `--distribute-saved-activations` is not supported because the retained segment
  input is not sharded through a checkpoint primitive.
- `attention_dropout` and `hidden_dropout` must both be zero; interleaved RNG replay
  for nonzero dropout is not implemented.
- Delayed-scaling FP8 is not supported because replay would update persistent
  `amax_history` twice. Current-scaling FP8 recipes (`tensorwise`, `mxfp8`, or
  `blockwise`) and supported FP4 modes rederive scales on replay.

When full recomputation is not enabled, leave `--recompute-method` and
`--recompute-num-layers` unset.

### Megatron-FSDP

Megatron-FSDP has version-specific integrations. The v1 adapter supports the
PP=1 combined schedule and rejects PP/VPP because that path does not manage its
root pre/post-backward lifecycle. It also requires:

- `cuda_graph_impl` equal to `none` or `full_iteration`, with no per-layer graphs.
- No full activation recomputation with Combined 1F1B.
- `fsdp_double_buffer=False` in the DDP configuration. Do not enable options that
  implicitly select double buffering, such as the max-pool double-buffer mode.
- Supported layer-level FSDP units when parameters are sharded.

Combined 1F1B executes layer submodules directly rather than calling the complete
`TransformerLayer.forward()`. The v1 adapter therefore manages parameter
availability for those submodule calls and releases gathered parameters through
callbacks from the overlap schedule. This integration is automatic; users do not
need to register parameter hooks.

For v2, the Combined 1F1B scheduler explicitly unshards and reshards
`FsdpModule` units to match its execution order. This integration does not use
v1's fine-grained parameter-gather mechanism.

The v2 restrictions still apply: disable gradient accumulation fusion, FP4, and
non-MXFP8 FP8 recipes. Its buffer settings and compatibility constraints are
separate from v1's. See [Megatron-FSDP](megatron_fsdp.md) for the broader setup.

### Other feature interactions

| Feature | Status / constraint |
| --- | --- |
| Fine-grained activation offloading | Supported without full recomputation. For TE 2.10+, set `NVTE_CPU_OFFLOAD_V1=1`. With `ncclep`, offloading `expert_fc1` is rejected by configuration validation. |
| Shared experts | Supported, but `--moe-shared-expert-overlap` must be disabled. Combined scheduling changes the order in which router/shared-expert gradients are added, so small numerical differences are expected. |
| MTP | Multiple MoE depths are supported, but with more than one depth disable `mtp_use_repeated_layer`, `mtp_hsm`, mHC, and full recomputation. The schedule callable also rejects sequence packing and cross-attention. |
| mHC | PP>1 with Combined 1F1B is rejected; multiple MTP depths also reject mHC. |
| Wide residual | `wide_residual` is rejected with Combined 1F1B. |
| FP8 / FP4 | Supported on top of BF16/FP16, subject to the full-recompute delayed-scaling restriction. |
| Hash MoE | `moe_num_hash_layers > 0` is not supported. |
| MOK megakernel | Not supported because MOK replaces the native dispatcher/expert/combine schedule. |
| MoT / Bagel MoT layer | Not supported by the combined schedule. |
| Model implementations | A custom or hybrid model must implement the schedule-plan interface and compatible per-layer callables; a normal `forward()` implementation alone is insufficient. |

## Performance results

The direct ablation reported in
[Scalable Training of Mixture-of-Experts Models with Megatron Core](https://arxiv.org/html/2603.07685#S4.SS2.SSS3)
is the relevant evidence for this feature:

| DeepSeek-V3 on H100 | Exposed expert-communication share |
| --- | ---: |
| Optimized dispatcher, without Combined 1F1B | 30–40% of iteration time |
| Combined 1F1B with forward/backward pairing and split dgrad/wgrad scheduling | Below 5% |

The reported communication overlap ratio reaches 93%. This is an ablation of
**communication exposure**, not a reported end-to-end iteration-speedup
percentage. It should not be converted directly into a speedup: communication
and GEMMs contend for GPU resources, and the first forward plus final backward
cannot be hidden.

These published results describe the report's configuration, not a new benchmark
of this branch. Recheck communication exposure and iteration time on the target
commit and hardware before extrapolating them.

### Reproduce the ablation

Use the same dispatcher and communication tuning for a controlled comparison,
subject to the platform-specific baseline requirement in step 2:

1. Fix the commit, container, model, topology, parallel mapping, batch shape,
   precision, CUDA Graph mode, and random seed.
2. Measure the dispatcher-optimized baseline without
   `--overlap-moe-expert-parallel-comm`.
   On Hopper and earlier, with TP>1 or CP>1 and without FSDP, this non-overlap
   baseline requires `CUDA_DEVICE_MAX_CONNECTIONS=1`. Use 1 in all runs for an
   otherwise-identical ablation, or explicitly report the connection-setting
   difference when comparing separately tuned configurations. Overlap-enabled
   runs may use 1 or 32; profiling 32 remains useful for EP overlap. This baseline
   restriction does not apply to Blackwell and newer, TP=CP=1, or FSDP. FSDP
   separately rejects `CUDA_DEVICE_MAX_CONNECTIONS=1`.
3. Enable only `--overlap-moe-expert-parallel-comm`.
4. Add `--delay-wgrad-compute` as a third run, not as part of the overlap
   baseline.
5. Exclude warmup iterations and report median iteration time, tokens/s,
   TFLOP/s/GPU, peak allocated/reserved memory, exposed A2A time, and overlap
   ratio.
6. Inspect an Nsight Systems trace. A2A kernels must overlap independent compute,
   not merely move to another exposed interval.
7. Run a short loss/gradient parity check before interpreting performance.

### Applicability and tuning

Use trace evidence to make each tuning decision:

- If A2A is already hidden, Combined 1F1B adds scheduling complexity without a
  communication critical path to remove.
- If only fill/drain A2A remains exposed, increase the number of microbatches
  only when the global-batch and optimization semantics permit it.
- If backward A2A remains exposed because there is not enough independent
  compute, delayed wgrad gives the scheduler another compute window by separating
  wgrad from dgrad. Keep it only when the trace shows wgrad covering otherwise
  exposed communication and end-to-end iteration time improves.
- Tune the dispatcher's SM allocation and
  `--high-priority-a2a-comm-stream` against end-to-end time. Both can improve
  communication progress while slowing concurrent GEMMs.
- With PP/VPP or hybrid dense/MoE stacks, inspect the actual per-stage layer
  balance and steady-state duration; VPP count alone does not predict overlap.
- When memory becomes the limit, use the tensor-lifetime evidence above to
  determine whether the peak comes from allocator pinning, attention
  activations, graph-fixed buffers, or concurrently live microbatches before
  selecting a mitigation.

## Troubleshooting and cautions

### The job runs but there is no speedup

- Confirm that the iteration has at least two microbatches and a nonempty 1F1B
  steady state after PP/VPP warmup.
- Confirm that `CUDA_DEVICE_MAX_CONNECTIONS` is greater than 1 and that A2A kernels
  actually overlap compute in a trace.
- Inspect the full MoE path for device-to-host copies, host synchronizations, or
  CPU-side shape/metadata decisions. A host sync serializes the schedule
  regardless of which dispatcher backend caused it.
- Check the CPU launch timeline. If the host enqueues a communication kernel too
  late, the GPU has no opportunity to overlap it with the preceding compute
  kernel even when their data dependencies permit overlap.
- If the communication kernel is enqueued on time but starts late, inspect the
  compute kernel's SM occupancy. A persistent compute kernel that occupies every
  SM can prevent the communication kernel from being scheduled. Launch the
  communication first when dependencies allow, or tune the compute kernel to
  leave enough SM capacity for concurrent communication.
- Check whether simultaneous TP/CP collectives or pipeline P2P consume the same
  network bandwidth, copy/communication resources, or SM capacity as EP A2A.
  This resource sharing can make both operations slower even though the trace
  shows temporal overlap.
- Compare traces with and without `--delay-wgrad-compute`. The flag is useful
  only when the separately scheduled wgrad kernels fill a window that would
  otherwise contain exposed A2A. Otherwise the extra host launches and schedule
  steps can leave iteration time unchanged or make it worse.
- Test the high-priority A2A stream only as a measured experiment. Stream
  priority can favor pending communication work, but it cannot preempt thread
  blocks that are already running.

### The job hangs

- Verify PyTorch 2.6 or newer.
- Reproduce with `alltoall`, no CUDA Graphs, no delayed wgrad, and no offloading.
- Check that every rank has the same effective pipeline layout, microbatch count,
  and dispatcher metadata.
- Check the first failing collective in an Nsight Systems trace or NCCL log; do not
  hide the issue by disabling validation.

### Peak memory increases

Combined 1F1B normally increases memory usage. Compute-stream and
communication-stream buffers are managed in different stream-specific allocator
reuse domains. A block released on one stream cannot immediately satisfy an
allocation on another stream while recorded work may still use it. The caching
allocator therefore reserves additional blocks and has fewer opportunities to
coalesce or reuse them across the two streams.

Consequently, a larger gap between **Max Reserved Memory** and
**Max Allocated Memory** is expected: it usually represents increased allocator
fragmentation and cached blocks that are temporarily unusable across streams,
not necessarily a tensor leak. Concurrently live forward/backward work can also
raise Max Allocated Memory. Compare both metrics and inspect a memory snapshot
before choosing a mitigation.

`--ep-overlap-early-attn-memory-release` moves attention backward before the
overlapped forward MLP so attention activations can be released sooner. This
explicitly trades overlap coverage—and potentially throughput—for lower peak
memory; it can expose dispatch/combine communication that the default order
would hide. Use it only when the memory saving is required and remeasure
end-to-end iteration time. Recompute or supported activation offloading are
alternative memory/compute tradeoffs. For Megatron-FSDP, follow the version-specific
buffer restrictions above.

### Losses differ slightly from the baseline

Different legal operation orderings can change floating-point accumulation order,
especially with shared experts, low precision, or nondeterministic kernels. Run the
A2A-overlap unit tests and a deterministic short A/B test. Large or growing
differences indicate a bug; do not dismiss them as expected reordering.
