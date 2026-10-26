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

This guide describes the current `dev` implementation, its configuration and
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
| `attn` | Attention, pre-MLP normalization, routing, and local dispatch preprocessing | Compute |
| `moe_dispatch` | Token dispatch A2A | Communication |
| `mlp` | Routed-expert computation | Compute |
| `moe_combine` | Token combine A2A and local postprocessing | Communication |
| `mhc_post` | Optional mHC MLP-side postprocessing | Compute |
| `mtp_post_process` | Optional MTP output and loss work | Compute |

Dense layers use no-op dispatch/combine nodes, so mixed dense/MoE decoder stacks
preserve a uniform scheduling interface.

A CUDA event shared by the schedule plan orders dependent nodes across the
compute and communication streams. Independent kernels may then run
concurrently. For one paired forward/backward layer, the issue order is:

```text
F.attn
B.combine
F.dispatch
B.expert dgrad [and delayed wgrad]
B.dispatch
F.expert
F.combine
B.attn [and delayed wgrad]
```

The dependency event makes `F.dispatch` wait for `F.attn`, `B.expert` wait
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
backward combine. mHC postprocessing has its own compute-stream node after
combine; any mHC group replay runs on that same stream immediately before its
backward. These placements avoid allocating a recomputed tensor on one stream
and consuming it on another.

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

1. A node with `free_input=True` calls `record_stream(consumer_stream)`, then
   resizes the input storage to zero after launching its forward work.
   `record_stream` prevents the caching allocator from reusing that block until
   work already queued on the consumer stream completes.
2. Inputs needed by autograd retain their storage. Dispatcher metadata is
   cleared only when full recomputation recreates it, and schedule-owned
   references are cleared after their final backward user.

`free_input` is selected per node rather than enabled globally:

| Node | `free_input` selection | Why |
| --- | --- | --- |
| Dense-layer nodes | `False` | Backward needs the original inputs. |
| `moe_combine` | `True` | The input is no longer needed after combine forward launches. |
| `moe_dispatch` | `True` only for the standard, non-DeepEP/DeepEPv2/HybridEP/NCCL-EP path and when `moe_preprocess` is not CUDA-graphed | Those Flex backends retain dispatch inputs for backward; CUDA Graph preprocess inputs are fixed buffers. |
| `mlp` | `True` for FP8/FP4; otherwise only when the input is not passed unchanged into GroupedGEMM | Low precision saves a cast tensor. The BF16/FP16 identity-postprocess cases—standard A2A with one local expert, HybridEP, and NCCL-EP—must retain the original input. |
| `mhc_post` and other nodes | `False` | Their backward path still owns or saves the input. |

The default path relies on allocator `record_stream` tracking. The latest
`dev` branch also includes the experimental schedule-aware release mode added by
[PR #7062](https://github.com/NVIDIA/Megatron-LM/pull/7062). Enable it through
the model-config field below; it defaults to `false` and requires
`overlap_moe_expert_parallel_comm: true`:

```yaml
ep_overlap_use_scheduled_tensor_release: true
```

This mode records the producer stream for every tensor owned by the schedule.
A same-stream consumer can apply the release immediately. For a cross-stream
consumer, the manager retains a strong reference until the producer stream
waits on the shared schedule event, then drains the deferred release. This
makes storage reusable at a schedule-defined handoff instead of waiting for the
allocator's more conservative cross-stream lifetime.

The release action still follows the node's `free_input` decision:
`free_input=True` empties forward-input storage, while `False` only ends plan
ownership and leaves storage under autograd or caller control. Backward output
gradients use a drop-reference action rather than resizing storage. Tensors that
enter outside the managed node chain retain the `record_stream` fallback; live
outputs are exported at plan boundaries, pending releases are drained at phase
finalization, and distinct tensor objects that alias one storage are rejected.

Enable this mode only on the Combined 1F1B path when investigating excess
allocator retention or a growing reserved-versus-allocated memory gap. Compare
both peak memory and iteration time against the default path. Its merged tests
cover the regular schedule, CUDA Graphs, delayed wgrad, and FSDP integration,
but they do not establish a general end-to-end speedup.

Do not add eager deletion, storage aliases, or new cross-stream work to a
schedule callable without updating these ownership rules.

## Requirements

### Hard requirements

These conditions are enforced for correctness:

| Area | Required setting | Reason |
| --- | --- | --- |
| Model path | `--use-mcore-models` and a training `forward_step` that accepts `return_schedule_plan=True` | Combined 1F1B asks the model for fine-grained nodes instead of running a normal opaque forward. `GPTModel.build_schedule_plan()` is the production implementation. |
| PyTorch | 2.6 or newer | Older PyTorch versions have a known hang in this overlap path. |
| EP | `--expert-model-parallel-size > 1` | With EP=1 there is no expert-parallel A2A to overlap. |
| Dispatcher | `--moe-token-dispatcher-type alltoall` or `flex` | The `allgather` dispatcher does not expose the required dispatch/combine node interface. |
| Base dtype | `--bf16` or `--fp16` | FP8/FP4 can be enabled on top, but the model still requires a supported base dtype. |
| Training mode | `forward_only=False` | The schedule pairs forward with backward and is not selected for inference/evaluation-only execution. |
| PP > 1 | Enable VPP with `--num-layers-per-virtual-pipeline-stage` or an equivalent pipeline layout | The multi-stage implementation extends the interleaved 1F1B schedule and therefore needs more than one virtual model chunk. |
| MTP | Zero or one MTP layer | Nested multi-layer MTP is not expanded by the current schedule plan. |

`--delay-wgrad-compute` additionally requires
`--transformer-impl transformer_engine` and TE 2.3 or newer. It is not a
requirement for Combined 1F1B itself.

### Conditions for useful overlap

These are performance conditions, not startup requirements:

- At least two microbatches are needed to form a forward/backward pair. With
  PP/VPP, enough microbatches must remain after warmup to create a steady state.
- `CUDA_DEVICE_MAX_CONNECTIONS > 1` lets compute and communication streams make
  progress concurrently. Official EP-overlap recipes commonly use 32.
- A trace should show exposed dispatch/combine communication before enabling the
  schedule. Already-hidden A2A leaves little work for Combined 1F1B to improve.

## Minimal configuration

The following YAML fragment shows the core settings for the standard All-to-All
dispatcher:

```yaml
ENV_VARS:
  CUDA_DEVICE_MAX_CONNECTIONS: '32'

ARGS:
  use_mcore_models: true
  transformer_impl: transformer_engine
  bf16: true

  num_experts: 128
  expert_model_parallel_size: 8
  expert_tensor_parallel_size: 1
  moe_token_dispatcher_type: alltoall

  overlap_moe_expert_parallel_comm: true

  micro_batch_size: 1
  global_batch_size: 256
```

The equivalent feature flags in a command-line training script are:

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=32

torchrun ... pretrain_gpt.py \
  --use-mcore-models \
  --transformer-impl transformer_engine \
  --bf16 \
  --num-experts 128 \
  --expert-model-parallel-size 8 \
  --expert-tensor-parallel-size 1 \
  --moe-token-dispatcher-type alltoall \
  --overlap-moe-expert-parallel-comm \
  ...
```

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
| `ep_overlap_use_scheduled_tensor_release` (model config) | `false` | Uses plan events and producer-stream ownership to release managed cross-stream tensors; external tensors keep the `record_stream` fallback. Requires Combined 1F1B. | Test when allocator retention or fragmentation increases the reserved-versus-allocated memory gap. Keep it only after measuring both memory and iteration time. |

`--overlap-dispatch-backward-with-experts-wgrad` is a different optimization.
It overlaps combine backward with expert wgrad and is mutually exclusive with
combined 1F1B and `--delay-wgrad-compute`.

### Parallelism and batch shape

| Setting | Effect and guidance |
| --- | --- |
| `--expert-model-parallel-size` | Must be greater than 1. Larger or cross-node EP domains often create more A2A latency to hide, but may also demand a faster dispatcher backend. |
| `--pipeline-model-parallel-size` | PP=1 uses the non-pipeline combined schedule. PP>1 uses the interleaved schedule and requires VPP. |
| `--num-layers-per-virtual-pipeline-stage` | Creates VPP chunks. Smaller chunks reduce the interleaved pipeline bubble but increase P2P frequency and scheduling overhead. Tune rather than minimizing blindly. |
| `--micro-batch-size` / `--global-batch-size` | Determine the number of microbatches. More microbatches improve fill/drain amortization and provide more overlap opportunities; they may increase iteration latency and interact with optimizer semantics. |
| `CUDA_DEVICE_MAX_CONNECTIONS` | Values greater than 1 allow the compute and communication streams to make progress concurrently; recipes commonly use 32. On Hopper and earlier, TP/CP overlap often prefers 1, so the best value must be profiled. |

On Blackwell and newer architectures the previous TP/CP requirement for
`CUDA_DEVICE_MAX_CONNECTIONS=1` no longer applies. On Hopper and earlier, combining
TP or CP with EP A2A overlap creates a tuning conflict: use 1 when prioritizing
TP/CP launch ordering or a larger value such as 32 when prioritizing EP overlap.

### CUDA Graphs

Combined 1F1B supports two different CUDA Graph strategies. They have different
capture boundaries and should not be configured as if they were interchangeable.

#### Partial CUDA Graphs

Partial capture graphs only static per-layer regions and leaves
`moe_dispatch`, expert compute, and `moe_combine` visible to the two-stream
scheduler. The tested TE path is:

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
- Capturing attention wgrad, or a non-overlapped shared-expert wgrad, requires
  TE 2.12 or newer and `--gradient-accumulation-fusion`. Attention bias is not
  supported in that combination.

See the
[Qwen3-235B partial-CG overlap recipe](../../../examples/moe_recipes/qwen3_235b/gb200/mxfp8_128GPU_TP1PP1EP64_partial_cg_overlap.yaml)
for a complete configuration.

#### Full-iteration CUDA Graphs

Full-iteration capture wraps the forward/backward training schedule, excluding
the optimizer step, while retaining the fine-grained compute and communication
streams inside the graph:

```bash
--cuda-graph-impl full_iteration \
--no-check-for-nan-in-loss-and-grad
```

`--cuda-graph-modules` must be empty in this mode. Dynamic dropless MoE routing
cannot be captured, so the reference path uses static, synchronization-free
routing: HybridEP plus an expert-rank capacity factor, paged stash, and the TE
op fuser. See [Paged Stash](paged_stash.md) and the
[Qwen3-235B full-CG overlap recipe](../../../examples/moe_recipes/qwen3_235b/gb200/mxfp8_128GPU_TP1PP1EP64_paged_stash_fullcg_overlap.yaml).

That recipe still uses the deprecated pair
`cuda_graph_impl: local` / `cuda_graph_scope: full_iteration`, which the
argument parser migrates automatically. New configurations should use
`cuda_graph_impl: full_iteration`.

Layer-level full activation recomputation can be combined with CUDA Graphs only
through full-iteration capture. Megatron-FSDP also permits full-iteration
capture on its supported PP=1 Combined 1F1B path.

#### CUDA Graph cautions

- Partial and full-iteration capture are alternative modes; do not supply
  per-layer modules with `--cuda-graph-impl full_iteration`.
- Use a graph-safe Transformer Engine RNG tracker and at least one warmup step.
  Reference overlap recipes use `--te-rng-tracker` and two warmup steps.
- Balanced DSA dynamic-pack routing is currently incompatible with CUDA-graphed
  Combined 1F1B plus delayed wgrad because its pack composition changes.
- Fixed graph input buffers cannot be resized or released. Revisit
  `free_input` whenever a callable is moved into or out of graph scope.
- Full-iteration MoE capture requires static shapes throughout the iteration;
  paged stash is the documented path, not a general guarantee for every
  dispatcher backend.
- The reference recipes use
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,graph_capture_record_stream_reuse:True`
  to reduce allocator fragmentation and enable graph-aware reuse. Treat this as
  part of the measured recipe rather than an unconditional requirement.

### Activation recomputation

Selective recomputation is supported, but `moe` must not appear in
`--recompute-modules` because the overlap schedule already decomposes the MoE
layer. Full recomputation is implemented as a backward-time replay of layer
segments rather than through the standard checkpoint primitive.

For `--recompute-granularity full`:

- Set both `--recompute-method` and `--recompute-num-layers` explicitly.
- `uniform` groups decoder layers by `recompute_num_layers`.
- `block` recomputes the first `recompute_num_layers` decoder layers, one segment
  per layer.
- MTP uses one segment per depth under either method.
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

Megatron-FSDP is supported on the PP=1 combined schedule. Interleaved PP/VPP with
FSDP is not supported because the multi-chunk path does not yet manage the FSDP
root pre/post-backward lifecycle.

Additional constraints are:

- Use no per-layer CUDA Graphs: `cuda_graph_impl` must be `none` or
  `full_iteration`.
- Full activation recomputation with combined 1F1B and Megatron-FSDP is not yet
  supported.
- `--megatron-fsdp-prefetch-recompute-forward-weights` is incompatible with the
  overlap schedule.
- If persistent FSDP communication buffers are enabled, set
  `--fsdp-buffer-count 3` or greater. A backward/recompute unit, the current
  forward unit, and a prefetched successor may be live concurrently.
- The overlap path installs fine-grained parameter gather/reshard hooks because it
  calls layer submodules directly instead of `TransformerLayer.forward()`.

### Other feature interactions

| Feature | Status / constraint |
| --- | --- |
| Fine-grained activation offloading | Supported and covered by compatibility tests. For TE 2.10+, set `NVTE_CPU_OFFLOAD_V1=1`. With `ncclep`, offloading `expert_fc1` is not supported because that path retains an input that the overlap schedule releases. |
| Shared experts | Supported, but `--moe-shared-expert-overlap` must be disabled. Combined scheduling changes the order in which router/shared-expert gradients are added, so small numerical differences are expected. |
| MTP | Supported with `mtp_num_layers` unset or equal to 1. |
| mHC | Selective mHC recomputation and supported attention CUDA Graph scopes are integrated into the schedule. |
| FP8 / FP4 | Supported on top of BF16/FP16, subject to the full-recompute delayed-scaling restriction. |
| Hash MoE | `moe_n_hash_layers > 0` is not supported. |
| MOK megakernel | Not supported because MOK replaces the native dispatcher/expert/combine schedule. |
| MoT / Bagel MoT layer | Not supported by the combined schedule. |
| HybridModel MTP | Not supported because the current schedule does not expand the nested `HybridStack`. |

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

The same analysis explains when the benefit changes:

- More microbatches increase the steady-state fraction and amortize the exposed
  first-forward/last-backward phases.
- Fine-grained MoE and cross-node EP tend to have a larger A2A fraction to hide.
- Splitting dgrad and wgrad increases the available backward compute window when
  forward compute alone is too short.
- Communication kernels consume SM resources. The report's DeepEP example
  reserves 20 SMs per GPU and observes about a 20% GEMM-efficiency cost, so
  maximizing communication progress does not necessarily minimize iteration
  time.
- For PP/VPP, one extra forward warmup microbatch removes the steady-state
  dependency. More virtual chunks create more pairing opportunities, but hybrid
  models still require a balanced pipeline layout.

Repository recipe TFLOP/s numbers are intentionally omitted here because those
recipes change precision, dispatcher, topology, CUDA Graph scope, offloading,
and other optimizations together. They do not isolate the effect of A2A overlap.

### Reproduce the ablation

Use the same dispatcher and communication tuning in every run:

1. Fix the commit, container, model, topology, parallel mapping, batch shape,
   precision, CUDA Graph mode, and random seed.
2. Measure the dispatcher-optimized baseline without
   `--overlap-moe-expert-parallel-comm`.
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
alternative memory/compute tradeoffs. For FSDP persistent pools, three live
buffer slots may still be required.

### Losses differ slightly from the baseline

Different legal operation orderings can change floating-point accumulation order,
especially with shared experts, low precision, or nondeterministic kernels. Run the
A2A-overlap unit tests and a deterministic short A/B test. Large or growing
differences indicate a bug; do not dismiss them as expected reordering.
