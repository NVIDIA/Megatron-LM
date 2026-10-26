# MoE load balancing (experimental)

MoE Scheduler addresses load imbalance in dropless MoE models by replicating
hot logical experts into idle physical expert slots and rerouting tokens to the
replicas. It is designed as a preprocessing stage between router output and the
existing Megatron token dispatcher.

The scheduler does not introduce planner-specific branches into the remaining
MoE execution path. After scheduling, `MoELayer` continues through its normal
token preprocessing, dispatch, expert computation, combine, and backward flow.

## Integration with upstream MoonEP

The implementation is based on dev `dd8f93b57` (including PR #7450). The existing
`moe_virtual_expert_load_balance` flag keeps the accepted compact-route,
GTP/TE, stream, and backward lifecycle. `MoELayer` exposes it through the same
`MoEScheduler` ownership interface as the configurable backends; the two enable
flags are mutually exclusive so one layer cannot install two expert runtimes.

The upstream implementation is separated into `moe_scheduler_plan.py`
(placement and per-forward execution state), `moe_scheduler_runtime.py`
(expert storage and GTP/TE binding), `moonep_peer_transport.py` (kernel submission),
and `moonep_execution_runtime.py` (execution ordering). The original
`VirtualExpertLoadBalancer` import is an alias of that execution class, sharing
its arena caches and teardown. Its kernels remain in `ops/virtual_expert_triton.py`.

The configurable MoonEP planner calls the same upstream cooperative kernel and
capacity policy. Compact IDs/scores remain compact through rerouting and retain
the probability autograd graph. Dense bool inputs, including mock routing, are
adapted explicitly. The former split planner is retained only under
`tests/unit_tests/transformer/moe/moonep_reference*.py` as an independent oracle.

ECHO/EPLB, arbitrary replica counts, shared/layerwise storage, and bounded NCCL
communication use the configurable replica execution backend. It is distinct
from the fixed-slot upstream MoonEP backend because those layouts and storage
policies have different contracts; only one is bound to any given layer.
Zero-replica EPLB keeps native compute, and home exchange/checkpoint boundaries
remain independent of replica transfer. Latest-dev TrainState is reset when
initializing home ownership from an ordinary logical checkpoint.

The unquantized native expert fallback trims unused rows from MoonEP's static
receive budget before TE splits expert segments, then zero-pads the output back
to the combine budget. This path already needs host counts; native grouped-tensor
and quantized kernels continue to consume device offsets without this host read. This also
handles BF16 overrides inside an otherwise MXFP8 MTP model.

## Design Goals

- Decouple load planning from expert-weight movement so planners and expert
  dispatch backends can be combined through one contract.
- Keep backend-native state, such as Echo offloading maps, EPLB replica tables,
  and MoonEP replica plans, private to concrete implementations.
- Split placement from token rerouting so asynchronous expert-weight dispatch
  can overlap reroute work without exposing planner-specific state.
- Make planners return final token reroute tensors so token dispatchers do not
  need to understand the selected planning algorithm.
- Keep the public planner/dispatcher boundary small enough to lower efficiently
  into different expert communication backends.

## Architecture

| Component | Responsibility |
| --- | --- |
| `SchedulerContext` | Carries layer, logical/local expert, EP rank/group, router top-k, training mode, and configuration context. |
| `MoEPlacementResult` | Opaque, explicit state passed between a planner's placement and reroute phases. |
| `MoELoadPlanner` | Updates the physical layout, and reroutes tokens against that placement. |
| `ExpertDispatch` | Validates/materializes a physical placement and owns forward/backward lifecycle hooks. |
| `MoEScheduler` | Orchestrates placement, asynchronous expert materialization, and token rerouting. |
| `MoELayer` | Invokes the scheduler between routing and token preprocessing. |

The planner contract is split into two phases:

```python
home_expert_placement, replica_expert_placement, placement_result = planner.update_placement(
    probs, routing_map, context
)
routing_map, probs = planner.reroute(
    probs, routing_map, placement_result, context
)
```

Logical mappings are private to the planner. `home_expert_placement` is an
optional `HomeExpertPlacement(source_slots, version)` with a `[EP, H]` source
permutation. `replica_expert_placement` is an optional `[EP, S]` integer table,
with `-1` for inactive replicas. Both tables are destination-major and encode
**physical** sources as `rank * H + local_home_slot`. `placement_result` is
opaque reroute state. `reroute()` returns either dense physical route/probability
tensors or compact physical IDs with matching top-k probabilities.

`MoEScheduler` calls `HomeExpertDispatch` and `ReplicaExpertDispatch` directly.
Home dispatch stages metadata until `scheduler.step()`; replica dispatch starts
weight prefetch immediately and retains the source table through backward. Both
use the same transport implementation. Zero-replica EPLB instead uses
`HomeOnlyExpertDispatch` with native expert execution and home-only transport
storage (see below). Replica runtime owns TE/GTP bindings and
gradient handoff. Transport never interprets logical identity or optimizer state.

## Class Diagram

```mermaid
classDiagram
    MoELayer --> MoEScheduler
    MoEScheduler o-- MoELoadPlanner
    MoEScheduler o-- ReplicaExpertDispatch
    MoEScheduler o-- HomeExpertDispatch
    HomeExpertDispatch --> ReplicaWeightTransport
    MoELoadPlanner <|-- EchoLoadPlanner
    MoELoadPlanner <|-- EPLBLoadPlanner
    MoELoadPlanner <|-- MoonEPLoadPlanner
    ExpertDispatch <|-- ReplicaExpertDispatch
    MoEPlacementResult <|-- EchoPlacementResult
    MoEPlacementResult <|-- EPLBPlacementResult
    MoEPlacementResult <|-- MoonEPPlacementResult
    EchoLoadPlanner --> EchoPlacementResult
    EPLBLoadPlanner --> EPLBPlacementResult
    MoonEPLoadPlanner --> MoonEPPlacementResult
    MoEScheduler --> ReplicaExpertDispatch
    ReplicaExpertDispatch o-- ReplicaExpertRuntime
    ReplicaExpertRuntime o-- ReplicaWeightTransport
    ReplicaWeightTransport <|-- PeerTmaTransport
    ReplicaWeightTransport <|-- HybridEPWeightTransport
    ReplicaWeightTransport <|-- NcclP2PTransport
    ReplicaWeightTransport --> ReplicaPreparedPlan
    ReplicaPreparedPlan --> ReplicaPlacement
    ReplicaPlacement --> ReplicaOwnership
```

## Implemented Components

| Type | Config value | Implementation | Status |
| --- | --- | --- | --- |
| Planner | `echo` | `EchoLoadPlanner` | CUDA/Triton assignment and token reroute aligned with Echo PR #2368. |
| Planner | `eplb` | `EPLBLoadPlanner` | Greedy replication, optional periodic home LPT permutations, and global round-robin rerouting. |
| Planner | `moon_ep` | `MoonEPLoadPlanner` | Upstream cooperative placement/rerouting kernel with symmetric-memory histogram exchange. |
| Expert dispatch | `replica_peer_tma` | `ReplicaExpertDispatch` | One replica lifecycle implementation; the type currently selects `PeerTmaTransport`. |
| Expert dispatch | `replica_hybridep` | Same dispatcher, `HybridEPWeightTransport` | Reserved; rejected during configuration validation. |
| Expert dispatch | `replica_nccl` | Same dispatcher, `NcclP2PTransport` | Packed NCCL P2P; BF16/MXFP8 weights, BF16/FP32 gradients, host planning. |

UltraEP is a planned integration. It should implement the same common planner
output or expert-dispatch input semantics instead of exposing UltraEP-native
metadata in the public interface.

## Execution Flow

1. The dispatcher wraps the MoE layer input when backward work must complete
   after router, shared-expert, and latent-projection backward.
2. The router produces logical-expert `probs` and `routing_map`.
3. `MoELayer` calls `MoEScheduler.schedule()`.
4. Every enabled scheduler invocation plans, dispatches, and reroutes.
5. `update_placement()` returns optional home/replica source tables and an
   opaque `MoEPlacementResult`.
6. The scheduler stages a home proposal and asynchronously starts current-home
   replica copies. Pending home ownership is never used by this forward.
7. `reroute()` converts the original logical routes to physical routes while
   expert-weight communication is in flight.
8. The existing token dispatcher consumes the physical routing tensors and
   runs the normal dispatch, expert compute, and combine stages.
9. During backward, the replica runtime starts FC2 reduction directly behind
   its wgrad GEMM, starts pending FC1/FC2 reductions after dispatch backward,
   and waits at the layer input before publishing source gradients.

## Configuration

A minimal Echo configuration is:

```yaml
moe_enable_scheduler: true
moe_scheduler_planner_type: echo
moe_scheduler_expert_dispatcher_type: replica_peer_tma
moe_scheduler_num_idle_experts: 4
```

**Configuration migration:** the former `replica_hybridep` value selected
Peer-TMA, despite its name. It has been renamed to `replica_peer_tma`, which
is also the default. Update existing YAML/CLI configurations to that value.
`replica_hybridep` is now reserved for the actual HybridEP weight backend and
fails with a migration message; it never silently selects a different data
path. Both implemented transports use `ReplicaExpertDispatch` from
`replica_expert_dispatch.py`; the common dispatcher has no transport-specific
name or compatibility alias.

Set `moe_scheduler_expert_dispatcher_type: replica_nccl` to use packed NCCL
P2P with BF16 or native MXFP8 weights and BF16 or FP32 gradient storage. It requires an
initialized NCCL EP group and the current CUDA device to match the runtime
device. Construction is collective across that group. CUDA graph capture
including MoE is rejected; attention-only graphs remain supported. MXFP8
requires `fp8: e4m3`, `fp8_recipe: mxfp8`, and `fp8_param: true`.

For MoonEP, set `moe_scheduler_planner_type: moon_ep`. The current MoonEP
implementation allocates one replica slot for every home expert, so
`moe_scheduler_num_idle_experts` must equal `num_moe_experts`.

For EPLB, set `moe_scheduler_planner_type: eplb`. `EPLBLoadPlanner` gathers the
current logical-expert loads across the EP group, greedily assigns redundant
instances according to `load / replica_count`, and LPT-packs those instances
into the fixed replica slots. It follows the core replication and packing
policy of [DeepSeek EPLB](https://github.com/deepseek-ai/EPLB). Homes remain fixed
by default; `moe_scheduler_home_update_interval` enables periodic migration.

Positive-replica configurations use the transport-backed `ReplicaExpertRuntime`. It supports an
`E + R` runtime layout, where `R` is positive and divisible by the EP size.
Every rank owns `E / EP` native experts followed by `R / EP` replica slots. It
requires the HybridEP flex token dispatcher, BF16 execution, TE grouped GEMM
with the operation fuser, and fused gradient accumulation. With Peer-TMA or NCCL, weights may be BF16
or native-parameter MXFP8 E4M3; FP4 and other FP8 recipes are rejected. The
current `replica_peer_tma` type selects `PeerTmaTransport`, which requires a
single NVLink domain and a PyTorch/NCCL build with working native NCCL
symmetric-memory support.

Current constraints:

- Dropless MoE only; expert capacity and capacity padding must be disabled.
- `num_moe_experts` and `moe_scheduler_num_idle_experts` must be divisible by
  the expert-model-parallel size.
- `add_bias_linear` must be disabled.
- Both replica expert dispatchers require discrete native expert
  weights and the Transformer Engine operation fuser; single grouped expert
  weights are unsupported.
- The HybridEP flex token dispatcher requires a build with HybridEP support.
- The MoonEP planner requires CUDA, initialized EP distributed groups, and one
  replica slot per local home expert. Its PR #7450 placement kernel uses an
  NCCL symmetric-memory histogram window; rerouting is a separate kernel.
- EPLB uses the current invocation's EP-global token counts for replica placement
  with fixed homes. With home migration enabled, it accumulates loads over the
  update interval and commits proposed homes after a successful optimizer step.
  Hierarchical multi-node expert-group placement is not implemented.
- MoE CUDA graph capture requires `replica_peer_tma`, an explicit
  `moe_expert_rank_capacity_factor`, and a storage policy without exact weight
  caching (shared storage, or MXFP8 layerwise storage with `reuse`). Separate
  `moe_router` and `moe_preprocess` scopes are rejected. Attention-only capture
  is allowed for both transports and all storage policies. MoE activation recompute is also rejected for this lifecycle.
  The physical-expert token-dispatcher config does not own graph capture;
  it clears its layer-level graph policy while the enclosing logical MoE
  layer retains the requested capture scopes. Dispatcher kernels execute
  within that enclosing graph. MoonEP converts replica placement to the common
  physical layout with fixed-shape selection, so changing inactive replica slots
  on graph replay never requires dynamic-size boolean indexing.
- When GTP exposes #6892's non-consuming peek protocol, the weight push peeks
  at gathered weights and the expert GEMMs perform the real consume. Older GTP
  implementations retain the pre-existing consume-at-push compatibility path.

Rank 0 logs the configured planner, dispatcher, and idle expert count at startup. The first scheduled forward also logs routing
shapes, physical expert and transfer counts, and whether planning and expert
materialization ran.

## Code Map

| Path | Purpose |
| --- | --- |
| `megatron/core/transformer/moe/moe_scheduler.py` | Shared two-phase planner interfaces and scheduler orchestration. |
| `megatron/core/transformer/moe/echo_moe_scheduler.py` | Echo planner and Triton reroute path. |
| `megatron/core/transformer/moe/eplb_moe_scheduler.py` | EPLB greedy replication, fixed-home LPT placement, and round-robin reroute. |
| `megatron/core/transformer/moe/moonep_moe_scheduler.py` | MoonEP/PR #7450 planner adapter and common-IR conversion. |
| `megatron/core/transformer/moe/moonep_replica_triton.py` | #7450 histogram/placement and route-mapping Triton kernels. |
| `megatron/core/transformer/moe/replica_expert_dispatch.py` | Common placement adapter and replica forward/backward lifecycle. |
| `megatron/core/transformer/moe/replica_expert_runtime.py` | TE/GTP runtime weights, gradient handoff, and transport coordination. |
| `megatron/core/transformer/moe/replica_weight_transport.py` | Backend-neutral replica weight/gradient transport contract and factory. |
| `megatron/core/transformer/moe/replica_peer_tma_transport.py` | Symmetric-memory peer-TMA transport and shared workspace. |
| `megatron/core/transformer/moe/replica_hybridep_transport.py` | Reserved HybridEP weight backend; not implemented. |
| `megatron/core/transformer/moe/replica_nccl_transport.py` | Packed NCCL P2P, host schedules, and FP32 gradient accumulation. |
| `megatron/core/transformer/moe/replica_weight_triton.py` | Peer-TMA weight transport and projection-selective gradient-reduction kernels, generalized to variable replica slots. |
| `megatron/core/transformer/moe/moe_layer.py` | Integration between logical routing and the existing token dispatcher. |
| `megatron/core/transformer/transformer_config.py` | Scheduler configuration and compatibility validation. |
| `tests/unit_tests/transformer/moe/test_moe_scheduler.py` | Common contract and orchestration tests. |
| `tests/unit_tests/transformer/moe/test_echo_moe_scheduler.py` | Echo planner, unified replica dispatch, and `MoELayer` integration tests. |
| `tests/unit_tests/transformer/moe/test_eplb_moe_scheduler.py` | EPLB placement, global reroute, gradient, and factory tests. |
| `tests/unit_tests/transformer/moe/test_moonep_moe_scheduler.py` | MoonEP planner and cross-component compatibility tests. |
| `tests/unit_tests/transformer/moe/test_replica_weight_transport.py` | Plan ownership, layout, completion lifetime, and placeholder contracts. |
| `tests/unit_tests/transformer/moe/test_replica_nccl_transport.py` | NCCL weight/gradient parity, strided EP subgroups, empty ranks, and stream completion. |

## Replica Transport Contract

The planner lowers logical choices to physical source slots. Replica dispatch
retains an immutable source table through forward/backward and assigns a
dispatcher-local generation. `ReplicaExpertRuntime` wraps that slot table in
`ReplicaPlacement`, with a separate `ReplicaOwnership` descriptor. The current
owner mapping is uniform and fixed; optional explicit owner tables reserve an
extension point and are rejected by Peer-TMA and NCCL P2P. Periodic home exchange
moves contents among these fixed physical sources, so replica transport needs
no logical ownership table. Each rank still has H home slots followed by S replicas.

`transport.prepare_plan(placement)` produces a `ReplicaPreparedPlan` belonging
to that exact transport instance. Peer-TMA keeps device metadata; NCCL
compiles host peer schedules, while a future HybridEP implementation can
compile chunk routing. The common layer never expands chunk routing or reads the
placement back to the CPU. Runtime caches are weak and keyed by plan identity,
not the address of a recycled slot. A future host-scheduled backend must
document synchronization and reject unsupported graph capture. CUDA graph
replay can update device metadata without executing Python or incrementing a
Python generation: cached host schedules must never assume otherwise.

Each `ReplicaWeightSource` exposes canonical local-expert data and optional
scales, with explicit plain/rowwise/columnwise layout. Tensor shape and dtype
describe the components; wire encoding and packing belong to the transport.
Optional pointer tables remain a Peer-TMA fast path. TE wrappers, GTP peeks,
direction selection and optimizer handoff remain in the runtime.

Each start returns a distinct `ReplicaTransferHandle`, separate from reusable
scheduling metadata. A weight completion covers unpack and final layout;
a gradient completion covers `native += sum(replicas)`, including local
replicas, with FP32 accumulation and one final cast. Native gradients are not
cleared. Reduction order can differ between backends; bitwise equivalence is
not promised. Every wait orders its calling stream, including repeated waits
for resident weights. Handles retain operation inputs, but backends must also
protect allocator lifetimes and callers must not overwrite sources or buffers
while the GPU uses them.

Peer-TMA brackets each weight push with device-side cross-rank rendezvous.
Callers order their previous local consumers before starting the next transfer;
the entry rendezvous extends that readiness to remote owners before they write
peer arenas. The exit rendezvous makes the new generation available to consumers.
A planner collective cannot substitute for entry readiness: cached plans and
rowwise/columnwise reuse can issue consecutive transfers without a new collective.
The protocol needs no additional host synchronization and remains graph capturable.

The storage regression suite also delays a recipient before reading its old
weights while other ranks start the next transfer. It checks both BF16 and
MXFP8 reuse arenas with shared and layerwise storage, and drains communication
before reporting corruption so a first assertion cannot hide behind a peer timeout.

Storage views remain stable until teardown. The existing Peer-TMA shared
workspace and its serialized layer lifecycle are preserved; the interface
does not promise independent concurrent storage per layer. New transports
should first use per-layer buffers before introducing shared pools. Only
initialized backends register finalizers, so cleanup does not import unused
Triton/symmetric-memory dependencies. Capabilities advertise implemented
weight formats, gradient dtypes, device planning, graph support and explicit
ownership; placeholders advertise none. Topology compatibility remains a
backend initialization requirement, not an automatic fallback policy.

NCCL P2P can move weights across nodes without symmetric-memory allocation.
Full scheduler execution also requires compatible planner and token-dispatch paths; current scheduler
configuration still requires HybridEP tokens and the MoonEP planner still
uses symmetric-memory histogram exchange.

## NCCL P2P Transport Design

The NCCL backend supports plain BF16 or MXFP8 weights and either BF16 or FP32
replica/native gradient storage. It does not import Peer-TMA, Triton, or
symmetric-memory helpers. TE parameter wrappers, GTP materialization, backward
weight selection and optimizer handoff remain in `ReplicaExpertRuntime`.

`prepare_plan()` performs one synchronous device-to-host copy of the global
`[EP, replica_slots]` placement per immutable microbatch plan. It validates
physical source IDs, skips `-1`, and derives owners using uniform canonical
ownership. The prepared schedule stores only peer ranks, native indices and
replica slots. It never retains source addresses, so backward may use newly
materialized weights. All ranks must provide the same placement; no extra
placement or count exchange occurs in the steady-state path.

For weights, each owner packs one projection/component tile at a time into a
bounded send arena. Receives are views into the matching receive arena, ordered
by directed peer edge and destination slot. Local replicas use device copies;
unused slots are untouched. MXFP8 data and padded scale bytes are transferred
as separate uint8 components without conversion or assumed scale/data ratios.
Rowwise and columnwise storage follows the `reuse` or `separate` policy.

`moe_scheduler_replica_exchange_buffer_mb` (default 256 MiB) bounds combined
send, receive and FP32 reduction scratch. Three equal, 8-byte-aligned segments
are shared by serialized layers on the same EP group/device/budget, independent
of expert shapes. The pool owns one stream, so tile and cross-layer reuse are
ordered without host synchronization. The global placement determines the
largest transfer-slot count on any rank; all ranks use that same tile width.
The gradient width also reserves one FP32 tile per home expert. There is no
per-peer full-expert allocation or full-expert BF16-to-FP32 temporary. This
budget excludes persistent replica weights/gradients and NCCL internal storage.

Gradients follow the same routes in reverse, only for the requested
projections. The owner adds local and received replica gradients to the
existing native gradient in FP32, then casts once into the supplied native
gradient destinations. Unreferenced native experts remain untouched. FC2 and
FC1 can start separately, preserving the runtime's FC2-first backward order.

Both transports support shared or layerwise replica storage. Neither transport
allocates native gradient staging. Native runtime leaves bind directly to the
owner's full `main_grad` views. DDP accumulates; FSDP overwrites its current
unsharded bucket before replica contributions are added and its hook performs
reduce-scatter. Eager FSDP bucket address changes refresh both runtime leaves and
transport pointer tables; changed addresses remain forbidden during capture.
Eager FSDP also refreshes native weight/gradient views when the address is
unchanged: the allocator can reuse an address while the previous Tensor's
storage has been retired. Pointer equality alone does not establish ownership.
Owners exposing only sharded GTP gradients must supply a full-sized wgrad
destination through `main_grad`/`get_main_grad`; otherwise binding fails before
GEMM. No fallback native-gradient arena is created.

Megatron-FSDP v1 validates replica gradient precision during wrapper initialization,
before allocating FSDP parameter/gradient buffers. The effective `main_grad` dtype
is `megatron_fsdp_main_grads_dtype` (BF16 when unset for the supported models),
except that sharded expert gradients with expert-DP size greater than one use
`megatron_fsdp_grad_comm_dtype` when specified. It must equal FP32 with
`grad_reduce_in_bf16=False`, or BF16 with `grad_reduce_in_bf16=True`. A mismatch
raises an actionable `ValueError`; the scheduler never changes precision settings
or allocates native gradient staging. Zero-replica EPLB uses native computation
and does not impose this replica dtype restriction. Megatron-FSDP v2 is rejected
at initialization because its parameter/gradient ownership and home-state bindings
are not integrated yet.

ECHO and EPLB retain Boolean logical routing. Both MoonEP entry points use
compact int64 IDs and top-k probabilities when index routing is enabled. Fused
TE routing returns full-width probabilities, which the router gathers in top-k
order without detaching gradients. Dense mock routes are adapted by the planner.
Token dispatch receives the representation supported by its backend.
`moe_megakernel_backend` must remain unset: that execution path bypasses the
scheduler lifecycle and has no integrated replica computation contract.

Gradient-handoff tests model transport completion by returning the owner's
`main_grad` itself after accumulation or overwrite. Publication must not copy or
add that result again. Storage tests pass the replica-slot count explicitly to
the shared GEMM reference, keeping transport-layout coverage independent of its
slot-count parameterization.
FSDP parity legs use the same enabled gradient clipping, ensuring the single
optimizer actually computes the norm checked by the test. With clipping
disabled it reports a placeholder zero, unlike the chained DDP optimizer.

The built-in training DataLoader owns a separate CPU generator for worker base
seeds, initialized from the caller's current RNG state. Creating a replacement
iterator after checkpoint restore does not consume the restored training RNG.
This also applies with zero workers, where PyTorch still draws a base seed.
This isolation does not serialize arbitrary worker-side dataset augmentation
state; the checkpoint continuation validation uses deterministic mock THD data.
Startup logging preserves accumulators only between regular log boundaries.
When the first step after restart is itself a boundary (notably with
`log_interval=1`), it resets the interval so the next reported loss is not an
average of two steps. This affects reporting, not the optimizer update.

A start waits for
its producer stream, packs messages, submits batched P2P, waits on NCCL Work
objects on the communication stream, and enqueues unpack or accumulation.
The returned event covers all of that work. Every wait inserts an event wait
on its calling stream, including repeated waits on different streams. Normal
NCCL waits order GPU work; enabling NCCL blocking-wait settings can also block
the host. Temporary buffers and input references are retained internally
until the event completes, even if the caller drops the handle. Tensor stream
tracking protects allocator reuse. Callers must still order consumers before
explicitly overwriting sources or reusable replica storage.

Construction must run in matching order on every EP member. It performs a
small all-reduce on the borrowed NCCL group before any sparse batched P2P, so
ranks without traffic need not participate in subsequent P2P batches. All
layers and other users of the same group must maintain matching communication
submission order. `destroy()` drains this layer's communication stream and
releases its storage references, without destroying the borrowed group.

Capabilities are conservative: host planning and dynamic message sizes mean
`device_plan=False`, `cuda_graph=False`, and `explicit_ownership=False`.
Both plan preparation and operation starts reject CUDA capture, even for
plans prepared before capture. `num_sms` is unused because the backend does
not expose a portable per-operation NCCL SM limit. Future optimizations can
add peer-local deduplication and fused pack/reduce
kernels after measuring the baseline. Static CUDA-graph schedules require a
separate implementation and validation.

## Extending MoE Scheduler

To add a planner:

1. Subclass `MoELoadPlanner` and implement `update_placement()` and `reroute()`
   for every enabled scheduler invocation.
2. Return optional home/replica physical source tables plus a `MoEPlacementResult`
   subclass from `update_placement()`. Keep logical layouts and backend-specific
   allocation state private to the planner.
3. Return dense physical `routing_map` and `probs` from `reroute()`.
4. Register the planner in `MoEScheduler.from_config()` and
   `TransformerConfig` validation.
5. Add contract tests and at least one planner/dispatcher compatibility test.

To add a replica communication mode while retaining the single expert dispatcher:

1. Implement `ReplicaWeightTransport`, including transport-owned storage views.
2. Keep TE, GTP, MXFP8, and optimizer semantics in `ReplicaExpertRuntime`.
3. Register the transport against an expert-dispatch type in
   `create_replica_weight_transport()`; all types instantiate
   `ReplicaExpertDispatch`.
4. Compile only backend-private schedules in `prepare_plan()` and preserve
   layout, ownership, storage lifetime and asynchronous completion semantics.
5. Add factory, forward/backward lifecycle, and multi-rank correctness tests.

Run the focused unit tests from the Megatron-LM repository root:

```bash
python -m torch.distributed.run --nproc-per-node 8 -m pytest \
  tests/unit_tests/transformer/moe/test_replica_weight_transport.py \
  tests/unit_tests/transformer/moe/test_replica_nccl_transport.py \
  tests/unit_tests/transformer/moe/test_moe_scheduler.py \
  tests/unit_tests/transformer/moe/test_echo_moe_scheduler.py \
  tests/unit_tests/transformer/moe/test_eplb_moe_scheduler.py \
  tests/unit_tests/transformer/moe/test_moonep_moe_scheduler.py
```


## Periodic Home Expert Exchange (NCCL)

Enable alongside the existing scheduler configuration:

```yaml
moe_enable_scheduler: true
moe_scheduler_planner_type: eplb
moe_scheduler_expert_dispatcher_type: replica_nccl
moe_scheduler_num_idle_experts: 4
moe_scheduler_home_update_interval: 50
```

The first 50 successful updates collect logical load history. The first forward
after that interval proposes a permutation; migration runs after that iteration's
optimizer update. Exactly H unique canonical homes remain on every rank. LPT uses
per-instance load after greedy replica selection. This is a constrained EPLB
policy: unrestricted placements cannot always preserve a fixed home capacity.

EPLB keeps active and pending home/replica mappings. In periodic mode both logical
placements are committed together at the successful-step boundary; unchanged
replica placements still issue fresh weight copies on every microbatch. Each
microbatch reroutes against the active mapping and current home slots.
`HomeExpertDispatch.dispatch` snapshots only the proposal, never stale weights.
At the drained optimizer boundary, `MoEScheduler.step()` calls home dispatch
first and `planner.step(completed_version)` second. Failed optimizer updates
retain both the old mapping and pending proposal.

`prepare_home_expert_step(model, optimizer)` and `finish_home_expert_step(...)`
wire this boundary into the Megatron training loop. Custom training loops must
use the same boundary, or bind a state adapter and arrange communication fences
before calling each scheduler's `.step()`. All replica microbatches must finish.
Next-step DDP/FSDP gathers are deferred during migration, including the chained
optimizer MXFP8 gather path, and restarted only after weights are refreshed.

NCCL migrates one typed state component at a time using bounded expert tiles.
`moe_scheduler_home_exchange_buffer_mb` (default 256 MiB) limits the combined
send/receive staging, shared across serialized layers on the same device and EP
group. The budget excludes NCCL internal storage and existing training buffers.
For H local experts and dtype element size d, each tile contains at most
floor(floor(B / 2 / 8) * 8 / (H * d)) elements per expert. Fixed-point slots are
skipped; all moved source tiles, including local cycles, are packed before any
in-place destination writes. Peer messages are views into the two reusable
buffers, with no full-state snapshot, concatenation or per-peer allocation.
Each tile orders communication and installation before storage reuse on the
current stream; it does not synchronize the host per tile. One final EP barrier
and stream fence precede cross-layer reuse and mapping commit. Optimizer option
lookups scan groups once, and step bounds/options share one object collective
per group while retaining all compatibility checks. Any uncertain partial exchange poisons
the dispatcher and requires checkpoint restoration.

The optimizer adapter moves authoritative parameters and tensor-valued optimizer
states in place. Parameter objects, master/moment addresses and bucket views are
preserved. Uneven or empty DistributedOptimizer/FSDP shards reconstruct only the
current tile over their optimizer DP group, then exchange over EP and write the
intersection with each destination shard. Plans and schemas are checked across
both groups before migration. Existing optimizer copy/cast functions regenerate
BF16/MXFP8 model weights and scales after all layers migrate. Migration remains
outside CUDA capture; the unbounded snapshot helper is diagnostic-only.

Supported adapter storage is resident, uncompressed elementwise optimizer state,
including ordinary Adam masters and moments, DDP mixed-precision masters,
DistributedOptimizer shards and Megatron-FSDP main-weight buffers. Precision-aware
compressed states, CPU/chunked optimizer offload, layer-wise optimizers and non-elementwise optimizer
states require dedicated adapters and are rejected. Peer-TMA exposes explicit
home-exchange placeholders; its existing replica APIs remain supported.

Checkpoints retain physical parameter slots and the EPLB `_extra_state` mapping,
step counter and load history. Resume requires the same EP/home-slot topology.
Parallel-topology changes are unsupported. To continue with the scheduler
disabled, first export an ordinary logical expert checkpoint as described below.

`test_home_expert_exchange.py` checks delayed reads of post-update weights,
local/cross-rank cycles, exact Adam continuation over repeated updates, mixed
tensor formats, repeated consumer-stream waits, uneven/empty optimizer shards,
active/pending routing, checkpoint metadata and explicit unsupported operations.

## Initialization and routing contract

Every enabled scheduler invocation executes `update_placement`, dispatch, and
`reroute`; there is no planning-skip hook. ECHO uses `one_shot_greedy` only.
There is no configurable ECHO assignment algorithm.
Unimplemented HybridEP weight transport, MoonEP/peer-TMA EP sizes above 64,
and MoonEP expert counts above 8192 are rejected before allocation. MoonEP
requires equal physical token counts on all EP ranks; post-planner variable
token equalization cannot satisfy this requirement and is rejected.

Learned and mock routing retain exactly top-k assignments on padding rows
with zero probabilities. Padding contributes physical work to the planner but
no expert output or gate gradient. Scheduler-disabled dropless HybridEP keeps
its existing sparse padding behavior.

CUDA, explicit NCCL EP groups, HybridEP, Triton, TE grouped ops, and the fused
GroupedMLP environment setting are checked before constructing expert weights.
MoonEP's symmetric histogram window is initialized at that boundary. Home optimizer
configuration is checked in `arguments.validate_args` after defaults and argument
normalization. Actual state representations and physical parameter bindings are
checked during optimizer initialization, not at the first migration. Supported home
optimizers are Adam and SGD with resident, uncompressed state.

## Checkpoint transitions

Loading a scheduler-disabled checkpoint with home migration enabled starts a
new training run: weights are loaded in identity home order, optimizer and
training/RNG progress are not restored, and counters start at zero. The source
must contain topology metadata and use the same EP/TP/PP/CP/world topology.
No exchange occurs during loading. The first forward initializes planner
history and normal `update_placement` proposals trigger later migrations at
successful optimizer boundaries. No planner state is required only for this
explicitly detected logical-weight initialization. Missing planner state in a
home checkpoint remains fatal, even with non-strict loading.

Resuming a home checkpoint requires unchanged topology and scheduler settings.
Changing parallel topology is unsupported. To disable the scheduler, first
export a canonical logical checkpoint using the original topology and scheduler
configuration:

```sh
# Append to the original model/training command; conversion saves and exits.
--load /path/to/home-checkpoint \
--ckpt-convert-format torch_dist \
--ckpt-convert-save /path/to/logical-export \
--moe-scheduler-export-logical-checkpoint
```

For Megatron-FSDP, use `--ckpt-convert-format fsdp_dtensor` instead; the output
is an ordinary logical FSDP checkpoint under the `fsdp_dtensor` subdirectory.
Export preserves the optimizer representation and saves synchronously before exit.

Export requires optimizer masters/state (`--finetune`, `--no-load-optim`, and
`--no-save-optim` are incompatible). It restores committed physical homes to
logical expert order together with master weights and optimizer tensors, then
regenerates compute weights and saves with scheduler metadata disabled. A
pending, uncommitted planner proposal is not applied. The output is an ordinary
checkpoint under `/path/to/logical-export/torch_dist`, loadable with the scheduler
disabled, including optimizer continuation. Export is terminal: the in-memory
token dispatcher is not reconfigured for continued execution in that process.

## Replica storage and caching

Peer TMA and NCCL P2P support two independent options:

| Option | Values | Default |
| --- | --- | --- |
| `--moe-scheduler-replica-storage` | `shared`, `layerwise` | `shared` |
| `--moe-scheduler-replica-mxfp8-storage` | `reuse`, `separate` | `reuse` |

`shared` reuses replica weights, scales, and gradients among compatible layers
on the same EP group/device. `layerwise` allocates private replica storage.
Native gradients alias owner-provided `main_grad` views, with the owner
controlling accumulation versus overwrite. There is no native staging allocation.
`reuse` aliases rowwise and columnwise MXFP8 storage; `separate` retains both
orientations. The second option has no effect on BF16 allocation.

For example, enable directional caching with:

```sh
--moe-scheduler-replica-storage layerwise \
--moe-scheduler-replica-mxfp8-storage separate
```

The runtime lazily fills each directional cache. A hit skips replica transport,
not native expert/GTP materialization or completion waits. Cache validity uses
exact placement and local source-byte/scale snapshots, not plan object identity
or Tensor version counters. This detects optimizer updates, checkpoint writes,
requantization, and home migration even when writes bypass version counters.
An EP-wide miss reduction ensures all senders and receivers agree. Snapshot
storage is an additional per-layer cost in this opt-in mode; comparisons and
the control collective also have a runtime cost. BF16 layerwise storage supports
the same cache with its single plain layout.

Shared storage and MXFP8 reuse mode do not enable this cache optimization.
The first backward must populate columnwise data; a cached rowwise tensor is
not a substitute for independently quantized columnwise bytes. An older
microbatch's placement can require a refresh after another placement overwrote
its directional cache. No gradients are cached: each microbatch still computes
and reduces its own replica gradients.

## Consumer lifetime

A runtime acquires a group/device consumer fence before prefetch and releases
it after token combine in forward, or after gradient publication in backward.
The next producer waits on the last consumer event. Communication completion
alone does not authorize overwrite. Unreleased/reentrant consumers raise an
error instead of overwriting shared storage. Source producers and GTP's
materialize/consume protocol remain intact on cache hits.

NCCL workspaces share the communication stream and completed-transfer cleanup,
so layer-local lists do not retain old temporary messages until that layer is
visited again. Peer-TMA symmetric windows have explicit transport reference
counts; destroying one shared user does not invalidate the others.
Completed NCCL handles retain only their completion event and transport;
the workspace releases packed messages after their GPU work completes, even
when a layer keeps an old handle for repeated waits.

## CUDA graphs and diagnostics

An unset `--moe-expert-rank-capacity-factor` preserves dropless dispatch for both
transports. ECHO/EPLB use HybridEP's dynamic output sizing. MoonEP supplies a
per-invocation bound on padded rows; HybridEP consumes it without adding padding
a second time and checks overflow on-device, avoiding dynamic output sizing in
eager execution. The scheduler does not imply a static
capacity of 1.0: a planner can leave a rank above the mean token count, and
per-expert padding can increase the required capacity further. Explicit static
capacity remains supported and requires the training overflow retry path when
the budget is exceeded. Capturing MoE requires an explicit static capacity;
the eager MoonEP bound does not relax that configuration restriction.

The exact cache requires host decisions and cannot run inside a captured MoE
region. Attention-only CUDA graphs remain supported. NCCL retains its existing
restriction against captured host-scheduled MoE communication.
`runtime.weight_cache.hits` and `.misses` expose collective cache decisions for
tests and profiling. Source snapshots and cache entries are ephemeral and are
not checkpoint state.

## Tests

The full-training checkpoint proxy disables evaluation with `eval_iters=0`.
Validation/test mock splits may then contain zero samples; the dataloader factory
uses upstream dev's empty-split handling, returning `None` before sampler
construction for any empty split. This keeps checkpoint continuation tests
independent of evaluation batch generation without overriding upstream loader
semantics.

Home optimizer binding inspects lazy state with non-inserting dictionary reads.
An untouched optimizer must remain empty until Adam initialization or checkpoint
loading allocates moments; inserting empty per-parameter entries makes the
distributed loader mistake missing moments for already initialized state.
Strict checkpoint loading also reads the rerun machine's ordinary sentinel
shard, while restoring fault-recovery behavior only for an active saved rerun.

`tests/unit_tests/transformer/moe/test_replica_storage.py` exercises both backends
and all four policies with BF16/MXFP8, exact bytes/scales, additive gradients,
consumer streams, rank-asymmetric cache invalidation, old placement restoration,
and real TE MXFP8 forward/dgrad GEMMs with repeated runtime prefetch requests.
The existing NCCL transport and runtime boundary suites remain regression tests.

`test_scheduler_bf16_layer_parity.py` compares a real scheduler-disabled
MoELayer against each transport, using identical initial weights, inputs and
upstream gradients. On four GB200 GPUs it checks forward, input gradients,
router/expert gradients, and one distributed Adam update with one or four
microbatches. It checks FP32 master-weight deltas as well as BF16 weights, so
the initial parameter magnitude cannot hide an incorrect optimizer update.
This is a paired production-path diagnostic, not an independent mathematical
reference. A separate insufficient-static-capacity control must report both
dispatch overflow and numerical disagreement. The six cases pass per rank;
dynamic-capacity forward and input gradients match exactly in this recipe.


## Module boundaries

`MoEScheduler` owns the training-facing prepare, step, logical-load, and export
operations. Planners provide logical export placements through `MoELoadPlanner`;
training code never interprets committed or pending planner layouts. Home dispatch
has its own typed contract because it stages `HomeExpertPlacement`, unlike replica
dispatch's Tensor placement. It owns installation health and terminal export.
Replica dispatch binds the borrowed transport and optimizer adapter without
exposing its runtime to scheduler orchestration.

Optimizer leaves expose `get_parameter_state_binding`,
`defer_param_sync_for_state_update`, and `refresh_model_weights`. Bucket maps,
master-to-compute casts, and FSDP shard offsets remain owned by optimizer/FSDP.
The home adapter receives source parameters and transport configuration, not the
whole replica runtime. Expert modules own runtime detachment and fused-op cleanup.
Scheduler diagnostics use module loggers at INFO level, following application
logging configuration; they do not print directly to stdout.


`test_scheduler_release_contracts.py` covers learned padding/autograd,
distributed MoonEP routing, logical checkpoint load and export with Adam
continuation, failed-install protection, module logging, optimizer hold release,
parameter bindings, and configuration capacity checks. These focused tests do
not establish full FSDP/MXFP8 training parity.


## Historical split-planner reference

The test-only split planner tracks PR #7450 at `242c8dbc63c17ec804f1e31974d3dec86e5ed56c`.
The following describes the former implementation preserved under
`tests/unit_tests/transformer/moe/moonep_reference*.py`. Its placement and reroute
are separate launches with dense routing. Production MoonEP instead uses the
cooperative kernel described above; reference benchmarks do not measure the new
scheduler adapter or establish its end-to-end performance.

Placement rounds each global logical expert count once into GEMM-aligned blocks
(256 for the supported fused grouped MLP), balances these blocks, and assigns
remainder blocks to lower ranks. Routes retain their source-rank/token order;
conceptual padding occurs only at the logical expert's end. Real route totals
can differ between destinations. This applies to BF16 as well as MXFP8.
The 32-program planner handles up to two EP ranks per program. Reroute uses
512-route sort/scan tiles and binary search while emitting the existing dense
probabilities with autograd. Compiled launchers are cached by shape, dtype,
alignment and input-pointer alignment. Token shape can change between eager
invocations; each shape gets its own specialization and capacity.

For R routes per source rank, P ranks, E logical experts and alignment A,
`blocks = (P*R + min(E, P*R)*(A-1)) // A` and the padded capacity is
`A * ceil(blocks/P)`. The optional padded-capacity metadata is independent of
an unpadded route bound. Explicit user capacity and overflow-retry policy take
precedence over both bounds. The existing explicit-capacity requirement for
captured MoE remains in place.

Peer-TMA pushes use the PR's TMA reads and coalesced uint64 peer stores for both
weight bytes and quantization scales. Existing transport completion and consumer
fences remain in force. Regular DDP natives accumulate directly into persistent
`main_grad`; virtual slots are zeroed before each accumulating backward. The
handoff triggers original parameter hooks without re-adding native gradients.
Runtime leaves do not request dummy gradients from TE. The original owner hooks
are triggered after replica reduction; no native-gradient staging is retained.

An unbound runtime can be destroyed without changing the expert layer's native
fused ops. Detaching a runtime owned by another layer remains an error.

The existing TE weighted squared-ReLU expert path is allowed by scheduler
configuration, with `ScaledSReLU` checked during environment initialization.
The historical alignment tests use a common supported recipe rather than GTP.
The retained upstream `test_virtual_experts.py` separately exercises quantile
routing, grouped-tensor/GTP, repeated MTP, recompute and offload through the
legacy entry point. These tests do not imply arbitrary planner/backend support
for every upstream GTP configuration.

`test_moonep_pr7450_alignment.py` checks the pinned PR's independent CPU oracle,
local EP33/64 layouts, real EP communication and graph replay.
`test_moonep_pr7450_training.py` checks DDP accumulation and BF16/MXFP8 Adam
updates with mock/learned routing and weighted squared-ReLU. Local EP64 simulation is
not a 64-GPU collective test. The optional `test_moonep_pr7450_benchmark.py`
requires the unmodified, hash-pinned PR kernel source and records eager and CUDA
graph timings, including equivalent dense-output measurements; it does not enforce noisy CI speed gates.

The FC2 wgrad callback belongs to `ReplicaExpertRuntime`. Expert construction
obtains it through `make_fc2_wgrad_store()`. It executes TE's wgrad callback
before starting FC2 replica reduction, preserving overlap with the remaining
backward work. Removing it would defer reduction until the later fallback
boundary. The callback does not defer the GEMM to another training phase.


## Cached mock routing for scheduler benchmarks

Set `--moe-router-mock-maxvio 1.5` to replace learned/Hash-MoE expert choices
with deterministic logical expert assignments. The target is the maximum EP
rank assignment count divided by the mean, measured against the **initial**
contiguous home layout, before scheduler rerouting and alignment padding.
Each token still selects `topk` distinct experts. Initial rank 0's expert group
is hot on every source rank; other groups share the remaining routes evenly.
For E experts, R EP ranks and K routes, the supported range is
`[1, R * min(K, E/R) / K]`. Integer counts can introduce rounding error.

`MockRouteCache` builds each token-shape/device template once without advancing
the training RNG. Gate projection and fresh differentiable probability scores
still run on every forward. The router emits this branch's existing dense bool
routing map and probability matrix. MoEScheduler's planner performs its normal
placement and reroute operations; dispatchers and transports need no mock mode.
EPLB home migration changes physical ownership without changing the cached
logical expert choices, so actual GPU load can differ from the initial target.

Padding uses device-side valid-token ordinals. With MoEScheduler enabled,
padded rows retain K dummy routes but have zero scores and gate gradients;
dummy assignments count toward the planner's physical work. Without the
scheduler, padded rows are removed from the route map. Warm every shape/device
before CUDA Graph capture; a cold cache during capture raises explicitly.
Caches live with the router, are not saved in checkpoints, and are rebuilt
deterministically. Changing the target requires a new router/cache.

This is a load-benchmark option. It disables routing balancing auxiliary losses
but retains Z-loss. It rejects force-balance, forced bias, expert-bias updates,
routing replay, group-limited selection, token dropping, expert TP > 1 and
Sinkhorn. Leave it unset for ordinary learned routing. Toolkit recipes must also
use `--no-force-balance` to avoid the launcher's default forced balancing.

## Empty inputs in fused experts

A rank can receive no tokens under an imbalanced MoE route, including when the
scheduler is disabled. TE activation kernels must not receive a zero-row TMA
descriptor. The fused expert path pads an empty input to one aligned group of
zero inputs and zero probabilities, executes normal parameter hooks and expert
operations, then returns the empty output slice. Input and probability gradients
retain their original empty shapes; expert gradients remain zero and participate
in the ordinary DDP/optimizer lifecycle.

This is separate from scheduler placement or dispatch capacity. Nonempty inputs
retain the existing path. `test_fused_expert_empty_rank.py` exercises real TE,
HybridEP, DDP and distributed Adam with all routed work concentrated on one of
four ranks and checks zero expert gradients on the other ranks.

## Replica parameter-hook lifetime across optimizer steps

Forward prefetch invokes the native expert parameter hooks before reading
source weights, then marks that preparation for consumption by the fused MLP
forward hook. Backward prefetch also prepares native storage, but must not
mark a subsequent forward as prepared: no fused forward hook follows it.
The `for_backward` phase explicitly clears the forward-prepared flag.
Otherwise, after optimizer update, the next forward can skip DDP
`finish_param_sync` and read incomplete MXFP8 all-gather data. This applies
to all planners and replica transports; the scheduler-disabled path retains
its existing forward-hook behavior.

A forward/backward/update/forward regression checks that each forward
prefetch waits for the current weight generation and that the fused forward
does not duplicate hooks already run by its own forward prefetch.

## EPLB without replicas

With `moe_scheduler_planner_type=eplb`, `moe_scheduler_expert_dispatcher_type=replica_nccl`,
`moe_scheduler_num_idle_experts=0` and a positive home update interval, the
scheduler selects `HomeOnlyExpertDispatch`. The configuration name retains
`replica_nccl` for transport compatibility, but this execution path does not
construct `ReplicaExpertRuntime` or attach runtime parameters to the experts.
Native TE expert forward, backward, parameter-gather hooks and optimizer gradient
accumulation remain authoritative. The existing BF16/MXFP8, optimizer-state and
no-MoE-recompute/capture constraints still apply.

The planner returns `None` for replica placement and an immutable cached home
column permutation. It bypasses replica-map construction, dense token ordinal
cumsum/modulo and scatter, using column selection for routes and masked
probabilities instead. The cache changes only at ownership commit or restore.
Load history stays local until an EP/optimizer-DP reduction at the proposal
boundary. Checkpoints record the history scope; legacy EP-summed histories
are distributed evenly on restore so their next global total is preserved.
Every token route is permuted
from logical IDs into the currently committed home slots; no replica dispatch,
weight prefetch or gradient reduction runs. EPLB still accumulates logical load
history and proposes capacity-constrained home permutations at its configured
successful-update interval. With no replicas, each expert has one instance.

The NCCL transport allocates only a communication stream at construction (plus
its small collective initialization token). Zero-slot workspaces hold no replica
weights/scales/gradients and no native wgrad staging. Replica operations are
explicitly rejected on this transport. The bounded home work area is allocated lazily on the first migration and reused
across subsequent migrations and compatible layers. Zero replicas does not
eliminate the bounded staging or NCCL internal storage.

`HomeOnlyExpertDispatch` borrows FC1/FC2 parameters without registering duplicate
parameters. It binds the same `OptimizerHomeExpertState` adapter and
`HomeExpertDispatch` as the replica path. A pass-through autograd boundary tracks
outstanding native microbatches until the whole backward graph completes; it does
not compute or redirect gradients. Pending native backward work rejects migration.
The existing training-step helpers fence GPU work, defer parameter gathers,
exchange post-update masters/moments, commit the logical mapping, and refresh
native BF16/MXFP8 weights before restarting gathers. Failed optimizer updates
retain the previous ownership and pending proposal. Custom loops must honor this
same drained optimizer boundary.

Checkpoint topology includes zero replicas and the empty `[EP, 0]` replica
layout. Active and pending home ownership, histories and step counters retain
the existing physical-checkpoint semantics. Transport cleanup participates in
`finalize_replica_weight_transports` even though no replica runtime exists.

`test_home_only_expert_dispatch.py` covers configuration/factory selection,
identity native gradient flow and outstanding-backward rejection, pending-home
checkpoint resume and routing gradients, no replica/native-gradient staging,
NCCL home swaps/cycles, and native Adam continuation against independent logical
experts across strided EP groups.

`test_home_chunk_exchange.py` checks a CPU cyclic tile oracle and GPU EP1/2/4/8
cycles, fixed points, same-rank swaps, mixed dtypes, tail tiles, shared bounded
storage, and failure before mapping commit. Existing Adam continuation and
uneven/empty optimizer-shard tests exercise multiple tiles with a 64-byte budget.


## Scheduler memory contract hardening

Scheduler configuration requires EP > 1 and rejects full activation recompute
as well as selective MoE recompute. Replay does not yet preserve scheduler
placement/dispatch lifetime, so enabling it must fail during configuration.

MXFP8 replica wrappers are created per slot, independently of the number of home
experts. Native GTP placeholders keep one wrapper per home and its own quantizer
metadata, temporarily borrowing a replica view until gather binding replaces it.

`test_scheduler_memory_contracts.py` covers configuration rejection, wrapper
cardinality, direct DDP/FSDP gradient addresses, zero-replica column routing and
legacy history restoration on CPU. A CPU global-table oracle exercises production
chunk packing/reduction with a simulated wire; live optimizer metadata tests
retain option/step mismatch rejection. The NCCL suite exercises slot counts below,
equal to and above home count with real TE, and small budgets force multi-tile
BF16/MXFP8 transfers and FP32 accumulation. Shared scratch lifetime is tested
across layers and producer streams. GPU execution results are recorded in the
accompanying toolkit experiment report; CPU checks alone do not validate NCCL.

### Determinism regression coverage

`tests/unit_tests/transformer/moe/test_scheduler_determinism.py` checks bit-exact
replay separately from scheduler-disabled numerical parity. It exercises live
EP placement and probability backward for ECHO, EPLB and MoonEP; BF16/MXFP8
replica weight/scales and additive owner gradients through Peer-TMA and NCCL;
and learned routing, HybridEP, grouped experts and two distributed Adam updates.
Replays use identical inputs with side-stream GEMM contention and include reused
transport buffers. The production-layer cells require four GPUs. This is a
fixed-topology contract, not a promise of identical results across different
EP sizes or different planner/transport configurations. Full GPT checkpoint
continuation remains a separate end-to-end test.
The GB200 unit-test recipe runs this replay file in its own bucket so pinned
process-global autotune state does not leak into ordinary MoE tests.
The platform selector excludes this file and the four-rank BF16 layer parity,
MoonEP dispatch-capacity, and PR7450 training suites from H100's eight-GPU
bucket. Those suites still run once on GB200; the replay file has a dedicated
process and the other three belong to the GB200 catch-all bucket. The
`launch_on_gb200` marker is an opt-in, not a general H100 exclusion: portable
planner, NCCL, and home-exchange coverage remains selected on both platforms.
`tests/test_utils/test_dev_unit_test_selection.py` checks these ownership rules
using both platform recipes and the production selector without importing CUDA.
