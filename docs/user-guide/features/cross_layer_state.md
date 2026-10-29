# Cross-layer tensor state

Cross-layer state lets one module produce tensors that later modules consume,
while preserving all consumer gradients to the producer. The same declared
tensor interface can cross a checkpoint, CUDA graph, or pipeline boundary.
The implementation is adapted from the general state and transport contracts
in [#7224](https://github.com/NVIDIA/Megatron-LM/pull/7224), by Hongxiao Bai.
It contains no model-specific attention, compression, or routing rules.

Packed (THD) batches cross pipeline boundaries through per-microbatch plans, whose
shapes may vary. Packed CP layout conversion remains model-owned: the CP helpers'
`thd_plans` argument is preliminary plumbing without end-to-end coverage.

## Declare the inputs and outputs of a region

`TensorField` declares a stable key, concrete shape, dtype, CP layout, gradient
eligibility, and presence. `TensorSchema` orders those fields and packs/unpacks
ordinary tensors. Integer fields cannot be differentiable. `present=False`
represents an absent field and differs from a present, zero-length tensor.
The layout describes sequence sharding under context parallelism: `contiguous`
and `zigzag` shard dimension zero, `replicated` is identical on every CP rank,
`local` is rank-specific, and `strided` declares no sequence semantics.

Gradient eligibility is a property of the model's interface, not the current
producer's `requires_grad` value: a frozen producer can still send a field that
a downstream trainable consumer differentiates with respect to. Fields declared
nondifferentiable are detached explicitly. No tensor is otherwise cloned or
detached merely because another layer reads it.

A region implements `forward(hidden, state) -> (hidden, state)`. Wrap it in
`StatefulModule` to expose a tensor-only execution boundary:

```python
import torch
from torch import nn
from megatron.core.transformer.state_boundary import TensorField
from megatron.core.transformer.stateful_module import StatefulModule

class MemoryRegion(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False).cuda()

    def forward(self, hidden, state):
        projected = self.projection(hidden)
        return projected + state["memory"], {"memory": projected}

field = TensorField("memory", (128, 2, 64), torch.float32, "contiguous", True)
region = StatefulModule(MemoryRegion(64), (field,), (field,))
hidden = torch.randn(128, 2, 64, device="cuda", requires_grad=True)
state = {"memory": torch.zeros_like(hidden)}
hidden, state = region.run(hidden, state, recompute=True)
```

Create fresh state at model entry for every microbatch. Region calls construct
fresh mappings; modules must not retain activations or mutate shared tensors in
place. Declared inputs omitted from the output schema are retired after their
last consumer. Unrelated fields belonging to other components remain in the
caller's state. Use namespaced keys when composing independent components.

Schemas describe one shape/layout profile. A different profile can wrap the
same parameter-bearing module without copying weights. Keep parameter ownership
with the model: registering a `StatefulModule` wrapper as a new child introduces
its `module.` prefix into checkpoint keys, whereas an execution adapter referencing
an already registered module need not replace that module in the model tree.

## Recomputation and CUDA graphs

`region.run(..., recompute=True)` checkpoints explicit input tensors and rebuilds
the mapping on replay. It preserves PyTorch's default RNG and Megatron's
model-parallel RNG, supports parameter gradients even when input tensors are
frozen, and preserves absent versus explicitly zero gradients. Quantized regions
pass an FP8-aware `checkpoint_fn(region, *tensors)`, e.g. one calling
`te_checkpoint(..., use_reentrant=False)`; the default checkpoint is rejected
while FP8 autocast is active, since its replay would run outside the forward's
FP8 recipe state.

`StatefulGraphs(region, sample_hidden, sample_state, slots=N, backend=...)`
captures the same tensor interface with `backend="torch"` or
`backend="transformer_engine"`. Replay with `graphs.run(hidden, state, slot=i)`.
For the TE backend, initialize and seed Megatron's TE RNG tracker during model
setup. Legacy byte-tensor RNG states are rejected before capture; the adapter
does not reset or replace the caller's RNG configuration. `capture_kwargs`
forward backend options such as TE's FP8 recipe. With MCore DDP's overlapped
parameter gather, pass `make_forward_pre_hook` so replays still run the
parameter owners' gather hooks.

Use independent slots for outstanding forwards; a slot cannot be reused until
its backward completes. Forward-only and frozen invocations return independent
copies, so another replay cannot overwrite outputs held by an asynchronous send.
Replays on different streams wait for the previous invocation to finish copying.

Input geometry, strides, dtypes, gradient profile, training mode and parameter
storage must match capture. Input signatures and the training mode of every
captured module are checked on each replay; capture a separate profile for
evaluation. The more costly storage walk is opt-in with `debug_checks=True`, or
call `validate_module()` at a configuration boundary. Optimizer updates to existing
parameter storage are allowed; FSDP storage replacement and MXFP8 parameter-gather
buffer reuse are unsupported.

Each slot still owns a separate graph memory pool. Memory grows with the number
of outstanding microbatches; size `slots` for the schedule and include these pools
in the activation-memory budget. Shared pools remain future work.

Warmup and capture preserve existing parameter gradients, fused `main_grad`
buffers and registered buffers (e.g. router statistics), and suppress MCore DDP
reductions during capture. CUDA graph creation must be serialized within a process.

Every differentiable captured output must participate in one backward
invocation. An unused output must be omitted from that capture profile; silently
turning a missing gradient into a numeric zero would change optimizer and
auxiliary-loss behavior. Full recomputation inside capture is rejected.

For a skipped or failed iteration, first drain communication and stop scheduling
work, then call `graphs.abort()` and discard its outputs. The slots can be reused;
backward through an aborted invocation raises an error.

After the last backward, discard its outputs and call `graphs.close()` before
destroying distributed process groups. This releases captured NCCL references
and breaks the graphed-forward closure cycles.

## Context parallelism

The helpers in `megatron.core.context_parallel.shared_state` accept an explicit
CP process group:

- `redistribute_state` reuses the existing contiguous/zigzag conversion. Its
  `thd_plans` argument (a `THDCPLayoutPlan` per field) is preliminary.
- `gather_state` gathers equal contiguous or zigzag sequence shards with one
  all-gather into a single buffer and a local reorder without host
  synchronization, so it also runs inside CUDA graph capture. Backward
  reduce-scatters, summing every consumer's contribution to its owner. A local
  slice of the result needs no additional gradient collective.
- With sequence parallelism, pass `sequence_parallel=True`, `tp_group`, and
  `tp_cp_group` to both helpers. Gathering then spans TP x CP and restores logical
  sequence order. Missing SP groups are rejected, including no-op conversions.
  If TP ranks consume the gathered state identically (a TP-replicated consumer),
  pass `tp_replicated=True` so their duplicate contributions are averaged; the
  default sum is correct for TP-partitioned consumers such as split heads.
- Fields marked `replicated`, `local`, or `strided` remain unchanged. Select the
  fields to communicate through the schema; do not gather query-local state
  simply because another field needs global visibility.

Peers must agree on shapes and collective/backward participation, as required by
the underlying distributed operations. The model owns compression groups, halo
ownership, packed positions, and TP/SP-specific layout rules.

## Pipeline parallelism

Stages opt into the existing ordinary or interleaved 1F1B scheduler by exposing
`pipeline_payload_factory = TensorStatePayload`. A nonterminal stage returns
`TensorStatePayload.from_state(hidden, state, output_schema, boundary_id=..., metadata=...)`.
Its successor receives that object through the normal `set_input_tensor` hook
and calls `payload.restore()` to obtain hidden and a fresh state mapping. The
terminal stage uses the normal scalar-loss reduction interface.

Every boundary message is described ahead of time by a `PipelinePayloadSpec`
(hidden field first, then the state fields, host metadata and a boundary
identity). Provide one incoming/outgoing spec per microbatch and virtual chunk
through `PipelineDataIterator` and `PipelinePayloadPlan`; the optional
`forward_step_func.prepare_pipeline_inputs` hook prepares iterators before
communication. Shapes may change between microbatches (e.g. packed sequences),
and all receives are posted without runtime shape discovery, so blocking,
batched and VPP-overlapped P2P share one protocol. Terminal directions use `None`.
Neighboring plans are compared once per schedule invocation, and each send is
checked against its plan before it is posted. Peers exchange only tensor data.

Gradient eligibility is static per microbatch. A field declared differentiable
must receive a gradient from its consumers; a missing gradient is an error rather
than a silent zero, because a zero would still run producer-side auxiliary-loss
hooks. Declare a branch that is unused for a microbatch non-differentiable in
that microbatch's plan. Empty fields carry no data.

Typed transport carries mixed dtypes and empty/absent fields. All differentiable
output roots of a stage participate in one autograd traversal; relay and
local-consumer contributions accumulate once. `from_state` wraps each
differentiable root in its own viewless tensor object without copying data; with
`deallocate_pipeline_outputs=True`, these roots are shrunk after the send is
posted, while detached wire buffers keep the storage alive until the transfer
completes. Caller aliases and storage saved by autograd are unaffected.
Non-differentiable references are dropped once the wire owns them.

Like the tensor-only communicator, PP=2 carries one physical direction on the
world group so that a posted receive never blocks a send to the same peer.
`overlap_p2p_comm_warmup_flush=True`, ring exchange, multi-module pipelines, the
combined MoE schedule and custom P2P communicator subclasses are rejected for
typed stages. Existing tensor-only models retain their current scheduling path.

## Validation

The tests include a shared-projection model independent of any attention family:

```bash
uv run python -m torch.distributed.run --standalone --nproc-per-node=1 \
  -m pytest tests/unit_tests/transformer/test_stateful_module.py

uv run python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m pytest tests/unit_tests/transformer/test_shared_state_cp.py \
  tests/unit_tests/pipeline_parallel/test_typed_pipeline.py

uv run python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m pytest tests/unit_tests/pipeline_parallel/test_shared_state_parallel.py
```

The four-rank cases compare CP2 x PP2/VPP against the same unpartitioned layer
chain, including model gradients, with eager execution, recomputation, and both
graph backends. Porting a particular model's state declarations and numerical
operations remains separate from this general feature.
