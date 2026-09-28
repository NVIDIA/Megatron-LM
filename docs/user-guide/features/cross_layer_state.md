# Cross-layer tensor state

Cross-layer state lets one module produce tensors that later modules consume,
while preserving all consumer gradients to the producer. The same declared
tensor interface can cross a checkpoint, CUDA graph, or pipeline boundary.
The implementation is adapted from the general state and transport contracts
in [#7224](https://github.com/NVIDIA/Megatron-LM/pull/7224), by Hongxiao Bai.
It contains no DeepSeek-specific attention, compression, or routing rules.

THD (packed sequences) support is still a work in progress and is not supported
by this feature's main-branch implementation. The existing layout-plan hooks
and variable-shape tensor transport do not provide end-to-end THD support.

## Declare the inputs and outputs of a region

`TensorField` declares a stable key, concrete shape, dtype, layout, gradient
eligibility, and presence. `TensorSchema` orders those fields and packs/unpacks
ordinary tensors. Integer fields cannot be differentiable. `present=False`
represents an absent field and differs from a present, zero-length tensor.

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
frozen, and preserves absent versus explicitly zero gradients.

`StatefulGraphs(region, sample_hidden, sample_state, slots=N, backend=...)`
captures the same tensor interface with `backend="torch"` or
`backend="transformer_engine"`. Replay with `graphs.run(hidden, state, slot=i)`.
Use independent slots for outstanding forwards; a slot cannot be reused until
its backward completes. Input geometry, strides, dtypes, gradient profile,
parameter storage, and training mode must match capture. Optimizer updates to
existing parameter storage are allowed.

This initial graph API validates FP32/BF16 execution. Every differentiable
captured output must participate in one backward invocation. An unused output
must be omitted from that capture profile; silently turning a missing gradient
into a numeric zero would change optimizer and auxiliary-loss behavior. Full
recomputation inside capture is rejected. FP8/FP4 capture recipes and automatic
integration with model-specific graph collection are outside this interface.

After the last backward, discard its outputs and call `graphs.close()` before
destroying distributed process groups. This releases captured NCCL references
and breaks the graphed-forward closure cycles.

## Context parallelism

The helpers in `megatron.core.context_parallel.shared_state` accept an explicit
CP process group:

- `redistribute_state` reuses the existing contiguous/zigzag conversion. Its
  `thd_plans` argument is preliminary plumbing for `THDCPLayoutPlan`; packed THD
  execution remains unsupported.
- `gather_state` gathers equal contiguous sequence shards and sums all consumer
  gradient contributions back to each owner. Integer fields use ordinary
  collectives. A local slice of the result needs no additional gradient
  collective; normal slicing avoids double reduction.
- Fields marked `replicated`, `local`, or `strided` remain unchanged. Select the
  fields to communicate through the schema; do not gather query-local state
  simply because another field needs global visibility.

Sequence is dimension zero. Peers must agree on shapes and collective/backward
participation, as required by the underlying distributed operations. The model
owns compression groups, halo ownership, packed positions, and TP/SP-specific
layout rules. The general state layer does not infer those semantics.

## Pipeline parallelism

Stages opt into the existing ordinary or interleaved 1F1B scheduler by exposing
`pipeline_payload_factory = TensorStatePayload`. A nonterminal stage returns
`TensorStatePayload.from_state(hidden, state, output_schema, boundary_id=...)`.
Its successor receives that object through the normal `set_input_tensor` hook
and calls `payload.restore()` to obtain hidden and a fresh state mapping. The
terminal stage uses the normal scalar-loss reduction interface.

CUDA stages also expose `pipeline_control_group`, created once during model
assembly with `create_pipeline_control_group(pg_collection.pp, world_group=world_group)`
from `typed_p2p_communication`. All global ranks call this in the same order with
their local PP group and the explicitly supplied world process group. It creates
every PP replica in a common global order. Reuse that
group across iterations and virtual chunks; destroy it after the schedules
and graph captures finish. Its Gloo backend carries host-derived descriptors,
identities, and gradient-activity bits. For PP=2, its NCCL backend carries one
physical direction while the original PP group carries the other. Separate
channels keep a prefetched receive from blocking a send to the same peer. Reading
activity bits from a CUDA tensor would synchronize the CPU with pending GPU work.

Typed transport carries mixed dtypes, empty/absent fields, and independent
gradient eligibility. A backward activity bitmap distinguishes a missing
gradient from an explicitly zero gradient. All active output roots participate
in one autograd traversal; relay and local-consumer contributions accumulate
once. The communicator owns detached wire buffers until their Work completes.
Shared output storage is retained for backward instead of being pseudo-freed.

There are two transport modes:

| Mode | Configuration and contract |
|---|---|
| Dynamic descriptors | Blocking, unbatched P2P (`batch_p2p_comm=False`, `overlap_p2p_comm=False`); shapes may vary between microbatches. |
| Prepared descriptors | Provide one incoming/outgoing `PipelinePayloadSpec` per microbatch and virtual chunk through `PipelineDataIterator` and `PipelinePayloadPlan`. Supports batched P2P without runtime shape discovery. |

Terminal directions use `None` in the plan. Neighboring descriptors must match
exactly, including field identity, presence and gradient eligibility. The
optional `forward_step_func.prepare_pipeline_inputs` hook prepares iterators
before communication; dataset interpretation remains outside core scheduling.

Batched and overlapped P2P **require prepared plans**: NCCL can return one coalesced Work for
the entire batch, which cannot be waited just for a dynamic header while its
data receives are still unknown. Asynchronous schedules likewise must post all receive buffers before waiting for future gradients. Overlapped P2P requires VPP and unbatched P2P.
`overlap_p2p_comm_warmup_flush=True` is rejected: warmup/flush prefetch still
hangs with NCCL and requires further work. Ordinary VPP communication overlap
remains available with that option disabled. Ring exchange, multi-module pipelines,
and the combined MoE schedule are not supported by this typed transport. Existing tensor-only models retain their
current scheduling path.

Custom P2P communicator subclasses cannot be combined with the automatic typed
adapter; the scheduler rejects them instead of silently replacing their behavior.

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
