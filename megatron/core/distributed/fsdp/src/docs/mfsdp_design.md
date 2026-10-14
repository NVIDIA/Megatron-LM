# Megatron FSDP Design

Contributors: @wujingyue, @cspades, @shjwudp, @Autumn1998

GitHub tracker: https://github.com/orgs/NVIDIA/projects/276

# Executive Summary

This design doc proposes MFSDP v2 to better satisfy
[Megatron FSDP Requirements](http://nv/mfsdp-requirements). In particular, the new
version enables:

- **Fine-grained control.** For FSDP, automatically determining the optimal bucketing
  and prefetching strategy has been challenging.
  - The proposed high-level `fully_shard` API, similar to
    [PyTorch FSDP2’s `fully_shard` API](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html),
    provides per-module control that is finer-grained than the production version.
  - In addition, we also plan to expose lower-level APIs (e.g., ParameterGroup and
    DBuffer) to give users even more fine-grained control. Spatially, this allows users
    to control which parameters belong to each “bucket”. Temporally, this allows users
    to control when unsharding and resharding occur during forward, backward, and
    optimization.
- **Emerging optimizers**. For example, it supports tensor-atomic sharding needed for
  the Muon optimizer.
- **Simplified lower-precision support.** Through a block-atomic sharding format, as
  motivated by [veScale-FSDP](https://arxiv.org/abs/2602.22437).

These capabilities are difficult to implement cleanly within the current codebase
architecture. Therefore, the code will be
[developed in branch `main`](#separate-code-paths-in-main) as a separate code path from
the existing `megatron_fsdp` implementation. The development will follow the
prototype-design-execute process that we’ll detail in
[this section](#development-process). Once the new code is on parity, we’ll gradually
migrate users over.

The remainder of the doc focuses on the core MFSDP abstractions and building blocks that
serve as the foundation for extensions. Capabilities such as MXFP8, CUDA Graphs, double
buffering, prefetching, NCCL user buffers, HFSDP, offloading, and checkpointing build on
these primitives and introduce additional design considerations. We will cover these
areas in dedicated follow-on design documents, leveraging the interfaces and mechanisms
established here.

# Subdesigns

- [Optimizer](optimizer.md)
- [Runtime schedule](runtime_schedule.md)

# API

```py
class Placement
class Replicate(Placement)
class Partial(Placement)
class Flat(Placement)
class TensorAtomic(Placement)


type MeshAxis = int | str


@dataclass
class Placements:
    dp_axes: list[MeshAxis]  # outer to inner
    parameter: list[Placement]  # same length as dp_axes
    gradient: list[Placement]
    optimizer: list[Placement]


def fully_shard(
    module: nn.Module,
    mesh: DeviceMesh,
    placements: Placements,
    mixed_precision_policy: MixedPrecisionPolicy | None,
    offload_policy: OffloadPolicy | None) -> None
```

Unlike MFSDP v1, `fully_shard` is expected to be called on each `nn.Module` that forms
an FSDP unit. Under the hood, `fully_shard` attaches the `FsdpModule` mixin to the
target module. FSDP units can also be nested: an outer unit owns the parameters within
its scope, excluding those managed by any inner FSDP units. This is compatible with
FSDP2’s behavior. For example,

```
FSDP Unit: RootModule
owns:
  RootModule.root_weight
  RootModule.root_bias

contains:
  FSDP Unit: SubmoduleA
  owns:
    SubmoduleA.weight
    SubmoduleA.bias

  FSDP Unit: SubmoduleB
  owns:
    SubmoduleB.weight
    SubmoduleB.bias

Ownership view:

RootModule FSDP unit
├── owns RootModule.* params
├── does NOT own SubmoduleA.* params
└── does NOT own SubmoduleB.* params

SubmoduleA FSDP unit
└── owns SubmoduleA.* params

SubmoduleB FSDP unit
└── owns SubmoduleB.* params
```

`fully_shard` sets each parameter in the given module to a shard. As a contract, no
parameters can escape its lowest FsdpModule ancestor to avoid issues like
https://github.com/NVIDIA/Megatron-LM/pull/4899. Without this contract, it can be unsafe
to unshard or reshard a parameter at the module boundary.

The placement is similar to DTensor’s placement but for the whole **unit** and per mesh
axis.

- `Replicate`. Not sharded.
- `Partial`. Used internally for pre-reduce-scatter gradients, which are unsharded and
  only partially accumulated; not user-facing.
- `Flat`. The current per-unit, dim-0 flat sharding. Good for elementwise optimizers.
- `TensorAtomic`. Don’t cut a parameter. For emerging optimizers that need full
  parameters.
- `BlockAtomic(block_size)`. Don’t cut a block of `block_size` rows. Simplifies
  blockwise quantization support. Currently, a 32x1 mxfp8 block may be sharded across
  ranks. This introduces complex host-side logic and custom quantization kernels to
  handle two levels of absmax reduction. Using block-atomic sharding with block_size=32
  ensures that every 32-row block is owned by a single rank.
- `PerTensor(dist.tensor.Placement)`. If needed. Per-tensor dim-0 sharding used in
  FSDP2. It leads to extra data copy so won’t be used by default.

```
Block-atomic sharding example
Input tensor: 8 rows × 4 columns
Block size: 2 rows
Earlier tensors in the parameter group occupy ranks 0, 1, and part of rank 2.

        c0   c1   c2   c3
      ┌────┬────┬────┬────┐
r0    │ x  │ x  │ x  │ x  │
r1    │ x  │ x  │ x  │ x  │
      ├────┼────┼────┼────┤  block 0: rows [0, 1] → rank 2
r2    │ x  │ x  │ x  │ x  │
r3    │ x  │ x  │ x  │ x  │
      ├────┼────┼────┼────┤  block 1: rows [2, 3] → rank 3
r4    │ x  │ x  │ x  │ x  │
r5    │ x  │ x  │ x  │ x  │
      ├────┼────┼────┼────┤  block 2: rows [4, 5] → rank 3
r6    │ x  │ x  │ x  │ x  │
r7    │ x  │ x  │ x  │ x  │
      └────┴────┴────┴────┘  block 3: rows [6, 7] → rank 4
```

`Placements` encodes the various grand [sharding strategies](#sharding-strategies) we
care about and gives users the flexibility to choose which sharding format/granularity
to use for each unit. We may want to pre-define a set of common placements for
convenience, e.g.,

```py
# Assuming `Flat` placement
def hfsdp(dp_outer: MeshAxis, dp_inner: MeshAxis) -> Placements:
    return Placements(dp_axes=[dp_outer, dp_inner],
                      parameter=[Replicate(), Flat()],
                      gradient=[Partial(), Flat()],
                      optimizer=[Flat(), Flat()])
```

If needed, grouping can be customized via the `fully_shard` API, similar to
[the `buckets` argument](https://github.com/pytorch/torchtitan/pull/2378/changes#diff-35ddb8c23734307a1b5fe23e06ffe8e0f2f2c84c58943380d137371e6e21e203R3289)
in the FlexShard proposal.

## Sharding Strategies

Unlike MFSDP v1, MFSDP v2 does not special-case named strategies such as HSDP or HFSDP.
Instead,
`fully_shard` receives a `Placements` configuration that independently specifies the
parameter, gradient, and optimizer placement for each data-parallel axis. The table
below illustrates familiar configurations; it is not an exhaustive list of supported
strategies.

`N` \= size of the inner DP shard dim (`dp_shard_dim`). `M` \= size of the outer DP dim
(`dp_outer_dim`), only present for HSDP/HFSDP. "Sharded" \= persistent state is
partitioned across that dim; "replicated" \= each rank holds a full copy.

| Strategy                                 | Parameters                  | Gradients                | Optimizer states                                    |
| :--------------------------------------- | :-------------------------- | :----------------------- | :-------------------------------------------------- |
| **DDP / `no_shard`**                     | replicated (N)              | partial (N)              | replicated (N)                                      |
| **ZeRO-1 / `optim`**                     | replicated (N)              | partial (N)              | sharded (N)                                         |
| **ZeRO-2 / `optim_grads`**               | replicated (N)              | sharded (N)              | sharded (N)                                         |
| **ZeRO-3 / `optim_grads_params` / FSDP** | sharded (N)                 | sharded (N)              | sharded (N)                                         |
| **HSDP** (FSDP inner, replicate outer)   | sharded (N), replicated (M) | sharded (N), partial (M) | sharded (N), replicated (M)                         |
| **HFSDP** (FSDP inner, `optim` outer)    | sharded (N), replicated (M) | sharded (N), partial (M) | sharded (N × M) — fully sharded across flattened DP |

The per-axis representation also expresses combinations beyond those in the table. For
example, a configuration with ZeRO-1 on the outer DP axis and ZeRO-2 on the inner axis
uses the following placement lists, ordered to match `dp_axes` from outer to inner:

```py
placements = Placements(
    dp_axes=[dp_outer, dp_inner],
    parameter=[Replicate(), Replicate()],
    gradient=[Partial(), Flat()],
    optimizer=[Flat(), Flat()],
)
```

## Compatibility

### FSDP2

Introduce a separate adapter API, `fully_shard_compat`, that mirrors the signature of
PyTorch’s `fully_shard` but omits certain MFSDP-specific features. This would give
existing FSDP2 users a low-friction migration path: they can first switch to
`fully_shard_compat`, and then optionally move to MFSDP’s `fully_shard` to take
advantage of the full feature set.

```py
def fully_shard_compat(...fsdp2 args...):
  convert the args
  fully_shard(...converted args...)
```

### MCore Adapter

This rewrite should be mostly transparent to users of
megatron/core/distributed/fsdp/mcore_fsdp_adapter.py. We’ll implement the adapter using
the new API.

However, certain features may behave differently. For example,
`enable_fine_grained_param_gather_hook` currently makes all-gather fine-grained (one per
submodule), but not reduce-scatter. With per-module control, users would instead apply
FSDP directly to individual submodules, causing both all-gather and reduce-scatter
operations to occur at the submodule level.

# Key Building Blocks

### Ownership and lifetime

The FSDP module tree owns its persistent runtime state:

```
nn.Module / FsdpModule
├── active nn.Parameter
├── FsdpParameterGroup
│   ├── paired sharded and unsharded nn.Parameters
│   └── DBuffers
└── shared FsdpContext
    ├── communication streams
    └── prefetch-order metadata
```

The module’s active parameter is one of the pair owned by its parameter group.

After construction is finalized, every backedge to the module tree **must use a weak
reference**: context prefetch metadata, parameter-group ownership markers, and hook
callbacks. Otherwise, deleting a model retains its persistent CUDA storage until cyclic
garbage collection; with weak backedges, storage is released immediately without
teardown. See https://github.com/NVIDIA/Megatron-LM/pull/6230.

### FsdpContext

- Created by `fully_shard_context` and shared by every `FsdpModule` constructed in that
  scope. On exit, it identifies FSDP roots and finalizes the static forward and backward
  prefetch orders.
- Per-device all-gather and reduce-scatter streams. Module compute runs on PyTorch’s
  current stream.
- Last-microbatch state for HSDP/HFSDP gradient accumulation.
- An optional PyTorch NCCL symmetric-memory pool for communication staging buffers.

### FsdpModule

A mixin attached in place to the original module, so its parent retains the same child
module reference.

- Registered forward and backward hooks drive parameter materialization, resharding,
  gradient reduction, and all-gather prefetching.
- `phase` tracks the module lifecycle: `RESTING` outside module computation, `FORWARD`
  between its forward hooks, and `BACKWARD` between its backward hooks. Activation
  recomputation preserves `BACKWARD` through its nested forward hooks.
- Parameter groups partition the module’s owned parameters by dtype and `requires_grad`.

### ParameterGroup

- dtype
- requires_grad: bool
- A sharded `nn.Parameter` for every logical parameter. Its `.data` is a DTensor backed
  by `main_weight`, and it is the parameter visible to the optimizer.
- The original `nn.Parameter` objects remain attached to the module. During compute,
  their `.data` views a temporary replicated buffer materialized from `model_weight`;
  their `.grad` is temporary full-gradient storage.
- `model_weight`: the persistent compute-dtype buffer, sharded according to
  `Placements.parameter`. It may alias `main_weight` when their dtype and placements
  match.
- `main_weight`: the persistent optimizer-dtype buffer, sharded according to
  `Placements.optimizer`.
- `main_grad`: the persistent gradient buffer for trainable groups, sharded according to
  `Placements.gradient` and allocated in the configured gradient dtype.

### DBuffer

Conceptually, a group of logical tensors, potentially with different shapes, stored in
one contiguous local buffer.

- `local_buffer`: a flat `torch.Tensor` holding this rank’s contiguous shard.
- `mesh` and a per-mesh-axis `placements` tuple. The current implementation requires the
  mesh to contain only data-parallel axes; callers extend returned DTensors with TP or
  EP axes when needed.
- `GlobalLayout`: global tensor shapes and stable offsets used to compute every rank’s
  local range.
- `redistribute(new_placements)`, with `allgather`, `allreduce`, `reduce_scatter`, and
  `scatter` convenience operations. Redistributing between sharded placements preserves
  the global layout; [the optimizer subdesign](optimizer.md) converts between `Flat` and
  `TensorAtomic` this way.
- `get_local_tensor(index)`: the local view for one logical tensor.
- `get_dtensor(index)`: the corresponding DTensor, used by the optimizer and distributed
  checkpointing.

# Flow

Below is what module parameters look like after each FSDP stage.

Key contract: an FsdpModule’s owned parameters are only unsharded during its forward and
backward.

| Stage                                        | Action                                                                                                                                            | param.data after action       | param.grad after action                      |
| :------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------ | :---------------------------- | :------------------------------------------- |
| After fully_shard / Start of a training loop | optimizer.zero_grad(set_to_none=True)                                                                                                             | DTensor backed by main_weight | None or a zeroed DTensor backed by main_grad |
| Pre-forward / during forward                 | Switch to the unsharded parameter; allgather model_weight into param.data                                                                         | A full-size plain Tensor      | None                                         |
| During forward                               | None                                                                                                                                              | Unchanged                     | Unchanged                                    |
| Post-forward                                 | Release param.data; switch to the sharded parameter                                                                                               | DTensor backed by main_weight | None or a DTensor backed by main_grad        |
| Pre-backward                                 | Switch to the unsharded parameter; allgather model_weight into param.data                                                                         | A full-size plain Tensor      | None                                         |
| During backward                              | Autograd sets param.grad                                                                                                                          | Unchanged                     | A full-size plain Tensor                     |
| Post-backward                                | Reduce-scatter param.grad; The result is written/accumulated to the sharded parameter’s grad; Release param.grad; switch to the sharded parameter | DTensor backed by main_weight | DTensor backed by main_grad                  |
| If more microbatches                         | Go back to pre-forward                                                                                                                            | Unchanged                     | Unchanged                                    |
| Optimizer step                               | optimizer.step()                                                                                                                                  | DTensor backed by main_weight | DTensor backed by main_grad                  |
| Post optimizer step                          | Quantize main_weight to model_weight                                                                                                              | DTensor backed by main_weight | DTensor backed by main_grad                  |

For HSDP and HFSDP, during post-backward for the last microbatch, we should further
reduce-scatter the sharded parameters according to `Placements.optimizer`. This way, the
reduce-scatters can be overlapped with backward compute instead of being exposed before
the optimizer step. Accordingly, prior to the forward pass of the first micro_batch, we
also need to all-gather the sharded parameters across the entire DP domain (outer \+
inner).

# Implementation Plan

## Separate code paths in main

### Production

- [`megatron_fsdp`](../megatron_fsdp/): the existing, non-experimental implementation
- Still maintained and occasionally optimized

### Experimental (this doc)

- [`megatron_fsdp/experimental`](../megatron_fsdp/experimental/): the long-term version
  of MFSDP that we want to maintain and use to support next generation of architectures
  and training techniques
- Development will be design driven and incremental with a peer review process
- Experimental will live alongside Production **in the `main` branch**
- Once battle-tested and demonstrating performance parity (e.g. by MLPerf models) on a
  per-model basis, onboard models and customers gradually.
- After enough adoption, production will become legacy and experimental will become
  production

### Prototype

A prototype implementation by @shjwudp and @Autumn1998 remains in
[@shjwudp's fork](https://github.com/shjwudp/Megatron-LM/tree/mfsdp_refactor). The
objective is to battle-test selected features—such as per-module control and
`TracePoolAllocator`—with early users and derisk this design. Once `main` reaches the
prototype feature set (see below), further prototype development and validation will
shift to `main` so we can focus on the same code path.

Current prototype features:

- MXFP8
- Overlapping
- Prefetching
- Checkpointing to DCP
- Composibility with EP
- Double buffering (through TracePoolAllocator)

## Development process

We’ll follow a standard prototype-design-execute process.

1. **Prototype**: Strictly optional. Make the feature work in a draft PR **only** to
   derisk the design.
2. **Design**: Update this design or create a subdesign to support a new feature. Draw
   and write documentation explaining the feature and how it works. Converge and align
   on the design change.
3. **Execute**: Update code and merge. Some general guidelines:
   - Code, review, and test incrementally. Keep
     [PRs small and focused](https://google.github.io/eng-practices/review/developer/small-cls.html).
   - Favor simplicity and maintainability by default. Any performance optimization that
     increases complexity should be justified with clear evidence and measurable impact.
   - Critical horizontal features (for example, CUDA Graphs and `torch.compile`) should
     be validated from the beginning. These integrations are easy to break and difficult
     to retrofit, so we should rely on CI coverage to catch regressions early.
