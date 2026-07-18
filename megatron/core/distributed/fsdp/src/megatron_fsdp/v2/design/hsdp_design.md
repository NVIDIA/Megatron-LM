# HSDP Design

FSDP v2 implements HSDP as layout transitions over a two-dimensional
data-parallel mesh.

## Notation

### Mesh and layout

| Symbol | Meaning |
| --- | --- |
| `O`, `I` | Outer and inner mesh sizes; the mesh shape is `(O, I)` |
| `o`, `i` | This rank's outer and inner mesh coordinates |
| `B` | Padded bucket size |
| Shard layout `(a, b)` | Outer- and inner-sharding flags; `1` is sharded and `0` is replicated |
| `unshard_dim`, `reduce_dim`, `shard_dim` | Mesh dimension ID: `0` is outer and `1` is inner |

Mesh shapes, rank coordinates, and shard layouts are distinct tuple types:
`(O, I)` is a mesh shape, `(o, i)` is a rank coordinate, and values such as
`(0, 1)` are shard layouts.

### Process groups

| Group | Role |
| --- | --- |
| `dp_cp` / `expt_dp` | Full flattened data-parallel group for dense/expert parameters |
| `intra_dp_cp` / `intra_expt_dp` | Inner-DP/EDP group when `O > 1`; otherwise the corresponding full group is used |
| `inter_dist_opt` | Outer-DP group when `O > 1`; otherwise a singleton group |

The mesh dimension names are `dp_outer` for the outer dimension and
`dp_or_edp` for the inner dense-DP or expert-EDP dimension.

### Sharding strategies

The inner `sharding_strategy` supports all four strategies below. The outer
`outer_dp_sharding_strategy` supports `no_shard` and `optim`; outer
`optim` requires inner `optim_grads_params`.

| Strategy | State sharded along the selected mesh dimension |
| --- | --- |
| `no_shard` | None |
| `optim` | Main weights and optimizer state |
| `optim_grads` | Main weights, optimizer state, and main gradients |
| `optim_grads_params` | Main weights, optimizer state, main gradients, and model/transpose weights |

See the [FSDP v2 design](design.md) for the complete strategy definitions.

## Mesh

```text
mesh dimensions: (dp_outer, dp_or_edp)
mesh shape:      (O, I)
rank coordinate: (o, i)
```

Dimension 0 is outer DP. Dimension 1 is inner DP for dense parameters and
inner EDP for expert parameters. EP remains independent.

| Parameters | Mesh | Inner group | Full flattened group |
| --- | --- | --- | --- |
| Dense | `(dp_outer, dp)` | `intra_dp_cp` or `dp_cp` | `dp_cp` |
| Expert | `(dp_outer, edp)` | `intra_expt_dp` or `expt_dp` | `expt_dp` |

For `O > 1`, dimension 0 uses `inter_dist_opt`; otherwise it is a singleton
group. Each parameter group is bound to either the dense or expert mesh.

The full group is the Cartesian product of outer and inner groups.
`mesh._flatten()` is explicitly bound to the existing `dp_cp` or
`expt_dp` process group after checking that rank order matches
`mesh.mesh.flatten()`. This is communicator binding, not data reordering.

## Layout and index

A buffer layout is:

```text
shard_layout = (outer_sharded, inner_sharded)
```

| Layout | Ownership |
| --- | --- |
| `(0, 0)` | Full bucket |
| `(0, 1)` | Inner shard |
| `(1, 0)` | Outer shard of the full bucket |
| `(1, 1)` | Outer shard of the inner shard |

`storage_shard_layout` describes persistent storage. In contrast,
`unshard_dim`, `reduce_dim`, and `shard_dim` are mesh dimension IDs:
0 selects outer and 1 selects inner.

Using the notation above, `BufferIndex` defines:

| Layout | Rank-owned global interval |
| --- | --- |
| `(0, 0)` | `[0, B)` |
| `(0, 1)` | `[i*B/I, (i+1)*B/I)` |
| `(1, 0)` | `[o*B/O, (o+1)*B/O)` |
| `(1, 1)` | `[i*B/I + o*B/(I*O), i*B/I + (o+1)*B/(I*O))` |

Thus:

```text
(1, 1) = shard(shard(full, inner_rank), outer_rank)
```

Both-dimension sharding is inner first, then outer. It is not the intersection
of the independent inner-only and outer-only intervals. This order matches
inner reduce-scatter followed by outer reduce-scatter.

Each buffer owns a `BufferIndex` built from the same parameter order and
alignment. It contains:

- `ItemIndex`: item offset, size, and shape in the full bucket.
- `ShardMeta`: this rank's interval for each shard layout.

For an item interval `P` and layout interval `S`:

```text
self range  = intersection(P, S) - P.start
local range = intersection(P, S) - S.start
```

This mapping lets different buffer roles address the same parameter while
using different physical layouts.

DTensor placements are indexed by mesh dimension, while the argument to
`Shard` is a logical tensor dimension. `[Shard(0), Shard(0)]` means that
both outer and inner mesh dimensions shard tensor dimension 0.
`BufferIndex` selects the physical `(1, 1)` local slice before it is wrapped
as a DTensor. `_shard_order = [1, 0]` is used only when computing uneven-
DTensor checkpoint chunk metadata, so global offsets follow the same
inner-then-outer layout. It does not reorder the mesh or control collectives.

## Buffer layouts and conversion

For either mesh dimension, the strategy-to-storage rule is:

| Buffer role | `no_shard` | `optim` | `optim_grads` | `optim_grads_params` |
| --- | ---: | ---: | ---: | ---: |
| Model/transpose weight | 0 | 0 | 0 | 1 |
| Main weight | 0 | 1 | 1 | 1 |
| Main gradient | 0 | 0 | 1 | 1 |

Outer and inner decisions are combined as
`(outer_sharded, inner_sharded)`. For outer `optim` plus inner
`optim_grads_params`:

| Data | Persistent layout | Active view |
| --- | --- | --- |
| Model/transpose weight | `(0, 1)` | Compute weight `(0, 0)` |
| Main weight | `(1, 1)` | Optimizer parameter `(1, 1)` |
| Main gradient | `(0, 1)` | Optimizer gradient `(1, 1)` |

`get_item()` and `set_item()` intersect the requested item range with the
range present in persistent storage, then translate it to storage-local
coordinates.

`fetch_buffer(layout)` follows three rules:

1. Return storage for an exact layout.
2. Return a view when storage contains the requested shard, such as
   `(0, 1) -> (1, 1)`.
3. Otherwise use a full temporary bucket, such as `(1, 1) -> (0, 1)`.

An all-gather changes one layout bit from 1 to 0. A reduce-scatter changes one
bit from 0 to 1. An all-reduce leaves the layout unchanged. Parameters are
bound only from `(0, 0)`; reshard releases temporary storage.

## State transitions

The flows below assume inner `optim_grads_params`. The optimizer-step
boundary is identified by `set_is_last_backward(True)`. The final backward
callback marks the model-weight refresh pending at that boundary. An integrated
optimizer may install updated model weights immediately; otherwise the next
normal root pre-forward installs them before any outer or inner unshard.
Activation-recompute forwards do not consume the pending refresh.

The unshard tables distinguish the persistent model-storage allocation from
the currently valid weight view. Under the assumed inner strategy,
`_inner_dirty` remains `False`, so only `_outer_dirty` is shown.

### Outer `no_shard`

#### Reduce grad

| Step | Operation | Resulting layout |
| --- | --- | --- |
| Backward gradient | — | `(0, 0)` |
| Reduce or accumulate inner gradient shard | Inner reduce-scatter | `(0, 1)` |
| Synchronize outer replicas at the step boundary | Outer all-reduce | `(0, 1)` |
| Optimizer consumes gradient | — | `(0, 1)` |

The outer all-reduce gives every outer rank the same inner gradient shard.
Main weights and optimizer state are also replicated on outer, so every outer
rank applies the same update.

#### Unshard

| Step | Operation | Persistent model storage | Current valid weight view | `_outer_dirty` |
| --- | --- | --- | --- | --- |
| Main weight, replicated on outer | — | — | `(0, 1)` | — |
| Copy the complete inner shard into model storage | Copy | `(0, 1)` | `(0, 1)` | `False` |
| Skip outer all-gather | — | `(0, 1)` | `(0, 1)` | `False` |
| Materialize compute weight | Inner all-gather | `(0, 1)` | `(0, 0)` | `False` |
| Bind parameters | — | `(0, 1)` | `(0, 0)` | `False` |

The model-weight storage is already complete on outer after the copy. Reshard
releases the temporary `(0, 0)` compute buffer and keeps the persistent
`(0, 1)` storage.

### Outer `optim`

#### Reduce grad

| Step | Operation | Resulting layout |
| --- | --- | --- |
| Backward gradient | — | `(0, 0)` |
| Reduce or accumulate inner gradient shard | Inner reduce-scatter | `(0, 1)` |
| Shard across outer ranks at the step boundary | Outer reduce-scatter | `(1, 1)` |
| Optimizer consumes gradient | — | `(1, 1)` |

The outer reduce-scatter leaves each outer rank with one slice of the inner
gradient shard. Main weights and optimizer state use the same `(1, 1)`
ownership.

#### Unshard

| Step | Operation | Persistent model storage | Current valid weight view | `_outer_dirty` |
| --- | --- | --- | --- | --- |
| Main weight | — | — | `(1, 1)` | — |
| Copy the local slice into model storage | Copy | `(0, 1)` | `(1, 1)` | `True` |
| Reconstruct the complete inner shard | Outer all-gather | `(0, 1)` | `(0, 1)` | `False` |
| Materialize compute weight | Inner all-gather | `(0, 1)` | `(0, 0)` | `False` |
| Bind parameters | — | `(0, 1)` | `(0, 0)` | `False` |

The persistent model-weight allocation has layout `(0, 1)`, but immediately
after the copy only its local `(1, 1)` slice is current. While
`_outer_dirty=True`, outer unshard treats that storage as a `(1, 1)` source.
The outer all-gather restores the complete `(0, 1)` inner shard and clears
`_outer_dirty`. Inner unshard and reshard leave it clear; the next
optimizer-to-model copy starts the next dirty cycle.
