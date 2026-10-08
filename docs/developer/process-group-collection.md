# Building a `ProcessGroupCollection` for your own grid

`initialize_model_parallel` builds one standard parallel grid and stores its groups in
`parallel_state`. A framework that places a model on its own grid can build a
`ProcessGroupCollection` for that grid instead, and pass it to the components that run on it.
This also covers several grids in one job, such as an encoder and a language model with different
parallelism, or a separate inference layout.

This page shows how to build a collection with `HyperCommGrid`
(`megatron/core/hyper_comm_grid.py`). For why this matters, see
[Deprecating `parallel_state`](parallel-state-deprecation.md).

## `HyperCommGrid` in brief

`HyperCommGrid(shape, dim_names, rank_offset=0, backend=None)` places `prod(shape)` consecutive
ranks, starting at `rank_offset`, on a grid with named dimensions. The first dimension varies
fastest, so `dim_names` plays the role of the `order` argument of `initialize_model_parallel`
(default `tp-cp-ep-dp-pp`).

- `grid.create_pg(dims)` creates the groups of ranks that differ only along `dims`, and returns
  the one this rank belongs to. Each combination of dimensions can be created only once.
- `grid.get_pg(dims)` returns a group that was already created.
- `grid.get_rank_enum(dims)` returns the rank lists of all those groups. Use it for groups that
  are not a product of dimensions, such as the embedding groups.

Creating a group is collective over the default process group. Every rank must make the same
`create_pg` and `torch.distributed.new_group` calls in the same order, including ranks outside
the grid.

## Example: a dense model on a TP × PP × DP grid

```python
import torch

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.process_groups_config import ProcessGroupCollection


def build_dense_pg_collection(tp: int, pp: int, dp: int, rank_offset: int = 0):
    """Process groups for a dense model (no CP, EP, or GTP) on a TP x PP x DP grid."""
    # Same rank order as the default order of initialize_model_parallel, tp-cp-ep-dp-pp.
    grid = HyperCommGrid(
        [tp, 1, 1, dp, pp], ["tp", "cp", "ep", "dp", "pp"], rank_offset=rank_offset
    )
    pgs = ProcessGroupCollection(
        tp=grid.create_pg("tp"),
        cp=grid.create_pg("cp"),
        pp=grid.create_pg("pp"),
        ep=grid.create_pg("ep"),
        dp=grid.create_pg("dp"),
        mp=grid.create_pg(["tp", "pp"]),
        tp_cp=grid.create_pg(["tp", "cp"]),
        dp_cp=grid.create_pg(["cp", "dp"]),
        tp_dp_cp=grid.create_pg(["tp", "cp", "dp"]),
        intra_dist_opt=grid.create_pg(["tp", "cp", "ep", "dp", "pp"]),
        # GTP weight rematerialization is off. Set these to None rather than leaving them
        # unset: a helper may fall back to the global grid for a field that was never set.
        gtp_remat=None,
        expt_gtp_remat=None,
    )
    # Without experts, the expert groups have the same ranks as the dense groups.
    pgs.expt_tp = pgs.tp
    pgs.tp_ep = pgs.tp
    pgs.tp_ep_pp = pgs.mp
    pgs.expt_dp = pgs.dp

    # Word embeddings live on the first and last pipeline stages, position embeddings on the
    # first. new_group is collective, so every rank creates every group.
    pgs.embd = pgs.pos_embd = None
    rank = torch.distributed.get_rank()
    for stage_ranks in grid.get_rank_enum("pp"):
        embd_ranks = sorted({stage_ranks[0], stage_ranks[-1]})
        embd = torch.distributed.new_group(embd_ranks)
        pos_embd = torch.distributed.new_group(stage_ranks[:1])
        if rank in embd_ranks:
            pgs.embd = embd
        if rank == stage_ranks[0]:
            pgs.pos_embd = pos_embd
    return pgs
```

On 8 ranks, `build_dense_pg_collection(tp=2, pp=2, dp=2)` gives rank 5 the tensor-parallel group
`[4, 5]`, the pipeline group `[1, 5]`, and the data-parallel group `[5, 7]`. These are the same
ranks that `initialize_model_parallel(tensor_model_parallel_size=2,
pipeline_model_parallel_size=2)` assigns.

## Which fields to set

Set every field that the components you call read. An unset field reads as `None`, and some
components still treat `None` as "use the global group". If an axis is not used, set its field
to `None` explicitly, as the example does for GTP.

The example sets the fields that a dense GPT model uses when it is trained with
`DistributedDataParallel`, `get_megatron_optimizer`, and the Megatron Core schedules:

| Fields | Purpose |
|---|---|
| `tp`, `cp`, `pp`, `tp_cp` | Model layers and pipeline schedules |
| `dp`, `dp_cp`, `tp_dp_cp` | Gradient reduction, distributed-optimizer sharding, schedules |
| `mp`, `intra_dist_opt` | Optimizer gradient norm (`intra_dist_opt` with the distributed optimizer) |
| `embd`, `pos_embd` | Embedding gradient reduction across pipeline stages; `None` on ranks outside the group |
| `ep`, `expt_tp`, `tp_ep`, `tp_ep_pp`, `expt_dp` | Expert groups; DDP and the optimizer require some of them even without experts |
| `gtp_remat`, `expt_gtp_remat` | GTP weight rematerialization; `None` when off |

Other layouts need more:

- **Context parallelism:** give the `cp` dimension a real size. `tp_cp`, `dp_cp`, and `tp_dp_cp`
  then differ from `tp`, `dp`, and the TP × DP group. Expert data parallelism spans the data- and
  context-parallel ranks, so `expt_dp` can no longer reuse `dp`.
- **Mixture of experts:** build the expert groups from an expert factorization of the same
  ranks. `build_inference_pg_collection` in `megatron/core/inference/shards.py` uses a second
  grid; `examples/mimo/training/topology.py` registers an expert view on the same grid with
  `HyperCommGrid.register_view`, and also covers GTP and one grid per module.
- **More than one distributed-optimizer instance:** also set `intra_dp_cp`, `intra_expt_dp`, and
  `inter_dist_opt`.

## Passing the collection

Pass the same collection to every component that runs on the grid:

```python
config = TransformerConfig(
    ...,
    tensor_model_parallel_size=pgs.tp.size(),
    pipeline_model_parallel_size=pgs.pp.size(),
)
config.finalize_model_grads_func = finalize_model_grads
model_parallel_cuda_manual_seed(
    seed, tp_rank=pgs.tp.rank(), ep_rank=pgs.ep.rank(), etp_rank=pgs.expt_tp.rank()
)

model = GPTModel(
    config=config,
    ...,
    pre_process=pgs.pp.rank() == 0,
    post_process=pgs.pp.rank() == pgs.pp.size() - 1,
    pg_collection=pgs,
)
model = DistributedDataParallel(config, ddp_config, model, pg_collection=pgs)
optimizer = get_megatron_optimizer(
    optimizer_config, [model], use_gloo_process_groups=False, pg_collection=pgs
)

forward_backward_func = get_forward_backward_func(pp_size=pgs.pp.size(), vp_size=None)
losses = forward_backward_func(
    ...,
    p2p_communicator=P2PCommunicator(pp_group=pgs.pp, config=config),
    pg_collection=pgs,
)
```

Passed this way, the collection is enough to build and train a dense GPT model with the
Transformer Engine layer spec, without calling `initialize_model_parallel`. Some details:

- Keep the parallel sizes in `TransformerConfig` equal to the group sizes.
- `model_parallel_cuda_manual_seed` and `get_forward_backward_func` read whatever you do not pass
  from `parallel_state`. Pass the ranks to the first (with GTP on, also the GTP ranks and sizes),
  and `pp_size` and `vp_size` to the second.
- Set `config.finalize_model_grads_func` as shown; the schedules call it with the collection.
- With pipeline parallelism, the schedules take a `P2PCommunicator` for the pipeline group
  together with the collection.
- The collection carries no Gloo groups. Pass `use_gloo_process_groups=False` to
  `get_megatron_optimizer`; it raises an error otherwise.
- Some layers still use [tier-4](parallel-state-deprecation.md#what-is-deprecated) global state,
  which the collection does not cover. For example, the local (non-Transformer Engine) attention
  uses the global memory buffer that `initialize_model_parallel` creates.
