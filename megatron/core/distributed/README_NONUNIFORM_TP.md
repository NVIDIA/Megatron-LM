# Nonuniform Tensor Parallelism

Nonuniform Tensor Parallelism (NTP) is an opt-in path for running tensor-parallel
training when one or more data-parallel replicas have fewer active TP ranks than
the healthy replicas. The implementation lives in `nonuniform_tp.py` and is used
by importing NTP-specific helpers and the NTP DDP wrapper from a training script.

NTP does not change the default Megatron training path. A training script must
explicitly opt in.

## Basic Usage

Use the NTP APIs instead of standard DDP for the affected model:

```python
from megatron.core.distributed.nonuniform_tp import (
    NonuniformTPConfig,
    NonuniformTPDistributedDataParallel,
    initialize_nonuniform_tp_process_groups,
    ntp_init,
    ntp_map,
)
```

Create the normal model-parallel groups with the healthy TP size, then apply the
NTP process-group adjustment:

```python
ntp_config = NonuniformTPConfig(
    tp_base=4,
    tp_spares=2,
    num_reduced_tp_dp_ranks=1,
    non_active_ranks_per_dp={(0, 0, 0): [2, 3]},
)

# Call after initialize_model_parallel(... tensor_model_parallel_size=4 ...).
initialize_nonuniform_tp_process_groups(ntp_config)
```

In this example, healthy replicas use TP4. DP replica 0 runs with TP2 because
local TP ranks 2 and 3 are non-active for `(dp=0, cp=0, pp=0)`.

After constructing the model, attach NTP shard mappings to tensor-parallel
parameters:

```python
for module in model.modules():
    if module.__class__.__name__ == "TransformerLayer":
        ntp_init(module, ntp_config)

ntp_map(model.embedding.word_embeddings, ntp_config, vocab_size)
ntp_map(model.output_layer, ntp_config, vocab_size)
```

Then wrap the model with `NonuniformTPDistributedDataParallel`:

```python
ddp_model = NonuniformTPDistributedDataParallel(
    config=config,
    ddp_config=ddp_config,
    module=model,
    disable_bucketing=False,
    pg_collection=pg_collection,
    ntp_config=ntp_config,
)
```

## Configuration Fields

- `tp_base`: TP size of healthy replicas.
- `tp_spares`: Number of local TP ranks removed from each reduced replica.
  Reduced TP size is `tp_base - tp_spares`.
- `num_reduced_tp_dp_ranks`: Number of DP replicas that use reduced TP when no
  explicit `non_active_ranks_per_dp` entry is provided.
- `non_active_ranks_per_dp`: Optional mapping from `(dp_rank, cp_rank, pp_rank)`
  to local TP ranks that are non-active. Legacy `dp_rank -> ranks` keys are also
  accepted.

Set `tp_spares=0` to make the NTP helpers a no-op.

## Rank Placement And Replica Mapping

NTP configuration separates two concepts:

- Global ranks are the process ranks assigned by the launcher.
- `non_active_ranks_per_dp` values are local TP slot IDs inside a nominal
  `tp_base` replica. They are not global rank IDs.

For the helper path, `initialize_nonuniform_tp_process_groups()` assumes global
ranks are laid out as contiguous nominal DP replicas:

```text
dp_replica_size = tp_base * context_parallel_size
dp_rank         = global_rank // dp_replica_size
local_tp_rank   = global_rank % tp_base
```

With `tp_base=4`, `tp_spares=2`, and
`non_active_ranks_per_dp={(0, 0, 0): [2, 3]}`, DP replica 0 is interpreted as a
nominal TP4 replica whose local TP slots 2 and 3 are non-active. The active
local TP slots are 0 and 1, so that replica runs as TP2.

On systems where physical placement matters, make the launcher rank order match
the desired replica layout. For example, if one NVL domain has 2 GPUs and
another NVL domain has 4 GPUs, and the intent is to run those as separate DP
replicas, use a packed rank layout like:

```text
global ranks 0,1     -> reduced TP2 replica on the 2-GPU NVL domain
global ranks 2,3,4,5 -> healthy TP4 replica on the 4-GPU NVL domain
```

That packed `TP2 + TP4` layout has only 6 physical ranks, so it should be built
with explicit process groups in the opt-in training script rather than with the
contiguous nominal-replica helper. The required groups are:

```text
TP groups: [0, 1], [2, 3, 4, 5]
DP groups: [0, 2], [1, 3], [4], [5]
```

Ranks 0 and 1 are the reduced replica's active local TP slots. Ranks 2 and 3
are the healthy replica's core local TP slots and participate in DP gradient
sync with ranks 0 and 1. Ranks 4 and 5 are healthy extra local TP slots; NTP
reshards their gradients through the core ranks, so their DP groups are
singletons.

Use launcher, hostfile, `CUDA_VISIBLE_DEVICES`, or cluster-specific placement
controls to ensure these global-rank ranges actually land in the intended NVL
domains. Pass the resulting `ProcessGroupCollection` to both the model and
`NonuniformTPDistributedDataParallel`.

## Gradient Sync Behavior

Healthy full-TP replicas contain extra TP ranks that have no peer in a reduced
TP replica. During gradient sync, NTP gathers those extra gradients onto healthy
core ranks, runs the data-parallel gradient reduction on peerable ranks, then
post-sync reshards gradients back to the extra ranks before optimizer step.

For overlapped gradient reduction, post-sync reshards are delayed until all
bucket reductions have been launched. This allows earlier post-sync reshards to
overlap with the final bucket reduction instead of running fully exposed after
all gradient sync work has completed.

Bucket size matters. Prefer a small number of large DDP buckets for NTP runs
when memory allows, since each bucket can trigger post-sync reshard work.

## Training Script Integration Notes

- Use an opt-in training script derived from `pretrain_gpt.py`; do not expect
  the standard CLI path to enable NTP automatically.
- Apply `ntp_init()` to transformer layers and `ntp_map()` to vocabulary
  parallel embeddings/output layers after model construction and before the NTP
  DDP wrapper is created.
- Pass the same `NonuniformTPConfig` to process-group setup, mapping helpers,
  and `NonuniformTPDistributedDataParallel`.
- Keep `overlap_grad_reduce=True` when measuring the optimized path.
- Validate topology, loss, and gradient parity against a uniform TP baseline
  before using a new NTP layout for performance measurements.
