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

## Launcher Rank To GPU Mapping

NTP does not assign global ranks to physical GPUs. It only observes the global
rank after `torch.distributed.init_process_group()`. The launcher must create
the desired mapping from global rank to node and GPU.

For `torchrun` with a fixed number of processes per node, global ranks are
assigned in node-major order:

```text
global_rank = node_rank * nproc_per_node + local_rank
gpu         = CUDA_VISIBLE_DEVICES[local_rank]
```

For example, with `--nnodes=2 --nproc_per_node=3`:

```text
node 0 local_rank 0 -> global rank 0 -> CUDA_VISIBLE_DEVICES[0]
node 0 local_rank 1 -> global rank 1 -> CUDA_VISIBLE_DEVICES[1]
node 0 local_rank 2 -> global rank 2 -> CUDA_VISIBLE_DEVICES[2]
node 1 local_rank 0 -> global rank 3 -> CUDA_VISIBLE_DEVICES[0]
node 1 local_rank 1 -> global rank 4 -> CUDA_VISIBLE_DEVICES[1]
node 1 local_rank 2 -> global rank 5 -> CUDA_VISIBLE_DEVICES[2]
```

If the desired physical layout is exactly:

```text
global ranks 0,1     -> 2-GPU NVL domain
global ranks 2,3,4,5 -> 4-GPU NVL domain
```

then use a launch that really creates two tasks in the first domain and four
tasks in the second domain. Plain `torchrun --nnodes=2 --nproc_per_node=N`
cannot express different process counts per node. Use a Slurm heterogeneous
launch, one task per rank, or another cluster-specific launcher that lets rank
0 and rank 1 bind to the 2-GPU domain and rank 2 through rank 5 bind to the
4-GPU domain.

With direct Slurm rank assignment, the common mapping is:

```text
global_rank = SLURM_PROCID
local_rank  = SLURM_LOCALID
world_size  = SLURM_NTASKS
```

The training script can validate the placement early with:

```python
import os
import socket
import torch
import torch.distributed as dist

local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
print(
    f"rank={dist.get_rank()} host={socket.gethostname()} "
    f"local_rank={local_rank} visible={os.environ.get('CUDA_VISIBLE_DEVICES')}",
    flush=True,
)
```

Check this printed table before trusting an NTP run. Cluster topology flags such
as NVL segment selection can constrain which GPUs are allocated, but the
launcher rank order still determines which global rank lands on each allocated
GPU.

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
