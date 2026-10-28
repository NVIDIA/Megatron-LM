<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Process groups

`ProcessGroupCollection` is the explicit handle for Megatron Core's model-parallel
and data-parallel communicators. Production `megatron/core` code should accept a
collection (or a single `torch.distributed.ProcessGroup`) from the caller and
pass it through, instead of reading MPU globals such as
`parallel_state.get_tensor_model_parallel_group()`.

Source:
[`megatron/core/process_groups_config.py`](../../../megatron/core/process_groups_config.py).

## What it holds

`ProcessGroupCollection` is a dataclass whose fields are `init=False`. Unset
fields resolve to `None` via `__getattr__`, so a caller can populate only the
groups a component needs.

| Field | Communicator |
| --- | --- |
| `tp`, `pp`, `mp` | Tensor, pipeline, and combined model-parallel groups |
| `embd`, `pos_embd` | Embedding / position-embedding groups |
| `cp`, `tp_cp`, `hcp` | Context-parallel groups (`hcp` is a list) |
| `ep`, `expt_tp`, `tp_ep`, `tp_ep_pp` | Expert-parallel groups |
| `dp`, `dp_cp`, `expt_dp` | Data-parallel groups |
| `intra_dp_cp`, `intra_expt_dp`, `inter_dist_opt`, `intra_dist_opt` | Distributed-optimizer groups |
| `gtp_remat`, `expt_gtp_remat` | GTP / EGTP weight-rematerialization groups |

See the class docstring in
[`process_groups_config.py`](../../../megatron/core/process_groups_config.py)
for the full field list, including GTP-remat variants.

`MultiModuleProcessGroupCollection` wraps several collections when one rank
hosts more than one module (for example a colocated encoder and LLM).

## How to construct it

**Set the fields you need.** This is the preferred construction for library
callers and tests that already have `ProcessGroup` handles:

```python
from megatron.core.process_groups_config import ProcessGroupCollection

pgs = ProcessGroupCollection()
pgs.tp = tp_group
pgs.pp = pp_group
pgs.dp = dp_group
```

Keyword construction is equivalent (`ProcessGroupCollection(tp=tp_group, ...)`).
Unknown names raise `ValueError`.

**Materialize from MPU globals** at bootstrap with
`ProcessGroupCollection.use_mpu_process_groups()`. Pass `required_pgs` to pull
only a subset; omit it to pull every mapped field:

```python
# All default MPU groups (training-loop / init path)
pgs = ProcessGroupCollection.use_mpu_process_groups()

# Only the groups this component will use
pgs = ProcessGroupCollection.use_mpu_process_groups(["tp", "pp", "cp"])
```

`use_mpu_process_groups` still reads `parallel_state`. It is a compatibility
shim for initialization, not a substitute for threading a collection through
`megatron/core` library code.

Then pass the same object into model, DDP, and gradient-finalization entry
points:

```python
model = TransformerBlock(..., pg_collection=pgs)
ddp_model = DistributedDataParallel(..., pg_collection=pgs)
finalize_model_grads(..., pg_collection=pgs)
```

## Why pass it through

`megatron.core.parallel_state` stores one global parallel grid. New
`megatron/core` production code that calls `parallel_state.get_*_group()` (or a
directly imported `get_*_group` helper) is tied to that grid and cannot run
under a caller-supplied or multi-grid configuration.

Passing a `ProcessGroupCollection` (or an explicit `ProcessGroup`) keeps the
communicators with the object that uses them, matches how
`TransformerBlock`, `DistributedDataParallel`, and `finalize_model_grads`
already take `pg_collection`, and avoids the class of bugs that appear when a
helper silently falls back to the global TP or DP group.

## Allowed compatibility points

Direct MPU-global reads remain acceptable only at:

- `megatron/core/parallel_state.py`
- `megatron/core/process_groups_config.py`
- initialization / bootstrap code that materializes a collection from MPU
  globals (for example `ProcessGroupCollection.use_mpu_process_groups()` in
  `megatron/training`)
- tests
- docs
- migration fallbacks that include an explicit comment

Do not add new `parallel_state.get_*_group()` reads in other `megatron/core`
library code. This guidance applies to Megatron Core, not to
`megatron/training`, unless a change opts into the same migration.

## Incorrect vs correct

```python
from megatron.core import parallel_state


class MyLayer(torch.nn.Module):
    def forward(self, x):
        # Incorrect: new megatron/core production code must not read MPU globals.
        group = parallel_state.get_tensor_model_parallel_group()
        torch.distributed.all_reduce(x, group=group)
        return x
```

```python
from megatron.core.process_groups_config import ProcessGroupCollection


class MyLayer(torch.nn.Module):
    def __init__(self, pg_collection: ProcessGroupCollection):
        super().__init__()
        # Correct: take the collection (or an explicit ProcessGroup) from the caller.
        self.tp_group = pg_collection.tp

    def forward(self, x):
        torch.distributed.all_reduce(x, group=self.tp_group)
        return x
```

A temporary `if pg_collection is None:` fallback that calls
`use_mpu_process_groups()` belongs only at a listed compatibility point, and
must say so in a comment.
