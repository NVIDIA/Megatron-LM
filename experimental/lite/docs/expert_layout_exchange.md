# Expert deployment layout exchange

`primitive.ckpt.expert_layout` redistributes already materialized expert tensors
between contiguous EP layouts. It preserves names, shapes, dtypes and bytes,
including quantized weights and their scales. It does not quantize tensors,
change training ownership, export ordinary parameters or load a rollout model.

For eight experts, source EP4 owns two experts per shard. On an ordered
four-rank group, target EP2 owns four experts per shard and has two replicas:

| Expert IDs | Source positions | Destination positions |
| --- | --- | --- |
| 0-3 | 0, 1 | 0, 2 |
| 4-7 | 2, 3 | 1, 3 |

The sender is selected once among source replicas, and every destination
replica receives its required tensors. Rank positions refer to the explicit
group, not global rank modulo EP. TP/ETP slices must be materialized or handled
outside this helper; repeated positions are replicas of complete experts.

The exchange group must contain all required source and destination positions.
For example, EP2 to EP4 on four ranks needs both source replicas in that group:
expert 4 selects source position 1 and destination position 2, while expert 6
selects source position 3 and destination position 3. A two-rank source EP group
alone cannot reach all four destination positions. The caller supplies an
existing suitable group; the helper does not create groups or change the
training parallel layout.

```python
import torch
import torch.distributed as dist

from megatron.lite.primitive.ckpt.expert_layout import (
    ExpertLayoutPlan,
    exchange_expert_layout_tensors,
)

plan = ExpertLayoutPlan(8, 4, 2, tuple(dist.get_process_group_ranks(group)))
weights = exchange_expert_layout_tensors(
    local_expert_weights_and_scales,  # Globally named, already quantized tensors.
    plan,
    group=group,
    device=torch.device("cuda", torch.cuda.current_device()),  # NCCL group.
    expert_id_from_name=extract_global_expert_id,
)
for name, tensor in weights:
    existing_rollout_weight_sender(name, tensor)
```

All group members must fully consume the iterator, including empty or exhausted
senders. Metadata is exchanged every round to coordinate termination and tensor
split sizes; the payload uses `all_to_all_single`. Tensor bytes can mix dtypes,
and scalar scales and noncontiguous inputs are supported. Invalid ownership,
duplicate names, inconsistent plans/budgets and group-order mismatches raise on
all participating ranks before the corresponding tensor collective.

The byte budget limits source tensors before replica fan-out. A single larger
tensor travels alone; replicated sends and contributions from multiple sources
can increase send/receive buffer sizes. This is not a hard receiver-memory cap.
Source replicas must contain identical values in real deployments.

CPU/Gloo coverage runs real multi-process exchanges, without CUDA or model
dependencies:

```bash
experimental/lite/tests/run_tests.sh \
  experimental/lite/tests/unit/primitive/ckpt/test_expert_layout.py
```

This primitive is opt-in. Model export adapters should call it after deployment
quantization and before their existing rollout sender. GPU/NCCL performance and
model update/export/reload validation are required before production enablement;
the CPU tests do not establish a throughput improvement.
