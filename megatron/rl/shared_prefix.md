# Shared-prefix packing API

`megatron.rl` owns the reusable shared-prefix packing implementation. It accepts
ordinary Python row metadata and tensors; it has no dependency on NeMo RL,
Ray, TransferQueue, a tokenizer, or a training configuration object. The
attention and Mamba execution kernels remain in `megatron.core`.

The supported data representation is a star, or a forest of independent stars.
This ownership move does not add arbitrary-depth trajectory-tree execution.
Rows share a prefix only when their group identities **and exact prompt token
sequences** match. Single-completion groups remain conventional dense units.
The representation is independent of the RL objective: GRPO and PPO adapters
can supply the same row contract without changing their losses or advantages.

## Modules

- `shared_prefix_packing`: immutable row, star and forest layouts; exact-prefix
  matching; physical padding; group subdivision and forest packing.
- `shared_prefix_tensors`: conventional-batch row construction; token gathering,
  completion/predecessor/scatter maps; reference causal masks; TP/CP alignment
  and zigzag context-parallel tensor shards. Dense masks are reference oracles;
  production callers can set `materialize_attention_mask=False`.
- `shared_prefix_metadata`: group-coherent sharding, stable inverse row order,
  equal real-row execution slots, and repeated rollout-group identities.
- `shared_prefix_cost`: backbone plus expanded-token estimates for balancing.
- `shared_prefix_execution`: execution units and plans; prescribed-slot
  resolution; training and evaluation budgets; dense fallback repacking and
  MTP-normalization-preserving fallback merging.
- `shared_prefix_alignment`: real-row splits to a caller-agreed distributed
  forward count, with conservative or prefix-preserving reconstruction.
- `shared_prefix_dense_bins`: expanded-row packing followed by shared-prefix
  reconstruction within aligned dense training bins.

The layout, execution, metadata, alignment and cost modules require only the
Python standard library. Tensor materialization additionally requires Torch.
Importing them does not initialize a model or load the optional Pydantic
generation request API. The `megatron-core` wheel includes `megatron.rl`.

## Minimal packing example

```python
from megatron.rl.shared_prefix_metadata import plan_fixed_execution_slots
from megatron.rl.shared_prefix_execution import plan_shared_prefix_execution_units
from megatron.rl.shared_prefix_packing import SharedPrefixRow

rows = [
    SharedPrefixRow(0, "prompt-a", (10, 11, 12), 2),
    SharedPrefixRow(1, "prompt-a", (10, 11, 12), 3),
]
schedule = plan_fixed_execution_slots(
    group_ids=[row.group_id for row in rows],
    sequence_lengths=[row.total_length for row in rows],
    bin_capacity=12,
)
slots = tuple(
    tuple(i for i, slot in enumerate(schedule.row_slot_ids) if slot == slot_id)
    for slot_id in range(schedule.units_per_group_by_chunk[0])
)
units = plan_shared_prefix_execution_units(
    rows, row_slots=slots, bin_capacity=12, padding_multiple=1
)
assert len(units) == 1
assert units[0].physical_length == 8  # prompt once plus both completions
```

Callers choose their conventional length-only packer and pass its `pack`
method as `dense_packer` when requesting dense fallback repacking. This keeps
framework-specific packing algorithm selection out of this library.

## Integration responsibilities

The caller validates its configuration and model capabilities, supplies the
TP/CP topology and padding multiple, transports metadata with source rows,
and agrees on a forward count across all participating model ranks. It then
calls the alignment helpers with that count. No helper launches collectives
or fabricates dummy training examples.

Training must fit both the physical shared backbone budget and the expanded
MTP budget. Physical-only evaluation packing requires uniformly disabling
MTP in the caller. Dense-bin reconstruction retains one MTP auxiliary-loss
normalization group per original dense bin. Prefix sharing in MTP itself is
outside this API's current execution scope.

NeMo RL supplies batch/configuration adapters and worker integration. Its
historical shared-prefix imports are compatibility exports of this canonical
implementation. Install the matching Megatron version in **both the driver
and model-worker environments** when using shared-prefix planning; ordinary
dense NeMo imports remain independent of the optional Megatron backend.

## Validation scope

The portable packing tests live under `tests/unit_tests/rl/test_shared_prefix*`.
They cover exact prompt identity, source-row coverage, causal isolation,
physical padding, CP shard reconstruction, fallback behavior, execution-count
alignment, and MTP normalization. Their pure CPU contracts can run without the
repository's distributed CUDA fixtures:

```bash
uv run python -m pytest --noconftest tests/unit_tests/rl/test_shared_prefix*.py
```

GPU kernel, distributed model and RL convergence qualification are separate
gates. The ownership refactor does not change the attention/Mamba kernels,
losses, or the feature's experimental numerical status.
