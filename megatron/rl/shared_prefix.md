# Shared-prefix packing API

`megatron.rl` owns the reusable shared-prefix packing implementation. It accepts
ordinary Python row metadata and tensors; it has no dependency on NeMo RL,
Ray, TransferQueue, a tokenizer, or a training configuration object. The
attention and Mamba execution kernels remain in `megatron.core`.

The canonical data representation is `tree_layout.PackedTreeLayout`: contiguous
token spans with parent indices, supporting multiple roots and arbitrary depth.
The current planner emits stars or forests of independent stars, and the fused
attention/Mamba execution backend supports these shapes. Arbitrary-depth
trajectory-tree execution still requires backend work.
Rows share a prefix only when their group identities **and exact prompt token
sequences** match. Single-completion groups remain conventional dense units.
The slot and shard planners require every prompt group within a logical batch
chunk to have the same number of rows. Dense-bin reconstruction and dense
fallback merging can still run one row as a single-completion forest root next
to other stars: a one-row fragment of a group split across bins, or a merged
fallback row.
The representation is independent of the RL objective: GRPO and PPO adapters
can supply the same row contract without changing their losses or advantages.

## Multi-level representation and execution boundary

`SharedPrefixLayout.tree_layout` and `SharedPrefixForestLayout.tree_layout`
expose the canonical descriptor. The star builder derives positions, first-token
predecessors and padding from it; forest composition rebases parent indices;
model-input lowering reads its node spans. Source-row gather/scatter and MTP
loss-group metadata remain in the packing wrappers.

For example, a prompt followed by a shared continuation and two independent
answers is a three-level tree:

```python
from megatron.rl.tree_layout import PackedTreeLayout

tree = PackedTreeLayout(
    node_start=(0, 2, 6, 8),
    node_len=(2, 4, 2, 3),
    node_parent=(-1, 0, 1, 1),
    logical_node_len=(2, 2, 2, 3),
)
assert tree.path_token_indices(2) == (0, 1, 2, 3, 6, 7)
assert tree.first_predecessors()[2] == 3  # excludes parent padding at 4, 5
```

The test-only `build_tree_attention_allow_mask` in
`tests/unit_tests/rl/shared_prefix_oracles.py` is a correctness oracle for these
deeper trees. It permits real ancestor tokens and causal tokens within the same
node, and isolates sibling branches and unrelated roots. Its use does **not**
qualify deeper fused attention or recurrent execution. The current Hybrid adapter
calls `iter_star_roots()`, which rejects deeper trees, interleaved star storage
and padded prompt roots before invoking the existing kernels. Future generalized
tree execution can reuse the descriptor, path and predecessor contracts while
adding backend support and generalized source-row/loss mappings.

## Modules

- `tree_layout`: immutable multi-level token-span forests; ancestry, logical
  positions, predecessor indices, dense path reconstruction, and parent rebasing.
  It has no Torch or model-runtime dependency.
- `shared_prefix_packing`: immutable row, star and forest layouts; exact-prefix
  matching; physical padding; group subdivision and forest packing.
- `shared_prefix_tensors`: conventional-batch row construction; token gathering,
  completion/predecessor/scatter maps; TP/CP alignment and zigzag context-parallel
  tensor shards. It builds no dense attention masks.
- `shared_prefix_metadata`: group-coherent sharding and equal real-row execution
  slots. `GroupCoherentShardPlan.rank_order_permutation` maps each position of
  the rank-major concatenation to its source row, the forward permutation that
  restores source order.
- `shared_prefix_cost`: backbone plus expanded-token estimates for balancing.
- `shared_prefix_execution`: execution units and plans; prescribed-slot
  resolution; training and evaluation budgets; dense fallback repacking and
  MTP-normalization-preserving fallback merging.
- `shared_prefix_alignment`: real-row splits to a caller-agreed distributed
  forward count, with conservative or prefix-preserving reconstruction.
- `shared_prefix_dense_bins`: expanded-row packing followed by shared-prefix
  reconstruction within aligned dense training bins.

`MAX_SHARED_PREFIX_BRANCHES` (16) is the default number of completions stored
behind one prompt copy. It is a planning policy, not a kernel limit, and planners
that take a branch limit default to it. Every planner splits a larger exact-prompt
group evenly: 17 rows become stars of 9 and 8, not a 16-completion star and an
unshared singleton.

The layout, execution, metadata, alignment and cost modules require only the
Python standard library. Tensor materialization additionally requires Torch.
Importing them does not initialize a model or load the optional Pydantic
generation request API. A built `megatron-core` wheel ships only these packing
modules from `megatron.rl`, plus `generation_api` and the lazy package
`__init__`; the RL runtime (`rl_utils`, agents, inference and server) is not in
the wheel. An editable install still exposes all of `megatron/rl`.

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
method as `dense_packer` when requesting dense fallback repacking or dense
training bins. This keeps framework-specific packing algorithm selection out of
this library. Every model-parallel rank plans its own microbatches, so
`dense_packer` must be a deterministic, pure function of the costs it receives:
TP/CP peers that agree on the forward count but not on bin composition would run
collectives over different tokens. `pack_dense_rows` calls the packer twice and
rejects differing bins, which catches a non-repeatable packer but not one that
differs between ranks.

## Integration responsibilities

The caller validates its configuration and model capabilities, supplies the
TP/CP topology and padding multiple, transports metadata with source rows,
and agrees on a forward count across all participating model ranks. It then
calls the alignment helpers with that count. No helper launches collectives
or fabricates dummy training examples.

TP greater than one requires sequence parallelism. The padding multiple must be
a multiple of the topology quantum from `get_shared_prefix_physical_alignment`:
1 at TP1/CP1, 2·CP at TP1 with CP greater than one, and 2·TP·CP whenever TP is
greater than one. At CP1 this is 2·TP, stricter than the TP alignment of
conventional sequence packing. `resolve_shared_prefix_physical_padding_multiple`
returns the quantum when the caller passes no multiple.

Training must fit both the physical shared backbone budget and the expanded
MTP budget. Physical-only evaluation packing requires uniformly disabling
MTP in the caller. Dense-bin reconstruction retains one MTP auxiliary-loss
normalization group per original dense bin. Prefix sharing in MTP itself is
outside this API's current execution scope.

NeMo RL supplies batch/configuration adapters and worker integration. Its
shared-prefix code imports `megatron.rl` directly, and only on shared-prefix
paths; it does not re-export these modules. Install the matching Megatron
version in **both the driver and model-worker environments** when using
shared-prefix planning; ordinary dense NeMo imports remain independent of the
optional Megatron backend.

## Validation scope

The portable packing tests live under `tests/unit_tests/rl/test_shared_prefix*`
and `tests/unit_tests/rl/test_tree_layout.py`. They cover exact prompt identity,
source-row coverage, causal isolation, physical padding, CP shard reconstruction,
fallback behavior, execution-count alignment, MTP normalization, seeded
planning and tensor properties, and the signatures NeMo RL calls.
Their pure CPU contracts can run without the repository's distributed CUDA
fixtures:

```bash
uv run python -m pytest --noconftest \
    tests/unit_tests/rl/test_shared_prefix*.py tests/unit_tests/rl/test_tree_layout.py
```

`test_shared_prefix_core_parity.py` additionally checks that the padding quantum
and CP ownership agree with `megatron.core`; it is skipped where the core
shared-prefix modules cannot be imported. GPU kernel, distributed model and RL
convergence qualification are separate gates; see the
[model execution notes](../../docs/developer/shared_prefix.md#validation-status).
