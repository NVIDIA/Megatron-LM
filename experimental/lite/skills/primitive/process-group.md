# Process Group Skill

Validate process-group ownership and collective participation.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.process_group", kind="state_machine", purpose="validate process groups and collectives",
    imports=["basic.constitution"], calls=[],
    inputs=["task", "groups", "collectives", "budget"],
    outputs=["mapping", "validation", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def process_group(task, groups, collectives, budget):
    mapping = derive_rank_mapping(groups)
    if mapping.has_overlap_without_owner():
        return blocked("rank belongs to overlapping groups without explicit owner")
    if not collectives.have_same_participants(mapping):
        return blocked("collective participants diverge", evidence=collectives.diff(mapping))

    validation = [
        smoke_collective(groups, collectives, shape="tiny"),
        vary_group_size_one_axis_at_a_time(groups, budget=budget),
    ]
    risks = ["NCCL hang from participant mismatch", "rank-order mismatch"]
    return done(mapping=mapping, validation=validation, risks=risks)
```
