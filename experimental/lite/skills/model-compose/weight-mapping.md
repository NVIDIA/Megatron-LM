# Weight Mapping Skill

Map checkpoint weights into Megatron Lite model state.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "model_compose.weight_mapping", kind="state_machine", purpose="map checkpoint weights into MLite",
    imports=["basic.constitution"], calls=["basic.find_reference", "basic.align_precision"],
    inputs=["task", "source_checkpoint", "target_model", "budget"],
    outputs=["mapping", "coverage", "evidence"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def weight_mapping(task, source_checkpoint, target_model, budget):
    candidates = [
        source_checkpoint,
        "mbridge",
        "megatron-bridge",
        *budget.reference.extra_checkpoint_tools,
    ]
    reference = basic.find_reference(task, layer="checkpoint", candidates=candidates, budget=budget.reference)
    if not reference.done:
        return blocked("checkpoint reference not found", evidence=reference)

    mapping = map_weight_names_shapes_dtypes(source_checkpoint, target_model, reference_tools=["mbridge", "megatron-bridge"])
    coverage = mapping.coverage()
    if not coverage.complete:
        return blocked("weight mapping incomplete", evidence=coverage.missing)

    precision = basic.align_precision(task, target=target_model, variables=mapping.variables, budget=budget.precision)
    if not precision.done:
        return blocked("mapped model precision failed", evidence=precision)

    return done(mapping=mapping, coverage=coverage, evidence=precision)
```
