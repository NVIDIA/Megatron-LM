# Config Mapping Skill

Map external model configs into Megatron Lite configs.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "model_compose.config_mapping", kind="state_machine", purpose="map model configs into MLite",
    imports=["basic.constitution"], calls=["basic.find_reference"],
    inputs=["task", "source_config", "target_config", "budget"],
    outputs=["mapping", "evidence", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def config_mapping(task, source_config, target_config, budget):
    reference = basic.find_reference(task, layer="config", candidates=[source_config], budget=budget.reference)
    if not reference.done:
        return blocked("config reference not found", evidence=reference)

    mapping = map_fields(
        source_config,
        target_config,
        required=["hidden_size", "num_layers", "num_heads", "dtype", "moe", "rope"],
    )
    if mapping.has_missing_required_fields():
        return blocked("config mapping incomplete", evidence=mapping.missing)

    evidence = record_evidence(task, run=mapping.check, comparison=mapping.comparison, environment=budget.env)
    return done(mapping=mapping, evidence=evidence, risks=["default mismatch", "alias mismatch"])
```
