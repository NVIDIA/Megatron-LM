# Basic Constitution Skill

Global constraints for all Megatron Lite skills.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "basic.constitution", kind="constitution", purpose="set global MLite constraints",
    imports=[], calls=[],
    inputs=["task", "layer", "reference"],
    outputs=["constraints", "validation", "stop"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def constitution(task, layer, reference):
    if reference is None:
        return blocked("no validation reference")
    if task.requires_undefined_skill:
        return out_of_scope("required procedural skill is undefined")

    constraints = [
        occam_razor("choose the smallest correct and reviewable design"),
        modularity("primitives are replaceable unless explicitly fused"),
        bounded_state_machine("every procedural skill has finite exits"),
    ]

    validation = [
        reference_order(["Megatron", "HuggingFace", "Torch", "first principles"]),
        require_bitwise_when_possible(reference),
        minimal_primitive_check(scope=["single_gpu", "single_node"], reduce=["layers", "experts"]),
        require_end_to_end_before_delivery(),
    ]

    stop = [
        "missing reference",
        "missing validation path",
        "required procedural skill is undefined",
    ]

    return done(constraints=constraints, validation=validation, stop=stop)
```
