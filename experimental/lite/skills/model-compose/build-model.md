# Build Model Skill

Compose a Megatron Lite model from validated primitives.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "model_compose.build_model", kind="state_machine", purpose="compose MLite model from primitives",
    imports=["basic.constitution"], calls=["primitive.select_for_compose", "primitive.validate", "basic.align_precision"],
    inputs=["task", "model_spec", "primitives", "budget"],
    outputs=["model", "validation", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def build_model(task, model_spec, primitives, budget):
    selected = primitive.select_for_compose(task, model_spec=model_spec, candidates=primitives, budget=budget.selection)
    if not selected.done:
        return blocked("primitive selection failed before model compose", evidence=selected)

    for selected_primitive in selected.selection:
        result = primitive.validate(task, primitive=selected_primitive, implementation=selected_primitive.impl, budget=budget.primitive)
        if not result.done:
            return blocked("primitive not validated before model compose", evidence=result)

    model = compose_layers(model_spec, selected.selection, boundary=["runtime", "model", "primitive"])
    precision = basic.align_precision(task, target=model, variables=model.variables, budget=budget.precision)
    if not precision.done:
        return blocked("model precision failed", evidence=precision)

    return done(model=model, validation=precision, risks=["composition can hide primitive boundary bugs"])
```
