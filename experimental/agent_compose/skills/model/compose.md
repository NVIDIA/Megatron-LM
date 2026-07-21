# Model Compose

Compose a model only from primitives with explicit review contracts.

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "model.compose", kind="state_machine", purpose="compose a model from validated primitives",
    imports=["basic.constitution", "primitive.contract"],
    calls=["basic.constitution", "primitive.contract"],
    inputs=["task", "model_spec", "primitives", "reference", "budget"],
    outputs=["model", "evidence", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def compose(task, model_spec, primitives, reference, budget):
    base = basic.constitution(task, layer="model", reference=reference)
    if not base.done:
        return blocked("model constitution failed", evidence=base)

    evidence = []
    selected = primitives[:budget.max_primitives]
    for candidate in selected:
        checked = primitive.contract(task, primitive=candidate, reference=candidate.reference)
        evidence.append(checked)
        if not checked.done:
            return blocked("primitive contract failed before composition", evidence=evidence)

    if not covers_required_features(selected, model_spec.required_features):
        return blocked("selected primitives do not cover the model spec", evidence=evidence)
    model = compose_layers(model_spec, selected, boundary=["model", "primitive"])
    return done(model=model, evidence=evidence, risks=["composition can hide boundary bugs"])
```
