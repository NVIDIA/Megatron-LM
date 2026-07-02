# Primitive Select For Compose Skill

Choose primitives for model composition without coupling their implementations.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.select_for_compose", kind="state_machine", purpose="choose primitives for model compose",
    imports=["basic.constitution"], calls=["primitive.principle"],
    inputs=["task", "model_spec", "candidates", "budget"],
    outputs=["selection", "rejected", "evidence", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def select_for_compose(task, model_spec, candidates, budget):
    selection = []
    rejected = []
    evidence = []

    for candidate in candidates[:budget.max_candidates]:
        principle = primitive.principle(candidate, reference=candidate.reference, constraints=model_spec.constraints)
        evidence.append(principle)
        if not principle.done:
            rejected.append((candidate, "principle missing"))
            continue
        if not candidate.supports(model_spec.required_features):
            rejected.append((candidate, "missing required model feature"))
            continue
        if candidate.creates_hidden_coupling(selection):
            rejected.append((candidate, "hidden primitive coupling"))
            continue
        selection.append(candidate)

    selection = minimize_complexity(selection, prefer=["single_gpu_proxy", "single_node_proxy", "existing_reference"])
    if not covers_required_features(selection, model_spec.required_features):
        return blocked("primitive selection does not cover model spec", evidence=[evidence, rejected])

    risks = ["selection can overfit one model family", "unsupported combinations must stay explicit"]
    return done(selection=selection, rejected=rejected, evidence=evidence, risks=risks)
```
