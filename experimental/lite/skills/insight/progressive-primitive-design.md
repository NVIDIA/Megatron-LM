# Progressive Primitive Design Skill

Plan a new primitive from first principles to performance.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "insight.progressive_primitive_design", kind="state_machine", purpose="stage new primitive design",
    imports=["basic.constitution"], calls=["primitive.design", "primitive.validate", "perf.optimize"],
    inputs=["task", "primitive", "budget"],
    outputs=["stages", "evidence", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def progressive_primitive_design(task, primitive, budget):
    stages = ["principle", "single_device", "single_node", "distributed", "model_compose", "perf"]
    evidence = []

    design = primitive.design(task, primitive, primitive.requirements, budget.design)
    evidence.append(design)
    if not design.done:
        return blocked("primitive design stopped at contract design", evidence=evidence)

    validation = primitive.validate(task, primitive=primitive, implementation=primitive.implementation, budget=budget.validation)
    evidence.append(validation)
    if not validation.done:
        return blocked("primitive design stopped at validation", evidence=evidence)

    optimization = perf.optimize(task, target=primitive.implementation, constraints=primitive.perf_constraints, budget=budget.perf)
    evidence.append(optimization)
    if not optimization.done:
        return blocked("primitive design stopped at performance", evidence=evidence)

    return done(stages=stages, evidence=evidence, risks=["performance stage must not rewrite correctness contract"])
```
