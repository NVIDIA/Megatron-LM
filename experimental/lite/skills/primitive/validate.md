# Primitive Validate Skill

Validate one primitive before using it in a model.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.validate", kind="state_machine", purpose="validate primitive correctness and precision",
    imports=["basic.constitution"], calls=["basic.construct_proxy_task", "basic.align_precision"],
    inputs=["task", "primitive", "implementation", "budget"],
    outputs=["validation", "evidence", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def validate(task, primitive, implementation, budget):
    proxy = basic.construct_proxy_task(
        task, target=primitive, reference=implementation.reference, variables=implementation.variables, budget=budget.proxy
    )
    if not proxy.done:
        return blocked("primitive proxy task not constructed", evidence=proxy)

    precision = basic.align_precision(task, target=primitive, variables=implementation.variables, budget=budget.precision)
    if not precision.done:
        return blocked("primitive precision not validated", evidence=precision)

    validation = [
        "static_contract",
        "single_gpu_or_node_proxy",
        "controlled_variable_precision",
        "composition_with_adjacent_primitives",
        "usage_example_runs",
    ]
    return done(validation=validation, evidence=[proxy, precision], risks=precision.next)
```
