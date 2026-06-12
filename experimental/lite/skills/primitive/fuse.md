# Primitive Fuse Skill

Decide whether primitive coupling is allowed as an explicit fused primitive.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.fuse", kind="state_machine", purpose="approve or reject primitive fusion",
    imports=["basic.constitution"], calls=["primitive.validate", "basic.align_precision"],
    inputs=["task", "primitives", "fused_design", "budget"],
    outputs=["decision", "validation", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def fuse(task, primitives, fused_design, budget):
    if not fused_design.names_all_coupled_primitives(primitives):
        return blocked("fusion hides primitive coupling")
    if not fused_design.has_independent_fallback_or_reference():
        return blocked("fusion needs an unfused reference")

    structure = primitive.validate(
        task,
        primitive=fused_design.primitive,
        implementation=fused_design.implementation,
        budget=budget.validate,
    )
    if not structure.done:
        return blocked("fused primitive structure not validated", evidence=structure)

    precision = basic.align_precision(task, target=fused_design, variables=fused_design.variables, budget=budget.precision)
    if not precision.done:
        return blocked("fused primitive precision not validated", evidence=precision)

    risks = ["fusion can mask individual primitive bugs", "fusion reduces replaceability"]
    return done(decision="approved_fused_primitive", validation=[structure, precision], risks=risks)
```
