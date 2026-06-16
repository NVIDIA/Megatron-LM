# Primitive Design Skill

Design a replaceable MLite primitive before implementation.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.design", kind="state_machine", purpose="design a modular primitive",
    imports=["basic.constitution"], calls=["primitive.principle", "primitive.contract", "basic.find_reference"],
    inputs=["task", "primitive", "requirements", "budget"],
    outputs=["design", "reference", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def design(task, primitive, requirements, budget):
    ref = basic.find_reference(task, layer=primitive.layer, candidates=requirements.references, budget=budget.reference)
    if not ref.done:
        return blocked("no primitive reference", evidence=ref)

    principle = primitive.principle(primitive, reference=ref.reference, constraints=requirements.constraints)
    if not principle.done:
        return blocked("primitive principle failed", evidence=principle)

    contract = primitive.contract(primitive, scope=requirements.scope, reference=ref.reference)
    if not contract.done:
        return blocked("primitive contract failed", evidence=contract)

    design = {
        "principle": principle.principle,
        "implementation_details": define_owned_modules_and_dataflow(primitive),
        "api": define_inputs_outputs_config_keys(primitive),
        "composition": declare_valid_and_invalid_combinations(primitive),
        "selection": decide_when_to_use_or_not_use(primitive),
        "replaceability": require_no_hidden_dependency(primitive),
    }
    return done(design=design, reference=ref, risks=contract.risks)
```
