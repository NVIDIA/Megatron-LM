# Find Reference Skill

Find the strongest checkable reference for a Megatron Lite task.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "basic.find_reference", kind="state_machine", purpose="select a checkable validation reference",
    imports=[], calls=[],
    inputs=["task", "layer", "candidates", "budget"],
    outputs=["reference", "contract", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def find_reference(task, layer, candidates, budget):
    if task.scope not in ["primitive", "model", "runtime", "application", "precision"]:
        return out_of_scope("unknown validation surface")

    ordered = [
        megatron_reference(task, layer),
        huggingface_reference(task, layer),
        torch_reference(task, layer),
        first_principles_formula_or_distributed_invariant(task, layer),
        *candidates,
    ]

    risks = []
    for ref in ordered[:budget.max_candidates]:
        if ref is None or not ref.exists():
            continue

        contract = extract_contract(
            ref,
            fields=["inputs", "outputs", "shape", "dtype", "seed", "variables", "tolerance"],
        )
        if not contract.is_checkable():
            risks.append((ref, "reference exists but contract is incomplete"))
            continue
        if ref.requires_unavailable_assets():
            risks.append((ref, "reference requires unavailable assets"))
            continue

        variables = freeze_variables(contract.variables, except_=task.variable_under_test)
        return done(reference=ref, contract=contract.with_variables(variables), risks=risks)

    return blocked("no checkable reference", risks=risks)
```
