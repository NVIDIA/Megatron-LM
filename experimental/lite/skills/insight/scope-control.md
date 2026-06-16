# Scope Control Skill

Decide when to split work instead of expanding a skill or task.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "insight.scope_control", kind="state_machine", purpose="prevent skill and task scope creep",
    imports=["basic.constitution"], calls=["basic.lint_skill"],
    inputs=["task", "change", "budget"],
    outputs=["decision", "split", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def scope_control(task, change, budget):
    if change.touches_multiple_namespaces() or change.exceeds(budget.max_files):
        return done(decision="split", split=propose_child_tasks(change), risks=["large review surface"])
    if change.adds_new_procedure_without_schema():
        return blocked("new procedure needs a skill schema")

    lint = basic.lint_skill(change.skill_file, registry=budget.registry, scenarios=[], budget=budget.lint)
    if not lint.done:
        return blocked("scope-control lint failed", evidence=lint)
    return done(decision="continue", split=[], risks=[])
```
