# Runtime Skill

Validate the MLite runtime path end to end.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "application.runtime", kind="state_machine", purpose="validate MLite runtime build/train/save/load",
    imports=["basic.constitution"], calls=["basic.align_precision"],
    inputs=["task", "runtime_config", "model", "budget"],
    outputs=["runtime", "evidence", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def runtime(task, runtime_config, model, budget):
    if runtime_config.backend != "mlite":
        return out_of_scope("not MLite runtime")

    run = execute_runtime_steps(["init", "build_model", "train_step", "save", "load"], runtime_config, model)
    if not run.return_code == 0:
        return blocked("runtime path failed", evidence=run)

    precision = basic.align_precision(task, target=model, variables=runtime_config.variables, budget=budget.precision)
    evidence = record_evidence(task, run=run, comparison=precision, environment=budget.env)
    return done(runtime=run, evidence=evidence, risks=["runtime success can hide model-local mismatch"])
```
