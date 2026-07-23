# Runtime Validate

Validate the runtime lifecycle through a composed model.

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "runtime.validate", kind="state_machine", purpose="validate the runtime lifecycle end to end",
    imports=["basic.constitution", "model.compose"],
    calls=["basic.constitution", "model.compose"],
    inputs=["task", "runtime_config", "model_spec", "primitives", "reference", "budget"],
    outputs=["runtime", "evidence", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def validate(task, runtime_config, model_spec, primitives, reference, budget):
    base = basic.constitution(task, layer="runtime", reference=reference)
    if not base.done:
        return blocked("runtime constitution failed", evidence=base)
    if not conforms_to_runtime_interface(runtime_config.backend):
        return blocked("backend does not implement the public Runtime interface")

    composed = model.compose(
        task,
        model_spec=model_spec,
        primitives=primitives,
        reference=reference,
        budget=budget.model,
    )
    if not composed.done:
        return blocked("model composition failed before runtime validation", evidence=composed)

    run = execute_runtime_steps(
        ["init", "build_model", "train_step", "save", "load"],
        runtime_config,
        composed.model,
        max_steps=budget.max_steps,
    )
    if run.return_code != 0:
        return blocked("runtime lifecycle failed", evidence=[composed, run])
    return done(
        runtime=run,
        evidence=[composed, run],
        risks=["runtime success can hide model-local mismatch"],
    )
```
