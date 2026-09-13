# Measure Skill

Measure runtime, memory, and quality metrics with a stable protocol.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "perf.measure", kind="state_machine", purpose="measure performance without losing precision context",
    imports=["basic.constitution"], calls=[],
    inputs=["task", "target", "workload", "budget"],
    outputs=["metrics", "run", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def measure(task, target, workload, budget):
    if workload is None:
        return blocked("missing workload")

    run = execute_with_protocol(target, workload, warmup=budget.warmup, repeats=budget.repeats)
    metrics = collect_metrics(run, fields=["tokens_per_sec", "step_time", "memory", "loss_or_reward"])
    if metrics.has_missing_fields():
        return blocked("performance evidence incomplete", evidence=metrics)

    risks = ["performance numbers without precision evidence are not sufficient"]
    return done(metrics=metrics, run=run, risks=risks)
```
