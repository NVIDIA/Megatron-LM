# Bench Skill

Run benchmark-style checks without weakening precision evidence.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "application.bench", kind="state_machine", purpose="benchmark with controlled variables and evidence",
    imports=["basic.constitution"], calls=["perf.measure"],
    inputs=["task", "bench_config", "target", "budget"],
    outputs=["bench", "evidence", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def bench(task, bench_config, target, budget):
    controlled = bench_config.variables.freeze_all_except(bench_config.axis)
    if controlled.has_unknown_unfrozen_axes():
        return blocked("bench has uncontrolled variables", risks=controlled.unknown_axes)

    measurement = perf.measure(task, target=target, workload=bench_config.workload, budget=budget.measure)
    if not measurement.done:
        return blocked("bench measurement failed", evidence=measurement)

    evidence = record_evidence(task, run=measurement.run, comparison=measurement.metrics, environment=budget.env)
    risks = ["benchmarks are not correctness proof", "uncontrolled variables can dominate"]
    return done(bench=measurement.metrics, evidence=evidence, risks=risks)
```
