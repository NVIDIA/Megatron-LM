# Fusion Skill

Evaluate fusion opportunities without breaking primitive boundaries.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "perf.fusion", kind="state_machine", purpose="evaluate safe fusion opportunities",
    imports=["basic.constitution"], calls=["primitive.fuse", "perf.measure"],
    inputs=["task", "candidates", "target", "budget"],
    outputs=["fusion", "evidence", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def fusion(task, candidates, target, budget):
    fused = propose_fusion(candidates, target)
    decision = primitive.fuse(task, primitives=candidates, fused_design=fused, budget=budget.fuse)
    if not decision.done:
        return blocked("fusion decision failed", evidence=decision)

    measurement = perf.measure(task, target=fused, workload=budget.workload, budget=budget.measure)
    if not measurement.done:
        return blocked("fused target measurement failed", evidence=measurement)

    risks = [*decision.risks, "fusion must not replace precision validation"]
    return done(fusion=fused, evidence=[decision, measurement], risks=risks)
```
