# Optimize Skill

Iterate on performance while preserving precision evidence.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "perf.optimize", kind="state_machine", purpose="optimize performance under precision guardrails",
    imports=["basic.constitution"], calls=["basic.align_precision", "perf.measure", "perf.fusion", "primitive.design"],
    inputs=["task", "target", "candidates", "budget"],
    outputs=["best", "evidence", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def optimize(task, target, candidates, budget):
    best = None
    evidence = []
    queue = candidates[:budget.max_candidates]

    for attempt in range(budget.max_attempts):
        if not queue:
            break
        candidate = queue.pop(0)

        experiment = run_experiment(candidate, workload=budget.workload, knobs=budget.tunable_knobs)
        profile = perf.measure(task, target=candidate, workload=experiment.workload, budget=budget.measure)
        evidence.append((candidate, experiment, profile))
        if not profile.done:
            continue

        opportunities = analyze_profile(
            profile.metrics,
            spaces=["compute_communication_overlap", "kernel_fusion", "schedule_tuning", "new_or_split_primitive"],
        )
        if opportunities.require_fusion:
            fusion = perf.fusion(task, candidates=opportunities.fusion_candidates, target=candidate, budget=budget.fusion)
            evidence.append(fusion)
            if fusion.done:
                queue.append(fusion.fusion)
        if opportunities.require_primitive_design:
            design = primitive.design(task, opportunities.primitive, opportunities.requirements, budget.primitive_design)
            evidence.append(design)
            if design.done:
                queue.append(design.design)

        tuned = tune_parameters(candidate, opportunities.knobs)
        precision = basic.align_precision(task, target=tuned, variables=tuned.variables, budget=budget.precision)
        if not precision.done:
            continue
        measurement = perf.measure(task, target=tuned, workload=budget.workload, budget=budget.measure)
        evidence.append((tuned, precision, measurement))
        best = choose_better(best, tuned, measurement.metrics)

    if best is None:
        return blocked("no optimized candidate preserved precision", evidence=evidence)
    return done(best=best, evidence=evidence, risks=["optimization search can overfit benchmark", "profiling must be repeated after every primitive change"])
```
