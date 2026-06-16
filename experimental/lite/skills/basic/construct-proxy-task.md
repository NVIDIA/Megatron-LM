# Construct Proxy Task Skill

Build the smallest task that can falsify a claim.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "basic.construct_proxy_task", kind="state_machine", purpose="minimize validation while preserving claim",
    imports=["basic.constitution"], calls=[],
    inputs=["task", "target", "reference", "variables", "budget"],
    outputs=["proxy", "controlled", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def construct_proxy_task(task, target, reference, variables, budget):
    if reference is None:
        return blocked("proxy task needs a reference")

    proxy = minimize(
        target,
        start=["single_gpu", "single_node"],
        reduce=["layers", "experts", "sequence", "hidden", "batch"],
        preserve=["operation", "shape_rules", "dtype_rules", task.variable_under_test],
    )
    if proxy is None or not proxy.still_tests(task.claim):
        return blocked("cannot build minimal proxy that preserves claim")

    controlled = variables.freeze_all_except(task.variable_under_test)
    if controlled.has_unknown_unfrozen_axes():
        return blocked("proxy has uncontrolled variables", risks=controlled.unknown_axes)

    risks = ["proxy may miss bugs that require full scale"]
    return done(proxy=proxy, controlled=controlled, risks=risks)
```
