# Align Precision Skill

Align Megatron Lite behavior against a reference, or localize the first precision
break with controlled variables.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "basic.align_precision", kind="state_machine", purpose="recursively align precision to reference",
    imports=["basic.constitution"],
    calls=[
        "basic.constitution", "basic.find_reference", "basic.construct_proxy_task",
        "basic.review_threshold", "basic.align_precision", "basic.align_e2e_precision",
    ],
    inputs=["task", "target", "variables", "budget"],
    outputs=["alignment", "evidence", "next"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def align_precision(task, target, variables, budget):
    if task.scope not in ["precision", "primitive", "model"]:
        return out_of_scope("not a precision-alignment task")
    if budget.depth > budget.max_depth or budget.attempts > budget.max_attempts:
        return blocked("precision recursion budget exhausted")

    ref = basic.find_reference(task, layer=target.layer, candidates=task.references, budget=budget.reference)
    if not ref.done:
        return blocked("no usable reference", evidence=ref)

    policy = basic.constitution(task, layer=target.layer, reference=ref.reference)
    if not policy.done:
        return blocked("precision policy failed", evidence=policy)

    proxy_result = basic.construct_proxy_task(
        task,
        target=target,
        reference=ref.reference,
        variables=variables,
        budget=budget.proxy,
    )
    if not proxy_result.done:
        return blocked("cannot construct minimal proxy task")
    proxy_task = proxy_result.proxy

    controlled = variables.freeze_all_except(task.variable_under_test)
    if controlled.has_unknown_unfrozen_axes():
        return blocked("unknown variables remain uncontrolled", risks=controlled.unknown_axes)
    axes = controlled.axes(one_at_a_time=True)

    deterministic = run_same_setting(proxy_task, runs=3)
    if not deterministic.bitwise_equal():
        proxy_task = remove_known_nondeterminism(proxy_task)
        deterministic = run_same_setting(proxy_task, runs=3)
        if not deterministic.bitwise_equal() and policy.validation.requires_bitwise:
            return blocked("reference or target is not deterministic", evidence=deterministic)

    threshold = basic.review_threshold(task, ref.contract, requested_threshold=task.tolerance)
    if not threshold.done:
        return blocked("precision threshold review failed", evidence=threshold)
    mode = threshold.compare_mode

    full = compare(proxy_task, target, ref.reference, mode=mode)
    if full.pass_:
        return done(alignment="aligned", evidence=[deterministic, full], next=[])

    evidence = [deterministic, full]
    for axis in axes:
        experiment = vary_one_axis(proxy_task, axis=axis)
        evidence.append(compare(experiment, target, ref.reference, mode=mode))
        if evidence[-1].first_failure:
            break

    for unit in decompose(target, order=["layer", "submodule", "primitive"]):
        child = basic.align_precision(
            task.narrow_to(unit),
            target=unit,
            variables=variables.freeze_except(unit.related_axes),
            budget=budget.child(increment=["depth", "attempts"]),
        )
        evidence.append(child)
        if child.alignment in ["aligned", "localized_mismatch"] or child.blocked:
            break

    if budget.e2e.high_cost_approved:
        e2e = basic.align_e2e_precision(task, target=target, modes=task.e2e_modes, budget=budget.e2e)
        evidence.append(e2e)
        if e2e.alignment == "e2e_aligned":
            return done(
                alignment="e2e_aligned_override",
                evidence=evidence,
                next=["local mismatch overridden by high-cost e2e evidence", *e2e.risks],
            )

    return done(alignment="localized_mismatch", evidence=evidence, next=owner_of_first_failure(evidence))
```
