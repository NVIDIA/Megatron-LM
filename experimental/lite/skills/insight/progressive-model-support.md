# Progressive Model Support Skill

Plan model support through either an HF-first path or a nearest-reference diff path.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "insight.progressive_model_support", kind="state_machine", purpose="stage model support with HF and bridge/reference paths",
    imports=["basic.constitution"],
    calls=[
        "basic.find_reference", "basic.align_precision", "basic.align_e2e_precision",
        "primitive.select_for_compose",
        "model_compose.config_mapping", "model_compose.weight_mapping", "model_compose.build_model",
    ],
    inputs=["task", "model", "budget"],
    outputs=["path", "stages", "evidence", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def progressive_model_support(task, model, budget):
    if model.is_pure_new or not model.has_bridge_or_nearest_reference():
        result = support_pure_new_model_from_hf(task, model, budget)
    else:
        result = support_model_from_bridge_reference(task, model, budget)

    if not result.done:
        return blocked("progressive model support failed", evidence=result)
    return done(path=result.path, stages=result.stages, evidence=result.evidence, risks=result.risks)


def support_pure_new_model_from_hf(task, model, budget):
    stages = ["hf_config", "hf_weights", "PP", "EP", "CP_if_needed", "TP_if_needed", "e2e"]
    evidence = []

    config = model_compose.config_mapping(task, model.hf.config, model.lite_config, budget.config)
    evidence.append(config)
    if not config.done:
        return blocked("pure-new HF path stopped at config mapping", evidence=evidence)

    weights = model_compose.weight_mapping(task, model.hf.checkpoint, model.lite_model, budget.weights)
    evidence.append(weights)
    if not weights.done:
        return blocked("pure-new HF path stopped at weight mapping", evidence=evidence)

    primitive_steps = [
        step("PP", required=True, features=["pipeline_parallel"]),
        step("EP", required=model.has_moe, features=["expert_parallel", "moe", "deepep"]),
        step("CP", required=model.needs_long_context or model.uses_thd, features=["context_parallel", "thd"]),
        step("TP", required=model.needs_tensor_parallel_for_memory_or_perf, features=["tensor_parallel"]),
    ]
    partial = model.empty_lite_model(config=config.mapping, weights=weights.mapping)

    for primitive_step in primitive_steps:
        if not primitive_step.required:
            continue
        selection = primitive.select_for_compose(
            task.narrow_to(primitive_step.name),
            model_spec=model.spec.require(primitive_step.features),
            candidates=model.primitive_candidates(primitive_step.features),
            budget=budget.primitive_step(primitive_step.name),
        )
        evidence.append(selection)
        if not selection.done:
            return blocked("pure-new primitive selection failed", evidence=evidence)

        partial = model_compose.build_model(
            task.narrow_to(primitive_step.name),
            model_spec=partial.spec.with_primitives(selection.selection),
            primitives=partial.primitives + selection.selection,
            budget=budget.build_step(primitive_step.name),
        )
        evidence.append(partial)
        if not partial.done:
            return blocked("pure-new incremental model build failed", evidence=evidence)

        precision = compare_model_precision_ladder(
            task.narrow_to(primitive_step.name),
            target=partial.model,
            reference=model.hf,
            variables=model.variables.freeze_all_except(primitive_step.features),
            budget=budget.precision_step(primitive_step.name),
        )
        evidence.append(precision)
        if not precision.done:
            return blocked("pure-new precision ladder failed", evidence=evidence)

    e2e = basic.align_e2e_precision(task, target=partial.model, modes=task.e2e_modes, budget=budget.e2e)
    evidence.append(e2e)
    if not e2e.done:
        return blocked("pure-new model stopped at e2e", evidence=evidence)

    risks = ["HF-first path can miss bugs shared by HF conversion and Lite compose"]
    return done(path="hf_first_pure_new", stages=stages, evidence=evidence, risks=risks)


def support_model_from_bridge_reference(task, model, budget):
    stages = ["nearest_reference", "diff_primitives", "bridge_precision", "hf_cross_check", "e2e"]
    evidence = []

    reference = basic.find_reference(
        task,
        layer="model",
        candidates=[model.bridge_reference, *model.nearest_existing_models, "mbridge", "megatron-bridge"],
        budget=budget.reference,
    )
    evidence.append(reference)
    if not reference.done:
        return blocked("no nearest model reference for diff path", evidence=evidence)

    diff = model.diff_against(reference.reference)
    selection = primitive.select_for_compose(
        task.narrow_to(diff),
        model_spec=diff.model_spec,
        candidates=diff.primitive_candidates,
        budget=budget.diff_primitives,
    )
    evidence.append(selection)
    if not selection.done:
        return blocked("diff primitive selection failed", evidence=evidence)

    config = model_compose.config_mapping(task, model.hf.config, model.lite_config, budget.config)
    weights = model_compose.weight_mapping(task, model.hf.checkpoint, model.lite_model, budget.weights)
    evidence.extend([config, weights])
    if not config.done or not weights.done:
        return blocked("HF config or weight mapping failed in reference path", evidence=evidence)

    candidate = model_compose.build_model(
        task,
        model_spec=model.spec.apply_diff(diff, config.mapping),
        primitives=reference.reference.primitives + selection.selection,
        budget=budget.build,
    )
    evidence.append(candidate)
    if not candidate.done:
        return blocked("reference-diff model build failed", evidence=evidence)

    bridge_precision = compare_model_precision_ladder(
        task.narrow_to("bridge_reference"),
        target=candidate.model,
        reference=reference.reference,
        variables=model.variables.freeze_all_except(diff.changed_features),
        budget=budget.bridge_precision,
    )
    evidence.append(bridge_precision)
    if not bridge_precision.done:
        return blocked("reference-diff precision failed", evidence=evidence)

    hf_precision = compare_model_precision_ladder(
        task.narrow_to("hf_cross_check"),
        target=candidate.model,
        reference=model.hf,
        variables=model.variables.freeze_all_except(diff.changed_features),
        budget=budget.hf_precision,
    )
    evidence.append(hf_precision)
    if not hf_precision.done:
        return blocked("HF cross-check precision failed", evidence=evidence)

    e2e = basic.align_e2e_precision(task, target=candidate.model, modes=task.e2e_modes, budget=budget.e2e)
    evidence.append(e2e)
    if not e2e.done:
        return blocked("reference-diff model stopped at e2e", evidence=evidence)

    risks = ["nearest-reference path can inherit reference bugs", "HF cross-check is still required"]
    return done(path="nearest_reference_diff", stages=stages, evidence=evidence, risks=risks)


def compare_model_precision_ladder(task, target, reference, variables, budget):
    forward = basic.align_precision(
        task.with_phase("forward").with_reference(reference).with_metrics(["bitwise_when_possible", "cos_sim"]),
        target=target.forward_proxy(),
        variables=variables.freeze_all_except("forward_math"),
        budget=budget.forward,
    )
    if not forward.done:
        return blocked("forward precision failed", evidence=forward)

    backward = basic.align_precision(
        task.with_phase("backward_grad").with_reference(reference).with_metrics(["bitwise_when_possible", "grad_cos_sim"]),
        target=target.backward_proxy(),
        variables=variables.freeze_all_except("backward_grad"),
        budget=budget.backward,
    )
    if not backward.done:
        return blocked("backward gradient precision failed", evidence=[forward, backward])

    grad_norm = compare_grad_norm(
        target,
        reference,
        mode=["bitwise_when_possible", "relative_threshold", "cos_sim_supporting_signal"],
        tolerance=budget.grad_norm_tolerance,
    )
    if not grad_norm.pass_:
        return blocked("grad norm precision failed", evidence=[forward, backward, grad_norm])

    return done(alignment="forward_backward_grad_norm_aligned", evidence=[forward, backward, grad_norm], risks=[])
```
