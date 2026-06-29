# Align E2E Precision Skill

Use pretrain, SFT, or RL as the strongest but highest-cost precision evidence.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "basic.align_e2e_precision", kind="state_machine", purpose="validate precision through pretrain/SFT/RL",
    imports=["basic.constitution"], calls=["basic.constitution", "basic.find_reference", "basic.review_threshold"],
    inputs=["task", "target", "modes", "budget"],
    outputs=["alignment", "evidence", "override", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def align_e2e_precision(task, target, modes, budget):
    if task.scope not in ["precision", "model", "runtime", "application"]:
        return out_of_scope("not an end-to-end precision task")
    if not budget.high_cost_approved:
        return blocked("e2e precision requires high-cost approval")

    mode = choose_first_available(modes, order=["pretrain", "SFT", "RL"])
    if mode is None:
        return blocked("no e2e mode selected")

    ref = basic.find_reference(task, layer=target.layer, candidates=task.references, budget=budget.reference)
    if not ref.done:
        return blocked("no usable e2e reference", evidence=ref)

    policy = basic.constitution(task, layer=target.layer, reference=ref.reference)
    if not policy.done:
        return blocked("e2e precision policy failed", evidence=policy)

    run = construct_e2e_run(
        mode,
        target,
        reference=ref.reference,
        freeze=["dataset", "tokenizer", "checkpoint", "schedule", "seed", "variables"],
        compare=["loss_curve", "grad_norm", "reward_or_metric", "checkpoint_delta"],
    )
    if run is None:
        return blocked("cannot construct comparable e2e run")

    threshold = basic.review_threshold(task, ref.contract, requested_threshold=task.tolerance)
    if not threshold.done:
        return blocked("e2e precision threshold review failed", evidence=threshold)
    compare_mode = threshold.compare_mode

    evidence = run_reference_and_target(run, compare_mode=compare_mode)
    risks = ["high cost", "may preserve a shared bug", "can mask local mismatches"]

    if evidence.pass_:
        return done(alignment="e2e_aligned", evidence=evidence, override=True, risks=risks)

    return done(alignment="e2e_mismatch", evidence=evidence, override=False, risks=risks)
```
