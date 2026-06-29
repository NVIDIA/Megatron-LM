# VERL Skill

Validate MLite through VERL SFT, RL, or GRPO workflows.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "application.verl", kind="state_machine", purpose="validate VERL workflows with MLite",
    imports=["basic.constitution"], calls=["basic.align_e2e_precision"],
    inputs=["task", "verl_config", "model", "budget"],
    outputs=["workflow", "evidence", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def verl(task, verl_config, model, budget):
    if verl_config.algorithm not in ["SFT", "RL", "GRPO"]:
        return out_of_scope("not a VERL SFT/RL/GRPO workflow")

    modes = {"SFT": ["SFT"], "RL": ["RL"], "GRPO": ["RL"]}[verl_config.algorithm]
    e2e = basic.align_e2e_precision(task, target=model, modes=modes, budget=budget.e2e)
    if not e2e.done:
        return blocked("VERL e2e precision failed", evidence=e2e)

    evidence = record_evidence(task, run=e2e.evidence.run, comparison=e2e.evidence, environment=budget.env)
    risks = [
        "SFT loss curve can pass while a lower-level bug remains shared",
        "RL reward variance can hide precision drift",
        "rollout backend can become the reference accidentally",
    ]
    return done(workflow=verl_config.algorithm, evidence=evidence, risks=risks)
```
