# Distributed Optimizer Skill

Define, implement, use, and validate distributed optimizer sharding.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.optimizer.distopt", kind="primitive", purpose="define and validate distributed optimizer primitive",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def distopt(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation.distopt, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("DistOpt contract not satisfied", evidence=contract)

    principle = {
        "semantics": "shard optimizer state and updates across data-parallel ranks",
        "invariants": ["global update equals unsharded optimizer", "state partition is deterministic"],
        "reference": reference or "single-rank optimizer with same grads",
    }
    implementation_contract = {
        "details": ["state partition", "grad shard ownership", "update and param sync"],
        "state": ["momentum/variance shard", "master param ownership", "offload policy"],
        "boundaries": ["optimizer primitive owns update state; data-parallel grad sync is a separate runtime contract"],
    }
    usage_contract = {
        "config": require_config_keys(config, ["use_distributed_optimizer", "data_parallel_size"]),
        "choose_when": ["optimizer state memory dominates", "replicated optimizer state is too expensive"],
        "avoid_when": ["single-rank proxy debugging", "FSDP already owns optimizer sharding"],
        "compose_with": ["data-parallel gradient path", "TP/EP/PP with clear DP group"],
    }
    validation = primitive.validate(task, primitive=implementation.distopt, implementation=implementation, budget=budget)
    risks = ["state shard drift", "param sync mismatch", "offload update device mismatch"]
    if not validation.done:
        return blocked("DistOpt validation failed", evidence=validation)
    return done(principle=principle, implementation_contract=implementation_contract, usage_contract=usage_contract, validation=validation, risks=risks)
```
