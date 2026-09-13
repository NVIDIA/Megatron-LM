# FSDP Skill

Define, implement, use, and validate fully sharded data parallelism.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.optimizer.fsdp", kind="primitive", purpose="define and validate FSDP optimizer primitive",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def fsdp(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation.fsdp, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("FSDP contract not satisfied", evidence=contract)

    principle = {
        "semantics": "shard parameters, gradients, and optimizer state while materialized computation matches reference",
        "invariants": ["materialized params equal reference", "optimizer update equals unsharded update"],
        "reference": reference or "single-rank optimizer update with same params/grads",
    }
    implementation_contract = {
        "details": ["wrap policy", "param shard", "all_gather before compute", "reduce_scatter grads"],
        "state": ["optimizer state shard", "param offload", "optimizer offload", "update-state device"],
        "boundaries": ["model modules expose params; optimizer primitive owns sharding/offload"],
    }
    usage_contract = {
        "config": require_config_keys(config, ["fsdp_size", "param_offload", "optimizer_offload"]),
        "choose_when": ["optimizer memory dominates", "param/optimizer offload is needed"],
        "avoid_when": ["debugging model math; use unsharded optimizer first"],
        "compose_with": ["TP/EP/PP through explicit param ownership", "DistOpt only with clear owner split"],
    }
    validation = primitive.validate(task, primitive=implementation.fsdp, implementation=implementation, budget=budget)
    risks = ["offload device mismatch", "optimizer state drift", "materialization timing bug"]
    if not validation.done:
        return blocked("FSDP validation failed", evidence=validation)
    return done(principle=principle, implementation_contract=implementation_contract, usage_contract=usage_contract, validation=validation, risks=risks)
```
