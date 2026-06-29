# Distopt Distributed Checkpoint Skill

Define, implement, use, and validate Megatron Core distributed checkpointing for MLite distopt continuity.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.checkpoint.distckpt", kind="primitive", purpose="checkpoint DistributedOptimizer state with mcore dist_checkpointing",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate", "primitive.optimizer.distopt"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def distckpt(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation.distckpt, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("Distopt distckpt contract not satisfied", evidence=contract)

    principle = {
        "semantics": "use mcore dist_checkpointing for DistributedOptimizer model and tensor optimizer state",
        "invariants": ["save-load-continue matches uninterrupted training", "model sharded_state_dict keys are MLite-local"],
        "reference": reference or "mcore DistributedOptimizer.sharded_state_dict roundtrip",
    }
    implementation_contract = {
        "owned_files": ["primitive.ckpt.distckpt", "model protocol distopt compose sites"],
        "state": ["model ShardedTensor metadata", "optimizer fp32 master params", "optimizer exp_avg", "optimizer exp_avg_sq", "optimizer step"],
        "boundaries": ["do not attach sharded_state_dict to generic model classes", "do not change FSDP2 or mfsdp checkpoint paths"],
    }
    usage_contract = {
        "choose_when": ["runtime use_dcp=True", "optimizer exposes sharded_state_dict", "model chunks expose sharded_state_dict"],
        "avoid_when": ["use_dcp=False local checkpoint", "non-distopt optimizer", "cross-tool mcore-MLite checkpoint interchange"],
        "compose_with": ["primitive.optimizer.distopt", "model-compose owned opt-in"],
    }
    validation = primitive.validate(task, primitive=implementation.distckpt, implementation=implementation, budget=budget)
    risks = ["missing optimizer tensor state", "model key drift", "replica_id or shard offset mismatch"]
    if not validation.done:
        return blocked("Distopt distckpt validation failed", evidence=validation)
    return done(principle=principle, implementation_contract=implementation_contract, usage_contract=usage_contract, validation=validation, risks=risks)
```
