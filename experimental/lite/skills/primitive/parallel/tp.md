# Tensor Parallel Skill

Define, implement, use, and validate tensor parallelism.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.parallel.tp", kind="primitive", purpose="define and validate TP primitive",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def tp(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation.tp, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("TP contract not satisfied", evidence=contract)

    principle = {
        "semantics": "split linear/vocab dimensions across tensor-parallel ranks",
        "invariants": ["TP=1 equals unsharded reference", "row/column shard axes are explicit"],
        "reference": reference or "Megatron tensor-parallel layers",
    }
    implementation_contract = {
        "details": ["tp_group", "ColumnParallelLinear", "RowParallelLinear", "VocabParallelEmbedding"],
        "collectives": ["all_gather output when needed", "reduce_scatter or all_reduce gradients"],
        "state": ["sharded weight layout", "bias ownership", "vocab padding"],
    }
    usage_contract = {
        "config": require_config_keys(config, ["tensor_model_parallel_size"]),
        "choose_when": ["large matmul or vocab projection", "model has TP-compatible dimensions"],
        "avoid_when": ["dimension not divisible and no padding contract", "debugging non-TP primitive"],
        "compose_with": ["FSDP/DistOpt through optimizer owner", "PP", "EP with explicit group nesting", "CP/THD with explicit head/context ownership"],
    }
    validation = primitive.validate(task, primitive=implementation.tp, implementation=implementation, budget=budget)
    risks = ["wrong shard axis", "missing collective", "dtype drift across shards"]
    if not validation.done:
        return blocked("TP validation failed", evidence=validation)
    return done(principle=principle, implementation_contract=implementation_contract, usage_contract=usage_contract, validation=validation, risks=risks)
```
