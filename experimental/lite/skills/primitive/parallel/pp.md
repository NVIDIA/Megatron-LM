# Pipeline Parallel Skill

Define, implement, use, and validate pipeline parallelism.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.parallel.pp", kind="primitive", purpose="define and validate PP primitive",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def pp(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation.pp, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("PP contract not satisfied", evidence=contract)

    principle = {
        "semantics": "partition ordered layers into pipeline stages",
        "invariants": ["stage concatenation equals full layer order", "microbatch schedule preserves gradients"],
        "reference": reference or "Megatron pipeline schedule",
    }
    implementation_contract = {
        "details": ["stage assignment", "send/recv activation tensors", "microbatch schedule"],
        "state": ["layer ownership", "activation shape", "loss stage"],
        "boundaries": ["first/last stage embeddings and heads", "MTP or auxiliary heads if present"],
    }
    usage_contract = {
        "config": require_config_keys(config, ["pipeline_model_parallel_size", "num_layers"]),
        "choose_when": ["model depth exceeds single-rank memory", "stageable layer stack"],
        "avoid_when": ["tiny proxy unless testing PP itself", "uneven stage ownership without explicit plan"],
        "compose_with": ["TP/EP inside each stage", "FSDP/DistOpt outside stage groups"],
    }
    validation = primitive.validate(task, primitive=implementation.pp, implementation=implementation, budget=budget)
    risks = ["stage boundary off by one", "activation shape mismatch", "schedule deadlock"]
    if not validation.done:
        return blocked("PP validation failed", evidence=validation)
    return done(principle=principle, implementation_contract=implementation_contract, usage_contract=usage_contract, validation=validation, risks=risks)
```
