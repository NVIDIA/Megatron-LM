# Expert Parallel Skill

Define, implement, use, and validate expert parallelism.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.parallel.ep", kind="primitive", purpose="define and validate EP primitive",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def ep(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation.ep, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("EP contract not satisfied", evidence=contract)

    principle = {
        "semantics": "partition experts across ranks while router semantics stay global",
        "invariants": ["EP=1 equals local MoE", "dispatch/combine preserve token order and weights"],
        "reference": reference or "Megatron MoE expert parallel path",
    }
    implementation_contract = {
        "details": ["ep_group", "expert placement", "token dispatcher", "combine weights"],
        "collectives": ["all_to_all token exchange", "optional grouped GEMM locality"],
        "state": ["expert ownership", "capacity/drop policy", "router dtype"],
    }
    usage_contract = {
        "config": require_config_keys(config, ["expert_model_parallel_size", "num_experts"]),
        "choose_when": ["num_experts exceeds local capacity", "MoE communication is acceptable"],
        "avoid_when": ["router nondeterminism is unresolved", "expert count cannot map cleanly"],
        "compose_with": ["TP only with explicit expert and tensor group nesting", "FSDP/DistOpt through optimizer owner"],
    }
    validation = primitive.validate(task, primitive=implementation.ep, implementation=implementation, budget=budget)
    risks = ["router tie instability", "token drop mismatch", "all-to-all participant mismatch"]
    if not validation.done:
        return blocked("EP validation failed", evidence=validation)
    return done(principle=principle, implementation_contract=implementation_contract, usage_contract=usage_contract, validation=validation, risks=risks)
```
