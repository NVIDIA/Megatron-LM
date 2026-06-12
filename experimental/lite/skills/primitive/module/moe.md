# MoE Primitive Skill

Define, implement, use, and validate mixture-of-experts primitives.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.module.moe", kind="primitive", purpose="define and validate MoE module primitive",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate", "primitive.parallel.ep"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def moe(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation.moe, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("MoE contract not satisfied", evidence=contract)

    principle = {
        "semantics": "router selects experts and combines weighted expert outputs",
        "invariants": ["router logits/topk match reference", "dispatch/combine preserves token order"],
        "reference": reference or "HuggingFace/Megatron MoE layer or first-principles weighted sum",
    }
    implementation_contract = {
        "details": ["router", "topk", "token dispatcher", "DeepEP dispatcher", "expert MLP", "combine weights"],
        "state": ["expert params", "router dtype", "capacity/drop policy", "DeepEP dispatch metadata"],
        "boundaries": ["module owns routing math; EP owns distributed expert placement; DeepEP owns dispatch/combine transport"],
    }
    usage_contract = {
        "config": require_config_keys(config, ["num_experts", "top_k", "expert_parallel_size", "use_deepep"]),
        "choose_when": ["model architecture has sparse experts", "expert count justifies EP or grouped GEMM"],
        "avoid_when": ["router tie behavior cannot be stabilized", "DeepEP metadata cannot be validated against all-to-all"],
        "compose_with": ["primitive.parallel.ep", "DeepEP dispatcher when EP>1", "primitive.parallel.tp for expert MLP if explicit"],
    }
    ep_validation = None
    if config.expert_parallel_size > 1:
        ep_validation = primitive.parallel.ep(task, implementation=implementation, config=config, reference=reference, budget=budget.ep)
        if not ep_validation.done:
            return blocked("MoE EP composition failed", evidence=ep_validation)

    validation = primitive.validate(task, primitive=implementation.moe, implementation=implementation, budget=budget)
    risks = ["router tie sensitivity", "expert capacity mismatch", "DeepEP metadata mismatch", "hidden state flattening bug"]
    if not validation.done:
        return blocked("MoE validation failed", evidence=validation)
    return done(
        principle=principle,
        implementation_contract=implementation_contract,
        usage_contract=usage_contract,
        validation=[ep_validation, validation],
        risks=risks,
    )
```
