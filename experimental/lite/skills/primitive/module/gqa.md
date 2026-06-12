# GQA Primitive Skill

Define, implement, use, and validate grouped-query attention primitives.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.module.gqa", kind="primitive", purpose="define and validate GQA module primitive",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def gqa(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation.gqa, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("GQA contract not satisfied", evidence=contract)

    principle = {
        "semantics": "many query heads share fewer key/value heads",
        "invariants": ["KV repeat mapping matches reference", "attention mask and rotary positions are unchanged"],
        "reference": reference or "HuggingFace attention implementation",
    }
    implementation_contract = {
        "details": ["qkv projection layout", "head mapping", "kv repeat", "attention call"],
        "state": ["num_attention_heads", "num_key_value_heads", "head_dim"],
        "boundaries": ["GQA owns head mapping; TP owns sharding of projection weights; THD owns packed sequence boundaries"],
    }
    usage_contract = {
        "config": require_config_keys(config, ["num_attention_heads", "num_key_value_heads"]),
        "choose_when": ["architecture declares grouped KV heads"],
        "avoid_when": ["num_attention_heads not divisible by num_key_value_heads"],
        "compose_with": ["TP projection sharding", "CP context sharding with explicit head/context mapping", "THD packed attention when use_thd=True"],
    }
    validation = primitive.validate(task, primitive=implementation.gqa, implementation=implementation, budget=budget)
    risks = ["head repeat mismatch", "qkv layout bug", "attention mask drift"]
    if not validation.done:
        return blocked("GQA validation failed", evidence=validation)
    return done(principle=principle, implementation_contract=implementation_contract, usage_contract=usage_contract, validation=validation, risks=risks)
```
