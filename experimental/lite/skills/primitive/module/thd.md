# THD Primitive Skill

Define, implement, use, and validate packed THD variable-length attention.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.module.thd", kind="primitive", purpose="define and validate THD packed-sequence primitive",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate", "primitive.parallel.cp"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def thd(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation.thd, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("THD contract not satisfied", evidence=contract)

    principle = {
        "semantics": "pack variable-length sequences as total_tokens x heads x dim for attention",
        "invariants": ["cu_seqlens define sequence boundaries", "pack/unpack never crosses sequence boundaries"],
        "reference": reference or "Megatron/Core TE THD packed-sequence attention contract",
    }
    implementation_contract = {
        "details": ["PackedSeqParams", "PackedTHDBatch", "pack_nested_thd", "unpack_packed_thd_to_nested"],
        "state": ["cu_seqlens", "cu_seqlens_padded", "max_seqlen", "qkv_format=thd"],
        "boundaries": ["THD owns packing; CP owns zigzag rank partition when cp_size > 1"],
    }
    usage_contract = {
        "config": require_config_keys(config, ["use_thd", "sequence_length", "context_parallel_size"]),
        "choose_when": ["variable-length SFT/RL data", "padding waste dominates", "attention supports packed THD"],
        "avoid_when": ["reference cannot expose cu_seqlens", "operator path lacks packed-sequence support"],
        "compose_with": ["primitive.parallel.cp via zigzag THD slicing", "primitive.module.gqa attention path"],
    }

    cp_validation = None
    if config.context_parallel_size > 1 and not task.stack.contains("primitive.parallel.cp"):
        cp_validation = primitive.parallel.cp(task.push("primitive.module.thd"), implementation=implementation, config=config, reference=reference, budget=budget.cp)
        if not cp_validation.done:
            return blocked("THD CP composition failed", evidence=cp_validation)

    validation = primitive.validate(task, primitive=implementation.thd, implementation=implementation, budget=budget)
    risks = ["cu_seqlens mismatch", "padding alignment drift", "CP zigzag pack/unpack bug"]
    if not validation.done:
        return blocked("THD validation failed", evidence=validation)
    return done(
        principle=principle,
        implementation_contract=implementation_contract,
        usage_contract=usage_contract,
        validation=[cp_validation, validation],
        risks=risks,
    )
```
