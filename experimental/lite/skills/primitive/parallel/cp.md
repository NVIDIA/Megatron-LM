# Context Parallel Skill

Define, implement, use, and validate context parallelism.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.parallel.cp", kind="primitive", purpose="define and validate CP primitive",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate", "primitive.module.thd"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def cp(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation.cp, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("CP contract not satisfied", evidence=contract)

    principle = {
        "semantics": "partition sequence context while preserving attention result",
        "invariants": ["CP=1 equals unsharded attention", "position and mask mapping are identical"],
        "reference": reference or "unsharded attention formula",
    }
    implementation_contract = {
        "details": ["cp_group", "zigzag_split_for_cp", "zigzag_reconstruct_from_cp_parts", "position ids", "mask partition"],
        "collectives": ["context gather/scatter or ring attention path"],
        "autograd": ["no non-differentiable fallback collectives"],
    }
    usage_contract = {
        "config": require_config_keys(config, ["context_parallel_size", "sequence_length"]),
        "choose_when": ["long context exceeds local memory", "attention implementation supports CP"],
        "avoid_when": ["reference cannot validate positions/masks", "model attention variant lacks CP support"],
        "compose_with": ["TP only with explicit head/context ownership", "PP by stage-local context policy", "THD packed sequences via zigzag THD slicing"],
    }
    thd_validation = None
    if config.use_thd and not task.stack.contains("primitive.module.thd"):
        thd_validation = primitive.module.thd(task.push("primitive.parallel.cp"), implementation=implementation, config=config, reference=reference, budget=budget.thd)
        if not thd_validation.done:
            return blocked("CP THD composition failed", evidence=thd_validation)

    validation = primitive.validate(task, primitive=implementation.cp, implementation=implementation, budget=budget)
    risks = ["zigzag reconstruction mismatch", "position mismatch", "mask mismatch", "non-differentiable gather fallback"]
    if not validation.done:
        return blocked("CP validation failed", evidence=validation)
    return done(principle=principle, implementation_contract=implementation_contract, usage_contract=usage_contract, validation=[thd_validation, validation], risks=risks)
```
