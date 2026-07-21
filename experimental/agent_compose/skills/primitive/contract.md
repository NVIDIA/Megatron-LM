# Primitive Contract

Define the minimum review contract for an upstreamed primitive.

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.contract", kind="constitution", purpose="define primitive review outputs",
    imports=["basic.constitution"], calls=["basic.constitution"],
    inputs=["task", "primitive", "reference"],
    outputs=["principle", "implementation", "usage", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def contract(task, primitive, reference):
    base = basic.constitution(task, layer="primitive", reference=reference)
    if not base.done:
        return blocked("primitive constitution failed", evidence=base)

    principle = require(["semantics", "invariants", "shape_dtype_rank_rules"])
    implementation = require([
        "owned_modules", "public_api", "state_and_config", "failure_modes",
    ])
    usage = require([
        "minimal_example", "selection_rules", "valid_combinations", "unsupported_combinations",
    ])
    validation = require([
        "single_gpu_or_single_node_proxy", "reference_comparison", "composition_test",
    ])
    risks = ["silent mismatch", "dtype drift", "hidden coupling", "wrong selection rule"]
    return done(
        principle=principle,
        implementation=implementation,
        usage=usage,
        validation=validation,
        risks=risks,
    )
```
