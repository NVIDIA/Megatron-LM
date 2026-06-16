# Primitive Contract Skill

Define the required outputs for every MLite primitive skill.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.contract", kind="constitution", purpose="define primitive skill outputs",
    imports=["basic.constitution"], calls=[],
    inputs=["primitive", "scope", "reference"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def contract(primitive, scope, reference):
    if reference is None:
        return blocked("primitive needs a reference or first-principles invariant")

    principle = require([
        "math_or_parallel_semantics",
        "invariants",
        "shape_dtype_rank_rules",
        "what_must_match_reference",
    ])
    implementation_contract = require([
        "owned_files_or_modules",
        "public_api",
        "state_and_config",
        "process_groups_or_device_placement",
        "forward_backward_update_details",
        "failure_modes",
    ])
    usage_contract = require([
        "config_keys",
        "minimal_example",
        "valid_combinations",
        "selection_rules",
        "unsupported_combinations",
    ])
    validation = require([
        "single_gpu_or_single_node_proxy",
        "controlled_variables",
        "precision_reference",
        "composition_test",
        "e2e_path_if_applicable",
    ])
    risks = ["silent mismatch", "dtype drift", "hidden coupling", "wrong selection rule"]

    return done(
        principle=principle,
        implementation_contract=implementation_contract,
        usage_contract=usage_contract,
        validation=validation,
        risks=risks,
    )
```
