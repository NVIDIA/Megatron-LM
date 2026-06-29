# Primitive Principle Skill

State the principle and invariants of a primitive before implementation choices.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.principle", kind="primitive", purpose="define primitive principle and invariants",
    imports=["basic.constitution"], calls=[],
    inputs=["primitive", "reference", "constraints"],
    outputs=["principle", "invariants", "reference", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def principle(primitive, reference, constraints):
    if reference is None and not constraints.has_first_principles:
        return blocked("primitive principle needs a reference or first-principles invariant")

    principle = state_math_or_parallel_semantics(primitive, reference=reference)
    invariants = [
        define_shape_dtype_rank_rules(primitive),
        define_forward_backward_update_equivalence(primitive),
        define_bitwise_or_threshold_contract(primitive, reference),
        define_single_gpu_or_single_node_proxy(primitive),
    ]
    risks = ["weak principle leads to implementation-specific tests", "missing invariant hides shared bugs"]
    return done(principle=principle, invariants=invariants, reference=reference, risks=risks)
```
