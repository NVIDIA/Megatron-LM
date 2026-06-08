# Review Threshold Skill

Choose the comparison mode and require human review for non-bitwise thresholds.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "basic.review_threshold", kind="state_machine", purpose="select bitwise or reviewed tolerance",
    imports=["basic.constitution"], calls=[],
    inputs=["task", "reference_contract", "requested_threshold"],
    outputs=["compare_mode", "review", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def review_threshold(task, reference_contract, requested_threshold):
    if reference_contract is None:
        return blocked("missing reference contract for threshold review")

    if reference_contract.bitwise_possible:
        return done(compare_mode=bitwise(), review=False, risks=[])

    threshold = requested_threshold or relative(0.01)
    review = human_review_required(
        reason="non-bitwise precision threshold; 1% relative is only a default",
        threshold=threshold,
    )
    risks = ["accepted threshold can hide real precision bugs"]
    return done(compare_mode=threshold, review=review, risks=risks)
```
