# Agent Compose Constitution

Global constraints for Megatron Lite work upstreamed through Agent Compose.

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "basic.constitution", kind="constitution", purpose="set global Agent Compose constraints",
    imports=[], calls=[],
    inputs=["task", "layer", "reference"],
    outputs=["constraints", "validation", "stop"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def constitution(task, layer, reference):
    if layer not in ["primitive", "model", "runtime"]:
        return out_of_scope("unknown Agent Compose layer")
    if reference is None:
        return blocked("no checkable validation reference")

    constraints = [
        occam_razor("choose the smallest correct and reviewable design"),
        modularity("keep primitives replaceable unless explicitly fused"),
        layering("runtime -> model -> primitive -> Megatron Core"),
        isolation("do not import the dev preview at runtime"),
    ]
    validation = [
        reference_order(["Megatron", "HuggingFace", "Torch", "first principles"]),
        require_bitwise_when_possible(reference),
        require_layer_test(layer),
        require_end_to_end_before_delivery(task),
    ]
    stop = ["missing reference", "missing validation path", "layer boundary violation"]
    return done(constraints=constraints, validation=validation, stop=stop)
```
