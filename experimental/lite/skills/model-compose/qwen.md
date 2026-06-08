# Qwen Compose Skill

Compose Qwen3 and Qwen3.5 Megatron Lite models.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "model_compose.qwen", kind="state_machine", purpose="compose Qwen3/Qwen3.5 MLite models",
    imports=["basic.constitution"], calls=["model_compose.config_mapping", "model_compose.weight_mapping", "model_compose.build_model"],
    inputs=["task", "hf_model", "lite_model", "budget"],
    outputs=["model", "mapping", "evidence", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def qwen(task, hf_model, lite_model, budget):
    if hf_model.family not in ["qwen3", "qwen3_5"]:
        return out_of_scope("not a Qwen3/Qwen3.5 model")

    config = model_compose.config_mapping(task, hf_model.config, lite_model.config, budget.config)
    if not config.done:
        return blocked("Qwen config mapping failed", evidence=config)

    weights = model_compose.weight_mapping(task, hf_model.checkpoint, lite_model, budget.weights)
    if not weights.done:
        return blocked("Qwen weight mapping failed", evidence=weights)

    model = model_compose.build_model(task, lite_model.spec, lite_model.primitives, budget.build)
    if not model.done:
        return blocked("Qwen model build failed", evidence=model)

    risks = ["Qwen alias mismatch", "MoE router drift", "GQA head mapping drift"]
    return done(model=model.model, mapping=[config.mapping, weights.mapping], evidence=[weights.evidence, model.validation], risks=risks)
```
