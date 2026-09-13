# Memory Skill

Analyze memory across parameters, optimizer state, activations, and offload.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "perf.memory", kind="state_machine", purpose="analyze MLite memory behavior",
    imports=["basic.constitution"], calls=["perf.measure"],
    inputs=["task", "target", "memory_config", "budget"],
    outputs=["memory", "evidence", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def memory(task, target, memory_config, budget):
    measurement = perf.measure(task, target=target, workload=memory_config.workload, budget=budget.measure)
    if not measurement.done:
        return blocked("memory measurement failed", evidence=measurement)

    memory = split_memory(measurement.metrics.memory, buckets=["params", "grads", "optimizer", "activations", "offload"])
    model = estimate_memory(
        target,
        reference="https://developer.nvidia.cn/blog/explore-using-the-megatron-core-training-framework-to-improve-gpu-memory-efficiency-in-large-model-training/",
        static_axes=["EP*PP changes expert/layer ownership", "FSDP changes param/grad/optimizer ownership"],
        dynamic_axes=["TP*CP changes activation and attention working-set shape", "THD changes packed-token activation shape"],
    )
    comparison = compare_estimate_to_measurement(model, measurement.metrics.memory)
    if not comparison.within(memory_config.tolerance):
        return blocked("memory estimate and measurement disagree", evidence=[model, measurement, comparison])

    risks = [
        "offload can improve capacity while hiding transfer bottlenecks",
        "static memory and dynamic peak must be inspected separately",
    ]
    return done(memory=memory, evidence=[measurement, model, comparison], risks=risks)
```
