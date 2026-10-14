# Runtime

Runtime backends provide the execution surface for Megatron Lite. A runtime is
created from `RuntimeConfig`:

```python
from megatron.lite.runtime import RuntimeConfig, create_runtime

runtime = create_runtime(RuntimeConfig(backend="mlite"))
```

The minimal runtime interface is:

- `build_model()`
- `train_mode(handle)`
- `eval_mode(handle)`
- `forward_backward(handle, data, loss_fn=None, ...)`
- `zero_grad(handle)`
- `optimizer_step(handle)`
- `lr_scheduler_step(handle)`

Checkpointing, offload, distributed execution, and external training framework
bridges are non-goals for this slice. They should extend the interface in later
PRs only when the behavior is implemented and validated.
