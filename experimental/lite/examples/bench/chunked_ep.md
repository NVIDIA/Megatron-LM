# Qwen3 MoE ChunkedEP

Token-wise DeepEP overlap is opt-in. The three primitives expose forward,
backward, and fused forward/backward; Qwen3 owns recomputation policy.
Normal training uses forward with saved context plus backward. Full layer
recomputation uses a graph-free forward followed by fused forward/backward.
The primitive owns the autograd bridge and buffer lifecycle. Qwen supplies
only the attention/residual/norm prefix and selects the composition; the
three OPs never read a recomputation setting. Model parameter paths are unchanged.
ChunkedEP subclasses reuse native setup and ordinary expert computation;
native Experts adds only a linear-construction hook, not a ChunkedEP branch.
Model-owned cleanup is registered
through the generic runtime `before_model_offload` callback.

```python
from megatron.lite.model.qwen3_moe.lite.protocol import ImplConfig
from megatron.lite.runtime.contracts import ParallelConfig

impl = ImplConfig(
    parallel=ParallelConfig(ep=8),
    use_deepep=True,
    enable_ep_chunk_overlap=True,
    ep_chunk_count=2,
    ep_chunk_max_token_rows_per_rank=32768,
    ep_chunk_full_recompute=True,
)
```

Use BF16 experts and DeepEP with EP>1, top-k<=EP. Logical chunk count defaults
to two; three and four are also representable. Set the input capacity to the
largest flattened local batch. Activation backing grows lazily. For a frozen
capacity, call each MoE layer's `chunked_ep.materialize` with
`expert_activation_max_rows` before each phase. Shared storage is guarded by
consumer events. `chunked_ep.release` releases it for offload; the
runtime invokes release before model unload. Capture owners must discard
graphs before explicit release. This PR does not claim CUDA graph validation.

MTP is rejected. Ordinary ChunkedEP rejects outer MoE/full checkpoints; select
the explicit full-recompute composition above, which replaces the outer
recompute list (including attention) with full-layer recomputation. Head/loss retains
the existing linear CE implementation and configuration, independently of
ChunkedEP. Other model families are not qualified.

Performance is not yet validated on this port. Compare native and ChunkedEP
with identical existing linear CE settings; older combined head/CE measurements
are not isolated ChunkedEP gains. Historical data and source commits remain in
the [review PR](https://github.com/ISEEKYAN/Megatron-LM/pull/225).
