# Qwen3 MoE ChunkedEP

Opt-in token-wise DeepEP overlap for Qwen3. Normal training uses saved-context forward + backward; full-layer recomputation uses graph-free forward + fused forward/backward. Qwen3 selects the composition and supplies the attention/residual/norm prefix; the three OPs own autograd and buffers without reading recomputation settings. Parameter paths are unchanged. ChunkedEP subclasses reuse native setup and expert computation through a linear-construction hook, without branching inside native Experts.

```python
from megatron.lite.model.qwen3_moe.lite.protocol import ImplConfig
from megatron.lite.runtime.contracts import ParallelConfig

impl = ImplConfig(parallel=ParallelConfig(ep=8), use_deepep=True,
    enable_ep_chunk_overlap=True, ep_chunk_count=2,
    ep_chunk_max_token_rows_per_rank=32768, ep_chunk_full_recompute=True,
)
```

Use BF16, DeepEP, EP>1 and top-k<=EP. Logical chunk count defaults to two; three and four are supported. Input capacity must cover the largest flattened local batch. Matching layers share an arena whose backing grows lazily, including after warmup when routing loads increase.
To freeze capacity, call a representative layer's `chunked_ep.materialize(expert_activation_max_rows=...)` for both `phase="forward"` and `phase="backward"`; over-capacity requests then fail. Event-guarded eager parking returns backing to the caching allocator before each full-recompute prefix backward, retaining capacity. The generic `before_model_offload` callback releases workspaces. Discard captured graphs before explicit release; CUDA graph capture/replay is not validated.

MTP is rejected. Normal ChunkedEP rejects outer MoE/full checkpoints. The explicit full-recompute composition above replaces the outer recompute list, including attention, with full-layer recomputation. Head/loss keeps the existing linear CE implementation and configuration, independently of ChunkedEP. Other model families are not qualified.
