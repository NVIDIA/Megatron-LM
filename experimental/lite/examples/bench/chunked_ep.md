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
largest flattened local batch. Activation backing grows lazily; new routing
loads can grow it after warmup. Matching layers share one arena, not one per
layer. To freeze capacity, call a representative layer's `chunked_ep.materialize`
with `expert_activation_max_rows` for both `phase="forward"` and `phase="backward"`
before execution; over-capacity requests then fail. Event-guarded eager parking
returns backing to the caching allocator while retaining capacity, before each
full-recompute prefix backward. The runtime's model-offload callback releases
the workspace. Discard captured graphs before explicit release; CUDA graph
capture/replay is not validated by this PR.

MTP is rejected. Ordinary ChunkedEP rejects outer MoE/full checkpoints; select
the explicit full-recompute composition above, which replaces the outer
recompute list (including attention) with full-layer recomputation. Head/loss retains
the existing linear CE implementation and configuration, independently of
ChunkedEP. Other model families are not qualified.

End-to-end measurements at `21d6918ab` used Qwen3, BF16, EP8, top-k8, two chunks,
full recomputation, identical existing linear CE, random weights, no optimizer
update, lazy activation capacity, and 3 warmup + 10 measured steps without a profiler.
Ranges below cover all eight ranks of one paired run, not confidence intervals.

| Layers / local tokens / microbatches | Speedup | Allocated peak reduction | Reserved peak reduction |
| --- | --- | --- | --- |
| 1 / 32768 / 1 | 1.0177–1.0185x | 12.67–16.11% | -12.95 to -10.03% |
| 48 / 16384 / 16 | 1.1331–1.1334x | 5.43–9.69% | 0.20 to 4.91% |

Negative reductions mean increased memory. Loss maximum absolute differences
were 0 and 5.44e-5, respectively; sampled gradient differences were at most
4.10e-7 and 2.83e-7. These are not full-tensor precision checks. Both scales
completed offload/recovery with no measured allocator retries or OOMs; lazy
capacity still grew after warmup. Normal-backward smoke tests at `2a2f98a03`
(1 layer, 32768 local tokens, 1 microbatch, EP8/top-k8, chunks 2/3/4,
1 warmup + 2 repeats, no optimizer update) completed on all eight ranks:
loss differences were zero and sampled gradient differences were at most
1.20e-7. The n=2 smoke repeated at `21d6918ab` with the same precision bounds;
allocated peak increased 10.43–10.81%. Smoke establishes no formal speedup or parity.
Final-source n=3/4 GPU qualification, optimizer updates, and isolated OP
activation savings remain incomplete. Older combined head/CE measurements
are not isolated ChunkedEP gains.
