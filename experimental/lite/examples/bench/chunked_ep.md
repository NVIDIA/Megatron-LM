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

End-to-end measurements at `a4f2e2b4a` used Qwen3, BF16, EP8, top-k8, two chunks,
full recomputation, identical existing linear CE, random weights, no optimizer
update, lazy activation capacity, and 3 warmup + 10 measured steps without a profiler.
Ranges below cover all eight ranks of one paired run, not confidence intervals.

| Layers / local tokens / microbatches | Speedup | Allocated peak reduction | Reserved peak reduction |
| --- | --- | --- | --- |
| 1 / 32768 / 1 | 1.0263–1.0283x | 12.67–16.11% | -12.95 to -10.03% |
| 48 / 16384 / 16 | 1.1302x | 5.48–9.70% | -2.36 to 4.91% |

Negative reductions mean increased memory. Loss maximum absolute differences
were 0 and 6.78e-5, respectively; sampled gradient differences were at most
3.66e-7 and 2.34e-7. These are not full-tensor precision checks. Neither scale
had measured allocator retries or OOMs; lazy capacity still grew after warmup.
Normal-backward smoke at `a4f2e2b4a` (1 layer, 32768 tokens, 1 microbatch,
EP8/top-k8/n=2, 1 warmup + 2 repeats, no update) had zero loss difference,
sampled gradient error <=1.20e-7 and allocated peak increase 10.43–10.81%.
Isolated OPs at `a4f2e2b4a` (32768 tokens, EP8/top-k8/n=2, 5 warmup + 30 repeats)
measured forward 16.563→15.005 ms (1.104x; allocated peak -17.38%) and fused
40.125→27.946 ms (1.436x; allocated peak -30.03%). These exclude the model head
and are not whole-model speedups or activation-only memory reductions.
Separate full-gradient diagnostics at `ba1709e5b` used 1 layer, 32768 local tokens,
1 microbatch, EP8/top-k8, n=2, full recomputation, and no optimizer update.
The paired run compared 1,245,452,544 finalized, optimizer-owned gradient elements
across 280 shards; maximum absolute error was 2.39e-7. The router weight's
relative L2 error was 0.2871%. This is measured disagreement, not accepted parity;
no non-bitwise tolerance has been approved, and 48-layer full gradients remain unvalidated.
Actual-update checks at `a4f2e2b4a` completed three steps per arm at 1 layer/32768 tokens/1 microbatch; at 48 layers/16384 tokens/16 microbatches, native OOMed after its first update, while ChunkedEP completed three updates but incurred allocator retries. One ChunkedEP offload/recovery run completed; another timed out, so recovery reliability remains unresolved.
These checks do not establish trained-model speedup or accepted numerical parity. Final-source n=3/4 qualification, normal-backward performance, and activation-only savings remain incomplete.
Both arms use the same contiguous router-index adapter; older combined head/CE gains are not isolated EP gains. No private backing-retention candidate is included in this implementation.
