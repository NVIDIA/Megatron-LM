# Qwen3 MoE ChunkedEP

Token-wise DeepEP overlap is opt-in. The three primitives expose forward,
backward, and fused forward/backward; Qwen3 owns recomputation policy.
Normal training uses forward with saved context plus backward. Full layer
recomputation uses a graph-free forward followed by fused forward/backward.
The primitive owns the autograd bridge and buffer lifecycle. Qwen supplies
only the attention/residual/norm prefix and selects the composition; the
three OPs never read a recomputation setting. Model parameter paths are unchanged.
ChunkedEP uses separate expert and transport implementations; the existing
EP experts and dispatcher are unchanged. Model-owned cleanup is registered
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
the explicit full-recompute composition above. Head/loss computation retains
the existing linear CE implementation and configuration, independently of
ChunkedEP. Other model families are not qualified.

## Historical measurements

Qwen3 MoE, EP8/top-k8, two chunks, full recomputation, fresh processes,
three warmups and ten iterations, random weights, no optimizer update.
These measurements belong to `33be6f1ef`, before this NVIDIA-dev port.
Those runs included a separate bounded LM-head/CE optimization, now removed
from this PR. They are not isolated ChunkedEP gains and do not establish this
PR's speed or memory benefit. New comparisons must use identical CE settings.

| Layers / tokens / microbatches | Native / chunked median | Speedup | Peak allocated | Peak reserved |
| --- | --- | --- | --- | --- |
| 1 / 32K / 1 | 450.770 / 326.005 ms | 1.3827x | -56.673% | -12.935% |
| 48 / 16K / 8 | 33514.279 / 29613.317 ms | 1.1317x | -9.809% | -13.355% |

Loss maximum absolute differences were 9.54e-7 and 5.72e-5 respectively.
Both runs had no allocator retries/OOMs and passed release/recovery checks.
Rerun on the final upstream candidate before treating these as current results.
