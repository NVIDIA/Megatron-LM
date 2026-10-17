# CUDA Graphs for Inference

Decode is launch-bound. A hybrid or MoE decode step issues hundreds of small
kernels whose combined GPU time is less than the CPU time to launch them, so the
device idles between kernels. CUDA graphs replace per-step launches with one
replay. Getting them to actually engage is most of the work.

Source commits: `32efeffd` (#3250), `fde3b90a` (#3527), `60a25aa6` (#3525),
`35f76df3` (#4440), `740c16e6` (#5797).

## Scopes

Two config fields control this today:

- `cuda_graph_impl`: `none` | `local` | `transformer_engine` | `full_iteration`.
  Inference graphs require **`local`**.
- `inference_cuda_graph_scope`: `none` | `layer` | `block`.

| Scope | Graphs per step | When to use |
|---|---|---|
| `block` | One, wrapping the whole decoder block | The latency target. Requires the entire block to be graph-safe. |
| `layer` | One per transformer or Mamba layer | Fallback when block capture is not safe. Still cuts launches substantially. |
| `none` | Eager | Baseline / debugging. |

`local` derives `layer` by default, so **set `block` explicitly** — a config that
merely says `cuda_graph_impl=local` is leaving the biggest win on the table.

`CudaGraphModule` (`attn`, `mlp`, `moe`, `moe_router`, …) is a separate,
training-oriented axis for capturing sub-layer regions. Not used for inference.

<details>
<summary>Deprecated API you will see in older commits and docs</summary>

Commits before the refactor use a `cuda_graph_scope` *list* holding
`CudaGraphScope.full_iteration_inference`. The migration guide lives in
[enums.py](megatron/core/transformer/enums.py):

- `full_iteration` → `cuda_graph_impl="full_iteration"` (training)
- `full_iteration_inference` → `inference_cuda_graph_scope=block`
- everything else → the equivalent `CudaGraphModule` member

`CudaGraphScope` is retained only for checkpoint deserialization. Do not write
new code against it.
</details>

## Who owns the graph

Two hooks decide, and they must agree — if both a block and its layers think they
own the graph you get nested capture.

**`create_mcore_cudagraph_manager(config)`** attaches the manager. Each module
checks whether the scope selects it:

```python
# megatron/core/models/hybrid/hybrid_model.py
def create_mcore_cudagraph_manager(self, config):
    if config.inference_cuda_graph_scope == InferenceCudaGraphScope.block:
        from megatron.core.transformer.cuda_graphs import CudaGraphManager
        self.cudagraph_manager = CudaGraphManager(config)
```

**`_should_call_local_cudagraph(*args, **kwargs)`** decides per call whether to
replay or run eager. It checks the scope, that a real inference context is
present, that `attention_mask is None`, and finally
`inference_context.using_cuda_graph_this_step()`.

Ownership by model family:

| Model | Owner under `block` scope |
|---|---|
| Hybrid | `HybridModel` — the *model*, so embedding + stack + output layer are one capture |
| GPT | `model.decoder`, the `TransformerBlock` |
| Mamba | The `MambaStack` |

### Widening the scope is itself an optimization

Commit `35f76df3` moved hybrid graph ownership up from `HybridStack` to
`HybridModel`, making `HybridStack` a plain `MegatronModule` and adding
`GraphableMegatronModule` to `HybridModel`. The embedding lookup and the final
projection are individually small kernels, but their launch overhead is on the
critical path of every decode step. Folding them into the existing capture cost
nothing and removed those launches.

**Generalizable:** when you add a new model class, ask what is still outside the
graph. Small kernels bracketing a graphed region are pure launch overhead.

### What a wide capture costs you

Widening is the right default, but it is a trade, not a free win, and the things it
forecloses are not obvious until you try them. On Qwen3-30B under full-iteration
inference capture, three separate levers turned out to be unreachable *because* the
capture was wide:

| Lever | Outcome under wide capture |
|---|---|
| flashinfer sampling backend | Server crash: `RuntimeError: Generator not registered with the capturing graph`. The sampler cannot be inside the capture — see the RNG hazard below. |
| Async scheduling (`async-sched-mode=serial`) | Guarded off for EP (`Async scheduling does not support expert parallelism`); opening the guard ran at batch, but deadlocked on the first single-request decode step. The guard encodes a real limitation. |
| Comm/compute overlap on separate streams | `CUDA_DEVICE_MAX_CONNECTIONS=8` measured flat. Overlap is bounded by the captured graph's structure and data dependencies, not by hardware queue count. Chunked collective pipelining needs concurrent streams under capture and hits the same wall. |

The async-scheduling result is worth internalizing beyond its own flag: async
overlap cannot hide the post-sampling host chain, because that chain is
**data-dependent on the current step's sampled tokens**. Measured end-to-end gain
from opening the guard was +0.85% — consistent with a genuine serial dependency
rather than a scheduling artifact.

**Generalizable:** before proposing overlap, streams, or asynchrony as the fix for
GPU idle, check whether the idle is a *data dependency*. If step N+1's input needs
step N's sampled token, no scheduler will overlap them, and the only fix is making
that host chain cheaper.

## Bucket coverage

A graph replays at a fixed shape, so every step is padded up to the smallest
captured bucket that fits. Two failure modes:

- **No bucket fits** → silent fallback to eager. A run with graphs "enabled" can
  still be launch-bound. This is the single most common reason a CUDA-graph
  change shows no improvement.
- **Nearest bucket is far too large** → wasted compute on padding.

All the logic is in
[batch_dimensions_utils.py](megatron/core/inference/batch_dimensions_utils.py).
The bucket unit is `InferenceBatchDimensions(token_count, prefill_req_count,
decode_req_count)`.

### Sizing distributions

`EXPONENTIAL` (default) halves down from the ceiling. This bounds the graph count
at roughly `log2(max_tokens)` and the relative padding at about 2x:

```python
sizes = set()
val = cuda_graph_max_tokens
for _ in range(num_cuda_graphs):
    rounded = max(rounder, (val // rounder) * rounder)
    rounded = math.ceil(rounded / tp_size) * tp_size
    sizes.add(rounded)
    val //= 2
    if val < 1:
        break
```

`LINEAR` gives the dense small-batch ladder introduced by `fde3b90a`, which
mirrors vLLM's:

```python
sizes = (
    [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, cuda_graph_max_tokens + 1, 16))
)
```

The reasoning: decode batches are usually small, so spend granularity at the low
end where relative padding hurts, and be coarse above 256 to bound the graph
count. Note the hygiene steps that follow in both paths — TP-align each entry,
clamp to the ceiling, and force the endpoints to be present.

`num_cuda_graphs=-1` auto-sizes. Under `EXPONENTIAL` it derives the count from
`log2(cuda_graph_max_tokens)` plus headroom, floored at 4.

### Two ceilings

Decode and prefill graphs are sized separately:

- **Decode** is always capped at `max_requests * (num_speculative_tokens + 1)` —
  one token per request, or 1 + speculative.
- **Prefill and mixed** are capped at `cuda_graph_max_tokens`, default **512**
  (commit `740c16e6`). Set `cuda_graph_all_prefills` to extend to `max_tokens`,
  at the cost of many large graphs.

At runtime `match_graph_config` picks `min()` over the applicable buckets and
returns `None` to signal eager.

### Cost of more graphs

Each bucket is a separate capture: its own activation memory in the graph mempool
plus capture time. `create_cuda_graphs` in
[dynamic_engine.py](megatron/core/inference/engines/dynamic_engine.py) runs a
full forward per bucket and logs elapsed time and memory deltas, so capture cost
is directly observable. `cuda_graph_max_tokens=512` exists to keep this bounded
by default.

## Making code graph-safe

### What breaks capture or replay

1. **Dynamic shapes** — handled by bucketing plus padding.
2. **Reallocated buffers.** A graph records raw pointers. If a buffer is freed and
   reallocated between capture and replay, replay writes into freed memory. Any
   grow-only buffer must be pre-sized to the worst case *before* capture:

```python
# megatron/core/inference/engines/dynamic_engine.py
# A forward larger than the capture-time size would reallocate (and free) the buffer
# whose address a captured graph still writes to on replay, corrupting whatever later
# reuses that freed block.
if getattr(model_config, "sequence_parallel", False):
    max_ag_numel = self.context.max_tokens * model_config.hidden_size
    get_global_memory_buffer().get_tensor((max_ag_numel,), model_config.params_dtype, "mpu")
```

3. **Host syncs and data-dependent Python** inside the captured region.
4. **Python-level assignment inside `forward`.** Under `block` scope it executes
   only during capture, so replays never see it. Use `copy_()` into a
   preallocated buffer instead — this is exactly why
   `mtp_decoder_hidden_states` exists.
5. **State baked in by value at capture.** The subtlest failure, because it
   produces no error. FlashInfer's sampling kernels bake the philox RNG state into
   the graph as a by-value constant, so a captured sampler replays identical
   "random" numbers every step. Its kernel choice is also data-dependent. For both
   reasons FlashInfer sampling is deliberately left eager — see
   [host-path.md](host-path.md). Before widening a graph over anything holding RNG
   or counter state, check what gets frozen.

### The toolkit

**One contiguous buffer, one memcpy.**
[ContextGPUView](megatron/core/inference/contexts/gpu_view.py) is the only
interface GPU code uses to read context state. Every field is a `view(dtype)`
onto a slice of a single `uint8` buffer, mirroring a pinned CPU buffer with the
identical layout, so publishing a step's bookkeeping is one `cudaMemcpyAsync`
rather than one per field. The convention is worth internalizing:

```
context.foo            -> CPU (source of truth, used by bookkeeping)
context.gpu_view.foo   -> GPU (snapshot, used by forward pass)
```

**Cache views instead of reconstructing them.** Slicing and unsqueezing per step
constructs new `TensorImpl`s at 30-60us each:

```python
# megatron/core/inference/contexts/dynamic_context.py
# Instead of slicing and unsqueezing on every new inference step (constructing
# new TensorImpls at 30-60 us), we fix the underlying storage so views are
# reusable across steps.
self._input_position_views: Dict[int, Tuple[Tensor, Tensor]] = {}
```

**Point padding at reserved storage.** Pad tokens get `dummy_block_idx` so the
KV-append kernel writes to a valid throwaway block. Pad rows get routing index
`-1` so they activate no expert.

**Keep EP sync off the compute stream.** `adjust_batch_dims_for_expert_parallelism`
can do the cross-rank max over ZMQ on the CPU, avoiding both a per-step NCCL
all-reduce on the compute stream and the H2D/D2H pair around it.

## Expert parallelism: idle ranks still have to run

Under EP, every rank must issue the same collectives in lockstep, and all ranks
must select the *same* graph bucket. So a rank with no real requests still runs a
forward. Commit `60a25aa6` found that idle rank was building its dummy batch
through the heavyweight graph-capture warmup path, constructing real request
objects and allocating KV blocks every step.

The fix, `add_dummy_requests_for_expert_parallel_step`, pokes only the
preallocated tensors that `initialize_attention_state` and the forward actually
read:

```python
N = smallest_cuda_graph_dimensions.decode_req_count
dummy_block_idx = self.block_allocator.dummy_block_idx
self.total_request_count = N
self.active_token_count = N
self.num_prefill_requests = 0
self.request_query_lengths[0:N].fill_(1)
self.request_kv_length_offsets[0:N].fill_(0)
self.request_to_kv_block_ids[0:N, 0] = dummy_block_idx
self.token_to_block_idx[0:N] = dummy_block_idx
```

Hybrid models also need Mamba slots, so `MambaMetadata.batch_allocate_slots(n)`
grabs them in a batch without allocating.

**Generalizable:** for any lockstep collective, idle participants must still run,
but give them the cheapest valid inputs. Reserve the dummy block and slots once,
then only `fill_` preallocated tensors. Never let the idle path allocate or
construct objects.

The EP token-count sync all-reduces the max so every rank agrees on the bucket,
and returns `None` (eager everywhere) if any rank is non-decode. Commit `c817dad2`
cut this back to the minimum after it had accumulated dead complexity:

```python
(max_token_count, max_is_non_decode) = ep_zmq_communicator.sync_all_reduce_max(
    local_batch_dims.token_count, int(is_non_decode)
)
```

## Expect golden values to change

Changing bucketization changes which padded shape each real step maps onto, which
changes padding tokens and MoE routing masks, which shifts logits. Commit
`60a25aa6` rewrote a 1448-line golden values file for exactly this reason. That is
expected — regenerate, and sanity-check that the *generated text* is still
coherent rather than assuming a diff means a bug.

## Validation gates

Under `block` scope, fp8 requires `--transformer-impl=inference_optimized` and
`--fp8-recipe=mxfp8`. `cuda_graph_impl=local` requires
`inference_dynamic_batching_num_cuda_graphs` to be positive or `-1`. Both are
asserted in [arguments.py](megatron/training/arguments.py).
