# The Host Path: Per-Step CPU Overhead and IPC

Once the GPU forward is graphed and fused, the bottleneck moves to the CPU. Every
engine step ends with bookkeeping, detokenization, serialization, and a ZMQ send,
all of which block the next step. This is where several of the largest wins came
from, and it is the part people forget to profile.

Source commits: `53a2b19a` (#2920), `602fad03` (#5918), `bcf4c8fb` (#5607),
`edd45620` (#5609), `650b7838` (#5791, co-authored).

## Where per-step host work lives

Always inspect these four places.

**The engine step tail** in
[dynamic_engine.py](megatron/core/inference/engines/dynamic_engine.py). The
critical section is bracketed by NVTX ranges so it is directly visible in a trace:
`bookkeeping` → `detokenization` → `coordinator_communication`. Anything added here
runs once per step.

**The controller**, `text_generation_controller.py`: routing-record bookkeeping,
log-prob detokenization, and building the step-result dict. CPU work here overlaps
already-enqueued GPU kernels, which makes it the right place to *put* unavoidable
work.

**Serialization** in
[inference_request.py](megatron/core/inference/inference_request.py):
`InferenceRequest.serialize`, `DynamicInferenceRequest.serialize`,
`DynamicInferenceRequestRecord.serialize` and `.merge()`. Runs once per finished
request per step.

**The coordinator** in
[data_parallel_inference_coordinator/](megatron/core/inference/data_parallel_inference_coordinator/):
per-request routing scoring in `coordinator.py`, pending-count updates in
`handlers.py`.

## Serialization: the two rules

### Never `dataclasses.asdict()` on anything holding tensors

`asdict()` recursively deepcopies every field. With CUDA tensors in the object
that is catastrophic. Use a shallow dict copy and handle fields explicitly:

```python
# megatron/core/inference/inference_request.py
# Dataclass to dict.
# do not use asdict(self) - it has very high CPU overheads
# and if there are tensors, it will try to deepcopy them
obj = self.__dict__.copy()  # shallow dict copy
obj["status"] = self.status.name if self.status else None
obj["sampling_params"] = self.sampling_params.serialize() if self.sampling_params else None
```

### Never `torch.save` for IPC

`torch.save` into a `BytesIO` pickles and writes the full tensor blob. For token
ids, a list is dramatically cheaper:

```python
def serialize_tensor(tensor: torch.Tensor) -> List:
    nvtx_range_push("serialize_tensor")
    # simply convert tensor into a list
    tensor = tensor.cpu().tolist()
    nvtx_range_pop("serialize_tensor")
    return tensor
```

Note the NVTX range on a function this small. That is deliberate — you cannot
attribute host cost you cannot see.

## Don't put large tensors on the wire by default

The engine was serializing and shipping the entire `prompt_tokens` tensor
engine → coordinator → API for every finished request. For long agentic or RL
prompts that dominates wire cost, and the client usually does not want the ids
echoed back.

Commit `602fad03` made it opt-in:

```python
# megatron/core/inference/sampling_params.py
# Echo prompt token ids back in the response. When False (default), the engine
# drops prompt_tokens before serializing the finished request, saving the ZMQ
# transmission cost for long prompts. Opt in when the client needs them.
return_prompt_tokens: bool = False
```

The API contract still needs `usage.prompt_tokens`, so the technique is to **keep
the scalar and drop the payload**: a `prompt_length: Optional[int]` field is always
populated during serialize even when the tensor is omitted. The drop is wire-only
— `serialize` nulls `self.prompt_tokens` around the `super().serialize()` call and
restores it, so the local object is unchanged.

Endpoints opt in where they must: `/v1/completions` always echoes
`prompt_token_ids`, so it sets `return_prompt_tokens=True` unconditionally.

**Generalizable:** for any per-request field on the IPC path, ask what the consumer
actually needs. Usually it is a count or a length, not the tensor.

## Move CPU work off the engine, don't just make it faster

The engine was detokenizing the full prompt plus generated sequence for every
finished request, inline in the step loop. Making detokenization faster would have
helped a little; moving it to a different process helped much more, because it then
overlaps the next engine step:

```python
# Detokenize all finished requests if not using
# the coordinator. Otherwise, the coordinator will
# overlap detokenization with the engine.
if not self.use_coordinator:
    nvtx_range_push("detokenization")
    ...
```

The coordinator gained a tokenizer and a `detokenize()` method; the engine ships
raw token ids. When a coordinator is present the engine does no detokenization at
all.

**Generalizable:** work that is not needed to compute the *next* step does not
belong in the step loop. Either defer it or move it to a process that runs
concurrently.

## Load-aware DP routing

With data-parallel engines, the coordinator decides which rank gets each request.
Round-robin ignores both prefix-cache affinity and current load, so it can send a
request to a busy rank while an idle rank holds exactly the KV blocks that request
needs.

`bcf4c8fb` replaced round-robin with `LOAD_BALANCED`, which is now the default
policy, and unified everything under one score:

```python
# megatron/core/inference/data_parallel_inference_coordinator/coordinator.py
if self.prefix_caching_coordinator_policy == PrefixCachingCoordinatorPolicy.LOAD_BALANCED:
    return self.get_least_loaded_data_parallel_rank()

# Without prefix caching (or when the request has no hashes to match on)
# fall back to load-balanced routing.
if not self.enable_prefix_caching or not request_hashes:
    return self.get_least_loaded_data_parallel_rank()

match, recency = self._match_vector(request_hashes)
alpha = self.prefix_caching_routing_alpha

# Vectorized score: alpha * match + (1-alpha) * free_capacity_fraction.
free_slots = np.maximum(0, self.max_requests - self._pending_counts).astype(np.float64)
scores = alpha * match + (1.0 - alpha) * (free_slots / self.max_requests)

# Tiebreak: highest score, then highest recency, then lowest rank index.
n_ranks = len(self._identities_list)
order = np.lexsort((np.arange(n_ranks), -recency, -scores))
```

`prefix_caching_routing_alpha` (default 0.5) is the tradeoff knob:

| alpha | Behavior | Risk |
|---|---|---|
| 0 | Pure load balance | Ignores cache; re-prefills work another rank already has |
| 0.5 | Balanced (default) | — |
| 1 | Pure cache affinity | Piles affine requests on one rank, starves the others |

Two implementation details worth copying. `match` is policy-dependent and
normalized to `[0, 1]` — binary for `first_prefix_block`, normalized prefix depth
for `longest_prefix` — so it is commensurable with the load term. And the whole
computation is vectorized numpy, because it runs per request on the coordinator's
hot path; a Python loop over ranks here would be its own bottleneck.

Pending counts are maintained incrementally in `handlers.py` (incremented on
dispatch, decremented on finish), and `_remove_engine` rebuilds them when an engine
disconnects mid-flight.

## Hold the dtype contract at kernel boundaries

FlashInfer's sampling kernels return sampled token ids as `int32`. The rest of the
pipeline — token buffers, `input_ids`, scatter into KV and embedding gather — is
`int64`. Every return in
[flashinfer_sampling.py](megatron/core/inference/sampling/flashinfer_sampling.py)
is cast with `.long()`.

A mismatch inserts implicit conversion kernels on the per-step path. Normalize
dtype where the external kernel enters rather than letting it propagate: the cast
is free once, the mismatch costs every step.

> Provenance note: Siddharth's `44334eda` ("Fix flashinfer sampling kernels to
> return int64 instead of int32") sits on the unmerged `fix_flashinfer_sampling`
> branch. The same casts reached `main` through #5791 (`650b7838`), which he
> co-authored.

### Corollary: not everything belongs in a graph

That same PR concluded FlashInfer sampling should run **eagerly**, and the reasons
generalize to any stateful or random kernel you are tempted to capture:

```
The sampler runs eagerly. Its kernel choice is data-dependent (it varies with
which filters the batch uses), so it cannot be captured in a CUDA graph; running
eagerly also lets the controller's seeded RNG generator advance its philox offset
normally between steps -- fresh randomness per step, reproducible from the seed.
(FlashInfer bakes the philox state into a graph as a by-value constant at capture,
so a captured sampler replays identical random numbers ...)
```

Two distinct hazards: **data-dependent kernel selection** cannot be captured at
all, and **state baked in by value at capture** produces a graph that replays
silently wrong — identical "random" numbers every step, with no error. Before
widening a graph over anything holding RNG or counter state, check what gets frozen.

Note also how the dispatch flags are read: from the pinned CPU sampling metadata,
so evaluating them costs no GPU sync. Same rule as everywhere else in this skill.

## Host-side rules, condensed

1. No `dataclasses.asdict()` on objects holding tensors.
2. No `torch.save` / pickle for IPC; use lists or ndarrays.
3. Drop large tensors from the wire by default; ship the scalar the contract needs.
4. Work not needed for the next step goes off the step loop, ideally to a
   concurrent process.
5. Route with load awareness, not blind round-robin; keep the scoring vectorized.
6. Normalize dtypes at external kernel boundaries.
7. NVTX-annotate anything you suspect, including small functions. Unmeasured host
   cost is invisible cost.
8. Any per-step recording must use preallocated static buffers and respect
   `using_cuda_graph_this_step()`.
