# MoE EP chunk overlap

`--moe-ep-chunk-overlap` replaces the **whole** routed-expert region of a MoE layer —
dispatch → FC1 / activation / FC2 → combine — with a token-chunked two-stream schedule, so each
chunk's expert-parallel all-to-all overlaps the previous chunk's expert GEMMs.

The GEMMs are not reimplemented: the schedule builds TE's op fuser
(`te_ops.Sequential(GroupedLinear, activation, GroupedLinear)`) and refuses to start unless TE
registers the fused `GroupedMLP_CuTeGEMM*` kernel, so it cannot silently fall back to three
unfused ops. The routing probability is folded into the activation rather than applied separately.

Default is off. This is training-only; under `torch.inference_mode()` the layer falls back to the
trunk dispatcher.

## Enabling it

| flag | meaning |
| --- | --- |
| `--moe-ep-chunk-overlap` | turn the path on |
| `--moe-ep-chunk-overlap-num-chunks` | source-token chunks per microbatch (default 4) |
| `--moe-flex-dispatcher-backend` | must be `hybridep` or `ncclep` (see below) |
| `--moe-flex-dispatcher-num-sms` | the communication-SM budget, shared with the flex dispatcher |

## Requirements

Each of these raises with a message naming the fix; none of them fails silently.

* **A flex backend that can stage several dispatches at once.** One dispatch manager is built per
  chunk, because chunk *c*'s combine runs while *c+1*'s dispatch is still in flight. `deepep` and
  `deepepv2` stage through process-wide buffers and are refused with `NotImplementedError`; use
  `hybridep` or `ncclep`.
* **`NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`, exported *before* TransformerEngine is imported.** Without
  it TE does not register the fused grouped-MLP kernel and `check_fused_grouped_mlp` raises.
* **An MXFP8 quantization autocast around the layer's forward.** This path implements MXFP8 block
  scaling only. No live autocast, or a non-MXFP8 recipe, raises rather than quietly discarding the
  configured one. If `first_last_layers_bf16` leaves a layer in bf16, that layer will raise.
* **An activation the routing probability can fold into**: `ScaledSwiGLU`, `ScaledClampedQGeGLU` or
  `ScaledSReLU` — i.e. gated SiLU, gated quick-GELU, or ungated squared ReLU.
* **`pg_collection.tp_ep`** must be present.

## What it refuses, and why

Refused at config time (`TransformerConfig.__post_init__`):

| configuration | reason |
| --- | --- |
| `recompute_granularity='full'` | re-running the chunked forward inside the backward reorders the wgrad FIFO silently |
| `'moe' in recompute_modules` | same; `'moe_act'` recomputes inside TE's own backward and **is** supported |
| `moe_paged_stash` | unsupported |
| `expert_tensor_parallel_size > 1` | the runtime's chunk capacity is derived from an unscaled `moe_router_topk` while the dispatch manager scales it by ETP, so the dispatch would be sized short — and the pinned permuted-token count would stop `check_dispatch_capacity` seeing it. Note ETP defaults to `tensor_model_parallel_size` when unset |
| `moe_expert_rank_capacity_factor` **and** `moe_pad_expert_input_to_capacity` | two capacity mechanisms over one dispatch |

Refused at construction or at runtime:

* expert GTP weight remat with fewer than 2 local experts — TE's `num_groups == 1` shortcuts read
  past `sum(split_sizes)`, giving wrong or NaN expert wgrads with a bit-identical forward and dgrad;
* `padding_mask`;
* `MoELayer.backward_dw()` — the schedule drains the expert weight gradients itself, in FIFO order.

Sequence packing with the `hybridep` backend is incompatible in practice, though nothing refuses
the combination by name: `TransformerConfig` requires `moe_hybridep_pad_uneven_dispatch_inputs=True`
for packed sequences, and that path's `setup_metadata` does an `all_reduce(...).item()` — a
device-to-host read CUDA-graph capture cannot serve.

## Running the tests

Two unit-test files cover this path. Neither needs more than one GPU.

**Seam tests — CPU only, no GPU required.** These pin the contract between the schedule and the
dispatch manager: which tensors cross streams, which node is differentiated to get each adjoint,
and which keyword surface the expert-compute step exposes.

```bash
python -m pytest tests/unit_tests/transformer/moe/test_moe_ep_overlap_backend_seam.py -q
```

Expected: `51 passed, 2 xfailed`.

**Schedule tests — one GPU.** These run the chunked schedule for real against a fake dispatch
manager, including the headline numerical gate
`test_chunked_gradients_match_a_single_shot_autograd_reference`, which compares the chunked
gradients against a single-shot autograd reference at 1, 2, 3 and 4 chunks.

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest \
    tests/unit_tests/transformer/moe/test_moe_ep_chunk_overlap.py -q -p no:randomly
```

Expected: `33 passed, 5 xfailed`.

### About the `xfail`s

They are **strict** and self-retiring, not skips. Each one asserts a guard or a fix that does not
exist yet and carries the evidence in its reason string — 5 on the schedule file (an eval-time
offload gate, and four parametrizations of an upstream activation-offload reload race), 2 on the
seam file (two absent checks in `end_wgrad_accumulation`). When the corresponding fix lands the
test starts passing, strict-xfail turns that into a failure, and the test asks to be deleted.
A *new* failure in either file is a real regression; an `xpass` is a fix landing.

### What the unit tests do not cover

They use a fake dispatch manager, so they exercise the schedule, not a real expert-parallel
collective. Covering the real thing needs ≥2 ranks with `hybridep` or `ncclep` available, MXFP8,
and `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`; the cheapest end-to-end check is to take a working MoE
training script and add `--moe-ep-chunk-overlap`, confirming the loss curve matches the run
without it.

No throughput comparison against the trunk dispatcher is published here. The schedule runs its
GEMMs over a fixed per-chunk capacity (padding included) rather than the actual routed row count,
and splits each microbatch into `num_chunks` smaller GEMMs, so the overlap has to pay for both
before it wins. Measure on your own shapes, sweeping `--moe-ep-chunk-overlap-num-chunks`, before
assuming a speedup.
