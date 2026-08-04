# Folding the CUDA-graph padding sentinel into the router's selection kernel

Status: **implemented, hangs in warmup, reverted.** The measured prize is real
(+1.02%, see QWEN-046) and the design below is sound as far as it was tested;
what is missing is the cause of the hang. Re-derive nothing — start from the
diagnostic ladder at the bottom.

## Why

`mask_routing_padding` writes `-1` into every topk slot of the CUDA-graph
padding rows so those tokens route to no expert. It is one launch per layer per
step (48/step here) on a tensor of 256x8 int64, so it is pure launch overhead.
Ablating it entirely measured **+1.02%** end to end (30,974 against 30,662
tok/s, replicates within 18 tok/s) — that is the ceiling for fusing it.

The router's `_softmax_topk_kernel` is one CTA per token and already stores that
token's index row, so the sentinel costs a scalar compare and a select there.
Zero extra launches, and the separate mask launch disappears.

## What was built

Three edits, all reverted in commit-time order:

1. `megatron/core/inference/moe/router_topk.py`
   - Module state `_graph_pad_count` plus `publish_graph_padding(tensor)` /
     `graph_padding_count()`. The tensor is the context's fixed-address
     `int32[1]` real-token count, so reading it inside the kernel is
     replay-safe: only the value behind the pointer changes.
   - `_softmax_topk_kernel` gains `real_cnt_ptr` and `MASK_PADDING: tl.constexpr`:

         if MASK_PADDING:
             is_padding = token_id >= tl.load(real_cnt_ptr).to(tl.int32)
         else:
             is_padding = False
         ...
         tl.store(indices_ptr + token_id * TOPK + k,
                  tl.where(is_padding, -1, best_idx))

   - `fused_softmax_topk(logits, topk, mask_padding=False)` passes the pointer,
     and on the fused path tags the output: `indices._mcore_padding_masked = True`.
   - `can_fuse_route_mask()` = gate `MCORE_FUSED_ROUTE_MASK` and a published tensor.

2. `megatron/core/transformer/moe/router.py` — pass
   `mask_padding=can_fuse_route_mask()` on the fused selection path.

3. `megatron/core/transformer/moe/token_dispatcher_inference.py`
   - `self._rows_unsharded = get_pg_size(pg_collection.tp) == 1`, because the
     kernel compares against *local* row ids. With an SP/TP shard the comparison
     needs this rank's row offset, which the router does not have; requiring
     group size 1 (rather than `sp_rank == 0`) keeps every rank on the same
     branch, which matters because the NVLS path barriers across ranks.
   - Publish the count in `dispatch_preprocess`, and skip the standalone mask
     when `getattr(self.routing_map, "_mcore_padding_masked", False)`.

Two properties worth keeping in any retry:

- **Skip on evidence, not on the gate.** The dispatcher keys off the tag the
  router leaves on the tensor it actually masked. The tag is set in Python, so
  the decision is what gets baked into the graph at capture. The one state that
  must never happen is *neither* side masking; ordering makes the first call
  (layer 0, before any publish) fall to the dispatcher, which is correct.
- **The tag survives the path.** `route()` -> `preprocess()` ->
  `dispatch_preprocess()` -> `self.routing_map = routing_map` passes the same
  object with no reshape, so the attribute is still there.

## The failure

With the gate *off*, the baseline arm hung in warmup iteration 1: server up and
listening, benchmark client past its banner, no progress for 28 min (arms
normally finish in 3.5). So the hang is in the forward, is reachable with
`MCORE_FUSED_ROUTE_MASK` unset, and therefore comes from the always-on parts:
the new import, `_rows_unsharded`, the unconditional publish, the `getattr`
skip, or the kernel's changed signature (`real_cnt_ptr=None`, and a
`tl.where` on a Python `False`) — not from the masking logic itself.

Reverting all three files and rerunning the same arm returned exactly the
baseline (30,661.9 against 30,662.5 and 30,661.0 before), so the hang is
attributable to these edits and nothing else on the node.

The leading suspect is the kernel change, because it is the only always-on item
that alters generated code: if `tl.where(False, -1, best_idx)` does not fold to
`best_idx`, every row gets `-1`, no token routes anywhere, and the NVLS
all-gather-v barrier can then deadlock on rank-divergent counts. That is a
guess; it was not measured.

## Diagnostic ladder for the retry

Cheapest first, each answering one question:

1. Run `dev/moe_fused/harness_routemask.py` (already written; needs an absolute
   path, see below). Bit-exactness against the standalone mask for real counts
   256/200/137/1/0 isolates kernel correctness from integration.
2. Print the compiled kernel for `MASK_PADDING=False` and confirm the `-1` store
   folded away, i.e. that the baseline kernel is unchanged. This directly tests
   the leading suspect.
3. Gate the kernel signature change itself: keep two `@triton.jit` kernels, the
   original untouched and a masking variant, so the gate-off path is provably
   byte-identical to today's. If the hang survives that, it is the publish or
   the skip, not the kernel.
4. Only then re-run the e2e A/B.

## Harness gotcha

The session runner executes inbox scripts with cwd = the *session* directory,
not the workspace, so `python dev/moe_fused/harness_routemask.py` fails with
`No such file or directory`. Use the absolute workspace path in the inbox
script. (The e2e arms get away with relative paths because `run_e2e_cfg.sh`
cds into the repo itself.)
