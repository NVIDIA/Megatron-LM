# te-graph-runtime (vendored)

Vendored from [buptzyb/te-graph-runtime](https://github.com/buptzyb/te-graph-runtime).

## Acknowledgements

- [@buptzyb](https://github.com/buptzyb) (Robin Zhang) — original `te-graph-runtime` implementation, `capture_time_hooks` support
- [@shjwudp](https://github.com/shjwudp) (Jianbin Chang) — MFSDP v2 CUDA graph integration, local modifications

TE-compatible CUDA graph callable runtime with `capture_time_hooks` support
— hooks that run outside CUDA graph capture (for FSDP unshard / reshard)
and are not replayed.

## Why vendored

The MFSDP v2 CUDA graph runner depends on `capture_time_hooks` (from TE PR #2831)
and the CUDA graph parameter-gradient lifetime fix (from TE PR #2937), which
are not yet available in upstream `torch.cuda.make_graphed_callables`.

## API

```python
from .te_graph_runtime import make_graphed_callables
```

## Local modifications

### Non-tensor `sample_kwargs` support
Frameworks pass mixed tensor/non-tensor kwargs (e.g. `attention_mask=None`,
`img_shapes=[[1,64,64]]`). `tree_flatten` passes `None` entries into
`static_input_surface`, which crashed `.requires_grad` and `.data_ptr()`.

- Added `is not None` / `isinstance(arg, torch.Tensor)` guards at 6 access points
  in warmup, backward capture, and `Graphed.forward`.
- `functionalized` reconstructs capture-time arg order from both `user_kwargs`
  (by name) and `user_args` (by position) to handle positional args during replay.

### Memory cleanup between warmup and capture
Two rounds of `gc.collect()` + `torch.cuda.empty_cache()` between warmup and
forward capture release cached blocks from the CUDA caching allocator, giving
the graph pool a clean, unfragmented state.

### Stream management
- `capture_stream` parameter accepted by `make_graphed_callables` and
  `_make_graphed_callables` for shared-pool serialization.
- **Warmup and capture share the same CUDA stream for activation reuse** —
  intermediate tensors allocated during warmup stay at the same addresses,
  so capture reuses them instead of freeing and reallocating.  Saves
  significant GPU memory vs. a throwaway stream.
- A throwaway warmup stream is used only as a workaround for `torch.compile`
  compatibility (see `cuda_graph_design.md` §9).  The ideal state removes
  this workaround once the underlying compile guard mismatch is fixed.

## Prefer pip install

If the fixes above are upstreamed, prefer `pip install te-graph-runtime` over
this vendored copy. See `cuda_graph_runner.py` for the fallback import logic.
