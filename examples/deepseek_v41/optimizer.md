# DSV4.1 optimizer prototype

The implementation targets the updated training branch at `166ec85a9d`. It leaves
its planned pipeline payload API, static packed scheduling, and MXFP8 model paths
in place; it does not introduce another training loop or MFSDP implementation.

Add these options to the [text-training example](README.md):

```text
--optimizer muon
--muon-momentum 0.95
--muon-nesterov
--muon-split-qkv-per-head
--muon-tp-mode duplicated
--muon-update-rms 0.18
--embedding-optimizer sinkhorn
--output-layer-optimizer sinkhorn
--engram-optimizer sinkhorn
--engram-lr-multiplier 5
--apply-wd-to-norm
--weight-decay 0.1
--adam-beta1 0.9
--adam-beta2 0.95
--adam-eps 1e-20
--sinkhorn-balance-mode global
```

The table selectors affect weights only: Engram projections still use Muon, and
its `q_weight`/`k_weight` normalization vectors use AdamW with decay. Biases and
scaling factors remain undecayed. The 5× LR multiplier applies to Engram tables,
including warmup and the LR floor; the Sinkhorn correction of 0.18 is applied
once to the direction, not again by the scheduler. This table-only 5× scope
follows the original Engram experiment, rather than scaling the entire module.

Query heads include both rotary and non-rotary features in one transform. The
shared Key projection is one matrix. Logical n-gram/hash-head tables are balanced
independently using their global row ranges, without unfusing parameter storage.
The per-head reconstruction helpers are adapted from PR #6326 at
`02237350dd9d743129b6a4769273546a36be2f0b`; generic MLA component annotations are
not imported, so this prototype does not change their default optimization.

Sinkhorn defaults are momentum 0.95, Nesterov enabled, 11 alternating axis steps,
row threshold 1e-3, epsilon 1e-20, correction 0.18, and zero decay. Global mode
reduces the initial row-norm sum and five feature-statistic vectors per logical
table. One parameter-activity reduction coordinates ranks without gradients.
There is no table/momentum all-gather. `--sinkhorn-balance-mode local` deliberately
uses each row partition's own threshold/column statistics instead, without those
reductions. This approximation depends on the partition and is recorded in the
optimizer checkpoint; changing modes on resume requires an explicit state reset.

## Prototype limits

- Use replicated optimizer state for the global reference mode. Do **not** enable
  `--use-distributed-optimizer` or MFSDP for that mode: LayerWise ownership can
  leave different EP peers without the parameter needed for a collective.
- Local balancing can use the legacy LayerWise whole-parameter ownership path,
  but not byte-sharded/padded DistOpt buffers. This path requires distributed
  qualification before production use; global Sinkhorn state sharding awaits
  the matrix-aware backend support planned for MFSDP.
- Feature-sharded or GTP-rematerialized Sinkhorn tables are rejected. Complete-row
  EP/TP table sharding is supported; actual V4.1 model TP restrictions still apply.
- Packed 3D expert matrices are not implemented by this prototype. Use per-expert
  2D weights; exact-RMS setup rejects unsupported packed expert tensors.
- FP32 master/state updates and BF16 model parameters are supported. Offload,
  optimizer graph capture, advanced batching/overlap, and vision setup are later
  additions. Model-layer capture is a separate feature.
- Newton–Schulz remains configurable, with the existing quintic/5-step defaults.
  The prototype does not assume DeepSeek-V4's coefficient sequence.

Emerging-Optimizers is pinned to the commit introducing Sinkhorn,
`36f70336da89dc80481c07dfb0af2c7333b9e5b3` (`0.5.0a0`). Its legacy SOAP import
is handled separately from the older package layout. Existing optimizer routing
and scaling remain unchanged when the new options are absent.

## Focused validation

Run inside the dev CI container:

```bash
uv run python -m torch.distributed.run --nproc-per-node 1 -m pytest -q \
  tests/unit_tests/optimizer/test_sinkhorn_optimizer.py
```

The suite checks multi-step updates, one momentum state, independent tables,
uneven EP shards with DP replicas (four CPU/Gloo workers), missing contributions,
TP/GTP per-head reconstruction, FP32/BF16 master metadata, LR/decay scheduling,
and same-layout resume. This is prototype validation, not a multi-node training
or performance qualification.
