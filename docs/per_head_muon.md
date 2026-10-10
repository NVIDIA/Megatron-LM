# Per-head Muon projection semantics

Enable `--muon-split-qkv-per-head` with the default QKV splitting enabled
(`muon_split_qkv=True`; do not pass `--muon-no-split-qkv`):

```bash
--optimizer muon --muon-split-qkv-per-head --use-distributed-optimizer
```

The flag defaults to false and leaves the existing Muon routing unchanged when
absent. Layer-wise data-parallel optimizer ownership is supported. The separate
`--muon-tp-mode layer_sharded` NS backend does not implement semantic splitting
and is rejected with this flag. Supported TP modes are `blockwise`, `duplicated`,
`distributed`, and `auto`; head fragments are reconstructed before local NS.

For Muon, the training CLI translates `--use-distributed-optimizer` into
`use_layer_wise_distributed_optimizer=True` and clears `use_distributed_optimizer`.
The optimizer factory assigns whole matrix parameters to DP ranks through
`LayerWiseDistributedOptimizer`; BF16 master copies retain their two-dimensional
shape. With the buffer-layout path, ordinary `DistributedOptimizer` instances
manage the separate scalar/embedding groups. Wrapping semantic Muon matrices in
the standard optimizer's flattened DP shards is unsupported and raises an error.

Per-head Muon with Kitchen is rejected by the optimizer factory. Kitchen uses
stride-1 FC1 storage and its backend is not publicly available for validation;
it must supply a verified layout contract before this combination is supported.
Kitchen parameters do not receive the stride-2 SwiGLU layout. The default Muon
path remains available when per-head mode is disabled.

This enables the following routing for module-owned fused projections:

| Projection | Muon matrices | AdamW rows |
| --- | --- | --- |
| Standard attention | Each Q, K and V head | Output gate, if present |
| GDN1 in_proj | Each Q, K and V head | z, beta, alpha |
| GDN2 in_proj | Each Q, K and V head | z, f, b, w |
| SwiGLU fc1 | Separate gate and up matrices | None |
| MLA up projections | Each head's non-RoPE Q/K, RoPE Q and V slices | None |
| MLA kv_down | Separate KV latent and shared RoPE-K matrices | None |
| Fused MLA qkv_down | Separate Q latent, KV latent and RoPE-K matrices | None |

SwiGLU's gate is a full feature projection; it is split from up but still uses Muon.
GDN and attention output gates instead use elementwise AdamW. Scalar model
parameters and embeddings retain their existing scalar-optimizer routing.

`MuonProjectionLayout` records physical row order. GDN uses the variant's actual
TP-local split table. SwiGLU accounts for the physical `[gate_local, up_local]`
layout. Fused MLA down projections reorder TP's physical
`[Q_rank, KV_rank]` blocks into global Q/KV order before NS and invert this
reordering before returning each shard. Complete local heads require no gathering.
Fragmented heads are restored
across TP/GTP before NS; padding is excluded from both NS and AdamW. Layout
mismatches raise an error instead of silently applying whole-matrix Muon.

The model keeps its fused weights, forward GEMMs and model checkpoint names.
Equal-sized matrices use batched NS, with independent Gram matrices and scaling.
Small unequal MLA down matrices can share a zero-padded batch when padding
preserves the NS orientation and the scale of each original matrix. Padding
is capped at twice the real row count; other shapes use separate calls.
Contiguous control rows share one AdamW update. AdamW uses the raw gradient,
`adam_beta1`, `adam_beta2`, `adam_eps`, and the parameter group's scheduled LR
and decoupled weight decay. Mixed routing currently supports plain Muon, not
AdaptiveMuon or coupled L2 decay. This routing does not change Megatron's global
gradient clipping or the scalar optimizer selected for other parameters.

## Optimizer state and resume

The mixed optimizer stores Adam first/second moments as parameter-shaped tensors,
with unused Muon rows zeroed, so Megatron can shard these tensors like the fused
weight. This costs extra optimizer memory compared with packed gate-only states.
New optimizer checkpoints carry `muon_semantic_version=3` and preserve each
parameter's update counter. Parameters with no gradient do not advance AdamW
bias correction. Mcore's distributed checkpoint format requires equal counters
across parameters; saving divergent counters fails explicitly instead of
silently restoring incorrect bias correction. Ordinary optimizer state dicts
preserve divergent counters. Version 2 used a group counter and cannot safely
recover per-parameter update counts; reset optimizer state when upgrading.
State saved by the earlier all-Muon implementation cannot be reused as
AdamW state: load model weights with `--no-load-optim`, or start a fresh experiment.
A changed routing rule also means old loss/throughput results do not evaluate
this implementation.

## Validation

CPU numerical and four-process TP/GTP tests can run without CUDA fixtures:

```bash
python -m unittest tests.unit_tests.test_muon_semantics -v
python -m unittest tests.unit_tests.test_muon_semantic_distributed -v
python -m unittest tests.unit_tests.determinism.kernels.test_per_head_muon -v
```

The tests compare multiple updates against separate Muon and PyTorch AdamW
optimizers, check that control rows never enter NS, verify resume and checkpoint
sharding, and exercise fragmented heads, GTP padding, replicated MLA down
projections, and SwiGLU's interleaved TP layout in all three explicit TP modes.

For GPU integration tests:

```bash
MUON_TEST_DEVICE=cuda python -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m pytest tests/unit_tests/test_muon_per_head.py tests/unit_tests/test_muon_semantics.py \
  tests/unit_tests/test_muon_layerwise_integration.py
```

## Validation scope and limitations

Numerical reference tests cover the fused layouts, per-parameter AdamW counters,
ordinary state-dict resume and sharded-state conversion. Four-process CPU/Gloo
regressions cover fragmented heads, TP/GTP padding, dense SwiGLU and fused MLA
down projections. Replay tests compare parameters and moment tensors byte for
byte, using side-stream contention when CUDA is available. GPU module tests
instantiate standard/output-gated attention, GDN1/GDN2, MLA and dense SwiGLU;
they skip cleanly without CUDA.

Layer-wise integration tests exercise the documented CLI conversion and the real
DDP/optimizer factory with both parameter-layout and legacy ping-pong ownership.
They supply controlled, already-reduced gradients, compare three updates with an
unsharded reference, check BF16 masters and moment shapes, and run DP parameter
synchronization. They isolate optimizer integration rather than full training.

Native distributed on-disk checkpoint save/load/resume and full GPU training
still require validation on the upstream integration. Grouped MoE expert FC1
modules do not currently attach this dense-MLP layout. Dense SwiGLU preserves
`blockwise` semantics (local gate/up matrices); `duplicated` and `distributed`
use each global gate/up matrix. Per-head layouts reconstruct complete heads in
all modes. No topology-changing resume or universal loss/speed gain is claimed.

## Credits and design reference

Developed in collaboration with [@sallyjunjun](https://github.com/sallyjunjun)
and [@JT-Ushio](https://github.com/JT-Ushio).
The layer-owned projection splitting design references
[InternLM/xtuner#2001](https://github.com/InternLM/xtuner/pull/2001), specifically
its MLA MuonSplit component boundaries and per-component scaling. Its separate
AdamW-only gradient-clipping change is not part of this implementation.
