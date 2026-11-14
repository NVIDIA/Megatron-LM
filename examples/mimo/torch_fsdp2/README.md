# Experimental native Torch FSDP2 for separate-rank MIMO

This prototype wraps the complete local language stage in native PyTorch
`fully_shard`, including embeddings and projectors used before the language
model's forward. Vision continues to use its existing DDP optimizer path.
It requires a PyTorch build exposing FSDP2 per-parameter mesh placement
(the experiment used PyTorch 2.13).

## Ownership and execution

Expert units are wrapped first on `pg_collection.expt_dp`, complete HybridStack
execution regions next on `pg_collection.dp_cp`, and remaining parameters last
on the root stage. Residual-shortcut children must stay in their execution
region because their `forward` may be called directly. Output-cloning hooks
place native FSDP's pre-backward gather before residual replay accesses weights;
MIMO communication metadata on those outputs is preserved.

Original BF16 and FP32 parameters remain in distinct native parameter groups.
A second same-membership process group permits native dtype grouping without
splitting an atomic replay region. This relies on private PyTorch FSDP APIs.
The extra groups and output copies have real memory and execution costs.

The caller must load pretrained parameters on CUDA before wrapping. Supply the
wrapped stage to the optimizer through `get_mimo_optimizer(...,
module_overrides={MIMO_LANGUAGE_MODULE_KEY: wrapped_stage})`.

## Precision and optimizer semantics

`DistributedDataParallelConfig.grad_reduce_in_fp32` controls native
`MixedPrecisionPolicy.reduce_dtype`. FP32 reduction does **not** make stored
gradients FP32: PyTorch 2.13 casts each reduced shard to its original parameter
dtype before accumulation. BF16 parameters therefore accumulate BF16 sharded
gradients. The optimizer converts those gradients to FP32 for local FP32-master
TE Adam updates, then copies updated masters back to model shards.

The numerical reference tests deliberately reproduce that per-microbatch
reduction and stored-gradient precision. They do not establish equivalence to
an implementation that accumulates gradients in FP32 throughout.

`no_sync()` reduces each microbatch so the adapter retains sharded rather than
full dense gradients. Norm calculation and clipping sum each local shard once
over the module's supplied statistics group, excluding unrelated vision ranks.

## Supported scope

- Separate language and encoder ranks; TP=PP=1.
- BF16 and mixed original BF16/FP32 parameters, resident Adam with FP32 masters.
- Native expert/dense meshes and repeated HybridStack/MTP execution regions.
- No checkpoint save/resume, CPU offload, FP8 or precision-aware optimizer.

This is a draft experiment. A prior integration snapshot completed 100 SFT
updates, but gradient norms differed materially from the MFSDP reference, and
one data sample differed following a separate loader repair. Close loss curves
are not evidence of gradient/clipping parity. This upstream port needs its own
validation; matched steady-state peak-memory measurement is outstanding.

## Companion integration

`bridge.patch` and `nemo_rl.patch` capture the experiment's downstream wiring.
They are review aids against the experiment's snapshots and may need rebasing
in their respective repositories. They enable
`MCORE_EXPERIMENTAL_TORCH_MIMO_FSDP2=1`, select this wrapper and optimizer, and
avoid moving owned DTensor storage through `Module.to()`. The Core PR alone
does not enable a Bridge/NeMo recipe. Dataset changes are excluded.

## Focused validation

Use the repository's GPU unit-test environment:

```bash
uv run python -m torch.distributed.run --nproc-per-node=2 -m pytest -q \
  tests/unit_tests/distributed/test_torch_mimo_fsdp.py \
  tests/unit_tests/determinism/kernels/test_torch_mimo_fsdp.py \
  tests/unit_tests/models/mimo/test_mimo_overlap_lifecycle.py
```

The numerical matrix checks three optimizer updates with four microbatches,
shared weights, direct child calls, checkpoint replay, empty local shards,
expert ownership and mixed parameter dtypes. The determinism test replays
native gradients, clipping, FP32 masters and Adam moments with stream contention.

`smoke_hybrid.py` additionally exercises actual HybridStack/MoE modules on two
ranks. Set `MIMO_TEST_WIDE_REPLAY=1`, `MIMO_TEST_MTP=1`, or `MIMO_TEST_MIMO=1` to
exercise the corresponding execution paths.
