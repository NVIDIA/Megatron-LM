# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GTP_remat must follow the caller's ``pg_collection``, not the MPU globals.

Two TransformerBlocks, same degrees (TP=1, CP=1, GTP_remat=2 over world=4), same weights,
same input: one built from ``parallel_state`` groups, one from a custom collection whose
``gtp_remat`` group is the PERMUTED pairing ([0,1],[2,3] vs [0,2],[1,3]). The forward
all-gathers every peer's shard, so both must produce identical output and gradients.

The MPU globals stay initialized as the first topology throughout: a module that reads the
global group instead of the collection it was handed then gathers the wrong peer's shard --
a valid-but-wrong group, which is the silent failure this test catches.
"""

import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTPShardedParam
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _requires_multi_gpu,
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)

WORLD = 4
GTP_SIZE = 2
HIDDEN = 256
NUM_HEADS = 8
FFN_HIDDEN = 512
NUM_LAYERS = 2
SEQ = 16
BATCH = 1
dtype = torch.bfloat16

# The two ways to split a 4-rank world into gtp_remat pairs. Whichever one the MPU picks,
# the test uses the other. Every rank creates every group in this fixed order so the NCCL
# group tags agree across ranks -- a per-rank "create only the group I belong to" idiom
# assigns mismatched tags and hangs.
_PAIRINGS = {"adjacent": [[0, 1], [2, 3]], "strided": [[0, 2], [1, 3]]}

# Forward is exact: all-gather is pure data movement, so both blocks feed bit-identical
# operands to identical GEMMs. Gradients additionally carry the attention backward's
# nondeterminism, hence the looser BF16-scale tolerance.
FWD_TOL = dict(atol=1e-5, rtol=1e-5)
GRAD_TOL = dict(atol=2e-2, rtol=2e-2)


def _make_config():
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_attention_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        ffn_hidden_size=FFN_HIDDEN,
        add_bias_linear=False,
        params_dtype=dtype,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bias_dropout_fusion=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )


def _build_block(pg_collection):
    """Build a GTP-sharded TransformerBlock wired to ``pg_collection``."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.transformer_block import TransformerBlock

    block = TransformerBlock(
        _make_config(), get_gpt_layer_with_transformer_engine_spec(), pg_collection=pg_collection
    ).cuda()
    assert any(
        isinstance(p, GTPShardedParam) for p in block.parameters()
    ), "GTP is not active: the block has no GTPShardedParam"
    return block


def _pick_permuted_gtp_group(rank, mpu_ranks):
    """Create both candidate pairings on every rank; return this rank's group in the other one.

    Returns the group whose membership differs from ``mpu_ranks``, so any module that reads
    the global group instead of the supplied one gathers a different peer's shard.
    """
    my_groups = {}  # sorted pair -> this rank's group in that pairing
    for pairs in _PAIRINGS.values():
        for pair in pairs:
            group = dist.new_group(ranks=pair)
            if rank in pair:
                my_groups[tuple(sorted(pair))] = group

    permuted = [g for pair, g in my_groups.items() if list(pair) != mpu_ranks]
    assert len(permuted) == 1, (
        f"rank {rank}: want exactly one pairing differing from the MPU group {mpu_ranks}, "
        f"got {list(my_groups)}"
    )
    return permuted[0]


def _canonical_full_weights(block, gtp_group):
    """Gather every parameter to full (unsharded) form, then broadcast rank 0's copy world-wide.

    Returns a name -> tensor dict that is bit-identical on every rank, so both blocks can be
    loaded with the same global model no matter how the shards are distributed.
    """
    full_weights = {}
    for name, param in block.named_parameters():
        if isinstance(param, GTPShardedParam):
            shards = [torch.empty_like(param.data) for _ in range(gtp_group.size())]
            dist.all_gather(shards, param.data.contiguous(), group=gtp_group)
            full = torch.cat(shards, dim=0)
        else:
            full = param.data.clone()
        dist.broadcast(full, src=0)
        full_weights[name] = full
    return full_weights


def _load_full_weights(block, full_weights, gtp_rank):
    """Load the canonical weights, slicing GTP params by ``gtp_rank`` and priming main_grad."""
    for name, param in block.named_parameters():
        full = full_weights[name]
        if isinstance(param, GTPShardedParam):
            shard = param.shape[0]
            param.data.copy_(full[gtp_rank * shard : (gtp_rank + 1) * shard])
            # GTP writes the reduce-scattered wgrad here; it must exist before backward.
            param.main_grad = torch.zeros(param.shape, dtype=dtype, device='cuda')
        else:
            param.data.copy_(full)


def _full_grads(block, gtp_group):
    """Full (unsharded) gradients keyed by parameter name, for cross-topology comparison."""
    grads = {}
    for name, param in block.named_parameters():
        if isinstance(param, GTPShardedParam):
            shards = [torch.empty_like(param.main_grad) for _ in range(gtp_group.size())]
            dist.all_gather(shards, param.main_grad.contiguous(), group=gtp_group)
            grads[name] = torch.cat(shards, dim=0).float().cpu()
        elif param.grad is not None:
            grads[name] = param.grad.detach().float().cpu()
    return grads


def _fwd_bwd(block, x):
    """Run one forward/backward; return (output, input gradient) on cpu in fp32."""
    out = block(hidden_states=x, attention_mask=None)
    out.sum().backward()
    return out.detach().float().cpu(), x.grad.detach().float().cpu()


def _worker_custom_pgs_match_mpu(rank, world_size, port):
    from megatron.core import parallel_state as ps
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    # ---------------- Topology 1: groups from parallel_state ----------------
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=GTP_SIZE
    )
    model_parallel_cuda_manual_seed(42)

    mpu_pgs = ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=['tp', 'cp', 'pp', 'gtp_remat', 'expt_gtp_remat']
    )
    mpu_gtp_group = mpu_pgs.gtp_remat
    assert (
        mpu_gtp_group.size() == GTP_SIZE
    ), f"GTP_remat inactive: group size {mpu_gtp_group.size()}, want {GTP_SIZE}"

    block_mpu = _build_block(mpu_pgs)

    # One canonical global model, shared by both topologies.
    full_weights = _canonical_full_weights(block_mpu, mpu_gtp_group)
    _load_full_weights(block_mpu, full_weights, mpu_gtp_group.rank())

    torch.manual_seed(1234)
    x = torch.randn(SEQ, BATCH, HIDDEN, dtype=dtype, device='cuda')
    dist.broadcast(x, src=0)  # identical input on every rank

    out_mpu, grad_in_mpu = _fwd_bwd(block_mpu, x.clone().requires_grad_(True))
    grads_mpu = _full_grads(block_mpu, mpu_gtp_group)

    del block_mpu
    GTPShardedParam._chain_state = {}

    # ---------------- Topology 2: custom collection, permuted gtp ranks ----------------
    mpu_ranks = sorted(dist.get_process_group_ranks(mpu_gtp_group))
    custom_gtp_group = _pick_permuted_gtp_group(rank, mpu_ranks)

    # Only gtp_remat differs; tp/cp/pp are size-1 groups, identical in both topologies.
    custom_pgs = ProcessGroupCollection(
        tp=mpu_pgs.tp,
        cp=mpu_pgs.cp,
        pp=mpu_pgs.pp,
        gtp_remat=custom_gtp_group,
        expt_gtp_remat=mpu_pgs.expt_gtp_remat,
    )
    # Seed from the custom topology's gtp rank rather than the global one. The weights are
    # overwritten below, so this only has to be self-consistent -- it also exercises the
    # explicit-rank arguments of model_parallel_cuda_manual_seed.
    model_parallel_cuda_manual_seed(
        42, gtp_remat_rank=custom_gtp_group.rank(), egtp_remat_rank=0, force_reset_rng=True
    )

    block_custom = _build_block(custom_pgs)
    _load_full_weights(block_custom, full_weights, custom_gtp_group.rank())

    out_custom, grad_in_custom = _fwd_bwd(block_custom, x.clone().requires_grad_(True))
    grads_custom = _full_grads(block_custom, custom_gtp_group)

    ps.destroy_model_parallel()
    ps.initialize_model_parallel()

    # ---------------- The two topologies must agree ----------------
    torch.testing.assert_close(
        out_custom,
        out_mpu,
        **FWD_TOL,
        msg="forward output differs between MPU and custom gtp_remat groups",
    )
    torch.testing.assert_close(
        grad_in_custom,
        grad_in_mpu,
        **GRAD_TOL,
        msg="input gradient differs between MPU and custom gtp_remat groups",
    )
    assert set(grads_custom) == set(grads_mpu), "parameter sets differ between the two blocks"
    for name in sorted(grads_mpu):
        torch.testing.assert_close(
            grads_custom[name],
            grads_mpu[name],
            **GRAD_TOL,
            msg=f"weight gradient for {name} differs between MPU and custom gtp_remat groups",
        )


def _worker_partial_pgs_fall_back_to_mpu(rank, world_size, port):
    """A collection that omits gtp_remat must fall back to the MPU group, not disable GTP.

    ``__getattr__`` returns None for unset fields, so ``hasattr`` lies: a resolver trusting it
    reads None and silently builds an unsharded block.
    """
    from megatron.core import parallel_state as ps
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=GTP_SIZE
    )
    model_parallel_cuda_manual_seed(42)

    partial_pgs = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp', 'pp'])
    assert 'gtp_remat' not in vars(partial_pgs), "this collection must omit gtp_remat"

    block = _build_block(partial_pgs)

    gtp_group = ps.get_gtp_weight_remat_group()
    sharded = [(n, p) for n, p in block.named_parameters() if isinstance(p, GTPShardedParam)]
    assert sharded, "no parameter was sharded: the resolver did not fall back to the MPU group"
    for name, param in sharded:
        assert param.gtp_remat_size == gtp_group.size(), (
            f"{name} was sharded over a size-{param.gtp_remat_size} axis, "
            f"want {gtp_group.size()} (the MPU gtp_remat group)"
        )

    ps.destroy_model_parallel()
    ps.initialize_model_parallel()


class TestGTPCustomProcessGroups:
    def test_custom_gtp_pg_collection_matches_mpu(self):
        """A permuted-but-equivalent gtp_remat group must give identical fwd/bwd results."""
        _requires_multi_gpu(WORLD)
        _run_distributed(_worker_custom_pgs_match_mpu, WORLD)

    def test_pg_collection_without_gtp_remat_falls_back_to_mpu(self):
        """Omitting gtp_remat must fall back to the MPU group, not silently disable sharding."""
        _requires_multi_gpu(WORLD)
        _run_distributed(_worker_partial_pgs_fall_back_to_mpu, WORLD)
