# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GDP must wait for conv1d's all-gather without changing the bucket dispatch order."""

import pytest
import torch

import megatron.core.ssm.gated_delta_product as gdp_module
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup
from megatron.core.models.hybrid.hybrid_layer_specs import gdp_stack_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_product import HAVE_FLA, GatedDeltaProductMixer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.utils import is_causal_conv1d_min_version
from tests.unit_tests.ssm.gdp_test_utils import HAVE_GDP_DEPS, HIDDEN_SIZE, _make_config
from tests.unit_tests.test_utilities import Utils

ITERATIONS = 3
SEQUENCE_LENGTH = 128
MICRO_BATCH_SIZE = 2

pytestmark = [
    pytest.mark.skipif(
        gdp_module.causal_conv1d_fn is None,
        reason="causal-conv1d is not installed; the mixer then runs conv1d's forward",
    ),
    pytest.mark.skipif(not HAVE_GDP_DEPS, reason="requires mamba-ssm, einops and FLA or CuTeDSL"),
]


@pytest.fixture(
    params=[
        (1, "headwise", False, False),
        (1, "headwise", False, True),
        (2, "headwise", False, False),
        (2, "chunkwise", True, False),
    ],
    ids=["cp1", "cp1-split", "cp2-headwise", "cp2-chunkwise-recompute"],
)
def ddp_mixer(request):
    """A GDP mixer under DDP with overlapped param gather and the distributed optimizer."""
    cp_size, cp_mode, recompute, split = request.param
    if Utils.world_size < 2 * cp_size:
        pytest.skip("requires at least two data-parallel ranks per CP rank")
    if cp_mode == "chunkwise":
        if not is_causal_conv1d_min_version("1.7.0"):
            pytest.skip("chunkwise CP requires causal-conv1d >= 1.7.0")
        if not (gdp_module.HAVE_FLA_GDP_CP if HAVE_FLA else gdp_module.HAVE_CUTEDSL_GDP_CP):
            pytest.skip("the selected GDP backend does not support chunkwise CP")
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=cp_size)
    model_parallel_cuda_manual_seed(123)
    try:
        config = _make_config(["gdp_in_proj", "gdp_qkv"] if recompute else [])
        config.context_parallel_size = cp_size
        config.linear_cp_mode = cp_mode
        config.linear_cp_layout = "contiguous"
        config.gdp_cutedsl_kernel = not HAVE_FLA
        mixer = GatedDeltaProductMixer(
            config,
            gdp_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
            config.hidden_size,
            conv_bias=True,
            layer_number=1,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=['tp', 'cp', 'dp_cp']
            ),
        ).cuda()

        # One bucket for the parameters registered after in_proj in conv1d's dtype: buckets are
        # filled in reverse registration order and closed once they reach bucket_size.
        params = [p for p in mixer.parameters() if p.requires_grad]
        in_proj_ids = {id(p) for p in mixer.in_proj.parameters()}
        last_in_proj = max(i for i, p in enumerate(params) if id(p) in in_proj_ids)
        conv_dtype = mixer.conv1d.weight.dtype
        bucket_size = sum(p.numel() for p in params[last_in_proj + 1 :] if p.dtype == conv_dtype)
        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_param_gather=True,
            overlap_grad_reduce=True,
            bucket_size=bucket_size,
        )
        layout = DistributedOptimizer.compute_full_param_layout(
            params, bucket_size, torch.distributed.get_world_size(), ddp_config
        )
        ddp = DistributedDataParallel(config, ddp_config, mixer, full_param_layout=layout)

        # Guard the premise, so that a changed layout cannot turn the tests into vacuous passes.
        groups = ddp.param_to_bucket_group
        conv_group = groups[mixer.conv1d.weight]
        assert groups[mixer.conv1d.bias] is conv_group
        assert groups[mixer.out_proj.weight] is conv_group, "no module after the conv waits"
        assert all(groups[p] is not conv_group for p in mixer.in_proj.parameters())

        optimizer = get_megatron_optimizer(
            OptimizerConfig(optimizer="adam", lr=1e-4, bf16=True, use_distributed_optimizer=True),
            [ddp],
        )
        yield ddp, optimizer, split
    finally:
        Utils.destroy_model_parallel()


def test_conv_kernel_reads_gathered_params(ddp_mixer, monkeypatch):
    """At the convolution entry point, conv1d's bucket group must be gathered and waited.

    Gathered and waited means dispatched with no pending handle. Before the fix the group's
    all-gather is still pending at every read, in every iteration.
    """
    ddp, optimizer, split = ddp_mixer
    mixer = ddp.module
    conv_params = {"conv1d.weight": mixer.conv1d.weight, "conv1d.bias": mixer.conv1d.bias}
    reads = []
    kernel_name = "causal_conv1d_cp" if mixer.chunkwise_context_parallel else "causal_conv1d_fn"
    kernel = getattr(gdp_module, kernel_name)
    dispatches = []
    start_param_sync = _ParamAndGradBucketGroup.start_param_sync

    def recording_start_param_sync(group, *args, **kwargs):
        dispatches.append(group)
        return start_param_sync(group, *args, **kwargs)

    def checked_kernel(*args, **kwargs):
        for name, param in conv_params.items():
            group = ddp.param_to_bucket_group[param]
            reads.append((name, group.param_gather_dispatched, group.param_gather_handle is None))
        return kernel(*args, **kwargs)

    monkeypatch.setattr(gdp_module, kernel_name, checked_kernel)
    monkeypatch.setattr(_ParamAndGradBucketGroup, "start_param_sync", recording_start_param_sync)
    for iteration in range(ITERATIONS):
        ddp.zero_grad_buffer()
        optimizer.zero_grad()
        torch.manual_seed(1000 * iteration + torch.distributed.get_rank())
        hidden_states = torch.randn(
            (SEQUENCE_LENGTH, MICRO_BATCH_SIZE, HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        if split:
            intermediate = mixer.forward_pre_attn_and_core_attn(hidden_states)
            output, _ = mixer.forward_post_core_attn(intermediate)
        else:
            output, _ = ddp(hidden_states)
        output.float().pow(2).mean().backward()
        ddp.finish_grad_sync()
        optimizer.step()

    # Each iteration gathers every bucket once in the same order. The mixer's own
    # FP32 parameters can trigger a different first bucket than split execution.
    first_iteration = dispatches[: len(ddp.bucket_groups)]
    assert set(first_iteration) == set(ddp.bucket_groups)
    assert dispatches == first_iteration * ITERATIONS

    passes = 2 if mixer.config.recompute_modules else 1
    assert len(reads) == ITERATIONS * len(conv_params) * passes, reads
    not_ready = [
        f"read {i // len(conv_params) + 1}: {name} dispatched={dispatched} waited={waited}"
        for i, (name, dispatched, waited) in enumerate(reads)
        if not (dispatched and waited)
    ]
    assert not not_ready, "conv1d read before its all-gather finished: " + "; ".join(not_ready)
