# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Same-allocation replay and independent accuracy for TP/SP NCCL mappings.

This does not establish cross-allocation, multi-node, quantized, GTP or overlapped
collective determinism. Explicit TP2/full-world groups never take a TP1 bypass.
"""

import hashlib
import os

import pytest
import torch

from megatron.core.tensor_parallel import mappings
from tests.performance_tests.shell_test_utils.determinism.kernel_case import kernel_policy
from tests.unit_tests.determinism.comparison import bytes_equal
from tests.unit_tests.determinism.kernels.harness import (
    _assert_replay_matches,
    assert_replays_bit_exact,
    replay_signature,
)
from tests.unit_tests.test_utilities import Utils
from tools.determinism.collective_reference import (
    MAPPINGS,
    collective_reference,
    collective_rtol,
    collective_shapes,
)
from tools.determinism.reference import assert_reference_close, assert_replay_sensitivity

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs NCCL GPUs"),
    pytest.mark.launch_on_gb200,
]


@pytest.fixture(scope="module", params=(2, "world"), ids=("tp2", "tpworld"))
def collective_group(request):
    """Create explicit equal-sized groups in a common global order on every rank."""
    Utils.initialize_distributed()
    world = torch.distributed.get_world_size()
    size = world if request.param == "world" else request.param
    if size < 2 or world % size:
        pytest.skip("requires a multi-rank allocation divisible by the requested TP group")
    global_rank = torch.distributed.get_rank()
    selected = None
    memberships = []
    for start in range(0, world, size):
        ranks = list(range(start, start + size))
        group = torch.distributed.new_group(ranks=ranks, backend="nccl")
        if global_rank in ranks:
            selected = group
            memberships = ranks
    try:
        assert selected is not None and selected.size() > 1
        assert torch.distributed.get_process_group_ranks(selected) == memberships
        yield selected, memberships
    finally:
        torch.cuda.synchronize()
        torch.distributed.barrier()
        if selected is not None:
            torch.distributed.destroy_process_group(selected)


def _rank_values(shape, dtype, rank, seed, strided):
    """Build independent rank inputs with random values and exact routing sentinels."""
    generator = torch.Generator(device="cpu").manual_seed(seed + rank)
    value = torch.randn(shape, generator=generator, dtype=torch.float64).to(dtype)
    value[0, 0] = rank + 1
    value[0, 1] = 0
    value[1, 0] = -1 if rank % 2 else 1
    value[1, 1] = (-1 if rank % 2 else 1) * (1 + rank / 8)
    return value.t().contiguous().t() if strided else value


def _hash(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            case,
            marks=pytest.mark.determinism_case(
                op_id="tensor_parallel_mappings", implementation="mcore:" + name
            ),
        )
        for case, name in MAPPINGS.items()
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "strided"])
@kernel_policy(torch, True)
def test_collective_mapping_evidence(collective_group, case, dtype, strided):
    """Compare every output and input gradient, including all-reduce backward."""
    group, ranks = collective_group
    rank, size = group.rank(), group.size()
    input_shape, output_shape = collective_shapes(case, size)
    inputs = [_rank_values(input_shape, dtype, value, 1700, strided) for value in ranks]
    upstream = [_rank_values(output_shape, dtype, value, 9100, strided) for value in ranks]
    local = inputs[rank].to("cuda").requires_grad_(True)
    grad_output = upstream[rank].to("cuda")
    assert local.is_contiguous() is not strided and grad_output.is_contiguous() is not strided
    configuration = {
        "collective": {
            "case": case,
            "group_ranks": ranks,
            "group_rank": rank,
            "size": size,
            "backend": str(torch.distributed.get_backend(group)),
            "nccl_version": torch.cuda.nccl.version(),
            "nccl_environment": {
                key: os.environ.get(key)
                for key in (
                    "NCCL_ALGO",
                    "NCCL_PROTO",
                    "NCCL_MAX_NCHANNELS",
                    "NCCL_MIN_NCHANNELS",
                    "NCCL_NVLS_ENABLE",
                    "NCCL_COLLNET_ENABLE",
                )
            },
            "input_seed_base": 1700,
            "gradient_seed_base": 9100,
            "input_sha256": _hash(inputs[rank]),
            "gradient_sha256": _hash(upstream[rank]),
            "upstream_gradient_shape": list(grad_output.shape),
            "upstream_gradient_stride": list(grad_output.stride()),
        }
    }
    function = getattr(mappings, MAPPINGS[case])
    actual = assert_replays_bit_exact(
        lambda x: function(x, group=group),
        (local,),
        replays=3,
        grad_outputs={"out": grad_output},
        contention=True,
        configuration=configuration,
        defer_comparison=True,
        what="collective:" + case,
    )
    assert bytes_equal(local.detach().cpu(), inputs[rank])
    assert bytes_equal(grad_output.cpu(), upstream[rank])
    signature = replay_signature((local,), backward=True, configuration=configuration)
    assert_replay_sensitivity(
        actual,
        lambda perturbed: _assert_replay_matches(2, *actual, *perturbed, what=case),
        signature=signature,
    )
    expected, reductions, mathematical = collective_reference(
        case, inputs, upstream, rank, device=local.device
    )
    assert_reference_close(
        actual,
        expected,
        signature=signature,
        reference_id="cpu_fp64_rank_inputs_and_gradients:v1:" + case,
        rtol=collective_rtol(dtype, size),
        atol=0,
        reductions=reductions,
        mathematical_reference=mathematical,
        numerical_controls=True,
    )
    for category, observed, reference in zip(("output", "gradient"), actual, expected):
        for name, tensor in observed.items():
            if f"{category}:{name}" not in reductions:
                assert bytes_equal(tensor, reference[name]), "Collective copy/gather changed bytes"
