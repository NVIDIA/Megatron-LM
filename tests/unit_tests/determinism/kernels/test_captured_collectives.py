# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay actual recipe tensors from an explicitly selected collective capture.

Set MCORE_DETERMINISM_COLLECTIVE_CAPTURE to a complete shared capture directory.
This instrumented same-allocation protocol is separate from recipe state replay
and performance. It never claims that unbound/native collectives were captured.
"""

import os
from pathlib import Path

import pytest
import torch

from tests.unit_tests.determinism.comparison import bytes_equal
from tests.unit_tests.determinism.kernels.harness import (
    _assert_replay_matches,
    assert_replays_bit_exact,
    replay_signature,
)
from tests.unit_tests.test_utilities import Utils
from tools.determinism.capture_recipe import source_context
from tools.determinism.collective_capture import load_captures, load_tensor, prepare_replay
from tools.determinism.collective_reference import collective_reference, collective_rtol
from tools.determinism.recipe_coverage import signature_key
from tools.determinism.reference import assert_reference_close, assert_replay_sensitivity

CAPTURE_PATH = os.environ.get("MCORE_DETERMINISM_COLLECTIVE_CAPTURE")
if not CAPTURE_PATH:
    pytest.skip("requires an explicit collective recipe capture", allow_module_level=True)
CAPTURE_ROOT = Path(CAPTURE_PATH)
MAX_BYTES = int(os.environ.get("MCORE_DETERMINISM_COLLECTIVE_MAX_BYTES", 256 * 1024 * 1024))
CAPTURES = load_captures(CAPTURE_ROOT, max_bytes=MAX_BYTES)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NCCL GPUs"),
    pytest.mark.launch_on_gb200,
]


@pytest.fixture(scope="module")
def replay_groups():
    """Create every recorded group in the same global order on every rank."""
    initialized = torch.distributed.is_initialized()
    Utils.initialize_distributed()
    rank = torch.distributed.get_rank()
    error = (
        None
        if source_context(torch) == CAPTURES[rank]["context"]
        else "Capture source/environment differs"
    )
    errors = [None] * len(CAPTURES)
    torch.distributed.all_gather_object(errors, error)
    assert not any(errors), errors
    memberships = sorted(
        {
            tuple(event["signature"]["configuration"]["collective"]["group_ranks"])
            for report in CAPTURES
            for event in report["events"]
        }
    )
    groups = {}
    for members in memberships:
        group = torch.distributed.new_group(ranks=list(members), backend="nccl")
        if rank in members:
            groups[members] = group
    try:
        yield groups
    finally:
        torch.cuda.synchronize()
        torch.distributed.barrier()
        for group in groups.values():
            torch.distributed.destroy_process_group(group)
        if not initialized:
            torch.distributed.destroy_process_group()


@pytest.mark.parametrize(
    "event_index",
    [
        pytest.param(
            index,
            id=f"event-{index}",
            marks=pytest.mark.determinism_case(
                op_id="tensor_parallel_mappings",
                implementation=event["signature"]["implementation"],
            ),
        )
        for index, event in enumerate(CAPTURES[0]["events"])
    ],
)
def test_captured_collective_replay(replay_groups, event_index):
    """Compare all replay bytes and independent FP64 references from real inputs."""
    rank = torch.distributed.get_rank()
    event = CAPTURES[rank]["events"][event_index]
    signature = event["signature"]
    collective = signature["configuration"]["collective"]
    members = collective["group_ranks"]
    group = replay_groups[tuple(members)]
    prepared, error = None, None
    try:
        prepared = prepare_replay(event, CAPTURE_ROOT / f"rank-{rank}", group, max_bytes=MAX_BYTES)
    except (ValueError, RuntimeError, OSError) as caught:
        error = f"{type(caught).__name__}: {caught}"
    errors = [None] * len(CAPTURES)
    torch.distributed.all_gather_object(errors, error)
    assert not any(errors), errors
    function, local, gradient = prepared
    backward = signature["phase"] == "forward_backward"
    with torch.set_grad_enabled(collective["grad_enabled"]):
        measured_signature = replay_signature(
            (local,), backward=backward, configuration=signature["configuration"]
        )
        assert signature_key(
            {
                **measured_signature,
                "op_id": signature["op_id"],
                "implementation": signature["implementation"],
            }
        ) == signature_key(signature)
        actual = assert_replays_bit_exact(
            lambda value: function(value, group=group),
            (local,),
            backward=backward,
            grad_outputs={"out": gradient} if backward else None,
            replays=3,
            contention=True,
            defer_comparison=True,
            configuration=signature["configuration"],
            what=f"captured-collective:{event_index}",
        )
    assert_replay_sensitivity(
        actual,
        lambda perturbed: _assert_replay_matches(
            2, *actual, *perturbed, what="captured collective"
        ),
        signature=measured_signature,
    )
    inputs, gradients = [], []
    for member in members:
        peer = CAPTURES[member]["events"][event_index]["signature"]["configuration"]["collective"]
        inputs.append(
            load_tensor(
                CAPTURE_ROOT / f"rank-{member}", peer["input"], max_bytes=MAX_BYTES
            ).detach()
        )
        gradients.append(
            load_tensor(
                CAPTURE_ROOT / f"rank-{member}", peer["gradient"], max_bytes=MAX_BYTES
            ).detach()
            if backward
            else torch.zeros_like(actual[0]["out"], device="cpu")
        )
    expected, reductions, mathematical = collective_reference(
        collective["case"], inputs, gradients, collective["group_rank"], device=local.device
    )
    if not backward:
        expected, mathematical = (expected[0], {}), (mathematical[0], {})
        reductions = {key: value for key, value in reductions.items() if key.startswith("output:")}
    assert_reference_close(
        actual,
        expected,
        signature=measured_signature,
        reference_id="captured_cpu_fp64_rank_inputs_and_gradients:v1:" + collective["case"],
        rtol=collective_rtol(local.dtype, len(members)),
        atol=0,
        reductions=reductions,
        mathematical_reference=mathematical,
        numerical_controls=True,
    )
    for category, observed, reference in zip(("output", "gradient"), actual, expected):
        for name, value in observed.items():
            if f"{category}:{name}" not in reductions:
                assert bytes_equal(value, reference[name]), "Collective routing changed bytes"
    assert bytes_equal(local.detach().cpu(), inputs[collective["group_rank"]])
    if backward:
        assert bytes_equal(gradient.detach().cpu(), gradients[collective["group_rank"]])
