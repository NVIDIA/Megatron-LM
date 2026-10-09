# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay actual recipe tensors from an explicitly selected collective capture.

Run through ``python -m tools.determinism.replay_collectives``, which passes
``--collective-capture`` and ``--collective-max-bytes``. Without a capture every
case is skipped. This instrumented same-allocation protocol is separate from
recipe state replay and performance. It never claims that unbound/native
collectives were captured.
"""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

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
from tools.determinism.collective_capture import (
    load_captures,
    load_tensor,
    prepare_replay,
    restore_group_options,
)
from tools.determinism.collective_reference import collective_reference, collective_rtol
from tools.determinism.recipe_coverage import signature_key
from tools.determinism.reference import assert_reference_close, assert_replay_sensitivity

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NCCL GPUs")]


def capture_options(config):
    """Return the selected capture, or None when the replay launcher did not pass one."""
    root = config.getoption("collective_capture", default=None)
    if root is None:
        return None
    max_bytes = config.getoption("collective_max_bytes")
    return SimpleNamespace(
        root=Path(root), max_bytes=max_bytes, reports=load_captures(Path(root), max_bytes=max_bytes)
    )


def pytest_generate_tests(metafunc):
    """Declare one case per captured event; skip when no capture was selected."""
    if "event_index" not in metafunc.fixturenames:
        return
    capture = capture_options(metafunc.config)
    if capture is None:
        skip = pytest.mark.skip(reason="requires an explicit collective recipe capture")
        metafunc.parametrize("event_index", [pytest.param(None, id="no-capture", marks=skip)])
        return
    metafunc.parametrize(
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
            for index, event in enumerate(capture.reports[0]["events"])
        ],
    )


@pytest.fixture(scope="module")
def capture(request):
    """The capture selected by the replay launcher."""
    return capture_options(request.config)


def group_key(collective):
    """Distinguish communicators with the same members but different options."""
    return json.dumps([collective["group_ranks"], collective["group_options"]], sort_keys=True)


@pytest.fixture(scope="module")
def replay_groups(capture):
    """Create every recorded group in the same global order on every rank."""
    captures = capture.reports
    initialized = torch.distributed.is_initialized()
    Utils.initialize_distributed()
    rank = torch.distributed.get_rank()
    assert torch.distributed.get_world_size() == len(captures), "Capture/replay world sizes differ"
    try:
        error = (
            None
            if source_context(torch) == captures[rank]["context"]
            else "Capture source/environment differs"
        )
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as caught:
        error = f"Capture provenance unavailable: {caught}"
    errors = [None] * len(captures)
    torch.distributed.all_gather_object(errors, error)
    assert not any(errors), errors
    specifications = {}
    for report in captures:
        for event in report["events"]:
            collective = event["signature"]["configuration"]["collective"]
            specifications[group_key(collective)] = collective
    groups = {}
    for key, collective in sorted(specifications.items()):
        members = collective["group_ranks"]
        options = restore_group_options(collective["group_options"])
        group = torch.distributed.new_group(ranks=members, backend="nccl", pg_options=options)
        if rank in members:
            groups[key] = group
    try:
        yield groups
    finally:
        torch.cuda.synchronize()
        torch.distributed.barrier()
        for group in groups.values():
            torch.distributed.destroy_process_group(group)
        if not initialized:
            torch.distributed.destroy_process_group()


def test_captured_collective_replay(capture, replay_groups, event_index):
    """Compare all replay bytes and independent FP64 references from real inputs."""
    captures, capture_root, max_bytes = capture.reports, capture.root, capture.max_bytes
    rank = torch.distributed.get_rank()
    event = captures[rank]["events"][event_index]
    signature = event["signature"]
    collective = signature["configuration"]["collective"]
    members = collective["group_ranks"]
    group = replay_groups[group_key(collective)]
    prepared, error = None, None
    try:
        prepared = prepare_replay(event, capture_root / f"rank-{rank}", group, max_bytes=max_bytes)
    except (ValueError, RuntimeError, OSError) as caught:
        error = f"{type(caught).__name__}: {caught}"
    errors = [None] * len(captures)
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
        peer = captures[member]["events"][event_index]["signature"]["configuration"]["collective"]
        inputs.append(
            load_tensor(
                capture_root / f"rank-{member}", peer["input"], max_bytes=max_bytes
            ).detach()
        )
        gradients.append(
            load_tensor(
                capture_root / f"rank-{member}", peer["gradient"], max_bytes=max_bytes
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
