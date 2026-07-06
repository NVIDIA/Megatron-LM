# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``--fake-process-group`` builds the parallel groups without any real comms.

The fake world PG replaces the process-wide default process group and is
single-process by construction, so it cannot share the world PG the rest of the
unit-test suite runs on. These cases therefore drive
``megatron.training.initialize._initialize_distributed`` in a fresh subprocess,
one per rank, each pinned to that rank's GPU.

Regression coverage for the nccl2 group-construction wiring:
  * the fake world PG must be device-bound, otherwise ``split_group`` raises
    "No device associated with the default pg, not safe to split any process
    groups";
  * the split backend filter must not be device-qualified over a fake parent,
    which registers the bare backend name ``fake`` for every device type;
  * the pipeline-parallel group must inherit the fake backend instead of asking
    for a real ``nccl-lazy`` one, which would rendezvous over the ``FakeStore``
    and hang.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import megatron.core.parallel_state as ps
from tests.unit_tests.test_utilities import Utils

# Runs in a subprocess: initialize the fake world exactly the way a
# `--fake-process-group` training job does, then report the resulting groups.
_FAKE_PG_DRIVER = textwrap.dedent(
    '''
    import json
    import sys

    import torch

    sys.argv = ["fake_pg_driver"] + json.loads(sys.argv[1])

    import torch.distributed as dist

    from megatron.core import parallel_state as ps
    from megatron.training.arguments import parse_args, validate_args
    from megatron.training.global_vars import set_args
    from megatron.training.initialize import _initialize_distributed

    args = parse_args()
    validate_args(args)
    set_args(args)
    _initialize_distributed(None, None, None)


    def backend_of(group):
        return type(group._get_backend(torch.device("cuda"))).__name__


    result = {
        "backend": dist.get_backend(),
        "bound_device_id": str(dist.distributed_c10d._get_default_group().bound_device_id),
        "tp_size": ps.get_tensor_model_parallel_world_size(),
        "tp_ranks": dist.get_process_group_ranks(ps.get_tensor_model_parallel_group()),
        "tp_backend": backend_of(ps.get_tensor_model_parallel_group()),
        "pp_size": ps.get_pipeline_model_parallel_world_size(),
        "pp_ranks": dist.get_process_group_ranks(ps.get_pipeline_model_parallel_group()),
        "pp_backend": backend_of(ps.get_pipeline_model_parallel_group()),
        "dp_size": ps.get_data_parallel_world_size(),
        "dp_backend": backend_of(ps.get_data_parallel_group()),
    }
    print("FAKE_PG_RESULT " + json.dumps(result))
    '''
)

_FAKE_WORLD_SIZE = 8
_MODEL_ARGS = [
    "--num-layers",
    "4",
    "--hidden-size",
    "64",
    "--num-attention-heads",
    "4",
    "--seq-length",
    "16",
    "--max-position-embeddings",
    "16",
    "--micro-batch-size",
    "1",
    "--global-batch-size",
    "2",
    "--train-iters",
    "1",
    "--lr",
    "1e-4",
    "--vocab-size",
    "128",
    "--tokenizer-type",
    "NullTokenizer",
    "--mock-data",
]


@pytest.fixture(scope="module")
def fake_pg_groups(tmp_path_factory):
    """Groups reported by a ``--fake-process-group`` init of an 8-rank world."""
    if not torch.cuda.is_available():
        pytest.skip("--fake-process-group binds a GPU")

    driver = tmp_path_factory.mktemp("fake_pg") / "fake_pg_driver.py"
    driver.write_text(_FAKE_PG_DRIVER)

    argv = [
        "--fake-process-group",
        "--tensor-model-parallel-size",
        "2",
        "--pipeline-model-parallel-size",
        "2",
        *_MODEL_ARGS,
    ]

    local_rank = Utils.rank % torch.cuda.device_count()
    env = dict(os.environ)
    env.update(
        {
            # The fake world pretends to be 8 ranks; the subprocess is its rank
            # 0 and uses this pytest rank's GPU.
            "WORLD_SIZE": str(_FAKE_WORLD_SIZE),
            "RANK": "0",
            "LOCAL_RANK": str(local_rank),
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        }
    )
    # The subprocess does not inherit the interpreter's sys.path. `megatron` is
    # a namespace package (no __file__), so locate it via one of its modules:
    # <root>/megatron/core/parallel_state.py -> <root>.
    megatron_root = str(Path(ps.__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in (megatron_root, env.get("PYTHONPATH", "")) if p]
    )

    completed = subprocess.run(
        [sys.executable, str(driver), json.dumps(argv)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, (
        f"fake process group init failed (rc={completed.returncode})\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )
    for line in completed.stdout.splitlines():
        if line.startswith("FAKE_PG_RESULT "):
            return json.loads(line[len("FAKE_PG_RESULT ") :])
    raise AssertionError(f"driver produced no result\n--- stdout ---\n{completed.stdout}")


class TestFakeProcessGroupInitialization:
    """``--fake-process-group`` builds every parallel group over a fake world PG."""

    def test_world_is_a_device_bound_fake_process_group(self, fake_pg_groups):
        assert fake_pg_groups["backend"] == "fake"
        # split_group refuses to split a parent with no bound_device_id.
        local_rank = Utils.rank % torch.cuda.device_count()
        assert fake_pg_groups["bound_device_id"] == f"cuda:{local_rank}"

    def test_parallel_groups_have_the_expected_ranks(self, fake_pg_groups):
        assert fake_pg_groups["tp_size"] == 2
        assert fake_pg_groups["tp_ranks"] == [0, 1]
        assert fake_pg_groups["pp_size"] == 2
        assert fake_pg_groups["pp_ranks"] == [0, 4]
        assert fake_pg_groups["dp_size"] == 2

    def test_no_real_backend_is_created(self, fake_pg_groups):
        """Every group stays fake -- notably the pipeline-parallel one.

        Asking for a real ``nccl-lazy`` pipeline group over a fake world builds
        an actual NCCL communicator and rendezvouses over the ``FakeStore``,
        which hangs.
        """
        assert fake_pg_groups["tp_backend"] == "FakeProcessGroup"
        assert fake_pg_groups["pp_backend"] == "FakeProcessGroup"
        assert fake_pg_groups["dp_backend"] == "FakeProcessGroup"


class TestSplitGroupBackendFilter:
    """``_split_group_backend`` only device-qualifies a real world PG."""

    def test_fake_world_inherits_the_parent_backend(self):
        # A fake parent registers the bare name "fake" for every device type, so
        # there is nothing to filter, and split_group validates the filter
        # against that bare name -- any device-qualified filter is rejected.
        with patch.object(ps, "is_fake_process_group", return_value=True):
            assert ps._split_group_backend(None) is None
            assert ps._split_group_backend("nccl") is None
            assert ps._split_group_backend("gloo") is None

    def test_real_world_is_device_qualified(self):
        Utils.initialize_distributed()
        cuda_backend = dict(
            part.split(":", 1) for part in torch.distributed.get_backend_config().split(",")
        )["cuda"]

        assert ps._split_group_backend(None) == f"cuda:{cuda_backend}"
        assert ps._split_group_backend("gloo") == f"cpu:gloo,cuda:{cuda_backend}"
