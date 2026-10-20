# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bit-exact replay of a bf16 AllReduce under ``NCCL_ALGO=Ring,NVLS``.

Deterministic mode accepts ``Ring,NVLS`` except on Hopper, where the result of an NVLS
reduction can depend on the multicast objects allocated before it in the same process
(NVIDIA/nccl#2360). The replay follows the pattern of that issue: before it creates the
communicator under test, it creates ``k`` other communicators that each run a small AllReduce,
then AllReduces a fixed 32 MiB bf16 tensor twice on the communicator under test. Every result
must have the same bits for every ``k``, on every rank, and in both repeats.

The unit-test ``conftest.py`` sets ``NCCL_NVLS_ENABLE=0`` and ``NCCL_MAX_NCHANNELS=1`` for the
suite, and NCCL reads both once per process, when it creates its first communicator. The test
therefore runs the replay as a separate single-node job on all GPUs of the node: rank 0 of the
pytest run starts it with ``torch.distributed.run``, with ``NCCL_ALGO=Ring,NVLS`` and without
those two variables, and the other ranks wait. NVLS needs NVLink multicast and more than two
GPUs, so the test is skipped elsewhere and on Hopper.

A pass shows that NVLS reductions are bit-exact only if NCCL used NVLS. NCCL can find multicast
unusable when it creates a communicator, and then logs ``Proceeding without NVLS support`` and
runs Ring alone; otherwise it chooses between Ring and NVLS for each call. To see which algorithm
ran, add ``NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING`` and look for
``AllReduce: 33554432 Bytes -> Algo NVLS`` in the replay's output. Run it from the repository
root with::

    torchrun --nproc_per_node 8 -m pytest -q -s \\
        tests/unit_tests/determinism/correctness/test_nccl_nvls_allreduce.py
"""

import ctypes
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.unit_tests.test_utilities import Utils

REPO_ROOT = Path(__file__).resolve().parents[4]
_MODULE = "tests.unit_tests.determinism.correctness.test_nccl_nvls_allreduce"

_IS_HOPPER = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9

# 32 MiB of bf16.
_NUMEL = 1 << 24

# CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED in cuda.h; NCCL queries it when it checks for NVLS.
_CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132


def _multicast_supported():
    libcuda = ctypes.CDLL("libcuda.so.1")
    device, supported = ctypes.c_int(), ctypes.c_int()
    if libcuda.cuDeviceGet(ctypes.byref(device), torch.cuda.current_device()) != 0:
        return False
    rc = libcuda.cuDeviceGetAttribute(
        ctypes.byref(supported), _CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device
    )
    return rc == 0 and supported.value == 1


def _order_sensitive_input(rank):
    """Normal values scaled by 2**e, e from -8 to 24, chosen to make the sum sensitive to the
    reduction order even with fp32 accumulation (which NCCL 2.27 and later use for NVLS)."""
    generator = torch.Generator(device="cuda").manual_seed(1234 + rank)
    values = torch.randn(_NUMEL, generator=generator, device="cuda")
    exponents = torch.randint(-8, 25, (_NUMEL,), generator=generator, device="cuda")
    return (values * torch.exp2(exponents.float())).to(torch.bfloat16)


def _digest(tensor):
    return hashlib.sha256(tensor.view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:16]


def _all_reduce_after(k, x):
    """Digests of two AllReduces of ``x`` on a new group created after ``k`` other groups."""
    ranks = list(range(torch.distributed.get_world_size()))
    groups = []
    for _ in range(k):
        groups.append(torch.distributed.new_group(ranks))
        torch.distributed.all_reduce(torch.ones(1024, device="cuda"), group=groups[-1])
    groups.append(torch.distributed.new_group(ranks))
    digests = []
    for _ in range(2):
        y = x.clone()
        torch.distributed.all_reduce(y, group=groups[-1])
        digests.append(_digest(y))
    for group in groups:
        torch.distributed.destroy_process_group(group)
    return digests


def _replay():
    """One rank of the replay job."""
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl")
    rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    x = _order_sensitive_input(rank)
    digests = {(k, repeat): d for k in range(4) for repeat, d in enumerate(_all_reduce_after(k, x))}
    per_rank = [None] * world_size
    torch.distributed.all_gather_object(per_rank, digests)
    torch.distributed.destroy_process_group()
    distinct = {d for rank_digests in per_rank for d in rank_digests.values()}
    assert len(distinct) == 1, f"results differ; (k, repeat) -> digest per rank: {per_rank}"
    if rank == 0:
        print(f"Ring,NVLS replay: {world_size} ranks, every result {distinct.pop()}", flush=True)


@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA GPUs")
@pytest.mark.skipif(
    _IS_HOPPER, reason="deterministic mode rejects Ring,NVLS on Hopper (NVIDIA/nccl#2360)"
)
def test_ring_nvls_all_reduce_is_bit_exact_across_communicator_histories():
    """The same bits for k = 0..3 earlier communicators, on every rank, and when repeated."""
    Utils.initialize_distributed()
    gpus = torch.cuda.device_count()
    if gpus <= 2:
        pytest.skip("NVLS needs more than two GPUs")
    if not _multicast_supported():
        pytest.skip("NVLink multicast, which NVLS needs, is not available")

    # A CPU group, so the waiting ranks do not keep a kernel spinning on the GPUs the job uses.
    cpu_group = torch.distributed.new_group(backend="gloo")
    returncode = [None]
    if torch.distributed.get_rank() == 0:
        dropped = ("NCCL_NVLS_ENABLE", "NCCL_MAX_NCHANNELS")
        env = {k: v for k, v in os.environ.items() if k not in dropped}
        env["NCCL_ALGO"] = "Ring,NVLS"
        # The job runs on this node only, so its workers use the loopback address rather than
        # the host name, which may not resolve.
        launch = [sys.executable, "-m", "torch.distributed.run", "--standalone"]
        launch += ["--local-addr=127.0.0.1", f"--nproc_per_node={gpus}", "-m", _MODULE]
        returncode[0] = subprocess.run(launch, cwd=REPO_ROOT, env=env, timeout=600).returncode
    torch.distributed.broadcast_object_list(returncode, src=0, group=cpu_group)
    torch.distributed.destroy_process_group(cpu_group)
    assert returncode[0] == 0, f"the replay job exited with {returncode[0]}; see this test's output"


if __name__ == "__main__":
    _replay()
