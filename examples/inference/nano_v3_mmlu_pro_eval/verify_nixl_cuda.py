# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Verify that NIXL can move data between CUDA buffers through UCX."""

import argparse
import importlib.metadata
import os
import time

import torch


def runtime_info() -> tuple[str, str, str]:
    from nixl import _api as nixl_api

    cuda_version = torch.version.cuda or "none"
    nixl_version = importlib.metadata.version("nixl")
    nixl_variant = nixl_api.__name__.split(".", maxsplit=1)[0]
    return cuda_version, nixl_version, nixl_variant


def loaded_transfer_libraries() -> list[str]:
    libraries = set()
    with open("/proc/self/maps", encoding="utf-8") as maps:
        for line in maps:
            path = line.rsplit(maxsplit=1)[-1]
            if any(name in path for name in ("libnixl", "libplugin_UCX", "libuc")):
                libraries.add(path)
    return sorted(libraries)


def verify_transfer() -> None:
    os.environ.setdefault("UCX_TLS", "tcp,cuda_ipc,cuda_copy,cma,shm,self")
    os.environ.setdefault("UCX_MEMTYPE_CACHE", "n")

    from nixl import _api as nixl_api

    cuda_version, nixl_version, nixl_variant = runtime_info()
    if cuda_version == "none" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; cannot validate NIXL GPU transfers")

    expected_variant = f"nixl_cu{cuda_version.split('.', maxsplit=1)[0]}"
    if nixl_variant != expected_variant:
        raise RuntimeError(
            f"PyTorch CUDA {cuda_version} requires {expected_variant}, but NIXL selected "
            f"{nixl_variant} (NIXL {nixl_version})"
        )

    agent = nixl_api.nixl_agent(f"megatron-nixl-preflight-{os.getpid()}")
    memory_types = {str(mem_type).lower() for mem_type in agent.get_backend_mem_types("UCX")}
    print(
        f"nixl_api={nixl_api.__file__} "
        f"NIXL_PLUGIN_DIR={os.environ.get('NIXL_PLUGIN_DIR')} "
        f"UCX_MODULE_DIR={os.environ.get('UCX_MODULE_DIR')} "
        f"UCX_TLS={os.environ.get('UCX_TLS')}",
        flush=True,
    )
    for library in loaded_transfer_libraries():
        print(f"loaded_library={library}", flush=True)
    if not any("cuda" in mem_type or "vram" in mem_type for mem_type in memory_types):
        raise RuntimeError(
            "NIXL's UCX backend does not advertise CUDA/VRAM support; "
            f"reported memory types: {sorted(memory_types)}"
        )

    source = torch.arange(256, dtype=torch.int64, device="cuda")
    destination = torch.zeros_like(source)
    registrations = [
        agent.register_memory(source, backends=["UCX"]),
        agent.register_memory(destination, backends=["UCX"]),
    ]
    try:
        transfer = agent.initialize_xfer(
            "READ",
            agent.get_xfer_descs(destination),
            agent.get_xfer_descs(source),
            agent.name,
            backends=["UCX"],
        )
        state = agent.transfer(transfer)
        deadline = time.monotonic() + 30
        while state not in {"DONE", "ERR"} and time.monotonic() < deadline:
            time.sleep(0.001)
            state = agent.check_xfer_state(transfer)
        if state != "DONE":
            raise RuntimeError(f"NIXL CUDA loopback transfer ended in state {state!r}")
        torch.cuda.synchronize()
        if not torch.equal(source, destination):
            raise RuntimeError("NIXL CUDA loopback transfer returned corrupted data")
    finally:
        for registration in registrations:
            agent.deregister_memory(registration, backends=["UCX"])

    print(
        f"NIXL CUDA preflight passed: torch_cuda={cuda_version} "
        f"nixl={nixl_version} variant={nixl_variant} ucx_memory_types={sorted(memory_types)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-info", action="store_true")
    args = parser.parse_args()
    if args.runtime_info:
        print("|".join(runtime_info()))
    else:
        verify_transfer()


if __name__ == "__main__":
    main()
