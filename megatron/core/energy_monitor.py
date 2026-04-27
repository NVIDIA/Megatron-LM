# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Megatron Energy Monitoring (NVML)"""

import torch
import torch.distributed as dist

try:
    from pynvml import (
        NVMLError,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetTotalEnergyConsumption,
        nvmlInit,
        nvmlShutdown,
    )

    has_nvml = True
except ImportError:
    has_nvml = False


class EnergyMonitor:
    """
    Energy monitoring using NVML.

    All ranks in the process group are expected to call functions lap() and get_total().
    Energy is monitored across all ranks and gathered with an all-gather.

    Per-GPU energy from the most recent lap() call is available via
    get_last_lap_per_gpu(), enabling per-GPU power analysis and CSV export.
    """

    def __init__(self) -> None:
        """Initialize EnergyMonitor."""
        self._total_energy = 0
        self._lap_energy = 0
        self._last_energy = 0
        self._handle = None
        self._last_lap_per_gpu: list[float] = []

    def setup(self) -> None:
        """Setup the NVML Handler."""
        if has_nvml:
            nvmlInit()
            self._handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    def shutdown(self) -> None:
        """Shutdown NVML."""
        if has_nvml:
            nvmlShutdown()

    def pause(self) -> None:
        """Pause energy monitor (must resume afterward)."""
        if has_nvml:
            energy = self._get_energy()
            self._lap_energy += energy - self._last_energy

    def resume(self) -> None:
        """Resume/start energy monitor."""
        if has_nvml:
            self._last_energy = self._get_energy()

    def _get_energy(self) -> int:
        """Get current energy consumption from NVML."""
        try:
            # Passing None to nvmlDeviceGetTotalEnergyConsumption can cause a core
            # dump, so short circuit if self._handle is None.
            if self._handle is not None:
                return nvmlDeviceGetTotalEnergyConsumption(self._handle)
            return self._last_energy
        except NVMLError:
            return self._last_energy  # return *something* if it errors

    def lap(self) -> float:
        """Returns lap (iteration) energy (J) and updates total energy.

        Also stores per-GPU energy (J) accessible via get_last_lap_per_gpu().
        """
        if not has_nvml:
            return 0.0

        energy = self._get_energy()
        lap_energy = self._lap_energy + (energy - self._last_energy)

        self._total_energy += lap_energy
        self._lap_energy = 0
        self._last_energy = energy

        lap_tensor = torch.tensor([lap_energy], dtype=torch.int64, device='cuda')

        # all_gather preserves per-GPU values for optional per-GPU analysis
        world_size = dist.get_world_size()
        gathered = [torch.zeros_like(lap_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, lap_tensor)

        # Store per-GPU energy in Joules
        self._last_lap_per_gpu = [g.item() / 1000.0 for g in gathered]

        return sum(self._last_lap_per_gpu)

    def get_last_lap_per_gpu(self) -> list[float]:
        """Returns per-GPU energy (J) from the most recent lap() call.

        Each element corresponds to one rank's GPU energy consumption
        since the previous lap() call. The list is ordered by rank.
        """
        return self._last_lap_per_gpu

    def get_total(self) -> float:
        """Get total energy consumption (J) across all GPUs."""
        if not has_nvml:
            return 0.0

        energy_tensor = torch.tensor([self._total_energy], dtype=torch.int64, device='cuda')
        dist.all_reduce(energy_tensor, op=dist.ReduceOp.SUM)

        return energy_tensor.item() / 1000.0
