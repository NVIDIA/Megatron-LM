# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

PLATFORMS = {}


def register_platforms() -> None:
    """
    Register all platforms

    """
    # Register CPU Platform
    from .platform_cpu import PlatformCPU
    platform_cpu = PlatformCPU()
    if platform_cpu.is_available():
        PLATFORMS["cpu"] = platform_cpu # use lower keys: cpu
        print(f"Megatron-LM-FL Platform: cpu Registered")

    # Register CUDA Platform
    from .platform_cuda import PlatformCUDA
    platform_cuda = PlatformCUDA()
    if platform_cuda.is_available():
        PLATFORMS["cuda"] = platform_cuda # use lower keys: cuda
        print(f"Megatron-LM-FL Platform: cuda Registered")
    
    # Register MUSA Platform
    from .platform_musa import PlatformMUSA
    platform_musa = PlatformMUSA()
    if platform_musa.is_available():
        PLATFORMS["musa"] = platform_musa # use lower keys: musa
        print(f"Megatron-LM-FL Platform: musa Registered")

    # Register TXDA Platform
    from .platform_txda import PlatformTXDA
    platform_txda = PlatformTXDA()
    if platform_txda.is_available():
        PLATFORMS["txda"] = platform_txda # use lower keys: txda
        print(f"Megatron-LM-FL Platform: txda Registered")

    # Register NPU Platform
    from .platform_npu import PlatformNPU
    platform_npu = PlatformNPU()
    if platform_npu.is_available():
        PLATFORMS["npu"] = platform_npu # use lower keys: npu
        print(f"Megatron-LM-FL Platform: npu Registered")

    # Register ENFLAME Platform
    from .platform_enflame import PlatformENFLAME
    platform_enflame = PlatformENFLAME()
    if platform_enflame.is_available():
        PLATFORMS["enflame"] = platform_enflame # use lower keys: enflame
        print(f"Megatron-LM-FL Platform: enflame Registered")
    