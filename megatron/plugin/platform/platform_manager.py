# Copyright (c) BAAI Corporation.

import os
from .platform_register import PLATFORMS

cur_platform = None


def is_current_platform_supported():
    return get_platform().device_name() in PLATFORMS.keys()


def get_platform():
    global cur_platform
    if cur_platform is not None:
        return cur_platform

    if "cuda" in PLATFORMS.keys() and PLATFORMS["cuda"].is_available():
        cur_platform = PLATFORMS["cuda"]
        print(f"Megatron-LM-FL Platform: cuda Selected")
    elif "musa" in PLATFORMS.keys() and PLATFORMS["musa"].is_available():
        cur_platform = PLATFORMS["musa"]
        print(f"Megatron-LM-FL Platform: musa Selected")
    elif "txda" in PLATFORMS.keys() and PLATFORMS["txda"].is_available():
        cur_platform = PLATFORMS["txda"]
        print(f"Megatron-LM-FL Platform: txda Selected")
    elif "npu" in PLATFORMS.keys() and PLATFORMS["npu"].is_available():
        cur_platform = PLATFORMS["npu"]
        print(f"Megatron-LM-FL Platform: npu Selected")
    elif "cpu" in PLATFORMS.keys() and PLATFORMS["cpu"].is_available():
        cur_platform = PLATFORMS["cpu"]
        print(f"Megatron-LM-FL Platform: cpu Selected")
    else:
        raise ValueError("No platform is available")
    
    return cur_platform


def set_platform(platform_obj):
    global cur_platform
    cur_platform = platform_obj
