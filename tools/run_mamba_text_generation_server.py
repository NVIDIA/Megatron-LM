# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from run_text_generation_server import main

if __name__ == "__main__":
    main(model_provider="mamba")
