# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Backwards-compatible entry point. The coordinator is now the default path
# in gpt_dynamic_inference.py.

from examples.inference.gpt.gpt_dynamic_inference import main

if __name__ == "__main__":
    main()
