<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# parallel_state module

This module manages model parallelism and data parallelism process groups in distributed training. It provides initialization and query functions for tensor parallelism (TP), pipeline parallelism (PP), data parallelism (DP), context parallelism (CP), and expert parallelism (EP) groups.

The key entry point is `initialize_model_parallel()`, which creates all necessary process groups based on the specified parallelism configuration. Once initialized, the module provides getter functions to retrieve the initialized groups and query rank information within each group.

See the [auto-generated API reference](https://nvidia.github.io/Megatron-LM/apidocs/core/core.parallel_state.html) for detailed function signatures and documentation.

