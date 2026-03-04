<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# dist_checkpointing.strategies package

Package defining different checkpoint formats (backends) and saving/loading algorithms (strategies).

Strategies can be used for implementing new checkpoint formats or implementing new (more optimal for a given use case) ways of saving/loading of existing formats.
Strategies are passed to `dist_checkpointing.load` and `dist_checkpointing.save` functions and control the actual saving/loading procedure.

