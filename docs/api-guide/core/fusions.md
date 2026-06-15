<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# fusions package

This package provides modules that provide commonly fused
operations. Fusing operations improves compute efficiency by
increasing the amount of work done each time a tensor is read from
memory. To perform the fusion, modules in this either rely on PyTorch
functionality for doing just-in-time compilation
(i.e. `torch.jit.script` in older PyTorch versions of `torch.compile`
in recent versions), or call into custom kernels in external libraries
such as Apex or TransformerEngine.

