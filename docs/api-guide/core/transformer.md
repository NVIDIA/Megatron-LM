<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# transformer package

The `transformer` package provides a customizable and configurable
implementation of the transformer model architecture. Each component
of a transformer stack, from entire layers down to individual linear
layers, can be customized by swapping in different PyTorch modules
using the "spec" parameters. The
configuration of the transformer (hidden size, number of layers,
number of attention heads, etc.) is provided via a `TransformerConfig`
object.
