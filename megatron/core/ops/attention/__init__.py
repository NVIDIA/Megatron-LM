# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Attention operation modules and kernels.

Organized by operation family:

* `dsa` -- DeepSeek sparse attention over MLA and over GQA: the indexer and attention
  modules, layout and masking helpers, the indexer loss, the TileLang and cuDNN
  backends and the minimum-activation DSA-GQA path.
* `csa` -- compressed sparse attention: compressors, indexers and attention modules
  with their SBHD and packed THD kernels.
* `mla` -- absorbed multi-latent attention.
* `dsv4` -- DeepSeek-v4 hybrid attention, built on the CSA modules.

Model assembly, module specs, layer configs and inference cache allocation stay outside
this package. Packages here do not re-export; import each name from the module that
defines it.
"""
