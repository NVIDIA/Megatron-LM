# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Normalization modules.

A norm is built through a ``torch_norm.LayerNormBuilder``,
``target(config=..., hidden_size=..., eps=...)``, and satisfies
``torch_norm.LayerNormInterface``: ``forward(x)`` normalizes over the last dimension and
``returns_residual`` says whether it also returns the residual it fused. A model's
``BackendSpecProvider.layer_norm`` chooses the builder. A norm owns its parameters
(``weight``, plus ``bias`` for LayerNorm; ``L2Norm`` has none) and no process groups.

* ``torch_norm`` -- the builder protocols, ``WrappedTorchNorm`` (PyTorch LayerNorm and
  RMSNorm) and ``L2Norm``.
* ``fused_layer_norm`` -- Apex's fused LayerNorm, ``FusedLayerNorm``.
* ``liger_norm`` -- optional Liger RMSNorm, ``LigerRMSNorm``.

Transformer Engine's norms stay in ``megatron.core.extensions.transformer_engine``.
Packages here do not re-export; import each name from the module that defines it.
"""
