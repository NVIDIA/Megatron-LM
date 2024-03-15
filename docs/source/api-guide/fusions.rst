fusions package
===============

This package provides modules that provide commonly fused
operations. Fusing operations improves compute efficiency by
increasing the amount of work done each time a tensor is read from
memory. To perform the fusion, modules in this either rely on PyTorch
functionality for doing just-in-time compilation
(i.e. `torch.jit.script` in older PyTorch versions of `torch.compile`
in recent versions), or call into custom kernels in external libraries
such as Apex or TransformerEngine.

Submodules
----------

fusions.fused\_bias\_dropout module
-----------------------------------

This module uses PyTorch JIT to fuse the bias add and dropout operations. Since dropout is not used during inference, different functions are used when in train mode and when in inference mode.

.. automodule:: core.fusions.fused_bias_dropout
   :members:
   :undoc-members:
   :show-inheritance:

fusions.fused\_bias\_gelu module
--------------------------------

This module uses PyTorch JIT to fuse the bias add and GeLU nonlinearity operations.

.. automodule:: core.fusions.fused_bias_gelu
   :members:
   :undoc-members:
   :show-inheritance:

fusions.fused\_layer\_norm module
---------------------------------

This module provides a wrapper around various fused LayerNorm implementation in Apex.

.. automodule:: core.fusions.fused_layer_norm
   :members:
   :undoc-members:
   :show-inheritance:

fusions.fused\_softmax module
-----------------------------

This module provides wrappers around variations of Softmax in Apex.

.. automodule:: core.fusions.fused_softmax
   :members:
   :undoc-members:
   :show-inheritance:

