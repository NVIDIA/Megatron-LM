# fusions package

This package provides modules that provide commonly fused
operations. Fusing operations improves compute efficiency by
increasing the amount of work done each time a tensor is read from
memory. To perform the fusion, modules in this either rely on PyTorch
functionality for doing just-in-time compilation
(i.e. `torch.jit.script` in older PyTorch versions of `torch.compile`
in recent versions), or call into custom kernels in external libraries
such as Apex or TransformerEngine.

