pipeline\_parallel package
==========================

This package contains implementations for two different pipeline parallelism
schedules (one without interleaving and one with interleaving, see `Efficient
Large-Scale Language Model Training on GPU Clusters Using Megatron-LM <https://arxiv.org/abs/2104.04473>`_
for details), and a default no-pipelining schedule. It also contains methods
for the point-to-point communication that is needed between pipeline stages.

Submodules
----------

.. mdinclude:: pipeline_parallel_layout.md

pipeline\_parallel.p2p\_communication module
--------------------------------------------

Contains implementations for the various point-to-point communication needed
(e.g., `recv_forward` and `recv_backward`) in the different pipeline parallelism
schedules.

.. automodule:: core.pipeline_parallel.p2p_communication
   :members:
   :undoc-members:
   :show-inheritance:

pipeline\_parallel.schedules module
-----------------------------------

Contains implementations for two pipeline parallelism schedules
(`forward_backward_pipelining_with_interleaving`for pipeline parallelism with
interleaving, `forward_backward_pipelining_without_interleaving` for pipeline
parallelism without interleaving) and a default no-pipelining schedule
(`forward_backward_no_pipelining`). `get_forward_backward_func` returns the right
scheduling function to use based on the configuration being trained
(e.g., if pipeline-parallel size is 1, use `forward_backward_no_pipelining`).

.. automodule:: core.pipeline_parallel.schedules
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: core.pipeline_parallel
   :members:
   :undoc-members:
   :show-inheritance:
