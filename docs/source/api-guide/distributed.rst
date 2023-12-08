distributed package
===================

This package contains various utilities to finalize model weight gradients
on each rank before the optimizer step. This includes a distributed data
parallelism wrapper to all-reduce or reduce-scatter the gradients across
data-parallel replicas, and a `finalize\_model\_grads` method to
synchronize gradients across different parallelism modes (e.g., 'tied'
layers on different pipeline stages, or gradients for experts in a MoE on
different ranks due to expert parallelism).

Submodules
----------

distributed.distributed\_data\_parallel
---------------------------------------

Model wrapper for distributed data parallelism. Stores gradients in a
contiguous buffer, and supports the option of overlapping communication
(all-reduce or reduce-scatter) with backprop computation by breaking up
full model's gradients into smaller buckets and running all-reduce /
reduce-scatter on each bucket asynchronously. 

.. automodule:: core.distributed.distributed_data_parallel
   :members:
   :undoc-members:
   :show-inheritance:

distributed.finalize\_model\_grads
----------------------------------

Finalize model gradients for optimizer step across all used parallelism modes.
Synchronizes the all-reduce / reduce-scatter of model gradients across DP replicas,
all-reduces the layernorm gradients for sequence parallelism, embedding gradients
across first and last pipeline stages (if not tied), and expert gradients for expert
parallelism.

.. automodule:: core.distributed.finalize_model_grads
   :members:
   :undoc-members:
   :show-inheritance:


Module contents
---------------

Contains functionality to synchronize gradients across different ranks before
optimizer step.

.. automodule:: core.distributed
   :members:
   :undoc-members:
   :show-inheritance:
