distributed package
===================

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

Finalize model grads for optimizer step across all used parallelism modes.
Synchronizes the all-reduce / reduce-scatter of model grads across DP replicas,
and all-reduces the layernorm grads for sequence parallelism, embedding grads
across first and last pipeline stages (if not tied), and expert grads for expert
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
