dist\_checkpointing.strategies package
======================================

Package defining different checkpoint formats (backends) and saving/loading algorithms (strategies).

Strategies can be used for implementing new checkpoint formats or implementing new (more optimal for a given use case) ways of saving/loading of existing formats.
Strategies are passed to `dist_checkpointing.load` and `dist_checkpointing.save` functions and control the actual saving/loading procedure.

Submodules
----------

dist\_checkpointing.strategies.base module
------------------------------------------

.. automodule:: core.dist_checkpointing.strategies.base
   :members:
   :undoc-members:
   :show-inheritance:

dist\_checkpointing.strategies.tensorstore module
-------------------------------------------------

.. automodule:: core.dist_checkpointing.strategies.tensorstore
   :members:
   :undoc-members:
   :show-inheritance:

dist\_checkpointing.strategies.two\_stage module
------------------------------------------------

.. automodule:: core.dist_checkpointing.strategies.two_stage
   :members:
   :undoc-members:
   :show-inheritance:

dist\_checkpointing.strategies.zarr module
------------------------------------------

.. automodule:: core.dist_checkpointing.strategies.zarr
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: core.dist_checkpointing.strategies
   :members:
   :undoc-members:
   :show-inheritance:
