dist\_checkpointing package
===========================

A library for saving and loading the distributed checkpoints.
A "distributed checkpoint" can have various underlying formats (current default format is based on Zarr)
but has a distinctive property - the checkpoint saved in one parallel configuration (tensor/pipeline/data parallelism)
can be loaded in a different parallel configuration.

Using the library requires defining sharded state_dict dictionaries with functions from  *mapping* and *optimizer* modules.
Those state dicts can be saved or loaded with a *serialization* module using strategies from *strategies* module.


Subpackages
-----------

.. toctree::
   :maxdepth: 4

   dist_checkpointing.strategies

Submodules
----------

dist\_checkpointing.serialization module
----------------------------------------

.. automodule:: core.dist_checkpointing.serialization
   :members:
   :undoc-members:
   :show-inheritance:

dist\_checkpointing.mapping module
----------------------------------

.. automodule:: core.dist_checkpointing.mapping
   :members:
   :undoc-members:
   :show-inheritance:

dist\_checkpointing.optimizer module
------------------------------------

.. automodule:: core.dist_checkpointing.optimizer
   :members:
   :undoc-members:
   :show-inheritance:

dist\_checkpointing.core module
-------------------------------

.. automodule:: core.dist_checkpointing.core
   :members:
   :undoc-members:
   :show-inheritance:

dist\_checkpointing.dict\_utils module
--------------------------------------

.. automodule:: core.dist_checkpointing.dict_utils
   :members:
   :undoc-members:
   :show-inheritance:


dist\_checkpointing.utils module
--------------------------------

.. automodule:: core.dist_checkpointing.utils
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: core.dist_checkpointing
   :members:
   :undoc-members:
   :show-inheritance:
