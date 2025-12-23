# dist_checkpointing.strategies package

Package defining different checkpoint formats (backends) and saving/loading algorithms (strategies).

Strategies can be used for implementing new checkpoint formats or implementing new (more optimal for a given use case) ways of saving/loading of existing formats.
Strategies are passed to `dist_checkpointing.load` and `dist_checkpointing.save` functions and control the actual saving/loading procedure.

