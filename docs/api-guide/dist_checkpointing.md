# dist_checkpointing package

A library for saving and loading the distributed checkpoints.
A "distributed checkpoint" can have various underlying formats (current default format is based on Zarr)
but has a distinctive property - the checkpoint saved in one parallel configuration (tensor/pipeline/data parallelism)
can be loaded in a different parallel configuration.

Using the library requires defining sharded state_dict dictionaries with functions from  *mapping* and *optimizer* modules.
Those state dicts can be saved or loaded with a *serialization* module using strategies from *strategies* module.

## Safe Checkpoint Loading

Since **PyTorch 2.6**, the default behavior of `torch.load` is `weights_only=True`.
This ensures that only tensors and allow-listed classes are loaded, reducing the risk of arbitrary code execution.

If you encounter an error such as:

```bash
WeightsUnpickler error: Unsupported global: GLOBAL argparse.Namespace was not an allowed global by default.
```

you can fix it by explicitly allow-listing the missing class in your script:

```python
import torch, argparse

torch.serialization.add_safe_globals([argparse.Namespace])
```

## Subpackages

```{toctree}
:maxdepth: 4

dist_checkpointing.strategies
```

## Submodules

### dist_checkpointing.serialization module

```{automodule} core.dist_checkpointing.serialization
:members:
:undoc-members:
:show-inheritance:
```

### dist_checkpointing.mapping module

```{automodule} core.dist_checkpointing.mapping
:members:
:undoc-members:
:show-inheritance:
```

### dist_checkpointing.optimizer module

```{automodule} core.dist_checkpointing.optimizer
:members:
:undoc-members:
:show-inheritance:
```

### dist_checkpointing.core module

```{automodule} core.dist_checkpointing.core
:members:
:undoc-members:
:show-inheritance:
```

### dist_checkpointing.dict_utils module

```{automodule} core.dist_checkpointing.dict_utils
:members:
:undoc-members:
:show-inheritance:
```

### dist_checkpointing.utils module

```{automodule} core.dist_checkpointing.utils
:members:
:undoc-members:
:show-inheritance:
```

## Module contents

```{automodule} core.dist_checkpointing
:members:
:undoc-members:
:show-inheritance:
```

