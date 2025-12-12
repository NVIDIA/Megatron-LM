# dist_checkpointing package

A library for saving and loading the distributed checkpoints.
A *distributed checkpoint* in Megatron Core uses the ``torch_dist`` format,
a custom checkpointing mechanism built on top of PyTorch's native
checkpointing capabilities.

A key property of distributed checkpoints is that a checkpoint saved under one
parallel configuration (tensor, pipeline, or data parallelism) can be loaded
under a different parallel configuration. This enables flexible scaling and
resharding of models across heterogeneous training setups.

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

Checkpointing Distributed Optimizer
-----------------------------------

Checkpoint Compatibility and Optimizer State Formats
####################################################

Beginning with **mcore v0.14**, the ``flattened_range`` attribute was removed from ``dist_checkpointing``. As a result:

- Optimizer states saved with mcore versions < 0.14 are no longer loadable. Loading these legacy optimizer states is not supported because the required sharded metadata is no longer available.
- Model weights from older checkpoints remain fully compatible. No additional work is required—model weights from checkpoints produced by earlier versions are loaded automatically.

Distributed Optimizer Checkpoint Formats
########################################

The refactor of the Distributed Optimizer introduces **two checkpoint formats**:

- dp_reshardable (Default)
   - Fast save/load performance.
   - Not reshardable — not possible to change model parallelism when using this format.
   - Recommended for general training when model parallelism changes are not needed.
- fully_reshardable
   - Fully reshardable — supports arbitrary changes in model parallelism.
   - Slower than dp_reshardable.
   - Enabled via the ``--dist-ckpt-optim-fully-reshardable`` flag.

Workflow for Changing Model Parallelism
#######################################

You can combine formats to optimize both flexibility and performance:

   1. Train using ``dp_reshardable`` (default) for faster checkpointing.
   2. When you need to change model parallelism:

      - Stop training.
      - Change model parallelism for train config.
      - Resume training with ``--dist-ckpt-optim-fully-reshardable``.

   3. Save at least one checkpoint under the new model parallel configuration.
   4. (Optional) To continue the training with updated model parallelism and better checkpointing performance, stop training and switch back to ``dp_reshardable`` format by removing ``--dist-ckpt-optim-fully-reshardable``.

## Subpackages

```{toctree}
:maxdepth: 4

dist_checkpointing.strategies
```

