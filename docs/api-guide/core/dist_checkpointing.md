<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

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

## Checkpointing Distributed Optimizer

### Checkpoint Compatibility and Optimizer State Formats

Beginning with **mcore v0.14**, the ``flattened_range`` attribute was removed from ``dist_checkpointing``. As a result:

- Optimizer states saved with mcore versions <= 0.14 can no longer be loaded directly. Loading these legacy optimizer states is not supported because the required sharded metadata is no longer available. If you need to continue training from older checkpoints, refer to the workaround described below.
- Model weights from older checkpoints remain fully compatible. No extra steps are needed—model weights from checkpoints created by earlier versions load automatically; simply add the ``--no-load-optim`` flag.

### Workaround: Loading legacy optimizer states with ToT MCore

**Step 1: Convert the legacy checkpoint using mcore v0.15.0**

Run a dummy training job with mcore v0.15.0 to re-save the checkpoint with new optimizer states format.

```bash
MODEL_TRAIN_PARAMS=(
    # Define model architecture and training parameters here
)
OLD_CKPT=/workspace/mcore_ckpt_old
CONVERTED_CKPT=/workspace/mcore_ckpt_0.15.0

torchrun --nproc_per_node=8 /opt/megatron-lm/pretrain_gpt.py \
   --save-interval 1 \
   --eval-interval 1 \
   --exit-interval 1 \
   --eval-iters 1 \
   --use-distributed-optimizer \
   --save ${CONVERTED_CKPT} \
   --load ${OLD_CKPT} \
   --ckpt-format torch_dist \
   "${MODEL_TRAIN_PARAMS[@]}"
```

**Step 2: Load the converted checkpoint with ToT MCore**

Use the converted checkpoint as the input for continued training with ToT MCore.

```bash
MODEL_TRAIN_PARAMS=(
    # Define model architecture and training parameters here
)
NEW_CKPT=/workspace/mcore_ckpt_new
CONVERTED_CKPT=/workspace/mcore_ckpt_0.15.0

torchrun --nproc_per_node=8 /opt/megatron-lm/pretrain_gpt.py \
   --use-distributed-optimizer \
   --save ${NEW_CKPT} \
   --load ${CONVERTED_CKPT} \
   --ckpt-format torch_dist \
   "${MODEL_TRAIN_PARAMS[@]}"
```

After this step, training can proceed normally using ToT MCore with fully supported optimizer state loading.

## Distributed Optimizer Checkpoint Formats

The refactor of the Distributed Optimizer introduces **two checkpoint formats**:

- dp_reshardable (Default)
   - Fast save/load performance.
   - Not reshardable — not possible to change model parallelism when using this format.
   - Recommended for general training when model parallelism changes are not needed.
- fully_reshardable
   - Fully reshardable — supports arbitrary changes in model parallelism.
   - Slower than dp_reshardable.
   - Enabled via the ``--dist-ckpt-optim-fully-reshardable`` flag.

### Workflow for Changing Model Parallelism

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

