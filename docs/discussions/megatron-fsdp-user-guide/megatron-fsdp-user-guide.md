# Megatron-FSDP User Guide

## Table of Contents

- [Megatron-FSDP Quick Start](#megatron-fsdp-quick-start)
- [Checkpoint Conversion from 3D-Parallel to Megatron-FSDP](#checkpoint-conversion-from-3d-parallel-to-megatron-fsdp)

## Megatron-FSDP Quick Start

For your reference, we provide an example launch script for DeepSeek-V3: [`sbatch_mfsdp_deepseek_v3.sh`](./example-scripts/sbatch_mfsdp_deepseek_v3.sh).

### Required Configurations

To enable Megatron-FSDP, add the following required flags to your training script:

```bash
--use-megatron-fsdp
--data-parallel-sharding-strategy optim_grads_params
--no-gradient-accumulation-fusion
--use-distributed-optimizer
--ckpt-format fsdp_dtensor
```

### Recommended Configurations

We also recommend adding the following configurations to further improve performance:

```bash
unset CUDA_DEVICE_MAX_CONNECTIONS
```
```bash
--calculate-per-token-loss
--init-model-with-meta-device
--grad-reduce-in-bf16
--fsdp-double-buffer
--use-nccl-ub
```

💡 **Detailed explanations of these configurations are provided below.**

#### 1. Disable `CUDA_DEVICE_MAX_CONNECTIONS`

To ensure full parallelization of FSDP communication and computation, disable the CUDA_DEVICE_MAX_CONNECTIONS environment variable. This step avoids potential bubbles in the CUDA stream. (But it may slow down TP and CP to some extent.)

#### 2. Add `--calculate-per-token-loss`

For gradients sharding mode optimization, include the `--calculate-per-token-loss` flag in your training script. This improves performance by reducing the frequency of gradient scaling, which is also a sizable drain on SM resources.

#### 3. Add `--init-model-with-meta-device`

Allows model initialization using meta device, followed by layer-by-layer initialization of distributed model weight buffers via the `Module.reset_parameters` API, facilitating the initialization of extremely large models.

#### 4. Add `--grad-reduce-in-bf16`

Enables gradient reduction in BF16 precision instead of FP32, reducing communication volume and accelerating the backward pass.

#### 5. Add `--fsdp-double-buffer`

Uses persistently allocated double buffers for temporarily-defined memory needed in `MegatronFSDP` communications. While having persistent double buffers may increase peak VRAM utilization, it is necessary to register NCCL user buffers (`nccl_ub=True`) for `MegatronFSDP`. Currently, this is supported only for simple repetitive model structures such as GPT.

- **Only effective when using Megatron-LM.**
- Defaults to `False`. Automatically overridden to `True` when `nccl_ub` is enabled.

#### 6. Add `--use-nccl-ub`

Allocates and registers NCCL user buffers for param and grad buffers. This option enables an SM-efficient NCCL algorithm that could improve the performance of overlapped computations. This flag will be much more effective when used together with SHARP if the FSDP communication includes both NVL and IB domains. Enabling this option will cause additional memory overhead due to the requirement to enable the `fsdp_double_buffer` option.

- **Only effective when using Megatron-LM.**
- Defaults to `False`.
- By default we try to use NCCL window (symmetric) registration if it is available. If not it falls back to conventional local registration.

## Checkpoint Conversion from 3D-Parallel to Megatron-FSDP

Megatron-FSDP introduces a new checkpoint format `fsdp_dtensor`. To help you smoothly transition from 3D-Parallel to Megatron-FSDP, we provide a script for converting checkpoints from the `torch_dist` format to the `fsdp_dtensor` format. Using DeepSeek-V3 as an example, the detailed conversion process is described below.

### Step 1: Generate 3D-Parallel Checkpoint with `param_to_param_group_map`

Run your 3D-parallel + EP training script to generate a `torch_dist` checkpoint along with a directory containing `param_to_param_group_map` files. Add the following flag to your training script:

```bash
--dump-param-to-param-group-map /path/to/param_to_param_group_map
```

If you already have a `torch_dist` checkpoint, simply specify the `--dump-param-to-param-group-map /path/to/param_to_param_group_map` flag and run a very short experiment-this will create the `param_to_param_group_map` you need without full pretraining.

### Step 2: Export `param_to_param_group_map` to a JSON File

Convert the `param_to_param_group_map` into a JSON file for easier processing by running:

```bash
python tools/checkpoint/checkpoint_inspector.py print-torch-dcp-in-json /path/to/param_to_param_group_map
```

This will create a `param_to_param_group_map.json` file in the `/path/to/param_to_param_group_map` directory.

### Step 3: Convert Checkpoint from `torch_dist` to `fsdp_dtensor`

Convert your `torch_dist` checkpoint to the `fsdp_dtensor` format using the parameter to `param_to_param_group_map` JSON file:

```bash
torchrun --nproc_per_node=8 --nnodes=1 \
    tools/checkpoint/checkpoint_inspector.py \
    convert-torch-dist-to-fsdp-dtensor --swiglu \
    /path/to/input_torch_dist_checkpoint \
    /path/to/output_fsdp_dtensor_checkpoint \
    --param-to-param-group-map-json /path/to/param_to_param_group_map.json
```

**Note:** For multi-node conversion tasks, please refer to the example script: [`sbatch_checkpoint_convert.sh`](./example-scripts/sbatch_checkpoint_convert.sh).

### Step 4: Launch Megatron-FSDP Training

Start your Megatron-FSDP training job using the converted `fsdp_dtensor` checkpoint.