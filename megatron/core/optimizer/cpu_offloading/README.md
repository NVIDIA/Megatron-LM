# Optimizer CPU Offloading

MCore provides two mutually exclusive options for reducing optimizer GPU memory usage by leveraging CPU memory. They cannot be combined.

## Option 1: `--optimizer-cpu-offload`

Runs the entire optimizer step on CPU. Gradients are copied to CPU, the optimizer updates parameters there, and the updated parameters are copied back to GPU.

```bash
--optimizer-cpu-offload
--optimizer-offload-fraction 1.0
--use-precision-aware-optimizer
```

Gradient copy, CPU optimizer step, and parameter copy can be time-consuming. Use `--overlap-cpu-optimizer-d2h-h2d` to overlap these transfers with computation.

## Option 2: `--offload-optimizer-states`

A lighter-weight alternative that keeps the optimizer step on GPU but moves optimizer states to CPU between steps to save GPU memory. Requires the distributed optimizer and Adam (TE FusedAdam).

```bash
--use-distributed-optimizer
--offload-optimizer-states
```
