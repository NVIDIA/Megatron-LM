## How to use ?

Add these flags to enable optimizer cpu offload in MCore.

```bash
--optimizer-cpu-offload
--optimizer-offload-fraction 1.0
--use-precision-aware-optimizer
```

## Configuration Recommendataions

Gradient copy from GPU to CPU, CPU optimizer step, and subsequent parameter copy from CPU to GPU can be time-consuming operations, and it is recommended to use the flag `--overlap-cpu-optimizer-d2h-h2d` to execute them concurrently.
