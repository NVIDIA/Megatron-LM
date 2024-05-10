
## Arguments Related with Quantization

### Weight Quantization Arguments
- `--quantized-weights`
    - Weight Communication will be quantized when this is enable
    - Default: not enabled
- `--weight-quantization-bits 4`
    - Specifies the number of bits used for quantizing weights.
    - Default: 4
- `--wq-group-size 2048`
    - Defines the group size for weight quantization.
    - Default: 2048

### Gradient Quantization Arguments
- `--quantized-gradients`
    - Gradient Communication will be quantized when this is enable
    - Default: not enabled
- `--gq-group-size-inter 128`
    - Defines the group size for gradient quantization between nodes (inter-node).
    - Default: 128
- `--gradient-quantization-bits-inter 4`
    - Specifies the number of bits used for inter-node gradient quantization.
    - Default: 4
- `--gq-group-size-intra 128`
    - Defines the group size for gradient quantization within nodes (intra-node).
    - Default: 512
- `--gradient-quantization-bits-intra 8`
    - Specifies the number of bits used for intra-node gradient quantization.
    - Default: 8
- `--hadamard-transform`
    - Enable this to reduce Gradient Quantization error.
    - Default: not enabled
- `--gradient-alltoall-pipeline 8`
    - Chunk gradients to overlap intra and inter node communication.
    - Default: 1
### Additional Settings
- `--no-async-tensor-model-parallel-allreduce`
    - To overlap intra and inter node all-to-all, this should be enabled to avoid setting CUDA_DEVICE_MAX_CONNECTIONS to 1.
- `--mock-data`
    - To avoid IO overhead during speed test