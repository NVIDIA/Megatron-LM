## How to use pytorch FSDP2?

Add these flag to enable Torch FSDP2.

```
--use-torch-fsdp2
--no-gradient-accumulation-fusion
--ckpt-format torch_dist
```

It is worth noting that CUDA_MAX_CONNECTIONS=1 should not be enabled to ensure that the communication of FSDP and the computation on the primary stream can be fully parallelized.
