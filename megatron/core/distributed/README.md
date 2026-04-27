# Distributed Data Parallelism

This module contains 

## Distributed Data Parallelism

This is the default data parallelism used with all parallelism topologies in Megatron-LM.

## Megatron-FSDP

To use Megatron-FSDP in Megatron-LM, enable the following arguments:

```
--use-megatron-fsdp
--ckpt-format fsdp_dtensor
--init-model-with-meta-device
```

## FSDP2

To use FSDP2 in Megatron-LM, enable the following arguments:

```
--use-torch-fsdp2
--no-gradient-accumulation-fusion
--ckpt-format torch_dist
```
