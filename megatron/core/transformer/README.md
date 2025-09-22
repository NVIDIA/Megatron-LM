<div align="center">

Fine-grained Activation Offloading
=============
<h4>NVIDIA, rednote</h4>
<div align="left">

# Quick Start

```bash
# Enable fine-grained activation offloading
--fine-grained-activation-offloading

# Specify which modules are going to be offloaded
# Choices: "attn_norm", "core_attn", "attn_proj", "mlp_norm", "expert_fc1", "moe_act".
--offload-modules core_attn
```

# Current status
## Features
* Support PP=1/PP/Interleaved PP
* Compatible with fine-grained recomputation
* Support FP8
* Support MTP
## Known issues
* `--offload-modules expert_fc1` doesn't work with TP>1

## WIP items
* Code refactor
* Support MTP
* Benchmark

# Methodology

## Offload/Reload the input of one module to/from CPU
Let's take the attention projection module as an example:
```
nvtx_range_push(suffix="linear_proj")
offload_context = contextlib.nullcontext()
if self.offload_attn_proj:
    core_attn_out = group_prefetch_offload_start(core_attn_out, name="attn_proj")
    offload_context = PipelineOffloadManager.get_instance()
with offload_context:
    output, bias = self.linear_proj(core_attn_out)
if self.offload_attn_proj:
    output, bias = group_prefetch_offload_commit(output, bias, release_tensors=[core_attn_out])
    offload_context = contextlib.nullcontext()
nvtx_range_pop(suffix="linear_proj")
```
The above code snippet could be divided into three parts in order:
1. Mark the starting point of offloading a new module;
2. Record the save_for_backward tensors in fprop and push it to a tensor buffer;
3. Offload the recorded tensors after the module's fprop finishes;

In bprop, the three parts above will:
1. Make sure the offloaded tensors are reloaded back to GPU;
2. Pop the corresponding tensors from the tensor buffer;
3. Reload the corresponding tensors of next module;

## Compatible with PP&Interleaved PP

`PipelineOffloadManager` is used to manage the chunks across different model chunks in fprop and bprop.
Before the model.forward() start, the `PipelineOffloadManager.get_instance().reset_chunk_handler` will be executed. In the fprop of this method, we create a `ChunkOffloadHandler` to handle the offloading context of one model chunk and then push it to a buffer, which will be popped out in a specific order in bprop.

<img width="1182" height="537" alt="image" src="https://github.com/user-attachments/assets/9d1655cc-d6d4-44de-acaf-35099cb902c2" />


## Compatible with fine-grained recomputation

<img width="2873" height="1494" alt="offload_and_recompute" src="https://github.com/user-attachments/assets/b857112f-4cf6-480f-aaf8-496bfe821faa" />


## A special case: attn_norm/mlp_norm

# Performance (WIP)

# Acknowledgement

This work refers to the previous work from Kuaishou: https://www.usenix.org/conference/atc24/presentation/yuan
