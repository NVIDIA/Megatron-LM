<!-- MegatronLM GPTModel Training Analysis Tool -->

# Introduction
Offline analysis of memory requirements and communication information of MegatronLM GPTModel training under hybrid parallel strategies.
# Features
Given the GPT model configuration and parallel training configuration, this tool will output the following:

* Detail the memory requirements for Parameters, Gradients, Optimizer States and Activations at the Transformer granularity level on each GPU.
* Provide an estimate predicting the least amount of memory a GPU needs to train the GPT model without causing Out-of-Memory (OOM) errors.
* Describe the communication requirements when implementing Data Parallelism, Pipeline Parallelism and Tensor Parallelism. State how many times each dimension needs to communicate, the amount of data transmitted each time and the members of the communication group, among others.
* Describe the changes in the size of the Transformer model before and after parallel and how these changes impact GPU utilization.

We randomly selected some parallel configurations and used the "Memory Requirement" output in this tool as the predicted value, and the output of "torch.cuda.max_memory_allocated()" in Megatron's [report_memory](../../megatron/utils.py#L81) after training several iterations as the actual value. The parallel configurations in the x-axis of the following figure correspond to the four model parallel configurations in the table below in order.

This can give users insight into whether their planned parallel configuration is trainable, and if it potentially could trigger OOM errors.

<div align="center">
<img src="https://raw.githubusercontent.com/NVIDIA/Megatron-LM/2ffa75a9bcf6afad563e9be06f6197d8bdb4a814/tools/get_training_info/Mem_Est_vs_Actual.png" alt="图片描述" style="width: 250px; object-fit: cover;">
</div>

<style>
table {
margin: auto;
}
</style>

<div  style="zoom:70%" >

Model |Precision | MBS   | GBS       | DP   | PP  | TP  | Peak_Memory_Actual |  Peak_Memory_Estimated | Error (%) 
:-: |:-: |:-: | :-: | :-:|:-: | :-: | :-: | :-: | :-:
Llama2 7B | bf16 | 2     |   2048    | 8    | 1   | 1   |  69.1|  68.8 |0.4 
Llama2 7B | bf16 | 4     |   512     | 4    | 1   | 2   | 55.7 |  55.5 |0.4
Llama2 7B | bf16 | 2     |   2048    | 4    | 2   | 1   | 49.8|  49.5|0.6
Llama2 7B | bf16 | 4     |   128     | 1    | 1   | 8   | 28.8 |  28.6 |0.7
In this table, MBS refers to micro batch size, GBS refers to global batch size, DP denotes data parallelism size, PP denotes pipeline parallelism size, and TP denotes tensor parallelism size.

</div>



# Calculation Method Explanation
We analyze the memory requirements of the model parameters, gradients, and optimizer states and the communication behavior of different parallel dimensions based on Megatron([1](https://arxiv.org/pdf/1909.08053.pdf), [2](https://arxiv.org/pdf/2104.04473.pdf), and [3](https://arxiv.org/pdf/2205.05198))

To estimate the memory requirements for the activation portion, given that Megatron supports FlashAttention and Fusion computations, we have adopted a distinctive approach. This method involves collecting the memory address and size information of the corresponding operations each time the cudaMalloc and cudaFree functions are executed, and then conducting line-by-line analysis of this information to derive a computational formula. To implement this method, we used the [torch.cuda.CUDAPluggableAllocator](https://pytorch.org/docs/stable/notes/cuda.html#using-custom-memory-allocators-for-cuda) to customize the memory allocator.

We will observe the changes in [torch.cuda.max_memory_allocated](https://pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html) during the model training process, then summarize these changes in order to estimate peak memory.

# Limitations
* Supported
  * GPTModel
  * Tensor parallelism, Pipeline parallelism, Data parallelism
  * Using `--bf16`, `--fp16`,  `--use-flash-attn`, `--use-distributed-optimizer`, `--swiglu`
* To be supported
  * Other Transformer-based models
  * Using `--sequence-parallel`, `--num_layers_per_virtual_pipeline_stage`, ` --recompute-activations`
  * isable `--use-flash-attn`, `--use-distributed-optimizer`, `--swiglu`，`--bf16`
# Usage
In the [`examples`](./examples) directory, we've provided scripts to get pretraining GPT information. Users can generate their scripts by using the following command:
```sh {.line-numbers}
sed 's%torchrun $DISTRIBUTED_ARGS pretrain_gpt.py%python ../get_training_info.py $DISTRIBUTED_ARGS %g' pretrain_gpt_distributed_with_mp.sh > get_pretrain_gpt_distributed_with_mp_info.sh
```
The function of this command is to replace "torchrun \$DISTRIBUTED_ARGS pretrain_gpt.py" with "python ../get_training_info.py \$DISTRIBUTED_ARGS" in the "pretrain_gpt_distributed_with_mp.sh" which is your script for launching the training.

Moreover, we've added the following training parameters:

* --use-flash-attn
* --use-distributed-optimizer
* --swiglu
* --bf16

## Example of output
```sh {.line-numbers}
GPUS_PER_NODE=8
NNODES=2

GPT_ARGS="
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --num-layers 24 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 4 \
    --global-batch-size 512 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --use-flash-attn \
    --use-distributed-optimizer \
    --swiglu \
    --bf16
"
```
Assuming there are two nodes, each equipped with eight cards, and training a model according to the above configuration, the following output will be produced.


#### Full Model without Parallel
Full model information without parallel training enabled.
```sh {.line-numbers}
***Full Model without Parallel***
===========================================================================================================
Layer                                      Param.(shape)           Param.(Mem. MB)  Act.(Mem. MB)        
----------------------------------------------------------------------------------------------------------
GPTModel                                                         
├─TransformerLanguageModel                 
│    └─Embedding                                                    	               	96.0           	
│    │    └─word_embeddings                w=[50432,4096]           	394.0          	  
│    │    └─position_embeddings            w=[2048,4096]            	16.0           	
│    └─ParallelTransformer: X 32(layer_num)                                        	1320.0/layer   	
│    │    └─input_layernorm                w=[4096],b=[4096]        	0.0            	64.0           	
│    │    └─self_attention                                          	               	384.0          	
│    │    |     └─query_key_value          w=[12288,4096],b=[4096]  	96.0           	
│    │    |     └─rearrange                                         	               	192.0          	
│    │    |     └─core_attention_flash                              	               	64.0           	
│    │    |     └─rearrange                                         	               	64.0           	
│    │    |     └─dense                    w=[4096,4096],b=[4096]   	32.0           	64.0           	
│    │    └─post_attention_layernorm       w=[4096],b=[4096]        	0.0            	64.0           	
│    │    └─mlp                                                     	               	744.0          	
│    │    |     └─dense_h_to_4h            w=[21760,4096],b=[21760] 	170.0          	
│    │    |     └─bias_glue                                         	               	680.0          	
│    │    |     └─dense_4h_to_h            w=[4096,10880],b=[4096]  	85.0           	64.0           	
│    │    └─drop_add_fusion                                         	               	96.0           	
-----------------------------------------------------------------------------------------------------------
Amount of Parameters: 6,642,245,632  
Parameters: 12.4GB
Gradients: 24.7GB
Optimizers(Adam) States: 74.2GB
Activations: 44.8GB
Total memory demand: 156.2GB
==============================================================================================================
```


#### Cluster Communication Summary
Given the model and parallel configuration, the total communication count and volume for each Pipeline Parallel, Data Parallel, and Tensor Parallel dimension in a single iteration, as well as the total communication count and volume for the entire cluster in the final training iteration.
```sh {.line-numbers}
***Cluster Communication Summary***
==============================
Pipeline Parallelism
│    └─frequency/iteration: 2048
│    └─volume/iteration: 128.0 GB
Data Parallelism
│    └─frequency/iteration: 2
│    └─volume/iteration: 12.8 GB
Tensor Parallelism
│    └─frequency/iteration: 32768
│    └─volume/iteration: 2048.0 GB
All Communication
│    └─frequency/iteration: 34818
│    └─volume/iteration: 2188.8 GB
==============================

```

#### Memory demand on each GPU in the cluster
Given the model and parallel configuration, the memory requirements on each GPU in the cluster for training one iteration.
```sh {.line-numbers} 
***Memory demand on each GPU in the cluster***
==============================
Amount of Parameters: 1,718,898,688  
Parameters: 3.2GB
Gradients: 6.4GB
Optimizers(Adam) States: 4.8GB
Activations: 25.8GB
Memory Requirement: 40.2GB
==============================
```
#### Pipeline Parallel Communication
```sh {.line-numbers} 
***Pipeline Parallel Communications***
========================================================================================
GPTModel                                                         
├─TransformerLanguageModel                 
│    └─Embedding                                              
│    │    └─word_embeddings                
│    │    └─position_embeddings            
│    └─Stage0: ParallelTransformerLayer_Index0-15
│    │    └─stage_device_mappings
│    │    │      └─[n0_g0 n0_g1 n0_g2 n0_g3 n0_g4 n0_g5 n0_g6 n0_g7]
│    │    └─each single communication on each gpu
│    │    │    └─shape: [4,2048,4096]              
│    │    │    └─volume: 64.0MB         
│    │    │    └─func: isend, irecv
│    │    │    └─location: between stage in forward and backward process
│    │    └─each iteration communication on each gpu
│    │    │    └─frequency: 128 (num_gradient_accumulation_steps * 4)
│    │    │    └─volume: 8192.0MB       
│    └─Stage1: ParallelTransformerLayer_Index16-31
│    │    └─stage_device_mappings 
│    │    │      └─[n1_g0 n1_g1 n1_g2 n1_g3 n1_g4 n1_g5 n1_g6 n1_g7]

----------------------------------------------------------------------------------------
8 Pipeline Parallel Communication Groups:
│    └─[n0_g0 n1_g0]
│    └─[n0_g1 n1_g1]
│    └─[n0_g2 n1_g2]
│    └─[n0_g3 n1_g3]
│    └─[n0_g4 n1_g4]
│    └─[n0_g5 n1_g5]
│    └─[n0_g6 n1_g6]
│    └─[n0_g7 n1_g7]
All Communication of Cluster in Pipeline Parallelism
│    └─frequency/iteration: 2048
│    └─volume/iteration: 128.0GB
========================================================================================
```

#### Data Parallel Communications
```sh {.line-numbers} 
***Data Parallel Communications***
========================================================================================
GPTModel                                                         
├─each iteration                
│    └─synchronize_gradient                                         
│    │    └─4 Data Parallel Groups 
│    │    │    └─[n0_g0 n0_g2 n0_g4 n0_g6]
│    │    │    └─[n0_g1 n0_g3 n0_g5 n0_g7]
│    │    │    └─[n1_g0 n1_g2 n1_g4 n1_g6]
│    │    │    └─[n1_g1 n1_g3 n1_g5 n1_g7]
│    │    └─communication 
│    │    │    └─volume: 6.4GB
│    │    │    └─func: reduce_scatter (using DistributedOptimizer) 
│    │    └─frequency/iteration: 1
│    │    └─location: after forward_and_backward_compute * 32 times/iteration 
│    └─gather_model_param (using DistributedOptimizer)                                          
│    │    └─4 Data Parallel Groups 
│    │    │    └─[n0_g0 n0_g2 n0_g4 n0_g6]
│    │    │    └─[n0_g1 n0_g3 n0_g5 n0_g7]
│    │    │    └─[n1_g0 n1_g2 n1_g4 n1_g6]
│    │    │    └─[n1_g1 n1_g3 n1_g5 n1_g7]
│    │    └─communication on each gpu
│    │    │    └─volume: 6.4GB
│    │    │    └─func: all_gather
│    │    └─frequency/iteration: 1
│    │    └─location: after optimizer.iteration
----------------------------------------------------------------------------------------
All Communication of Cluster in Data Parallelism
│    └─frequency/iteration: 2
│    └─volume/iteration: 12.8GB
========================================================================================
```
#### Tensor Parallel Communications
```sh {.line-numbers} 
***Tensor Parallel Communications***
=================================================================================================================================================================================================================
Layer                                      Param(shape)           Param(Mem. MB)  Activations(Mem. MB)   TP_Fw.(Comm. Shape)  TP_Fw.(Comm. Mem. MB)   TP_Bw.(Comm. Shape)  TP_Bw.(Comm. Mem. MB)   TP(Comm. func)
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
GPTModel                                                         
├─TransformerLanguageModel                 
│    └─Embedding                                                    	               	96.0           	                    
│    │    └─word_embeddings                w=[25216],b=[4096]       	394.0          	               	[4,2048,4096]            	64.0           	                         	               	allreduce                	
│    │    └─position_embeddings            w=[2048],b=[4096]        	16.0           	
│    └─ParallelTransformer: X 16(layer_num)                                        	1320.0/layer   	
│    │    └─input_layernorm                w=[4096],b=[4096]        	0.0            	64.0           	
│    │    └─self_attention                                          	               	384.0          	
│    │    |     └─query_key_value          w=[6144,4096],b=[4096]   	48.0           	               	                         	               	[4,2048,4096]            	64.0           	allreduce                	
│    │    |     └─rearrange                                         	               	96.0           	
│    │    |     └─core_attention_flash                              	               	32.0           	
│    │    |     └─rearrange                                         	               	32.0           	
│    │    |     └─dense                    w=[2048,4096],b=[4096]   	16.0           	64.0           	[4,2048,4096]            	64.0           	                         	               	allreduce                	
│    │    └─post_attention_layernorm       w=[4096],b=[4096]        	0.0            	64.0           	
│    │    └─mlp                                                     	               	744.0          	
│    │    |     └─dense_h_to_4h            w=[10880,4096],b=[10880] 	85.0           	               	                         	               	[4,2048,4096]            	64.0           	allreduce                	
│    │    |     └─bias_glue                                         	               	680.0          	
│    │    |     └─dense_4h_to_h            w=[4096,5440],b=[4096]   	85.0           	64.0           	[4,2048,4096]            	64.0           	                         	               	allreduce                	
│    │    └─drop_add_fusion                                         	               	96.0           	
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
8 Tensor Parallel Communication Groups:
│    └─[n0_g0 n0_g1]
│    └─[n0_g2 n0_g3]
│    └─[n0_g4 n0_g5]
│    └─[n0_g6 n0_g7]
│    └─[n1_g0 n1_g1]
│    └─[n1_g2 n1_g3]
│    └─[n1_g4 n1_g5]
│    └─[n1_g6 n1_g7]
Communication in Tensor Parallel
│    └─each gpu:
│    │    └─each micro_batch:
│    │    │    └─frequency: 64
│    │    │    └─volume: 4.0GB
│    │    │    └─each transformer:
│    │    │    │    └─frequency: 2(forward)+2(backward)=4
│    │    │    │    └─volume: 0.25GB
│    │    └─each iteration:
│    │    │    └─frequency: 2048
│    │    │    └─volume: 128.0GB
│    └─cluster:
│    │    └─each micro_batch:
│    │    │    └─frequency: 1024
│    │    │    └─volume: 64.0GB
│    │    └─each iteration:
│    │    │    └─frequency: 32768
│    │    │    └─volume: 2048.0GB
=======================================================================================================================================================================================================================

```
