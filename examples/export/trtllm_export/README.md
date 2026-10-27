# Megatron Core To TRTLLM Export Documentation
This guide will walk you through how you can use the megatron core export for exporting models to trtllm format

### Contents
- [Megatron Core To TRTLLM Export Documentation](#megatron-core-to-trtllm-export-documentation)
- [Contents](#contents)
  - [1. Quick Start](#1-quick-start)
    - [1.1 Understanding The Code](#11-understanding-the-code)
    - [1.2 Running The Code](#12-running-the-code)
  - [2. GPU Export](#2-gpu-export)
  - [3. Future work](#4-future-work)

#### 1. Quick Start
This will walk you through the flow of converting an mcore gpt model to trtllm format using single device mode. The file can be found at [gpt_single_device_cpu_export.py](./single_device_export/gpt_single_device_cpu_export.py)

NOTE: For faster performance, if your entire model will fit into gpu memory, pre transfer the model state dict to gpu and then call the get_trtllm_pretrained_config_and_model_weights function.

<br>

##### 1.1 Understanding The Code
***STEP 1 - We initialize model parallel and other default arguments***
We initalize tp and pp to 1 so that we can get the full model state dict on cpu
```python
    initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
```

***STEP 2 - We load the model using the model_provider_function***
NOTE: We create a simple gpt model

```python
    transformer_config = TransformerConfig(
        num_layers=2, 
        hidden_size=64, # Needs to be atleast 32 times num_attn_heads
        num_attention_heads=2, 
        use_cpu_initialization=True, 
        pipeline_dtype=torch.float32,
    )

    gpt_model = GPTModel(
        config=transformer_config, 
        transformer_layer_spec=get_gpt_layer_local_spec(), 
        vocab_size=100, 
        max_sequence_length=_SEQUENCE_LENGTH,
    )

    # Optionally you can also load a model using this code 
    # sharded_state_dict=gpt_model.sharded_state_dict(prefix='')
    # checkpoint = dist_checkpointing.load(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)
    # gpt_model.load_state_dict(checkpoint)

```

***STEP 3 - Instantiate the TRTLLM Helper***
We instantiate the [TRTLLM Helper](../../../megatron/core/export/trtllm/trtllm_helper.py)  For the GPT model we instantiate trtllm_helper as shown below.
```python
    if hasattr(gpt_model, "rotary_pos_emb"):
        seq_len_interpolation_factor =  gpt_model.rotary_pos_emb.seq_len_interpolation_factor

    trtllm_helper = TRTLLMHelper(
                        transformer_config=gpt_model.config, 
                        model_type=ModelType.gpt,
                        position_embedding_type = gpt_model.position_embedding_type, 
                        max_position_embeddings = gpt_model.max_position_embeddings, 
                        rotary_percentage = gpt_model.rotary_percent,
                        rotary_base = gpt_model.rotary_base,
                        moe_tp_mode = 2,
                        multi_query_mode = False,
                        activation = "gelu", 
                        seq_len_interpolation_factor = seq_len_interpolation_factor,
                        share_embeddings_and_output_weights=gpt_model.share_embeddings_and_output_weights
                    )   
```

***STEP 4 - Get the TRTLLM Weights and configs***
To convert model weights to trtllm weights and configs, we use the [single_device_converter](../../../megatron/core/export/trtllm/trtllm_weights_converter/single_device_trtllm_model_weights_converter.py). We pass as inputs the model state dict, and export config. In this example we use inference tp size as 2 for the export. 

```python
    model_state_dict={}
    for key , val in gpt_model.state_dict().items():
        # val is non for _extra_state layers . We filter it out
        if val is not None:
            model_state_dict[key] = val

    export_config = ExportConfig(inference_tp_size = 2)
    weight_list, config_list = trtllm_helper.get_trtllm_pretrained_config_and_model_weights(
        model_state_dict= model_state_dict,
        dtype = DataType.bfloat16,
        export_config=export_config
    )
```

***STEP 5 - Build the TRTLLM Engine***
Following code is used to build the TRTLLM Engine. 

```python
    for trtllm_model_weights, trtllm_model_config in zip(weight_list, config_list):
        trtllm_helper.build_and_save_engine(
            max_input_len=256,
            max_output_len=256,
            max_batch_size=8,
            engine_dir='/opt/megatron-lm/engine',
            trtllm_model_weights=trtllm_model_weights,
            trtllm_model_config=trtllm_model_config,
            lora_ckpt_list=None,
            use_lora_plugin=None,
            max_lora_rank=64,
            lora_target_modules=None,
            max_prompt_embedding_table_size=0,
            paged_kv_cache=True,
            remove_input_padding=True,
            paged_context_fmha=False,
            use_refit=False,
            max_num_tokens=None,
            max_seq_len=512,
            opt_num_tokens=None,
            max_beam_width=1,
            tokens_per_block=128,
            multiple_profiles=False,
            gpt_attention_plugin="auto",
            gemm_plugin="auto",
        )
```
<br>

##### 1.2 Running The Code
An example run script is shown below. 

```
# In a workstation 
MLM_PATH=/path/to/megatron-lm
CONTAINER_IMAGE=gitlab-master.nvidia.com:5005/dl/joc/nemo-ci/trtllm_0.12/train:pipe.17669124-x86

docker run -it --gpus=all --ipc=host -v $MLM_PATH/:/opt/megatron-lm $CONTAINER_IMAGE bash

# Inside the container run the following. 

cd /opt/megatron-lm/

CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1  examples/export/trtllm_export/single_device_export/gpt_single_device_cpu_export.py
```

<br>

#### 2. GPU Export
You can use the [gpt_distributed_gpu_export.py](./distributed_export/gpt_distributed_gpu_export.py) to run a more optimized on device distributed. version of trtllm export. Internally this uses the [distributed_converter](../../../megatron/core/export/trtllm/trtllm_weights_converter/distributed_trtllm_model_weights_converter.py) to convert model weights on device. 
In the single device version you collect all the model weights on CPU/GPU, convert it to trtllm format, and then store the engine back on disk. In the GPU version you load each individual state dict on the gpus, convert it on the device itself and store the engine on disk. 

To run the gpu version 

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2  examples/export/trtllm_export/distributed_export/gpt_distributed_gpu_export.py
```

<br>

#### 3. Future work
The following are planned for the future releases . 
* Pipeline parallellism for export (Work in progress) 
* GPU Export for more models (Work in progress for some models)
* Refit functionality
* VLLM Support