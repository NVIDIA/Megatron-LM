import os
import torch
from megatron.core import parallel_state
from megatron.core import dist_checkpointing
from megatron.core.export.model_type import ModelType
from megatron.core.export.data_type import DataType
from megatron.core.export.export_config import ExportConfig
from megatron.core.export.trtllm.trtllm_helper import TRTLLMHelper
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec


_SEQUENCE_LENGTH = 64


def initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1):
    parallel_state.destroy_model_parallel()

    # Torch setup for distributed training
    rank = int(os.environ['LOCAL_RANK'])
    world_size = torch.cuda.device_count()
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(world_size=world_size, rank=rank)

    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size)

def model_provider():
    """Build the model."""

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

    return gpt_model

def load_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict=gpt_model.sharded_state_dict(prefix='')
    checkpoint = dist_checkpointing.load(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)
    gpt_model.load_state_dict(checkpoint)
    return gpt_model

if __name__ == "__main__":
    # Need to use TP1 PP1 for export on single device
    initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)

    gpt_model = model_provider()

    # Optionally you can also load a gpt model from ckpt_path using this code below
    # gpt_model = load_distributed_checkpoint(gpt_model=gpt_model, checkpoint_path=ckpt_path)

    seq_len_interpolation_factor = None
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
    

    export_config = ExportConfig(inference_tp_size = 2)
    # NOTE : For faster performance, if your entire model will fit in gpu memory, transfer model state dict to GPU and then call this api
    weight_list, config_list = trtllm_helper.get_trtllm_pretrained_config_and_model_weights(
        model_state_dict= gpt_model.state_dict(),
        dtype = DataType.bfloat16,
        export_config=export_config
    )

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