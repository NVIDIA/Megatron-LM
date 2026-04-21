# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""GPT zero-shot evaluation."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))

from megatron.training import get_args
from megatron.elastification.arguments import add_flextron_args
from megatron.training.initialize import initialize_megatron
from megatron.training import print_rank_0
from megatron.training.checkpointing import (
    load_checkpoint,
    save_checkpoint
)
from megatron.core.utils import get_model_config
from megatron.training import get_model
from megatron.training.training import (
    evaluate_and_print_results,
    update_train_iters,
    build_train_valid_test_data_iterators,
)



def main():
    """Main program."""
    args = get_args()

    assert args.hybrid_layer_pattern is not None, "Hybrid override pattern is required"
    
    if args.hybrid_layer_pattern is not None:
        from functools import partial
        from pretrain_mamba import (
            train_valid_test_datasets_provider,
            model_provider,
            forward_step
        )
        from mamba_builders import mamba_builder
        model_provider = partial(model_provider, mamba_builder)
    else:
        from pretrain_gpt import (
            train_valid_test_datasets_provider,
            model_provider,
            forward_step
        )

    def get_model_provider():
        return model_provider

    # Set up model and load checkpoint.
    model = get_model(get_model_provider(), wrap_with_ddp=False)
    config = get_model_config(model[0])
    if args.hybrid_layer_pattern is not None and getattr(config, 'flextron', False):
        from megatron.elastification.flextron_utils import setup_flextron_model, inject_flextron_forward_logic

        setup_flextron_model(model[0].module)
        inject_flextron_forward_logic(model[0].module)

    if args.load is not None:

        iteration, num_floating_point_operations_so_far = load_checkpoint(model,
                        None,
                        None,
                        strict=True)
    import torch

    assert len(model) == 1, "Above condition should have caught this"
    args.iteration = 0
    args.curr_iteration = 0
    
    # Import the parameter calculation function
    from megatron.elastification.router.flex_budget_utils import get_num_parameters
    
    model[0].eval()
    for override_budget in args.override_selected_budget:
        with torch.no_grad():
            mlp_forward_outputs, skipping_forward_outputs, emb_forward_outputs, mamba_forward_outputs, head_forward_outputs, moe_expert_forward_outputs = model[0].module.router.forward(override_budget)

        # Extract router outputs and convert to actual model dimensions
        flex_ffn_hidden_size = torch.tensor(mlp_forward_outputs[-1]) * config.ffn_hidden_size
        flex_hidden_size = torch.tensor(emb_forward_outputs[-1]) * config.hidden_size
        flex_mamba_num_head = torch.tensor(mamba_forward_outputs[-1]) * config.mamba_num_heads
        flex_num_attention_heads = torch.tensor(head_forward_outputs[-1]) * config.num_attention_heads
        flex_moe_expert = torch.tensor(moe_expert_forward_outputs[-1]) * config.num_moe_experts
        
        if not model[0].module.config.flex_hetero_ffn and not model[0].module.config.add_skipping:
            flex_ffn_hidden_size = flex_ffn_hidden_size.unsqueeze(-1)
        if not model[0].module.config.flex_hetero_mamba and not model[0].module.config.add_skipping:
            flex_mamba_num_head = flex_mamba_num_head.unsqueeze(-1)
        if not model[0].module.config.flex_hetero_head and not model[0].module.config.add_skipping:
            flex_num_attention_heads = flex_num_attention_heads.unsqueeze(-1)
        if not model[0].module.config.flex_hetero_moe_expert and not model[0].module.config.add_skipping:
            flex_moe_expert = flex_moe_expert.unsqueeze(-1)

        # Calculate current parameter count based on router outputs

        num_mlp_layers = model[0].module.hybrid_layer_pattern.count('-') - 1
        if flex_ffn_hidden_size.shape[0] != 1 and model[0].module.config.flex_hetero_ffn:
            flex_ffn_hidden_size = torch.cat([torch.tensor([1.0*model[0].module.config.ffn_hidden_size]).to(flex_ffn_hidden_size.device, flex_ffn_hidden_size.dtype), flex_ffn_hidden_size], dim=0)
        if flex_moe_expert.shape[0] != 1 and model[0].module.config.flex_hetero_moe_expert:
            flex_moe_expert = torch.cat([torch.tensor([1.0*model[0].module.config.num_moe_experts]).to(flex_moe_expert.device, flex_moe_expert.dtype), flex_moe_expert], dim=0)

        total_param, active_param = get_num_parameters(
                hybrid_pattern=model[0].module.hybrid_layer_pattern,
                mamba_num_heads=flex_mamba_num_head.float(),
                mamba_d_head=config.mamba_head_dim,
                mamba_d_state=config.mamba_state_dim,
                num_attention_heads=flex_num_attention_heads.float(),
                num_query_groups=config.num_query_groups,
                ffn_hidden_size=flex_ffn_hidden_size.float(),
                hidden_size=flex_hidden_size.float(),
                kv_channels=config.kv_channels,
                vocab_size=model[0].module.vocab_size,
                tied_vocab=model[0].module.share_embeddings_and_output_weights,
                num_experts=flex_moe_expert.float(),
                moe_shared_expert_expand=config.moe_shared_expert_intermediate_size/config.ffn_hidden_size,
                moe_router_topk=config.moe_router_topk
            )
            
        processed_router_output = {
            'flex_ffn_hidden_size': flex_ffn_hidden_size.flatten().int().tolist(),
            'flex_hidden_size': flex_hidden_size.flatten().int().tolist(),
            'flex_mamba_num_head': flex_mamba_num_head.flatten().int().tolist(),
            'flex_num_attention_heads': flex_num_attention_heads.flatten().int().tolist(),
            'flex_moe_expert': flex_moe_expert.flatten().int().tolist(),
        }
        if args.add_skipping:
            processed_router_output['router_skip'] = skipping_forward_outputs[-1]
        
        if torch.distributed.get_rank() == 0:
            print(f"Override budget: {override_budget}")
            print(f"processed_router_output: {processed_router_output}")
            print(f"Total parameter count: {total_param.item()/1e9} B")
            print(f"Active parameter count: {active_param.item()/1e9} B")
            print("="*100)
    
if __name__ == '__main__':
    initialize_megatron(extra_args_provider=add_flextron_args)
    main()
