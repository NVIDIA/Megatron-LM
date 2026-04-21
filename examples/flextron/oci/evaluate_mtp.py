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
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.transformer import TransformerConfig
from megatron.core.models.mamba import MambaModel
import modelopt.torch.prune as mtp
# Import the mcore_minitron plugin to register the pruning mode
import modelopt.torch.prune.plugins.mcore_minitron

from pretrain_mamba import (
            train_valid_test_datasets_provider,
            count_parameters_in_layer,
            forward_step
        )

def add_pruning_args(parser):
    """Add pruning-specific arguments."""
    group = parser.add_argument_group(title='pruning')
    
    group.add_argument('--prune-hidden-size', type=int, default=None,
                       help='Hidden size for pruned model')
    group.add_argument('--prune-num-moe-experts', type=int, default=None,
                       help='Number of MoE experts for pruned model')
    group.add_argument('--prune-moe-shared-expert-intermediate-size', type=int, default=None,
                       help='MoE shared expert intermediate size for pruned model')
    group.add_argument('--prune-ffn-hidden-size', type=int, default=None,
                       help='FFN hidden size for pruned model')
    group.add_argument('--prune-moe-ffn-hidden-size', type=int, default=None,
                       help='MoE FFN hidden size for pruned model')
    group.add_argument('--prune-mamba-head-dim', type=int, default=None,
                       help='Mamba head dimension for pruned model')
    group.add_argument('--prune-mamba-num-heads', type=int, default=None,
                       help='Number of Mamba heads for pruned model')
    group.add_argument('--prune-skip-sorting', action='store_true', default=False,
                       help='Skip sorting during pruning')
    group.add_argument('--prune-scores-path', type=str, default=None,
                       help='Path to pruning scores file')
    group.add_argument('--prune-drop-layers', nargs='+', type=int, default=None,
                       help='Layers to drop for pruned model')
    
    return parser


def evaluate_mtp_extra_args_provider(parser):
    """Pruning args plus Flextron CLI when not already on the parser."""
    parser = add_pruning_args(parser)
    if not any(getattr(action, "dest", None) == "flextron" for action in parser._actions):
        parser = add_flextron_args(parser)
    return parser


def model_provider(pre_process=True, post_process=True, vp_stage = None):
    """Builds the model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        MambaModel: The returned model
    """
    args = get_args()

    print_rank_0('building Mamba model ...')
    config = core_transformer_config_from_args(args, TransformerConfig)

    assert args.use_legacy_models == False, "Mamba only supported in Mcore!"
    from megatron.core.post_training.modelopt.mamba.model_specs import get_mamba_stack_modelopt_spec
    mamba_stack_spec = get_mamba_stack_modelopt_spec(local_core_attention=False, 
                                                    remap_te_layernorm=True)

    model = MambaModel(
        config=config,
        mamba_stack_spec=mamba_stack_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        hybrid_layer_pattern=args.hybrid_layer_pattern,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        vp_stage=vp_stage
    )

    for l in range(model.decoder.num_layers_per_pipeline_rank):
        layer_params = count_parameters_in_layer(model, f'decoder.layers.{l}.')
        print_rank_0(f" == params layer {l}: {layer_params}")

    return model

def main():
    """Main program."""
    args = get_args()

    assert args.hybrid_layer_pattern is not None, "Hybrid override pattern is required"

    def get_model_provider():
        return model_provider

    # Set up model and load checkpoint.
    model = get_model(get_model_provider(), wrap_with_ddp=False)
    config = get_model_config(model[0])

    # if args.hybrid_layer_pattern is not None and getattr(config, 'flextron', False):
    #     from flextron_utils import setup_flextron_model, inject_flextron_forward_logic
    #     from flextron_elasticity_hooks import apply_flextron_elasticity_to_model
        
    #     setup_flextron_model(model[0].module)
    #     apply_flextron_elasticity_to_model(model[0].module, config)
    #     inject_flextron_forward_logic(model[0].module)

    if args.load is not None:

        iteration, num_floating_point_operations_so_far = load_checkpoint(model,
                        None,
                        None,
                        strict=True)
    import torch

    assert len(model) == 1, "Above condition should have caught this"

    # Data stuff.
    args.iteration = 0
    args.curr_iteration = 0
    update_train_iters(args)
    train_valid_test_datasets_provider.is_distributed = True
    train_iterator, _, _ = build_train_valid_test_data_iterators(train_valid_test_datasets_provider)

    # Run evaluation.
    def forward_loop(model):
        prefix = f'iteration {args.iteration}'
        evaluate_and_print_results(prefix=prefix,
                                forward_step_func=forward_step,
                                data_iterator=train_iterator,
                                model=[model],
                                iteration=args.iteration,
                                process_non_loss_data_func=None,
                                config=config,
                                verbose=True,
                                write_to_tensorboard=False)

    # Specify the pruning constraints

    export_config = {
        "hidden_size": args.prune_hidden_size,
        "num_moe_experts": args.prune_num_moe_experts,
        "moe_shared_expert_intermediate_size": args.prune_moe_shared_expert_intermediate_size,
        "ffn_hidden_size": args.prune_ffn_hidden_size,
        "moe_ffn_hidden_size": args.prune_moe_ffn_hidden_size,
        "mamba_head_dim": args.prune_mamba_head_dim,
        "mamba_num_heads": args.prune_mamba_num_heads
    }


    # Run the pruning process
    output = mtp.prune(
        model[0],
        mode="mcore_minitron",
        constraints={"export_config": export_config},
        dummy_input=None,  # Not used
        config={"forward_loop": forward_loop, "skip_sorting": args.prune_skip_sorting,
        "scores_path": args.prune_scores_path},
    )
    

    if args.prune_drop_layers:
        from modelopt.torch.nas.plugins.megatron import drop_mcore_language_model_layers
        drop_mcore_language_model_layers(model[0], layers_to_drop=args.prune_drop_layers) # 1-indexed layer number


    from modelopt.torch.opt import ModeloptStateManager

    if ModeloptStateManager.has_state_for_mode_type("prune", model=model[0]):
        try:
            ModeloptStateManager.remove_state(model[0])
        except:
            pass


    # Save the fresh checkpoint with new configuration
    print_rank_0('done :-)')
    if args.save !=  None:
        save_checkpoint(1, model, None, None, 0, {})  
        torch.distributed.barrier()
        import sys
        sys.exit()
    

if __name__ == '__main__':
    initialize_megatron(extra_args_provider=evaluate_mtp_extra_args_provider)
    main()
