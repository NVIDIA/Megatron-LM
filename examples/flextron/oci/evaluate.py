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
    if args.hybrid_layer_pattern is not None and getattr(args, 'flextron', False):
        from megatron.elastification.flextron_utils import setup_flextron_model, inject_flextron_forward_logic

        setup_flextron_model(model[0].module)
        inject_flextron_forward_logic(model[0].module)

    if args.load is not None:

        iteration, num_floating_point_operations_so_far = load_checkpoint(model,
                        None,
                        None,
                        strict=True)
    assert len(model) == 1, "Above condition should have caught this"

    # Data stuff.
    args.iteration = 0
    args.curr_iteration = 0
    update_train_iters(args)
    train_valid_test_datasets_provider.is_distributed = True
    _, valid_iterator, _ = build_train_valid_test_data_iterators(train_valid_test_datasets_provider)

    # Run evaluation.
    prefix = f'iteration {args.iteration}'
    evaluate_and_print_results(prefix=prefix,
                               forward_step_func=forward_step,
                               data_iterator=valid_iterator,
                               model=model,
                               iteration=args.iteration,
                               process_non_loss_data_func=None,
                               config=config,
                               verbose=True,
                               write_to_tensorboard=False)
    print_rank_0('done :-)')
    if args.save !=  None:
        save_checkpoint(iteration, model, None, None, num_floating_point_operations_so_far, {})  

if __name__ == '__main__':
    initialize_megatron(extra_args_provider=add_flextron_args)
    main()
