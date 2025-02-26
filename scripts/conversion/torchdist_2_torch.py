'''
This script converts a megatron torch-dist checkpoint to a megatron torch checkpoint.
Necessary before conversion to HF.
'''
from megatron.core.enums import ModelType
from megatron.training.training import setup_model_and_optimizer
from megatron.training.initialize import initialize_megatron
from megatron.training.global_vars import get_args
# from megatron.training.utils import unwrap_model
from pretrain_gpt import model_provider


def main():
    args_defaults = {
        "transformer_impl": "transformer_engine",
        "use_checkpoint_args": True,
        "ckpt_format": "torch_dist",
        "ckpt_convert_format": "torch",
        "no_load_rng": True,
        "no_load_optim": True,
        "no_save_optim": True,
        "--untie-embeddings-and-output-weights": True,
        # fake args
        "micro_batch_size": 1,
        "train_iters": 1,
        "lr": 0.0,
    }
    initialize_megatron(
        args_defaults=args_defaults,
    )
    args = get_args()
    assert args.load is not None, "You must specify --load"
    assert args.ckpt_convert_save is not None, "You must specify --ckpt-convert-save"
    setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)


if __name__ == "__main__":
    main()