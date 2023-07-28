import torch
import deepspeed
import megatron
from megatron import get_args
from megatron.core import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.arguments import core_transformer_config_from_args
from megatron.text_generation_utils import generate_samples_eval


def model_provider(pre_process=True, post_process=True):

    config = core_transformer_config_from_args(get_args())

    model = GPTModel(
        config=config,
        num_tokentypes=0,
        parallel_output=False,
        pre_process=pre_process,
        post_process=post_process,
        return_moe_loss=False,
    )
    return model


def add_text_generate_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title="text generation")

    group.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature."
    )
    group.add_argument(
        "--greedy", action="store_true", default=False, help="Use greedy sampling."
    )
    group.add_argument("--top_p", type=float, default=0.0, help="Top p sampling.")
    group.add_argument("--top_k", type=int, default=0, help="Top k sampling.")
    group.add_argument(
        "--out-seq-length",
        type=int,
        default=1024,
        help="Size of the output generated text.",
    )
    group.add_argument(
        "--sample-input-file",
        type=str,
        default=None,
        help="Get input from file instead of interactive mode, "
        "each line is an input.",
    )
    group.add_argument(
        "--sample-output-file",
        type=str,
        default=None,
        help="Output file got from --sample-input-file",
    )
    group.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of samples to generate unconditionally, "
        "defaults to 0 and interactive conditional sampling",
    )
    group.add_argument(
        "--genfile", type=str, help="Output file when generating unconditionally"
    )
    group.add_argument(
        "--recompute",
        action="store_true",
        help="During generation recompute all attention "
        "instead of using previously computed keys/values.",
    )
    group.add_argument(
        "--context-tokens", type=str, default="DeepSpeed is the greatest"
    )
    group.add_argument("--max-tokens", type=int, default=50)

    return parser


if __name__ == "__main__":
    # initialize megatron
    initialize_megatron(
        extra_args_provider=add_text_generate_args,
        args_defaults={
            "tokenizer_type": "GPT2BPETokenizer",
            "no_load_rng": True,
            "no_load_optim": True,
        },
    )
    args = get_args()

    # setup model
    model = get_model(model_provider)
    _ = load_checkpoint(model, None, None)
    model = model[0]
    if args.ds_inference:
        engine = deepspeed.init_inference(
            model=model,
            mp_size=args.tensor_model_parallel_size,
            tensor_parallel={"mpu": mpu},
            dtype=torch.half,
            replace_with_kernel_inject=True,
            moe_experts=args.num_experts,
            moe_type=args.mlp_type,
        )
        model = engine.module

    # generate output
    generate_samples_eval(
        model, args.context_tokens, 1, 0
    )  # Just so we don't get log output from DeepSpeed (this should be removed once we improve logging in DeepSpeed)
    print("===START OUTPUT===")
    print(generate_samples_eval(model, args.context_tokens, args.max_tokens, 0))
    print("===END OUTPUT===")
