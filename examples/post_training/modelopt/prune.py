# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Example script for pruning a GPT / Mamba model using TensorRT Model Optimizer (ModelOpt).

Read more about ModelOpt pruning at https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/pruning
"""

import functools
import os
import sys
import warnings

import torch
from datasets import load_dataset
from tqdm import tqdm

import modelopt.torch.prune as mtp
from modelopt.torch.export import import_mcore_gpt_from_hf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from megatron.post_training.arguments import add_modelopt_args
from megatron.post_training.checkpointing import load_modelopt_checkpoint
from megatron.post_training.generate import simple_generate
from megatron.post_training.model_provider import model_provider
from megatron.post_training.utils import report_current_memory_info
from megatron.training import get_args, get_model, get_tokenizer, initialize_megatron
from megatron.training.checkpointing import save_checkpoint
from megatron.training.utils import print_rank_0, unwrap_model

warnings.filterwarnings("ignore")


SUPPORTED_PRUNING_HPARAMS = {
    # Width pruning
    "ffn_hidden_size",
    "hidden_size",
    "num_attention_heads",
    "num_query_groups",
    "mamba_num_heads",
    "mamba_head_dim",
    # Depth pruning
    "num_layers",
}


def add_prune_args(parser):
    """Add additional arguments for ModelOpt pruning."""
    group = parser.add_argument_group(title="ModelOpt pruning")
    group.add_argument(
        "--calib-size",
        type=int,
        default=1024,
        help="Samples to use for pruning calibration.",
    )
    group.add_argument(
        "--prompts",
        type=str,
        default=("Hello!|Born in California, Soyer trained as a"),
        help="Input texts. Please use | to separate different batches.",
    )
    group.add_argument(
        "--references",
        type=str,
        default="",
        help="Reference texts. Please use | to separate different batches.",
    )
    group.add_argument(
        "--pretrained-model-path",
        type=str,
        default=None,
        help="HuggingFace pretrained model",
    )
    # Pruning parameters
    parser.add_argument(
        "--target-ffn-hidden-size",
        type=int,
        help="Prune MLP FFN hidden size to this value",
    )
    parser.add_argument(
        "--target-hidden-size",
        type=int,
        help="Prune hidden size (embedding dim) to this value",
    )
    parser.add_argument(
        "--target-num-attention-heads",
        type=int,
        help="Prune number of attention heads to this value. Must be supplied with --target-num-query-groups",
    )
    parser.add_argument(
        "--target-num-query-groups",
        type=int,
        help="Prune number of query groups to this value. Must be supplied with --target-num-attention-heads",
    )
    parser.add_argument(
        "--target-mamba-num-heads",
        type=int,
        help="Prune number of Mamba attention heads to this value",
    )
    parser.add_argument(
        "--target-mamba-head-dim",
        type=int,
        help="Prune dimension of Mamba attention heads to this value",
    )
    parser.add_argument(
        "--target-num-layers",
        type=int,
        help="Prune number of transformer layers to this value based on "
        "Block Influence metric (cosine similarity) as per https://arxiv.org/abs/2403.03853",
    )
    parser.add_argument(
        "--layers-to-drop",
        type=int,
        metavar="N",
        nargs="*",
        help="Drop specific model layers (1-indexed). Cannot be used with rest of the pruning options",
    )
    add_modelopt_args(parser)
    return parser


def check_arguments(args):
    """Checking user arguments."""
    if args.layers_to_drop:
        other_params = [
            args.target_ffn_hidden_size,
            args.target_hidden_size,
            args.target_num_attention_heads,
            args.target_num_query_groups,
            args.target_mamba_num_heads,
            args.target_mamba_head_dim,
            args.target_num_layers,
        ]
        if any(p is not None for p in other_params):
            raise ValueError("--layers_to_drop cannot be used with other pruning parameters")


def get_calib_dataloader(calib_size=1024, max_sequence_length=512):
    """Return a dataloader for calibration."""
    dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
    text_column = "article"

    calib_size = min(len(dataset), calib_size)
    for i in range(calib_size):
        yield dataset[i][text_column][:max_sequence_length]


if __name__ == "__main__":
    initialize_megatron(
        extra_args_provider=add_prune_args,
        args_defaults={
            "tokenizer_type": "HuggingFaceTokenizer",
            "no_load_rng": True,
            "no_load_optim": True,
        },
    )

    args = get_args()
    check_arguments(args)

    tokenizer = get_tokenizer()._tokenizer
    model = get_model(functools.partial(model_provider, parallel_output=True), wrap_with_ddp=False)
    unwrapped_model = unwrap_model(model)[0]

    report_current_memory_info()

    if args.load is not None:
        load_modelopt_checkpoint(model, strict=not args.untie_embeddings_and_output_weights)
        print_rank_0("Done loading checkpoint")

    if args.pretrained_model_path is not None:
        workspace_dir = os.environ.get("MLM_WORK_DIR", "/tmp")
        import_mcore_gpt_from_hf(unwrapped_model, args.pretrained_model_path, workspace_dir)

    def _custom_prompt_forward_loop_func(model):
        all_prompts = args.prompts.split("|")
        if args.references == "":
            all_references = [None] * len(all_prompts)
        else:
            all_references = args.references.split("|")

        for idx, prompt in tqdm(enumerate(all_prompts), disable=torch.distributed.get_rank()):
            tokens = tokenizer(prompt, return_tensors="pt")
            generated_ids = simple_generate(model, tokens.input_ids.cuda(), osl=32)
            generated_texts = tokenizer.batch_decode(generated_ids)
            print_rank_0("{}".format(generated_texts))
            if all_references[idx] is not None:
                assert all_references[idx] == generated_texts[0], all_references[idx]

    def _hf_dataset_forword_loop_func(model):
        dataloader = get_calib_dataloader(args.calib_size)

        for prompt in tqdm(dataloader, total=args.calib_size, disable=torch.distributed.get_rank()):
            tokens = tokenizer(prompt, return_tensors="pt")
            simple_generate(model, tokens.input_ids.cuda(), osl=1)

    if args.layers_to_drop:
        mtp.plugins.drop_mcore_language_model_layers(model, layers_to_drop=args.layers_to_drop)
    else:
        print_rank_0("Pruning model...")
        export_config = {
            k: getattr(args, f"target_{k}")
            for k in SUPPORTED_PRUNING_HPARAMS
            if getattr(args, f"target_{k}") is not None
        }
        mtp.prune(
            unwrapped_model,
            mode="mcore_minitron",
            constraints={"export_config": export_config},
            dummy_input=None,  # Not used
            config={"forward_loop": _hf_dataset_forword_loop_func},
        )
    print_rank_0(f"Pruned Model:\n {unwrapped_model}")

    _custom_prompt_forward_loop_func(unwrapped_model)

    if args.save is not None:
        save_checkpoint(1, model, None, None, 0)

    print_rank_0("Done")
