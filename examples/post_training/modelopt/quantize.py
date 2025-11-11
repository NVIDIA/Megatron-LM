# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate GPT."""

import functools
import os
import sys
import warnings

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import modelopt.torch.quantization as mtq
from modelopt.torch.export import import_mcore_gpt_from_hf

from megatron.core.transformer.moe.router import TopKRouter
from megatron.post_training.arguments import add_modelopt_args
from megatron.post_training.checkpointing import load_modelopt_checkpoint
from megatron.post_training.generate import simple_generate
from megatron.post_training.model_builder import modelopt_gpt_mamba_builder
from megatron.post_training.utils import report_current_memory_info
from megatron.training import get_args, get_model, get_tokenizer, initialize_megatron
from megatron.training.checkpointing import save_checkpoint
from megatron.training.utils import print_rank_0, unwrap_model
from model_provider import model_provider

warnings.filterwarnings("ignore")


QUANT_CFG_CHOICES = {
    "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
    "fp8": mtq.FP8_DEFAULT_CFG,
    "fp8_blockwise": mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
}


def add_text_generate_ptq_args(parser):
    """Add additional arguments for ModelOpt text generation PTQ."""
    group = parser.add_argument_group(title="ModelOpt text generation ptq")
    group.add_argument(
        "--calib-size", type=int, default=512, help="Samples to use for ptq calibration."
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
        "--pretrained-model-path", type=str, default=None, help="HuggingFace pretrained model"
    )
    group.add_argument(
        "--compress",
        action="store_true",
        help="Enable real low-bit quantization.",
    )
    group.add_argument(
        "--disable-qkv-quant",
        action="store_true",
        help="Disable q, k, v linear from being quantized.",
    )
    group.add_argument(
        "--weight-only",
        action="store_true",
        help="Disable input quantization.",
    )
    group.add_argument(
        "--force-all-expert-routing",
        action="store_true",
        help="Forcing all experts to be routed during the calibration.",
    )
    add_modelopt_args(parser)
    return parser


def check_arguments():
    """Checking user arguments."""
    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print_rank_0("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    if hasattr(args, "moe_grouped_gemm") and args.moe_grouped_gemm == True:
        print_rank_0("WARNING: Forcing moe_grouped_gemm to False for PTQ and export.")
        args.moe_grouped_gemm = False


def get_modelopt_torch_quantization_config():
    """Return a quantization config."""
    args = get_args()
    mtq_config = QUANT_CFG_CHOICES[args.export_quant_cfg]
    fp8_config = {"enable": True, "num_bits": (4, 3), "axis": None}
    fp4_config = {
        "num_bits": (2, 1),
        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        "axis": None,
        "enable": True,
    }
    # Disable mamba-mixer quantization for now.
    mtq_config["quant_cfg"]["*mixer.*"] = {"enable": False}
    if args.export_quant_cfg == "fp8":
        # Enable Medusa heads and kv-cache quantization
        mtq_config["quant_cfg"]["*medusa_heads**"] = fp8_config
    if "fp4" in args.export_quant_cfg:
        # Enable Medusa heads and kv-cache quantization
        mtq_config["quant_cfg"]["*medusa_heads**"] = fp4_config
    if "awq" in args.export_quant_cfg:
        weight_quantizer = mtq_config["quant_cfg"]["*weight_quantizer"]  # type: ignore
        if isinstance(weight_quantizer, list):
            weight_quantizer = weight_quantizer[0]
        weight_quantizer["block_sizes"][-1] = 128

    # Customization
    if args.disable_qkv_quant:
        mtq_config["quant_cfg"]["*self_attention*"] = {"enable": False}
    if args.export_kv_cache_quant and not args.compress:
        mtq_config["quant_cfg"]["*linear_qkv.output_quantizer"] = fp8_config
    if args.weight_only:
        mtq_config["quant_cfg"]["*input_quantizer"] = {"enable": False}

    return mtq_config


def get_calib_dataloader(calib_size=512, max_sequence_length=512):
    """Return a dataloader for calibration."""
    dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
    text_column = "article"

    calib_size = min(len(dataset), calib_size)
    for i in range(calib_size):
        yield dataset[i][text_column][:max_sequence_length]


if __name__ == "__main__":
    initialize_megatron(
        extra_args_provider=add_text_generate_ptq_args,
        args_defaults={
            "tokenizer_type": "HuggingFaceTokenizer",
            "no_load_rng": True,
            "no_load_optim": True,
        },
    )

    check_arguments()

    args = get_args()

    tokenizer = get_tokenizer()._tokenizer
    model = get_model(functools.partial(model_provider, modelopt_gpt_mamba_builder), wrap_with_ddp=False)

    report_current_memory_info()

    if args.load is not None:
        load_modelopt_checkpoint(model, strict=not args.untie_embeddings_and_output_weights)
        print_rank_0("Done loading checkpoint")

    if args.pretrained_model_path is not None:
        from modelopt.torch.export import import_mcore_gpt_from_hf
        import_dtype = torch.float16 if args.fp16 else torch.bfloat16
        unwrapped_model = unwrap_model(model)[0]
        workspace_dir = os.environ.get("MLM_WORK_DIR", "/tmp")
        import_mcore_gpt_from_hf(
            unwrapped_model,
            args.pretrained_model_path,
            workspace_dir,
            dtype=import_dtype,
        )

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

        if args.force_all_expert_routing:
            for name, module in model.named_modules():
                if isinstance(module, TopKRouter):
                    module.topk = module.num_experts

        for prompt in tqdm(dataloader, total=args.calib_size, disable=torch.distributed.get_rank()):
            tokens = tokenizer(prompt, return_tensors="pt")
            generated_ids = simple_generate(model, tokens.input_ids.cuda(), osl=1)

            if args.force_all_expert_routing:
                for name, module in model.named_modules():
                    if isinstance(module, TopKRouter):
                        module.topk = module.config.moe_router_topk

    unwrapped_model = unwrap_model(model)[0]

    if args.export_quant_cfg in QUANT_CFG_CHOICES:
        print_rank_0("Quantizing the model...")
        mtq_config = get_modelopt_torch_quantization_config()
        ptq_forward_loop_func = _hf_dataset_forword_loop_func

        if args.weight_only:
            mtq.quantize(unwrapped_model, mtq_config)
        elif hasattr(unwrapped_model, "calibration_mode"):
            unwrapped_model.calibration_mode = True
            mtq.quantize(unwrapped_model, mtq_config, ptq_forward_loop_func)
            unwrapped_model.calibration_mode = False
        else:
            mtq.quantize(unwrapped_model, mtq_config, ptq_forward_loop_func)

        if args.compress:
            mtq.compress(unwrapped_model)
            print_rank_0("Weights are now compressed to low-bit!")

    print_rank_0(f"Fake Quantized Model:\n {unwrapped_model}")

    if torch.distributed.get_rank() == 0:
        for k, v in unwrapped_model.state_dict().items():
            if "amax" not in k and "_scale" not in k:
                continue
            if isinstance(v, torch.Tensor):
                v_amax = torch.max(torch.abs(v.clone().detach().to(torch.bfloat16)))
                print("{:80} {:32} {:32} max {:.4e}".format(k, str(v.dtype), str(v.shape), v_amax))
            else:
                print("{:80}".format(k))

    _custom_prompt_forward_loop_func(unwrapped_model)

    if args.save is not None and args.export_quant_cfg in QUANT_CFG_CHOICES:
        save_checkpoint(1, model, None, None, 0, release=True)
