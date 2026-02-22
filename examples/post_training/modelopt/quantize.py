# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate GPT."""

import copy
import functools
import inspect
import json
import os
import random
import sys
import warnings

import modelopt.torch.quantization as mtq
import torch
import torch.distributed
from modelopt.torch.export import import_mcore_gpt_from_hf
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader
from tqdm import tqdm

try:
    import modelopt.torch.quantization.plugins.psx_formats as mtq_psx
except ImportError:
    mtq_psx = None
    warnings.warn(
        "psx_formats is not installed. PSX formats quantization configs will not be available."
    )
try:
    import modelopt.torch.quantization.plugins.luts as mtq_luts
except ImportError:
    mtq_luts = None
    warnings.warn("luts is not installed. LUTs quantization configs will not be available.")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from megatron.core.utils import get_batch_on_this_cp_rank
from megatron.post_training.arguments import add_modelopt_args
from megatron.post_training.checkpointing import load_modelopt_checkpoint
from megatron.post_training.generate import simple_generate
from megatron.post_training.model_builder import modelopt_gpt_mamba_builder
from megatron.post_training.utils import (
    print_distributed_quant_summary,
    report_current_memory_info,
)
from megatron.training import get_args, get_model, get_tokenizer, initialize_megatron
from megatron.training.checkpointing import save_checkpoint
from megatron.training.utils import print_rank_0, unwrap_model
from model_provider import model_provider

warnings.filterwarnings("ignore")

QUANT_CFG_CHOICES = {}

# Auto-load all quant configs by full name
for k in mtq.config.choices:
    QUANT_CFG_CHOICES[k] = getattr(mtq, k)

KV_QUANT_CFG_CHOICES = {
    "none": "none",
    "fp8": "FP8_KV_CFG",
    "fp8_affine": "FP8_AFFINE_KV_CFG",
    "nvfp4": "NVFP4_KV_CFG",
    "nvfp4_affine": "NVFP4_AFFINE_KV_CFG",
    "nvfp4_rotate": "NVFP4_KV_ROTATE_CFG",
}

if mtq_psx is not None:
    QUANT_CFG_CHOICES.update({k: getattr(mtq_psx, k) for k in mtq_psx.choices})

if mtq_luts is not None:
    QUANT_CFG_CHOICES.update({k: getattr(mtq_luts, k) for k in mtq_luts.choices})


def add_text_generate_ptq_args(parser):
    """Add additional arguments for ModelOpt text generation PTQ."""
    group = parser.add_argument_group(title="ModelOpt text generation ptq")
    group.add_argument(
        "--calib-size", type=int, default=512, help="Number of samples to use for ptq calibration."
    )
    group.add_argument(
        "--calib-dataset-path-or-name",
        type=str,
        default="cnn_dailymail",
        help="Path to local calibration dataset file (.jsonl) or HuggingFace dataset name.",
    )
    group.add_argument(
        "--calib-max-sequence-length",
        type=int,
        default=512,
        help="Maximum sequence length for calibration.",
    )
    group.add_argument(
        "--calib-use-random-offset",
        action="store_true",
        help="Use random offsets when slicing sequences for calibration. (Only for local files)",
    )
    group.add_argument(
        "--calib-batch-size", type=int, default=1, help="Batch size for calibration."
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
    group.add_argument("--compress", action="store_true", help="Enable real low-bit quantization.")
    group.add_argument(
        "--disable-qkv-quant",
        action="store_true",
        help="Disable q, k, v linear from being quantized.",
    )
    group.add_argument("--weight-only", action="store_true", help="Disable input quantization.")
    group.add_argument(
        "--force-all-expert-routing",
        action="store_true",
        help="Forcing all experts to be routed during the calibration.",
    )
    group.add_argument(
        "--num-first-layers-to-skip-quant",
        type=int,
        default=None,
        help="Number of first layers to skip quantization.",
    )
    group.add_argument(
        "--num-last-layers-to-skip-quant",
        type=int,
        default=None,
        help="Number of last layers to skip quantization.",
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


def _is_first_layers(name: str, num_layers: int = 1, num_layers_to_disable: int = 1) -> bool:
    if "layers." not in name:
        return False
    try:
        layer_idx = int(name.split("layers.")[-1].split(".")[0])
    except ValueError:
        return False
    return layer_idx < num_layers_to_disable


def _is_last_layers(name: str, num_layers: int = 1, num_layers_to_disable: int = 1) -> bool:
    if "layers." not in name:
        return False
    try:
        layer_idx = int(name.split("layers.")[-1].split(".")[0])
    except ValueError:
        return False
    return layer_idx >= num_layers - num_layers_to_disable


def get_first_layers_disabled_config(config, num_layers: int = 1, num_layers_to_disable: int = 1):
    """Get a config for `mtq.quantize` with first & last `num_layers_to_disable` layers disabled.

    The layers to disable are the first & last `num_layers_to_disable` layers.
    """
    config = copy.deepcopy(config)
    quant_cfg = config.get("quant_cfg", {})
    quant_cfg.update(
        {
            functools.partial(
                _is_first_layers, num_layers=num_layers, num_layers_to_disable=num_layers_to_disable
            ): {"enable": False}
        }
    )
    config["quant_cfg"] = quant_cfg
    return config


def get_last_layers_disabled_config(config, num_layers: int = 1, num_layers_to_disable: int = 1):
    """Get a config for `mtq.quantize` with last `num_layers_to_disable` layers disabled.

    The layers to disable are the last `num_layers_to_disable` layers.
    """
    config = copy.deepcopy(config)
    quant_cfg = config.get("quant_cfg", {})
    quant_cfg.update(
        {
            functools.partial(
                _is_last_layers, num_layers=num_layers, num_layers_to_disable=num_layers_to_disable
            ): {"enable": False}
        }
    )
    config["quant_cfg"] = quant_cfg
    return config


def get_modelopt_torch_quantization_config():
    """Return a quantization config."""
    args = get_args()
    if args.export_quant_cfg not in QUANT_CFG_CHOICES:
        raise ValueError(f"Unsupported quantization config {args.export_quant_cfg}.")
    mtq_config = QUANT_CFG_CHOICES[args.export_quant_cfg]

    fp8_config = {"enable": True, "num_bits": (4, 3), "axis": None}
    fp4_config = {
        "num_bits": (2, 1),
        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        "axis": None,
        "enable": True,
    }
    if args.export_quant_cfg == "FP8_DEFAULT_CFG":
        # Enable Medusa heads and kv-cache quantization
        mtq_config["quant_cfg"]["*medusa_heads**"] = fp8_config
    if "FP4" in args.export_quant_cfg:
        # Enable Medusa heads and kv-cache quantization
        mtq_config["quant_cfg"]["*medusa_heads**"] = fp4_config
    if "AWQ" in args.export_quant_cfg:
        weight_quantizer = mtq_config["quant_cfg"]["*weight_quantizer"]  # type: ignore
        if isinstance(weight_quantizer, list):
            weight_quantizer = weight_quantizer[0]
        weight_quantizer["block_sizes"][-1] = 128

    # Customization
    if args.disable_qkv_quant:
        mtq_config["quant_cfg"]["*self_attention*"] = {"enable": False}

    # KV Cache Quantization
    enable_quant_kv_cache = args.export_kv_cache_quant != "none"
    if enable_quant_kv_cache and not args.compress:
        kv_cache_quant_cfg = getattr(mtq, KV_QUANT_CFG_CHOICES[args.export_kv_cache_quant])[
            "quant_cfg"
        ]
        mtq_config = mtq.utils.update_quant_cfg_with_kv_cache_quant(mtq_config, kv_cache_quant_cfg)

    # Weight Only Quantization
    if args.weight_only:
        mtq_config["quant_cfg"]["*input_quantizer"] = {"enable": False}
    if args.num_first_layers_to_skip_quant is not None:
        mtq_config = get_first_layers_disabled_config(
            mtq_config,
            num_layers=args.num_layers,
            num_layers_to_disable=args.num_first_layers_to_skip_quant,
        )
    if args.num_last_layers_to_skip_quant is not None:
        mtq_config = get_last_layers_disabled_config(
            mtq_config,
            num_layers=args.num_layers,
            num_layers_to_disable=args.num_last_layers_to_skip_quant,
        )

    return mtq_config


def get_calib_dataloader(
    dataset_path_or_name,
    tokenizer,
    calib_size=512,
    max_sequence_length=512,
    use_random_offset=False,
    batch_size=1,
):
    """Return a dataloader/iterator for calibration using SFT or HF datasets.

    Supports either a local path (.jsonl) or a HuggingFace dataset name.
    """
    if os.path.isfile(dataset_path_or_name):
        # Local file
        print_rank_0(f"Loading calibration dataset from local file: {dataset_path_or_name}")
        all_texts = []
        with open(dataset_path_or_name) as f:
            for i, line in enumerate(f):
                if len(all_texts) == calib_size:
                    break
                sample = json.loads(line)

                # Extract text field from various possible keys
                if isinstance(sample, dict) and "text" in sample:
                    if not sample["text"]:
                        warnings.warn(f"Sample {i} has empty text, skipping")
                        continue
                    full_text = sample["text"]
                elif isinstance(sample, dict) and "messages" in sample:
                    conversations = sample["messages"]
                    assert "role" in conversations[0] and "content" in conversations[0]
                    full_text = "".join([f"{msg['role']}: {msg['content']}" for msg in conversations])
                elif isinstance(sample, list) and isinstance(sample[0], dict):
                    assert "role" in sample[0] and "content" in sample[0]
                    full_text = "".join([f"{msg['role']}: {msg['content']}" for msg in sample])
                else:
                    raise ValueError(f"Sample {i} has unexpected format")

                # Slice text
                max_text_length = int(max_sequence_length / 0.75)  # tokenized text is roughtly ~75% length of original
                start_idx = 0
                if use_random_offset and len(full_text) > max_text_length:
                    start_idx = random.randint(0, len(full_text) - max_text_length)
                text = full_text[start_idx : start_idx + max_text_length]
                all_texts.append(text)

        print_rank_0(f"Loaded calibration dataset ({dataset_path_or_name}) with {len(all_texts)} samples")
        print_rank_0(f"Actual num samples: {len(all_texts)}, max seq length: {max_sequence_length}")
        print_rank_0(f"Sampling Strategy: {'Random Index' if use_random_offset else 'From Beginning'}")

        # Tokenize all texts at once and move to device
        tokens = tokenizer(
            all_texts, return_tensors="pt", padding="max_length", max_length=max_sequence_length, truncation=True
        )
        all_input_ids = tokens.input_ids.cuda()
        return [{"input_ids": all_input_ids[i:i+batch_size]} for i in range(0, len(all_input_ids), batch_size)]
    else:
        # HuggingFace dataset
        if use_random_offset:
            warnings.warn("Random offset is not supported for HuggingFace datasets.")
        print_rank_0(f"Loading calibration dataset from HuggingFace: {dataset_path_or_name}")
        return get_dataset_dataloader(
            dataset_name=dataset_path_or_name,
            tokenizer=tokenizer,
            num_samples=calib_size,
            max_sample_length=max_sequence_length,
            batch_size=batch_size,
            device="cuda",
        )


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
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    model = get_model(
        functools.partial(model_provider, modelopt_gpt_mamba_builder), wrap_with_ddp=False
    )

    report_current_memory_info()

    if args.load is not None:
        load_modelopt_checkpoint(model, strict=not args.untie_embeddings_and_output_weights)
        print_rank_0("Done loading checkpoint")

    if args.pretrained_model_path is not None:
        from modelopt.torch.export import import_mcore_gpt_from_hf

        import_dtype = torch.float16 if args.fp16 else torch.bfloat16
        unwrapped_model = unwrap_model(model)[0]
        workspace_dir = os.environ.get("MLM_WORK_DIR", "/tmp")

        import_kwargs = {"dtype": import_dtype}
        if "trust_remote_code" in inspect.signature(import_mcore_gpt_from_hf).parameters:
            import_kwargs.update({"trust_remote_code": args.trust_remote_code})
        import_mcore_gpt_from_hf(
            unwrapped_model, args.pretrained_model_path, workspace_dir, **import_kwargs
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

    def _dataset_forward_loop_func(model):
        dataloader = get_calib_dataloader(
            dataset_path_or_name=args.calib_dataset_path_or_name,
            tokenizer=tokenizer,
            calib_size=args.calib_size,
            max_sequence_length=args.calib_max_sequence_length,
            use_random_offset=args.calib_use_random_offset,
            batch_size=args.calib_batch_size,
        )
        for sample in tqdm(dataloader, disable=torch.distributed.get_rank()):
            sample = get_batch_on_this_cp_rank(sample)
            simple_generate(model, sample["input_ids"], osl=1, calibration_mode=True)

    unwrapped_model = unwrap_model(model)[0]

    if args.force_all_expert_routing:
        warnings.warn(
            "--force-all-expert-routing will be deprecated in the next release and is no longer needed."
        )

    if args.export_quant_cfg is not None:
        print_rank_0("Quantizing the model...")
        mtq_config = get_modelopt_torch_quantization_config()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # better for calibration

        if args.weight_only:
            mtq.quantize(unwrapped_model, mtq_config)
        elif hasattr(unwrapped_model, "calibration_mode"):
            unwrapped_model.calibration_mode = True
            mtq.quantize(unwrapped_model, mtq_config, _dataset_forward_loop_func)
            unwrapped_model.calibration_mode = False
        else:
            mtq.quantize(unwrapped_model, mtq_config, _dataset_forward_loop_func)

        if args.compress:
            mtq.compress(unwrapped_model)
            print_rank_0("Weights are now compressed to low-bit!")

        print_distributed_quant_summary(model, "Quantized Model:")

    if args.save is not None:
        save_checkpoint(1, model, None, None, 0, release=True)

    # Do this after saving in case it causes issues
    _custom_prompt_forward_loop_func(unwrapped_model)
