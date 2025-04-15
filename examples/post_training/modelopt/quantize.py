# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate GPT."""
import functools
import os
import sys
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import modelopt
import modelopt.torch.quantization as mtq
import torch
from datasets import load_dataset
from tqdm import tqdm

from megatron.core import mpu
from megatron.core.inference.communication_utils import broadcast_from_last_pipeline_stage
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.post_training.arguments import add_modelopt_args
from megatron.post_training.checkpointing import load_modelopt_checkpoint
from megatron.post_training.model_provider import model_provider
from megatron.training import get_args, get_model, get_tokenizer, initialize_megatron
from megatron.training.checkpointing import save_checkpoint
from megatron.training.utils import get_ltor_masks_and_position_ids, print_rank_0, unwrap_model

warnings.filterwarnings('ignore')


QUANT_CFG_CHOICES = {
    "int8": mtq.INT8_DEFAULT_CFG,
    "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
    "fp8": mtq.FP8_DEFAULT_CFG,
    "fp8_real_quant": mtq.FP8_PER_TENSOR_REAL_QUANT_CFG,
    "fp8_blockwise_real_quant": mtq.FP8_2D_BLOCKWISE_REAL_QUANT_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
    "int4": mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
    "fp4": mtq.NVFP4_DEFAULT_CFG,
}


def add_text_generate_ptq_args(parser):
    """Add additional arguments for ModelOpt text generation PTQ."""
    group = parser.add_argument_group(title='ModelOpt text generation ptq')
    group.add_argument(
        "--calib-size", type=int, default=512, help="Samples to use for ptq calibration."
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=("Hello!|Born in California, Soyer trained as a"),
        help="Input texts. Please use | to separate different batches.",
    )
    parser.add_argument(
        "--references",
        type=str,
        default="",
        help="Reference texts. Please use | to separate different batches.",
    )
    parser.add_argument(
        "--pretrained-model-path", type=str, default=None, help="HuggingFace pretrained model"
    )
    add_modelopt_args(parser)
    return parser


def check_arguments():
    """Checking user arguments."""
    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print_rank_0("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    if hasattr(args, 'moe_grouped_gemm') and args.moe_grouped_gemm == True:
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
    if "fp8" == args.export_quant_cfg:
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
    if args.export_kv_cache_quant:
        mtq_config["quant_cfg"]["*linear_qkv.output_quantizer"] = fp8_config

    return mtq_config


def get_calib_dataloader(calib_size=512, max_sequence_length=512):
    """Return a dataloader for calibration."""
    dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
    text_column = "article"

    calib_size = min(len(dataset), calib_size)
    for i in range(calib_size):
        yield dataset[i][text_column][:max_sequence_length]


def get_current_memory_info():
    remaining_mem, total_mem = torch.cuda.mem_get_info()
    info = "rank {:02}  memory remaining {:03}% ({}/{} MB) ".format(
        torch.distributed.get_rank(),
        int(remaining_mem * 100 / total_mem),
        remaining_mem // 1048576,
        total_mem // 1048576,
    )
    return info


def report_current_memory_info():
    """Report current memory usage."""
    print(get_current_memory_info())
    torch.distributed.barrier()


def eager_generate_no_kv_cache(model, input_ids, osl):
    """A simple generate function for post-training calibration forward."""

    model.eval()
    eos_token_ids = get_tokenizer().eod

    def _dummy_loss_func(output_tensor, non_loss_data=True):
        return output_tensor

    def _forward_step_func(data, model):
        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            data["tokens"], eos_token_ids, True, True, True
        )
        output_tensor = model(data["tokens"], position_ids, attention_mask)
        return output_tensor, _dummy_loss_func

    output_ids = None

    step_pbar = tqdm(range(osl), disable=torch.distributed.get_rank(), leave=False)

    for step in step_pbar:
        step_pbar.set_description(get_current_memory_info())

        # When --sequence-parallel is used, sequence_len must be a multiple of
        # --tensor-parallel. We pad eos tokens on the left to be multiple of 32.
        num_pad_tokens = input_ids.shape[-1] % 32

        if num_pad_tokens > 0:
            num_pad_tokens = 32 - num_pad_tokens
            padding_shape = (input_ids.shape[0], num_pad_tokens)
            padded_tokens = torch.full(
                padding_shape, eos_token_ids, dtype=input_ids.dtype, device=input_ids.device
            )
            tokens = torch.cat((padded_tokens, input_ids), dim=-1)
        else:
            tokens = input_ids

        list_of_logits = get_forward_backward_func()(
            forward_step_func=_forward_step_func,
            data_iterator=[{"tokens": tokens}],
            model=model,
            num_microbatches=1,
            seq_length=tokens.shape[-1],
            micro_batch_size=1,
            decoder_seq_length=tokens.shape[-1],
            forward_only=True,
            collect_non_loss_data=True,
        )

        if mpu.is_pipeline_last_stage():
            logits = gather_from_tensor_model_parallel_region(list_of_logits[0])
            eager_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True).detach()
        else:
            eager_ids = None

        eager_ids = broadcast_from_last_pipeline_stage(
            [input_ids.shape[0], 1], input_ids.dtype, eager_ids
        )

        input_ids = torch.cat([input_ids, eager_ids], dim=-1)

        if output_ids is None:
            output_ids = eager_ids
        else:
            output_ids = torch.cat([output_ids, eager_ids], dim=-1)

    return output_ids


if __name__ == "__main__":
    initialize_megatron(
        extra_args_provider=add_text_generate_ptq_args,
        args_defaults={
            'tokenizer_type': 'HuggingFaceTokenizer',
            'no_load_rng': True,
            'no_load_optim': True,
        },
    )

    check_arguments()

    args = get_args()

    tokenizer = get_tokenizer()._tokenizer
    model = get_model(functools.partial(model_provider, parallel_output=True), wrap_with_ddp=False)

    report_current_memory_info()

    if args.load is not None:
        load_modelopt_checkpoint(model, strict=not args.untie_embeddings_and_output_weights)
        print_rank_0("Done loading checkpoint")

    if args.pretrained_model_path is not None:
        from modelopt.torch.export import import_mcore_gpt_from_hf

        unwrapped_model = unwrap_model(model)[0]
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
            generated_ids = eager_generate_no_kv_cache(model, tokens.input_ids.cuda(), 32)
            generated_texts = tokenizer.batch_decode(generated_ids)
            print_rank_0("{}".format(generated_texts))
            if all_references[idx] is not None:
                assert all_references[idx] == generated_texts[0], all_references[idx]

    def _hf_dataset_forword_loop_func(model):
        dataloader = get_calib_dataloader(args.calib_size)
        for prompt in tqdm(dataloader, total=args.calib_size, disable=torch.distributed.get_rank()):
            tokens = tokenizer(prompt, return_tensors="pt")
            generated_ids = eager_generate_no_kv_cache(model, tokens.input_ids.cuda(), 1)

    unwrapped_model = unwrap_model(model)[0]

    if args.export_quant_cfg in QUANT_CFG_CHOICES:
        print_rank_0("Quantizing the model...")
        mtq_config = get_modelopt_torch_quantization_config()
        ptq_forward_loop_func = _hf_dataset_forword_loop_func
        if hasattr(unwrapped_model, "calibration_mode"):
            unwrapped_model.calibration_mode = True
            mtq.quantize(unwrapped_model, mtq_config, ptq_forward_loop_func)
            unwrapped_model.calibration_mode = False
        else:
            mtq.quantize(unwrapped_model, mtq_config, ptq_forward_loop_func)

    print_rank_0(f"Fake Quantized Model:\n {unwrapped_model}")

    _custom_prompt_forward_loop_func(unwrapped_model)

    if args.save is not None and args.export_quant_cfg in QUANT_CFG_CHOICES:
        save_checkpoint(1, model, None, None, 0)
