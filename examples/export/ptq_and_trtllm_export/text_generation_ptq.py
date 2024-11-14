# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate GPT."""
import functools
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import modelopt.torch.quantization as mtq
import torch
from datasets import load_dataset
from tqdm import tqdm

# [ModelOpt]: changing the default model provider to the ModelOpt version
from megatron.core import mpu
from megatron.inference.arguments import add_modelopt_args
from megatron.inference.checkpointing import load_modelopt_checkpoint
from megatron.inference.gpt.model_provider import model_provider
from megatron.inference.text_generation import generate_and_post_process
from megatron.training import get_args, get_model, initialize_megatron
from megatron.training.checkpointing import save_checkpoint
from megatron.training.utils import print_rank_0, unwrap_model

QUANT_CFG_CHOICES = {
    "int8": mtq.INT8_DEFAULT_CFG,
    "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
    "fp8": mtq.FP8_DEFAULT_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
    "int4": mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
}


def add_trtllm_ckpt_export_args(parser):
    """Add additional arguments for TensorRT-LLM."""
    group = parser.add_argument_group(title="trtllm")

    group.add_argument(
        "--export-dir", type=str, help="The output TensorRT-LLM checkpoint.",
    )
    group.add_argument(
        "--decoder", type=str, choices=["gptnext", 'llama'], help="The decoder type of the model.",
    )
    group.add_argument(
        "--inference-tensor-parallel",
        type=int,
        help="Tensor parallel for the inference time, can be different from the training config.",
        default=1,
    )


def add_text_generate_ptq_args(parser):
    """Add additional arguments for ModelOpt text generation PTQ."""
    group = parser.add_argument_group(title='ModelOpt text generation ptq')
    group.add_argument(
        "--calib-dataset",
        type=str,
        default="cnn_dailymail",
        help="Calibration datasets from HuggingFace datasets.",
    )
    group.add_argument(
        "--calib-batch-size", type=int, default=4, help="Batch size to use for ptq calibration."
    )
    group.add_argument(
        "--calib-size", type=int, default=512, help="Samples to use for ptq calibration."
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=(
            "Born in north-east France, Soyer trained as a|Born in California, Soyer trained as a"
        ),
        help="Input texts. Please use | to separate different batches.",
    )
    add_modelopt_args(parser)
    add_trtllm_ckpt_export_args(parser)
    return parser


def get_calib_dataloader(
    data="cnn_dailymail", batch_size=4, calib_size=512, max_sequence_length=512
):
    if data == "pileval":
        dataset = load_dataset(
            "json", data_files="https://the-eye.eu/public/AI/pile/val.jsonl.zst", split="train"
        )
        text_column = "text"
    elif data == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
        text_column = "text"
    elif data == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
        text_column = "article"

    calib_size = max(min(len(dataset), calib_size), batch_size)
    for i in range(calib_size // batch_size):
        batch = dataset[i * batch_size : (i + 1) * batch_size][text_column]
        for j in range(len(batch)):
            batch[j] = batch[j][:max_sequence_length]
        yield batch



if __name__ == "__main__":
    initialize_megatron(
        extra_args_provider=add_text_generate_ptq_args,
        args_defaults={
            'tokenizer_type': 'GPT2BPETokenizer',
            'no_load_rng': True,
            'no_load_optim': True,
        },
    )

    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print_rank_0("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    print_rank_0("WARNING: Forcing exit_on_missing_checkpoint to True for text generation.")
    args.exit_on_missing_checkpoint = True
    if hasattr(args, 'moe_grouped_gemm') and args.moe_grouped_gemm == True:
        print_rank_0("WARNING: Forcing moe_grouped_gemm to False for PTQ and export.")
        args.moe_grouped_gemm = False

    # Set up model and load checkpoint
    # [ModelOpt]: make sure that output logits are allgathered.
    text_generation_model_provider = functools.partial(model_provider, parallel_output=False)
    model = get_model(text_generation_model_provider, wrap_with_ddp=False)

    if args.load is not None:
        load_modelopt_checkpoint(model, strict=not args.untie_embeddings_and_output_weights)
        print_rank_0("Done loading checkpoint")

    # Removing virtual pipeline parallel and other wrapper
    assert len(model) == 1, "Above condition should have caught this"
    unwrapped_model = unwrap_model(model)

    all_prompts = args.prompts.split("|")

    def custom_prompt_forward_loop_func(model):
        for prompt in tqdm(all_prompts):
            if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
                (
                    prompts_plus_generations,
                    prompts_plus_generations_segments,
                    logprobs,
                    _,
                ) = generate_and_post_process(
                    model,
                    prompts=[prompt],
                    tokens_to_generate=128,
                    return_output_log_probs=True,
                    temperature=1.0,
                )
                print_rank_0(prompts_plus_generations)
            else:
                generate_and_post_process(model)

    def hf_dataset_forword_loop_func(model):
        dataloader = get_calib_dataloader(args.calib_dataset, args.calib_batch_size, args.calib_size)
        for prompts in tqdm(dataloader, total=args.calib_size//args.calib_batch_size):
            if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
                (
                    prompts_plus_generations,
                    prompts_plus_generations_segments,
                    logprobs,
                    _,
                ) = generate_and_post_process(
                    model,
                    prompts=prompts,
                    tokens_to_generate=0,
                    return_output_log_probs=False,
                    temperature=1.0,
                )
            else:
                generate_and_post_process(model)

    ptq_forward_loop_func = custom_prompt_forward_loop_func
    if args.calib_dataset is not None:
        ptq_forward_loop_func = hf_dataset_forword_loop_func

    if args.export_quant_cfg in QUANT_CFG_CHOICES:
        mtq_config = QUANT_CFG_CHOICES[args.export_quant_cfg]
        if "*output_layer*" not in mtq_config["quant_cfg"]:
            mtq_config["quant_cfg"]["*output_layer*"] = {"enable": False}
        if "awq" in args.export_quant_cfg:
            weight_quantizer = mtq_config["quant_cfg"]["*weight_quantizer"]  # type: ignore
            if isinstance(weight_quantizer, list):
                weight_quantizer = weight_quantizer[0]
            weight_quantizer["block_sizes"][-1] = 128
        print_rank_0("Quantizing the model...")
        mtq.quantize(unwrapped_model[0], mtq_config, ptq_forward_loop_func)

    custom_prompt_forward_loop_func(model[0])

    if args.save is not None and args.export_quant_cfg in QUANT_CFG_CHOICES:
        save_checkpoint(1, unwrapped_model, None, None, 0)

    print_rank_0(f"Fake Quantized Model:\n {unwrapped_model[0]}")

    if args.export_dir:
        assert args.decoder in ["gptnext", "llama"], f"Decoder type {args.decoder} not supported."
        Path(args.export_dir).mkdir(parents=True, exist_ok=True)
        print_rank_0("Exporting TensorRT-LLM checkpoints.")

        from modelopt.torch.export import export_tensorrt_llm_checkpoint

        # In TRT LLM, squared relu activation does not support bf16. So we use fp16 by default.
        export_tensorrt_llm_checkpoint(
            unwrapped_model[0],
            args.decoder,
            torch.bfloat16 if args.bf16 else torch.float16,
            export_dir=args.export_dir,
            inference_tensor_parallel=args.inference_tensor_parallel,
            inference_pipeline_parallel=1,
            use_nfs_workspace=True,
        )

        print_rank_0(f"TensorRT-LLM checkpoints saved to {args.export_dir}")
        torch.distributed.barrier()
