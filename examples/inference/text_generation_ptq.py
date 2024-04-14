# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate GPT."""
import functools
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import ammo.torch.quantization as atq
import torch
from datasets import load_dataset

# [ModelOpt]: changing the default model provider to the AMMO version
from megatron.training import get_args, print_rank_0
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.core import mpu
from megatron.core.dist_checkpointing import load
from megatron.inference.arguments import add_ammo_args
from megatron.inference.gpt.model_provider import model_provider
from megatron.training.initialize import initialize_megatron
from megatron.inference.text_generation import generate_and_post_process
from megatron.training import get_model
from megatron.training.utils import unwrap_model

QUANT_CFG_CHOICES = {
    "int8": atq.INT8_DEFAULT_CFG,
    "int8_sq": atq.INT8_SMOOTHQUANT_CFG,
    "fp8": atq.FP8_DEFAULT_CFG,
    "int4_awq": atq.INT4_AWQ_CFG,
    "w4a8_awq": atq.W4A8_AWQ_BETA_CFG,
}


def add_trtllm_args(parser):
    """Add additional arguments for TensorRT-LLM."""
    group = parser.add_argument_group(title="trtllm")

    group.add_argument(
        "--engine-dir", type=str, help="The output TensorRT-LLM engine dir.",
    )
    group.add_argument(
        "--decoder", type=str, choices=["gptnext", 'llama'], help="The decoder type of the model.",
    )
    group.add_argument("--max-input-len", type=int, help="Max input sequence length.", default=2048)
    group.add_argument(
        "--max-output-len", type=int, help="Max output sequence length.", default=512
    )
    group.add_argument("--max-batch-size", type=int, help="Max batch size.", default=32)
    group.add_argument(
        "--inference-tensor-parallel",
        type=int,
        help="Tensor parallel for the inference time, can be different from the training config.",
        default=1,
    )


def add_text_generate_ptq_args(parser):
    """Add additional arguments for AMMO text generation PTQ."""
    group = parser.add_argument_group(title='AMMO text generation ptq')
    group.add_argument(
        "--calib-dataset",
        type=str,
        default="cnn_dailymail",
        help="Calibration datasets from HuggingFace datasets.",
    )
    group.add_argument(
        "--calib-steps", type=int, default=512, help="Steps to perform atq.quantize calibration."
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=(
            "Born in north-east France, Soyer trained as a|Born in California, Soyer trained as a"
        ),
        help="Input texts. Please use | to separate different batches.",
    )
    add_ammo_args(parser)
    add_trtllm_args(parser)
    return parser


def get_calib_dataloader(
    data="cnn_dailymail", batch_size=4, calib_size=512, max_sequence_length=512
):
    if data == "wikitext":
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


def ammo_load_checkpoint(
    model, optimizer=None, opt_param_scheduler=None, strict=True, additional_sharded_prefix=""
):
    """Load a megatron checkpoint depending its format.

    Args:
        model: MCoreGPTModel instance
        optimizer: Megatron optimizer instance
        opt_param_scheduler: Megatron scheduler instance
        strict: if True, no extra or missing keys are allowed while loading the state_dict 
        additional_sharded_prefix (str): Append additional prefix to align the sharded checkpoint keys. When loading
        an .nemo sharded checkpoint, this is usually `model.`. Otherwise, this is typically an empty string.
    """

    def _remove_prefix_state_dict_pre_hook(
        state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
    ):
        """Pytorch _load_state_dict_pre_hook to remap the state_dict with the additional sharded prefix."""
        if additional_sharded_prefix is None:
            return
        key_rewrite_list = []
        for key, _ in state_dict.items():
            if key.startswith(additional_sharded_prefix):
                key_rewrite_list.append(key)
        for old_key in key_rewrite_list:
            new_key = old_key[len(additional_sharded_prefix) :]
            state_dict[new_key] = state_dict.pop(old_key)

    args = get_args()
    load_dir = args.load

    shared_model_state_dir = "model_weights"
    sharded_load_dir = Path(load_dir + "/" + shared_model_state_dir)

    if sharded_load_dir.exists() and optimizer is None and opt_param_scheduler is None:
        unwrapped_model = unwrap_model(model)
        shareded_state_dict = unwrapped_model[0].sharded_state_dict(
            prefix=additional_sharded_prefix
        )
        if additional_sharded_prefix:
            unwrapped_model[0]._register_load_state_dict_pre_hook(
                _remove_prefix_state_dict_pre_hook
            )
        unwrapped_model[0].load_state_dict(load(shareded_state_dict, sharded_load_dir))
    else:
        _ = load_checkpoint(model, optimizer, opt_param_scheduler, strict=strict)


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
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    text_generation_model_provider = functools.partial(model_provider, parallel_output=False)
    model = get_model(text_generation_model_provider, wrap_with_ddp=False)
    assert len(model) == 1, "Above condition should have caught this"

    if args.load is not None:
        _ = ammo_load_checkpoint(
            model,
            None,
            None,
            strict=not args.untie_embeddings_and_output_weights,
            additional_sharded_prefix="model.",
        )
    else:
        print_rank_0("WARNING: No checkpoint is loaded for PTQ! The process will still continue.")

    all_prompts = args.prompts.split("|")

    def custom_prompt_forward_loop_func():
        for prompt in all_prompts:
            if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
                (
                    prompts_plus_generations,
                    prompts_plus_generations_segments,
                    logprobs,
                    _,
                ) = generate_and_post_process(
                    model[0],
                    prompts=[prompt],
                    tokens_to_generate=128,
                    return_output_log_probs=True,
                    temperature=1.0,
                )
                print_rank_0(prompts_plus_generations)
            else:
                generate_and_post_process(model[0])

    def hf_dataset_forword_loop_func():
        dataloader = get_calib_dataloader(args.calib_dataset, calib_size=args.calib_steps)
        for prompts in dataloader:
            if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
                (
                    prompts_plus_generations,
                    prompts_plus_generations_segments,
                    logprobs,
                    _,
                ) = generate_and_post_process(
                    model[0],
                    prompts=prompts,
                    tokens_to_generate=0,
                    return_output_log_probs=True,
                    temperature=1.0,
                )
            else:
                generate_and_post_process(model[0])

    ptq_forward_loop_func = custom_prompt_forward_loop_func
    if args.calib_dataset is not None:
        ptq_forward_loop_func = hf_dataset_forword_loop_func

    if args.ammo_quant_cfg in QUANT_CFG_CHOICES:
        atq_config = QUANT_CFG_CHOICES[args.ammo_quant_cfg]
        if "awq" in args.ammo_quant_cfg:
            weight_quantizer = atq_config["quant_cfg"]["*weight_quantizer"]  # type: ignore
            if isinstance(weight_quantizer, list):
                weight_quantizer = weight_quantizer[0]
            weight_quantizer["block_sizes"][-1] = 128
        atq_config["quant_cfg"]["*.output_layer.*"] = {"enable": False}
        print_rank_0("atq.quantize: output_layer quantization is disable")
        atq.quantize(model[0], atq_config, ptq_forward_loop_func)
        custom_prompt_forward_loop_func()
        if args.save:
            save_checkpoint(1, model, None, None)
    else:
        custom_prompt_forward_loop_func()

    if args.engine_dir:
        from ammo.deploy.llm import model_config_to_tensorrt_llm
        from ammo.torch.export import torch_to_model_config

        assert args.decoder in ["gptnext", "llama"], f"Decoder type {args.decoder} not supported."

        Path(args.engine_dir).mkdir(parents=True, exist_ok=True)

        print_rank_0("Exporting model_configs for TRT LLM.")
        model = unwrap_model(model)
        model = model[0]

        # In TRT LLM, squared relu activation does not support bf16. So we use fp16 by default.
        model_configs = torch_to_model_config(
            model,
            args.decoder,
            torch.float16,
            inference_tensor_parallel=args.inference_tensor_parallel,
        )

        print_rank_0("Building TRT LLM engines.")
        for model_config in model_configs:
            model_config_to_tensorrt_llm(
                model_config,
                args.engine_dir,
                max_input_len=args.max_input_len,
                max_output_len=args.max_output_len,
                max_batch_size=args.max_batch_size,
                max_beam_width=1,
                num_build_workers=1,
                inflight_batching=False,
                enable_sparsity=False,
            )
        print_rank_0(f"TRT LLM engines saved to {args.engine_dir}")
