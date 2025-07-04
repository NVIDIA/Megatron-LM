# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
"""Generate text using a vision language model."""
import json
import logging
import os
import sys
from functools import partial
from typing import List, Dict

# Add megatron to the path.
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

import torch
import yaml
from config import EvaluationConfig
from evaluation.evaluation_datasets import get_evaluation_dataset
from model import model_provider
from multimodal_args import add_multimodal_extra_args

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.models.multimodal.llava_model import IMAGE_TOKEN
from megatron.core.models.vision.clip_vit_model import get_num_image_embeddings
from megatron.inference.text_generation.api import generate_and_post_process
from megatron.inference.text_generation.forward_step import ForwardStep
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.engines import StaticInferenceEngine
from megatron.core.inference.inference_request import InferenceRequest, VLMInferenceRequest
from megatron.core.inference.text_generation_controllers.vlm_text_generation_controller import (
    VLMTextGenerationController,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.model_inference_wrappers.multimodal.vlm_inference_wrapper import (
    VLMInferenceWrapper,
)
from megatron.training import get_args, get_model, get_tokenizer, print_rank_0, is_last_rank
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron


def is_first_rank():
    """First tensor and pipeline parallel rank."""
    return (
        parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        and parallel_state.get_tensor_model_parallel_rank() == 0
    )


def add_text_generation_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='Vision language model text generation arguments')

    group.add_argument("--temperature", type=float, default=1.0, help='Sampling temperature.')
    group.add_argument("--top_p", type=float, default=0.0, help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0, help='Top k sampling.')
    group.add_argument(
        "--out-seq-length", type=int, default=128, help='Length of the output generated text.'
    )
    group.add_argument("--output-path", type=str, help='Output file path')
    group.add_argument('--input-image-path', type=str, help="Input image directory")
    group.add_argument(
        '--num-partitions', type=int, default=0, help="Number of partitions for inputs."
    )
    group.add_argument('--partition-id', type=int, default=0, help="Partition index")
    group.add_argument("--gt-path", type=str, help="Optional ground truth file")
    group.add_argument(
        "--task",
        type=str,
        choices=[
            "captioning",
            "TextVQA",
            "VQAv2",
            "ChartQA",
            "MMMU",
            "OCRBench",
            "OCRBench_v2",
            "MathVista",
            "AI2D",
            "InfoVQA",
            "SPDocVQA",
            "RD_TableBench",
            "VideoMME",
            "PerceptionTest",
            "MotionBench",
            "PhysGameBench",
            "MVBench",
            "inference",
        ],
        help="Generation task to run",
    )
    group.add_argument(
        "--num-samples-per-partition", type=int, default=0, help="Number of samples per partition"
    )
    group.add_argument("--config-path", type=str, help="Evaluation config file to use.")

    # Add common multimodal arguments needed for e.g. building the model.
    parser = add_multimodal_extra_args(parser)

    return parser


def get_evaluation_dataloader(
    task,
    input_image_path,
    gt_path,
    img_h,
    img_w,
    use_tiling,
    max_num_tiles,
    use_thumbnail,
    num_samples_per_partition,
    num_partitions,
    partition_id,
    num_frames,
    num_workers,
    vision_model_type,
    split="validation"
):
    """Build evaluation dataset."""
    dataset = get_evaluation_dataset(
        task,
        input_image_path,
        gt_path,
        img_h,
        img_w,
        use_tiling,
        max_num_tiles,
        use_thumbnail,
        num_samples_per_partition,
        num_partitions,
        partition_id,
        num_frames,
        vision_model_type,
        split=split
    )

    dp_rank = parallel_state.get_data_parallel_rank()
    dp_world_size = parallel_state.get_data_parallel_world_size()

    sampler = torch.utils.data.DistributedSampler(
        dataset, shuffle=False, num_replicas=dp_world_size, rank=dp_rank
    )
    # TODO: Batched inference is not supported yet.
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=num_workers, sampler=sampler, pin_memory=True
    )

    return dataloader


def generate_samples(model, config: EvaluationConfig, print_output):
    """Text generation using a trained vision language model."""
    args = get_args()

    dataloader = get_evaluation_dataloader(
        config.task,
        config.input_image_path,
        config.gt_path,
        args.img_h,
        args.img_w,
        args.use_tiling,
        args.max_num_tiles,
        args.use_thumbnail,
        config.num_samples_per_partition,
        config.num_partitions,
        config.partition_id,
        args.num_frames,
        args.num_workers,
        args.vision_model_type,
        config.split
    )

    num_img_embeddings_per_tile = get_num_image_embeddings(
        args.img_h,
        args.img_w,
        args.patch_dim,
        args.vision_model_type,
        args.disable_vision_class_token,
        1,
        args.pixel_shuffle,
        args.use_tile_tags,
        args.max_num_tiles,
        args.tokenizer_prompt_format,
    )

    if args.use_mcore_inference:
        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=args.hidden_size,
            inference_batch_times_seqlen_threshold=args.inference_batch_times_seqlen_threshold,
            fp32_residual_connection=args.fp32_residual_connection,
            params_dtype=args.params_dtype,
            padded_vocab_size=args.padded_vocab_size,
        )
        inference_wrapped_model = VLMInferenceWrapper(model, inference_wrapper_config)
        tokenizer = get_tokenizer()
        controller = VLMTextGenerationController(
            inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer
        )
        inference_engine = StaticInferenceEngine(
            controller, max_batch_size=1, random_seed=args.seed
        )
        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            num_tokens_to_generate=config.out_seq_length,
        )

    for idx, (imgs, num_tiles, sample_id, question, answers, metadata) in enumerate(dataloader):
        imgs = imgs.to("cuda")
        num_tiles = num_tiles.to("cuda")

        conv = get_conversation(config.task, question, metadata)

        if not args.use_mcore_inference:
            forward_step = partial(VLMForwardStep, num_img_embeddings_per_tile, imgs, num_tiles, args.decoder_seq_length)

        inference_context = StaticInferenceContext(max_batch_size=1, max_sequence_length=args.inference_max_seq_length)
        if is_first_rank():

            if args.use_mcore_inference:
                inference_request = VLMInferenceRequest(
                   request_id=inference_engine.get_new_request_id(),
                   prompt=conv,
                   prompt_tokens=controller.tokenize_prompt(conv),
                   sampling_params=sampling_params,
                   num_img_embeddings_per_tile=num_img_embeddings_per_tile,
                   imgs=imgs,
                   num_tiles=num_tiles,
                   decoder_seq_length=args.decoder_seq_length,
                )
                results: List[InferenceRequest] = inference_engine.generate(
                    inference_requests=[inference_request]
                )

                resp_sentences = [
                    tokenizer.detokenize(result.prompt_tokens) + result.generated_text
                    for result in results
                ]
            else:
                resp_sentences, _, _, _ = generate_and_post_process(
                    model, inference_context,
                    forward_step=forward_step,
                    prompts=[conv],
                    tokens_to_generate=config.out_seq_length,
                    top_k_sampling=config.top_k,
                    top_p_sampling=config.top_p,
                    add_BOS=False,
                    temperature=config.temperature,
                    random_seed=args.seed,
                    detokenize_segments=False,
                    data_parallel=True,
            )

            for generation in resp_sentences:
                if isinstance(sample_id, torch.Tensor):
                    sample_id = sample_id.item()

                output = {"sample_id": sample_id}

                output_name = ""
                if config.task == "captioning":
                    output_name = "caption"
                elif config.task in (
                    "TextVQA",
                    "VQAv2",
                    "ChartQA",
                    "OCRBench",
                    "MathVista",
                    "AI2D",
                    "RealworldQA",
                    "MotionBench",
                    "PhysGameBench",
                    "MVBench",
                    "InfoVQA",
                    "SPDocVQA",
                    "inference",
                ):
                    output_name = "answer"
                elif config.task in ("MMMU"):
                    output_name = "text"
                elif config.task == "VideoMME":
                    output_name = "response"
                    output = question
                elif config.task in ["OCRBench_v2", "RD_TableBench"]:
                    output_name = "predict"
                else:
                    raise NotImplementedError("no output name defined for", config.task)

                prompt, generated = get_prompt_and_generated(
                    generation, args.tokenizer_prompt_format
                )
                if config.task == "VideoMME":
                    output["questions"][0][output_name] = generated
                else:
                    output["prompt"] = prompt
                    output[output_name] = generated

                if config.task in ["captioning", "RD_TableBench"]:
                    output["ground_truth"] = answers
                elif config.task in (
                    "TextVQA",
                    "VQAv2",
                    "ChartQA",
                    "OCRBench",
                    "OCRBench_v2",
                    "MathVista",
                    "AI2D",
                    "PerceptionTest",
                    "RealworldQA",
                    "MotionBench",
                    "PhysGameBench",
                    "MVBench",
                    "InfoVQA",
                    "SPDocVQA",
                    "inference",
                ):
                    if isinstance(answers, str):
                        answers = [answers]
                    output["gt_answer"] = answers

                    if len(metadata) > 0:
                        output.update(metadata)
                elif config.task == "MMMU":
                    output["prediction"] = generated
                    output.update(metadata)
                elif config.task == "VideoMME":
                    pass
                else:
                    raise NotImplementedError("no output processing defined for", config.task)

                if print_output:
                    print(output)

                yield output
                idx += 1
        else:
            if args.use_mcore_inference:
                inference_request = VLMInferenceRequest(
                   request_id=inference_engine.get_new_request_id(),
                   prompt=conv,
                   prompt_tokens=controller.tokenize_prompt(conv),
                   sampling_params=sampling_params,
                   num_img_embeddings_per_tile=num_img_embeddings_per_tile,
                   imgs=imgs,
                   num_tiles=num_tiles,
                   decoder_seq_length=args.decoder_seq_length,
                )
                inference_engine.generate(
                    inference_requests=[inference_request]
                )
            else:
                generate_and_post_process(
                    model, inference_context, forward_step=forward_step, detokenize_segments=False, data_parallel=True
                )

            idx += 1


def get_evaluation_configs(config_path=None) -> Dict[str, EvaluationConfig]:
    """Get evaluation config(s) from a config file or command-line arguments.

    Args:
        config_path: Optional path to config file. If not provided, will check args.config_path
                    or fall back to command-line arguments.

    Returns:
        Dict[str, EvaluationConfig]: dict of configs.
    """
    args = get_args()
    configs = {}

    # Use provided config_path or fall back to args.config_path
    config_file = config_path or args.config_path

    # We check if we're trying to run a single config evals by checking for the task and output_path
    # args.
    if hasattr(args, "task") and args.task and hasattr(args, "output_path") and args.output_path:
        # Single config from args
        config = EvaluationConfig(
            task=args.task,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            out_seq_length=args.out_seq_length,
            output_path=args.output_path,
            input_image_path=args.input_image_path,
            gt_path=args.gt_path,
            num_partitions=args.num_partitions,
            partition_id=args.partition_id,
            num_samples_per_partition=args.num_samples_per_partition,
        )
        if not config.output_path:
            default_output_dir = args.output_path if args.output_path else "generated"
            os.makedirs(default_output_dir, exist_ok=True)
            config.output_path = os.path.join(default_output_dir, args.language_model_type)
        return {args.task: config}
    elif config_file:
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)
        if 'datasets' not in config_data:
            print("Error: 'datasets' key not found in config file for batch mode.")
            sys.exit(1)
        config_dict = config_data['datasets']
        for key, value in config_dict.items():
            config = EvaluationConfig(**value)
            config.dataset = key
            if not config.output_path:
                # Use args.output_path if available, otherwise use "generated"
                default_output_dir = getattr(args, 'output_path', None) or "generated"
                os.makedirs(default_output_dir, exist_ok=True)
                config.output_path = os.path.join(default_output_dir, f"{args.language_model_type}")
            configs[key] = config
        return configs
    else:
        raise ValueError("No config file provided and no task specified.")


def get_output_path(config, dp_rank):
    """Generation output path."""

    ckpt_step = None
    try:
        args = get_args()
        ckpt_step = args.ckpt_step
    except Exception as e:
        print(f"Failed getting args: {type(e).__name__} - {e}")
    if ckpt_step is not None:
        return f"{config.output_path}-{config.task}-dprank={dp_rank}-partition={config.partition_id}-step={args.ckpt_step}.jsonl"
    else:
        return f"{config.output_path}-{config.task}-dprank={dp_rank}-partition={config.partition_id}.jsonl"


def generate_and_write_samples(model, config, print_output=True):
    """Generate text and write to an output file."""
    dp_rank = parallel_state.get_data_parallel_rank()

    if is_first_rank():
        output_path = get_output_path(config, dp_rank)
        output_file = open(output_path, "w")
        print(f"output path: {output_file.name}")

    with torch.no_grad():
        for output in generate_samples(model, config, print_output):
            if is_first_rank():
                output_file.write(json.dumps(output) + "\n")
                output_file.flush()

    if is_first_rank():
        output_file.close()

class VLMForwardStep(ForwardStep):
    """Inference forward step for a multimodal model."""

    def __init__(
        self,
        num_img_embeddings_per_tile,
        images,
        num_tiles,
        decoder_seq_length,
        model,
        inference_context,
    ):
        """Create multimodal forward step."""
        total_num_tiles = torch.sum(num_tiles).item()
        num_img_embeddings = num_img_embeddings_per_tile * total_num_tiles

        super().__init__(model, inference_context)
        self._images = images
        self._num_tiles = num_tiles
        self._num_img_embeddings = num_img_embeddings
        self.decoder_seq_length = decoder_seq_length

        self._recv_only_vision_embeds = False
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        # Checks if the previous stage only has a vision encoder, and that the current stage has part of the LM decoder.
        # In this case, the current stage should only receive vision embeddings.
        if pp_rank > 0:
            self._recv_only_vision_embeds = parallel_state.is_inside_encoder(pp_rank - 1) and (not parallel_state.is_inside_decoder(pp_rank - 1)) and parallel_state.is_inside_decoder()

        # Checks if the current stage only has a vision encoder
        self._encoder_only = parallel_state.is_inside_encoder() and not parallel_state.is_inside_decoder()

    def _forward(self, tokens, position_ids, attention_mask):
        return self.model(
            self._images,
            tokens,
            position_ids,
            attention_mask=None,
            inference_context=self.inference_context,
            num_image_tiles=self._num_tiles,
            runtime_gather_output=True,
        )

    def __call__(self, tokens, position_ids, attention_mask):
        num_image_tokens = (tokens == self.model.module.image_token_index).sum().item()
        num_tokens = tokens.size(1)
        recv_buffer_seq_length = None
        if num_image_tokens > 0:
            # When there are image tokens and this stage only receives vision embeddings, adjust the recv buffer seq length to match the image embeddings sequence length.
            # If there are image tokens and this stage receives full embeddings, make sure we compensate for expansion of image tokens.
            # Note that this will set a recv_buffer_seq_length for the encoder stage, this length is irrelevant since that recv buffer is never allocated.
            if self._recv_only_vision_embeds:
                recv_buffer_seq_length = self._num_img_embeddings
            else:
                recv_buffer_seq_length = min(self._num_img_embeddings + num_tokens - num_image_tokens, self.decoder_seq_length)
        elif self._recv_only_vision_embeds:
            # If this stage only receives vision embeddings and there are no image tokens we won't run the encoder and therefore shouldn't try to recv.
            recv_buffer_seq_length = 0

        # If the pipeline stage only has a vision encoder, then it only needs to run when there are image tokens
        if not (self._encoder_only and num_image_tokens == 0):
            output = super().__call__(tokens, position_ids, attention_mask, recv_buffer_seq_length=recv_buffer_seq_length)
        else:
            output = None
        if isinstance(output, tuple):
            logits, _ = output
        else:
            logits = output

        # On the first inference iteration, we compute image tokens.
        # On every PP stage(although inference params should only matter for decoder),
        # update the sequence length offset by the number of image tokens.
        if num_tokens > 1 and num_image_tokens > 0:
            if "image_tokens_count" not in self.inference_context.key_value_memory_dict:
                self.inference_context.key_value_memory_dict["image_tokens_count"] = self._num_img_embeddings

            if self._num_img_embeddings + num_tokens - num_image_tokens > self.decoder_seq_length:
                self.inference_context.sequence_len_offset += self.decoder_seq_length - num_tokens
            else:
                self.inference_context.sequence_len_offset += (
                    self.inference_context.key_value_memory_dict["image_tokens_count"] - num_image_tokens
                )

        return logits


def get_conversation(task, question, metadata=None):
    """Get a conversation for a given task and evaluation question."""
    conversation = []

    # In all cases, the tokenizer adds possible header tokens for the assistant.
    if task == "captioning":
        conversation = [
            {"role": "system", "content": "Answer the questions."},
            {
                "role": "user",
                "content": f"{IMAGE_TOKEN}\nGive a brief description of this image in one sentence.",
            },
        ]
    elif task in ("TextVQA", "InfoVQA", "SPDocVQA"):
        conversation = [
            {"role": "system", "content": "Follow the user's instruction and answer questions."},
            {
                "role": "user",
                "content": f"{IMAGE_TOKEN}\n{question}\nAnswer the question using a single word, phrase, or number.",
            },
        ]
    elif task == "VQAv2":
        conversation = [
            {"role": "system", "content": "Follow the user's instruction and answer questions."},
            {
                "role": "user",
                "content": f"{IMAGE_TOKEN}\n{question}\nAnswer the question using a single word or phrase.",
            },
        ]
    elif task == "ChartQA":
        conversation = [
            {"role": "system", "content": "Follow the user's instruction and answer questions."},
            {
                "role": "user",
                "content": f"{IMAGE_TOKEN}\n{question}\nAnswer the question using a single word or phrase.",
            },
        ]
    elif task == "MMMU":
        conversation = [
            {"role": "system", "content": "Answer the questions."},
            {"role": "user", "content": f"{IMAGE_TOKEN}\n{question}"},
        ]
    elif task == "VideoMME":
        q = (
            "Select the best answer to the following multiple-choice "
            "question based on the video. Respond with only the letter "
            "(A, B, C, or D) of the correct option.\n"
        )
        q += question["questions"][0]["question"] + "\n"
        q += question["questions"][0]["choices"][0] + "\n"
        q += question["questions"][0]["choices"][1] + "\n"
        q += question["questions"][0]["choices"][2] + "\n"
        q += question["questions"][0]["choices"][3] + "\n"

        conversation = [
            {"role": "system", "content": "Answer the questions."},
            {"role": "user", "content": f"{IMAGE_TOKEN}\n{q}"},
        ]
    elif task in ("OCRBench", "OCRBench_v2", "RD_TableBench"):
        conversation = [
            {"role": "system", "content": "Follow the user's instruction and answer questions."},
            {"role": "user", "content": f"{IMAGE_TOKEN}\n{question}"},
        ]
    elif task == "MathVista":
        conversation = [
            {"role": "system", "content": "You are math expert. Use your math knowledge to calculate the answer."},
            {"role": "user", "content": f"{IMAGE_TOKEN}\n{question}"},
        ]
    elif task == "RealworldQA":
        conversation = [
            {"role": "system", "content": "Follow the user's instruction and answer questions."},
            {"role": "user", "content": f"{IMAGE_TOKEN}\n{question}"},
        ]
    elif task == "AI2D":
        conversation = [
            {"role": "system", "content": "Follow the user's instruction and answer questions."},
            {"role": "user", "content": f"{IMAGE_TOKEN}\n{question}"},
        ]
    elif task == "MotionBench":
        extra_instruction = "Respond with only the letter choice (A, B, C, or D) of the correct option.\n"
        conversation = [
            {"role": "system", "content": "Answer the questions."},
            {"role": "user", "content": f"{IMAGE_TOKEN}\n{question}\n{extra_instruction}"},
        ]
    elif task == "PhysGameBench":
        extra_instruction = "Respond with only the letter choice (A, B, C, or D) of the correct option.\n"
        conversation = [
            {"role": "system", "content": "Answer the questions."},
            {"role": "user", "content": f"{IMAGE_TOKEN}\n{question}\n{extra_instruction}"},
        ]
    elif task == "MVBench":
        conversation = [
            {"role": "system", "content": "Answer the questions."},
            {"role": "user", "content": f"{IMAGE_TOKEN}\n{question}\nAnswer the question using a single word or phrase."},
        ]
    elif task in ["PerceptionTest"]:
        conversation = [
            {"role": "system", "content": "Answer the questions."},
            {"role": "user", "content": f"{IMAGE_TOKEN}\n{question}"},
        ]
    elif task == "inference":
        conversation = [
            {"role": "system", "content": "Answer the questions."},
            {"role": "user", "content": f"{question}"},
        ]
    else:
        raise NotImplementedError(f"No prompting support for task {task}")


    return conversation


def get_prompt_and_generated(prompt_and_generation, prompt_format):
    """Strip prompt and other unnecessary text from generation."""
    if prompt_format in ("llama3", "llama3p1"):
        splitted = prompt_and_generation.split("<|start_header_id|>assistant<|end_header_id|>\n\n")
        prompt = splitted[0]
        generated = splitted[1]
        generated = generated.split("<|eot_id|>")[0]
    elif prompt_format == "mistral":
        splitted = prompt_and_generation.split("[/INST]")
        prompt = splitted[0]
        generated = splitted[1]
        generated = generated.split("</s>")[0]
    elif prompt_format == "chatml":
        splitted = prompt_and_generation.split("<|im_start|> assistant\n")
        prompt = splitted[0]
        generated = splitted[1]
        generated = generated.split("<|im_end|>")[0]
    elif prompt_format in ("nvlm-yi-34b", "qwen2p0", "qwen2p5"):
        splitted = prompt_and_generation.split("<|im_start|>assistant\n")
        prompt = splitted[0]
        generated = splitted[1]
        generated = generated.split("<|im_end|>")[0]
    elif prompt_format in ("nemotron5"):
        splitted = prompt_and_generation.split("<SPECIAL_14>assistant\n")
        prompt = splitted[0]
        generated = splitted[1]
        generated = generated.split("<SPECIAL_15>")[0]
    elif prompt_format in ("nemotron5-aligned"):
        splitted = prompt_and_generation.split("Assistant\n")
        prompt = splitted[0]
        generated = splitted[1]
        generated = generated.split("[PREFIX]")[0]
        generated = generated.split("\\n")[0]
    else:
        raise ValueError(f"Prompt format {prompt_format} is not supported.")

    # Remove possible garbage.
    generated = generated.strip()

    return prompt, generated


def run_eval(config, iteration=None):
    # Run evaluation.
    print(f"====== {config.task} {config.dataset} at iteration={iteration} scores ======")

    if config.task == "TextVQA":
        from evaluation.evaluate_textvqa import textvqa_eval
        avg_acc = textvqa_eval(config.output_path)

        score = {"TextVQA accuracy": avg_acc}
        with open(config.output_path + "-scores.txt", "a") as f:
            f.write(f"{config.task} {config.dataset} at iteration={iteration} TextVQA accuracy: {score}\n")

    elif config.task == "OCRBench":
        from evaluation.evaluate_ocrbench import ocrbench_eval
        log, avg_acc = ocrbench_eval(config.output_path)

        score = {"OCRBench accuracy": avg_acc}
        with open(config.output_path + "-scores.txt", "a") as f:
            f.write(f"{config.task} {config.dataset} at iteration={iteration} OCRBench accuracy: {score}\n")
            f.write(f"{log}\n")

    elif config.task == "MathVista":
        from evaluation.evaluate_mathvista import mathvista_eval
        avg_acc = mathvista_eval(config.output_path)

        score = {"MathVista accuracy": avg_acc}
        with open(config.output_path + "-scores.txt", "a") as f:
            f.write(f"{config.task} {config.dataset} at iteration={iteration} MathVista accuracy: {score}\n")

    elif config.task == "ChartQA":
        from evaluation.evaluate_chartqa import chartqa_eval
        avg_acc = chartqa_eval(config.output_path)

        score = {"ChartQA accuracy": avg_acc}
        with open(config.output_path + "-scores.txt", "a") as f:
            f.write(f"{config.task} {config.dataset} at iteration={iteration} ChartQA accuracy: {score}\n")

    elif config.task == "SPDocVQA":
        from evaluation.evaluate_spdocvqa import spdocvqa_eval
        avg_acc = spdocvqa_eval(config.output_path)

        score = {"SPDocVQA accuracy": avg_acc}
        with open(config.output_path + "-scores.txt", "a") as f:
            f.write(f"{config.task} {config.dataset} at iteration={iteration} SPDocVQA accuracy: {score}\n")

    elif config.task == "RealworldQA":
        from evaluation.evaluate_realworldqa import realworldqa_eval
        avg_acc = realworldqa_eval(config.output_path)

        score = {"RealworldQA accuracy": avg_acc}
        with open(config.output_path + "-scores.txt", "a") as f:
            f.write(f"{config.task} {config.dataset} at iteration={iteration} RealworldQA accuracy: {score}\n")

    elif config.task == "AI2D":
        from evaluation.evaluate_ai2d import ai2d_eval
        avg_acc = ai2d_eval(config.output_path)

        score = {f"AI2D {config.dataset} accuracy": avg_acc}
        with open(config.output_path + "-scores.txt", "a") as f:
            f.write(f"{config.task} {config.dataset} at iteration={iteration} AI2D accuracy: {score}\n")

    elif config.task == "MMMU":
        from evaluation.evaluate_mmmu import convert_to_mmmu_format
        from examples.multimodal.evaluation.mmmu_utils import mmmu_main_eval
        result_file = convert_to_mmmu_format(config.output_path)
        result = json.load(open(result_file))
        mmmu_results = mmmu_main_eval(result, {"answer_dict": config.gt_path})
        with open(config.output_path + "-scores.txt", "a") as f:
            f.write(f"{config.task} {config.split} at iteration={iteration} :\n")
            for cat, cat_val in mmmu_results.items():
                if 'Overall' in cat:
                    cat = cat.replace("Overall-", "")
                    print(f'{cat}: {cat_val["acc"] * 100:.2f}')
                    f.write(f'{cat}: {cat_val["acc"] * 100:.2f}\n')

        score = {"MMMU val accuracy": mmmu_results['Overall']['acc']}
    elif config.task == 'captioning':
        from evaluation.evaluate_coco import coco_captioning_eval
        cider_score = coco_captioning_eval(config.output_path, config.gt_path)
        score = {f"{config.task} {config.dataset} CIDEr": cider_score}

        with open(config.output_path + "-scores.txt", "a") as f:
            f.write(f"{config.task} {config.dataset} CIDEr scores at iteration={iteration}: {cider_score}\n")
    elif config.task == 'MotionBench':
        from evaluation.evaluate_video_motionbench import motionbench_eval
        avg_acc = motionbench_eval(config.output_path)

        score = {f"MotionBench accuracy": avg_acc}
        with open(config.output_path + "-scores.txt", "a") as f:
            f.write(f"{config.task} {config.dataset} scores at iteration={iteration}: {score}\n")
    elif config.task == 'PhysGameBench':
        from evaluation.evaluate_video_phys_game_bench import phys_game_bench_eval
        avg_acc_dict = phys_game_bench_eval(config.output_path)

        score = {f"PhysGame Total accuracy": avg_acc_dict['Physgame-Total-Acc']}
        with open(config.output_path + "-scores.txt", "a") as f:
            f.write(f"{config.task} {config.dataset} scores at iteration={iteration}: {avg_acc_dict}\n")
    elif config.task == "MVBench":
        from evaluation.evaluate_video_mvbench import mvbench_eval
        avg_acc_dict = mvbench_eval(config.output_path)

        score = {f"MVBench accuracy": avg_acc_dict['total-acc']}
        with open(config.output_path + "-scores.txt", "a") as f:
            f.write(f"{config.task} {config.dataset} scores at iteration={iteration}: {avg_acc_dict}\n")
    elif config.task == "inference":
        score = {"Inference accuracy:": None}
        pass
    else:
        raise NotImplementedError(f"Evaluation of {config.task} not implemented yet")

    print(score)
    return score


def run_evaluation_loop(model, configs, output_dir_override=None, iteration=None, print_output=True):
    """
    Common evaluation loop used by both online evaluation during training and standalone evaluation.

    Args:
        model: The model to evaluate
        configs: Dict[str, EvaluationConfig] - dictionary of evaluation configs
        output_dir_override: Optional directory to override the output path in configs
        iteration: Optional iteration number for logging
        print_output: Whether to print generation output

    Returns:
        Dict[str, float]: Dictionary of evaluation scores
    """
    args = get_args()
    scores = {}

    for key, config in configs.items():
        # Handle output path override for online evaluation
        if output_dir_override:
            config.output_path = os.path.join(output_dir_override, args.language_model_type)

        # Generate samples and write to file
        generate_and_write_samples(model, config, print_output=print_output)

        # Synchronize before evaluation
        torch.distributed.barrier()

        # Run evaluation on the last rank
        if is_last_rank():
            task_scores = run_eval(config, iteration=iteration)
            scores.update(task_scores)

        # Synchronize after evaluation
        torch.distributed.barrier()

    return scores


def eval_tasks():
    """Vision language model text generation for single or batch tasks."""
    initialize_megatron(extra_args_provider=add_text_generation_args)

    args = get_args()

    def wrapped_model_provider(pre_process, post_process, add_encoder=True, add_decoder=True):
        return model_provider(pre_process, post_process, add_encoder=add_encoder, add_decoder=add_decoder,
                              parallel_output=False)

    # Set up model and load checkpoint.
    model = get_model(wrapped_model_provider, model_type=ModelType.encoder_and_decoder, wrap_with_ddp=False)

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    model = model[0]
    model.eval()

    configs = get_evaluation_configs()

    # Use the common evaluation loop
    run_evaluation_loop(model, configs, iteration=args.ckpt_step)


if __name__ == "__main__":
    eval_tasks()
