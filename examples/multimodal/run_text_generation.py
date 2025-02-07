# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
"""Generate text using a vision language model."""
import json
import logging
import os
import sys
from functools import partial

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
from megatron.inference.text_generation.communication import broadcast_int_list
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.engines.mcore_engine import MCoreEngine
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
from megatron.training import get_args, get_model, get_tokenizer, print_rank_0
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron


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
            "VideoMME",
            "OCRBench",
            "MathVista",
            "AI2D",
        ],
        help="Generation task to run",
    )
    group.add_argument(
        "--num-samples-per-partition", type=int, default=0, help="Number of samples per partition"
    )
    group.add_argument("--config-path", type=str, help="Evaluation config file to use.")

    group.add_argument("--use-mcore-inference", action="store_true", default=False, help="Use the MCore inference API")

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
        inference_engine = MCoreEngine(
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

        conv = get_conversation(config.task, question)

        if not args.use_mcore_inference:
            forward_step = partial(VLMForwardStep, num_img_embeddings_per_tile, imgs, num_tiles, args.decoder_seq_length)


        if is_first_rank():

            if args.use_mcore_inference:
                inference_request = VLMInferenceRequest(
                   request_id=inference_engine.get_new_request_id(),
                   prompt=conv,
                   prompt_tokens=controller.tokenize_prompt(conv),
                   inference_parameters=sampling_params,
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
                model,
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
                ):
                    output_name = "answer"
                elif config.task in ("MMMU"):
                    output_name = "text"
                elif config.task == "VideoMME":
                    output_name = "response"
                    output = question
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

                if config.task == "captioning":
                    output["ground_truth"] = answers
                elif config.task in (
                    "TextVQA",
                    "VQAv2",
                    "ChartQA",
                    "OCRBench",
                    "MathVista",
                    "AI2D",
                ):
                    if isinstance(answers, str):
                        answers = [answers]
                    output["gt_answer"] = answers

                    if len(metadata) > 0:
                        output.update(metadata)
                elif config.task == "MMMU":
                    output["prediction"] = generated
                    output.update(metadata)
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
                   inference_parameters=sampling_params,
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
                    model, forward_step=forward_step, detokenize_segments=False, data_parallel=True
                )

            idx += 1


def get_evaluation_config():
    """Get evaluation config from a config file or command-line arguments."""
    args = get_args()
    if args.config_path:
        with open(args.config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        config = EvaluationConfig(**config_dict)
    else:
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

    # Default output path if not defined...
    if not config.output_path:
        os.makedirs("generated", exist_ok=True)
        config.output_path = "generated/" + args.language_model_type

    return config


def is_first_rank():
    """First tensor and pipeline parallel rank."""
    return (
        parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        and parallel_state.get_tensor_model_parallel_rank() == 0
    )


def get_output_path(config, dp_rank):
    """Generation output path."""
    return (
        f"{config.output_path}-{config.task}-dprank={dp_rank}-partition={config.partition_id}.jsonl"
    )


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
        max_batch_size,
        max_sequence_length,
    ):
        """Create multimodal forward step."""
        total_num_tiles = torch.sum(num_tiles).item()
        num_img_embeddings = num_img_embeddings_per_tile * total_num_tiles

        super().__init__(model, max_batch_size, max_sequence_length + num_img_embeddings)
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
            inference_params=self.inference_params,
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
            if "image_tokens_count" not in self.inference_params.key_value_memory_dict:
                self.inference_params.key_value_memory_dict["image_tokens_count"] = self._num_img_embeddings

            if self._num_img_embeddings + num_tokens - num_image_tokens > self.decoder_seq_length:
                self.inference_params.sequence_len_offset += self.decoder_seq_length - num_tokens
            else:
                self.inference_params.sequence_len_offset += (
                    self.inference_params.key_value_memory_dict["image_tokens_count"] - num_image_tokens
                )

        return logits


def get_conversation(task, question):
    """Get a conversation for a given task and evaluation question."""
    conversation = []

    # In all cases, the tokenizer adds possible header tokens for the assistant.
    if task == "captioning":
        conversation = [
            {"role": "system", "content": "Answer the questions."},
            {
                "role": "user",
                "content": f"{IMAGE_TOKEN}\nProvide a one-sentence caption for provided image.",
            },
        ]
    elif task in ("TextVQA", "VQAv2", "ChartQA"):
        conversation = [
            {"role": "system", "content": "Answer the questions."},
            {
                "role": "user",
                "content": f"{IMAGE_TOKEN}\n{question}\nAnswer the question using a single word or phrase.",
            },
        ]
    elif task in ("OCRBench", "MathVista", "AI2D"):
        conversation = [
            {"role": "system", "content": "Answer the questions."},
            {"role": "user", "content": f"{IMAGE_TOKEN}\n{question}"},
        ]
    elif task == "MMMU":
        conversation = [
            {"role": "system", "content": "Answer the questions."},
            {"role": "user", "content": question},
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
    else:
        raise ValueError(f"Prompt format {prompt_format} is not supported.")

    # Remove possible garbage.
    generated = generated.strip()
    generated = generated.split("\n\n")[0]
    generated = generated.split("\n")[0]

    return prompt, generated


def main():
    """Vision language model text generation."""
    initialize_megatron(extra_args_provider=add_text_generation_args)

    if torch.distributed.get_rank() == 0:
        logging.getLogger(__name__).warning(
            "Models using pipeline parallelism are not supported yet."
        )

    args = get_args()

    def wrapped_model_provider(pre_process, post_process, add_encoder, add_decoder):
        return model_provider(pre_process, post_process, add_encoder, add_decoder, parallel_output=False)

    # Set up model and load checkpoint.
    model = get_model(wrapped_model_provider, model_type=ModelType.encoder_and_decoder, wrap_with_ddp=False)

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    model = model[0]

    model.eval()

    config = get_evaluation_config()

    generate_and_write_samples(model, config)


if __name__ == "__main__":
    main()
