# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import ast
import inspect
import json
from typing import Any, Dict, Iterable, List, Optional, Union
import librosa
import numpy as np

from megatron.core import mpu
from megatron.core.inference.inference_request import InferenceRequest, AVLMInferenceRequest
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.inference.text_generation.communication import broadcast_float_list
from megatron.inference.text_generation.tokenization import tokenize_prompts
from megatron.training import get_args
from megatron.training.tokenizer.multimodal_tokenizer import MultimodalTokenizer

class ModelInferenceWrapperServer(GPTInferenceWrapper):
    def __init__(self, model, inference_wrapper_config):
        super().__init__(model, inference_wrapper_config)

    def get_batch_for_context_window(
        self,
        inference_input: Dict[str, Any],
        context_start_position: int,
        context_end_position: int,
    ) -> Dict[str, Any]:
        """
        Slices out the tokens, position ids, and masking for the specific context window.

        Args:
            inference_input (Dict[str, Any]): The inference input for the batch.
            context_start_position (int): Start of the context window.
            context_end_position (int): End of the context window.

        Returns:
            Dict[str, Any]: Inputs used in the forward call.
        """
        inference_input = super().get_batch_for_context_window(
            inference_input, context_start_position, context_end_position
        )
        return inference_input


def run_mcore_engine(
    engine,
    prompts=None,
    temperature=1.0,
    top_k=0,
    top_p=0.0,
    logprobs=True,
    tokens_to_generate=0,
    top_n_logprobs=0,
    echo=False,
    random_seed=-1,
):
    """Server-compatible version of the MCore Engine, used in
    tools/run_text_generation_server.py."""

    values = [tokens_to_generate, logprobs, top_k, top_p, temperature, top_n_logprobs, random_seed]
    values_float_tensor = broadcast_float_list(len(values), float_list=values, data_parallel=False)
    tokens_to_generate = int(values_float_tensor[0].item())
    return_output_log_probs = bool(values_float_tensor[1].item())
    top_k = int(values_float_tensor[2].item())
    top_p = values_float_tensor[3].item()
    temperature = values_float_tensor[4].item()
    top_n_logprobs = int(values_float_tensor[5].item())
    random_seed = int(values_float_tensor[6].item())

    if random_seed > 0:
        engine.text_generation_controller.sampling_rng.manual_seed(random_seed)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        return_segments=True,
        return_log_probs=return_output_log_probs,
        num_tokens_to_generate=tokens_to_generate,
        top_n_logprobs=top_n_logprobs,
        return_prompt_top_n_logprobs=True
    )

    context_tokens_tensor, context_length_tensor = tokenize_prompts(
        prompts=prompts, tokens_to_generate=tokens_to_generate, add_BOS=False, data_parallel=False
    )

    tokenized_prompts = []
    for p, l in zip(context_tokens_tensor, context_length_tensor):
        tokenized_prompts.append(p[:l].cpu().numpy().tolist())

    tokenizer = engine.text_generation_controller.tokenizer

    # detect if detokenize supports skip_special_tokens or **kwargs
    sig_params = inspect.signature(tokenizer.detokenize).parameters.values()
    accepts_skip = any(
        p.name == "skip_special_tokens" or p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig_params
    )

    # Detokenize prompts into strings to pass through the engine
    detokenized_prompts = [
        (
            tokenizer.detokenize(p, skip_special_tokens=True)
            if accepts_skip
            else tokenizer.detokenize(p)
        )
        for p in tokenized_prompts
    ]

    requests = []
    for i in range(len(tokenized_prompts)):
        req = InferenceRequest(
            prompt=detokenized_prompts[i],
            prompt_tokens=tokenized_prompts[i],
            sampling_params=sampling_params,
            request_id=engine.get_new_request_id(),
        )
        requests.append(req)

    result = engine.generate(inference_requests=requests)

    # Only post-process on first stage.
    if mpu.is_pipeline_first_stage():
        response_dict = {
            "text": [x.prompt + x.generated_text for x in result],
            "tokens": [x.prompt_tokens + x.generated_tokens.tolist() for x in result],
        }
        if sampling_params.return_log_probs:
            response_logprobs = [x.prompt_log_probs + x.generated_log_probs for x in result]
            response_dict["logprobs"] = response_logprobs
        if sampling_params.return_segments:
            response_dict["segments"] = [x.segments for x in result]
        if sampling_params.top_n_logprobs > 0:
            # TODO(ksanthanam): Support disabling `return_prompt_top_n_logprobs`
            assert sampling_params.return_prompt_top_n_logprobs
            response_dict["top_n_logprobs"] = [
                x.prompt_top_n_logprobs + x.generated_top_n_logprobs for x in result
            ]

        return response_dict
    return None


def is_chat_completion_request(prompt: str) -> bool:
    """
    Check if the prompt is a chat completion request.
    The prompt is a chat completion request if it is a valid list of dicts.
    If the prompt is a completion request, it is a string, and ast.literal_eval() will raise a SyntaxError.
    """
    try:
        data = ast.literal_eval(prompt)
        return isinstance(data, list) and all(isinstance(item, dict) for item in data)
    except SyntaxError:
        return False


def get_audio_samples(data: dict, sample_rate: int, audio_feature_duration: float, audio_pad_duration: float = None, audio_file_field: str = "path") -> np.ndarray:
    """
    Get the audio samples from the data.
    Supports both file paths and base64-encoded audio (OpenAI format).
    """
    audio_file = data.get(audio_file_field, None)
    
    # Support base64-encoded audio: {"data": "<base64>", "format": "wav"}
    if audio_file is None and "data" in data:
        import base64
        import io
        import soundfile as sf
        try:
            audio_bytes = base64.b64decode(data["data"])
            wav, orig_sr = sf.read(io.BytesIO(audio_bytes))
            if orig_sr != sample_rate:
                wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sample_rate)
        except Exception as e:
            print(f"Error decoding base64 audio: {e}")
            return None, 0, 0
    elif audio_file is not None:
        wav, _ = librosa.load(audio_file, sr=sample_rate)
    else:
        return None, 0, 0
    
    wav_len = len(wav)
    dur = librosa.get_duration(y=wav, sr=sample_rate)
    num_audio_embeddings = int(dur / audio_feature_duration) + 1
    if audio_pad_duration is not None and audio_pad_duration > 0:
        num_audio_embeddings = int(audio_pad_duration / audio_feature_duration) + 1
        max_audio_samples = int(audio_pad_duration * sample_rate)
        if dur < audio_pad_duration:
            wav = np.pad(wav, (0, max_audio_samples - len(wav)))
        elif dur > audio_pad_duration:
            wav = wav[:max_audio_samples]
            wav_len = max_audio_samples
    return wav, wav_len, num_audio_embeddings


def collate_audio_wav_list(audio_wav_list: List[np.ndarray]) -> np.ndarray:
    """
    Collate the audio wav list into a single tensor of shape (num_audio_clips, max_audio_length).
    """
    max_audio_length = max(len(wav) for wav in audio_wav_list)
    audio_clips = np.zeros((len(audio_wav_list), max_audio_length))
    for i, wav in enumerate(audio_wav_list):
        audio_clips[i, :len(wav)] = wav
    return audio_clips


def parse_avlm_chat_completion_request_from_prompt(prompt: str, tokenizer, sampling_params: SamplingParams, request_id: int) -> AVLMInferenceRequest:
    args = get_args()

    from megatron.core.models.multimodal.llava_model import SOUND_TOKEN
    messages = ast.literal_eval(prompt)  # list of dicts

    new_messages = []
    audio_wav_list = []
    audio_len_list = []
    total_num_sound_embeddings = 0
    for message in messages:
        if message["role"] == "user":
            content = message.get("content", "")
            audio_data = None
            
            # Handle OpenAI multimodal format where content is a list
            # e.g. [{"type": "text", "text": "..."}, {"type": "input_audio", "input_audio": {...}}]
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "input_audio":
                            audio_data = item.get("input_audio", {})
                        elif item.get("type") == "audio_url":
                            audio_data = item.get("audio_url", {})
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = " ".join(text_parts)
                message["content"] = content
            
            # Check for audio in separate "audio" field (original format)
            if "audio" in message:
                audio_data = message["audio"]
            
            # Process audio if found
            if audio_data:
                audio_wav, audio_len, num_audio_embed = get_audio_samples(
                    data=audio_data,
                    sample_rate=args.audio_sample_rate,
                    audio_feature_duration=args.audio_feature_duration,
                    audio_pad_duration=args.audio_pad_duration
                )
                if audio_wav is not None:
                    audio_wav_list.append(audio_wav)
                    audio_len_list.append(audio_len)
                    sound_tokens = "".join([SOUND_TOKEN] * num_audio_embed)
                    total_num_sound_embeddings += num_audio_embed
                    message["content"] = f"{message['content']}{args.audio_start_token}{sound_tokens}{args.audio_end_token}"
            elif "image" in message:
                raise ValueError("Image is not yet supported in chat completion request")
        new_messages.append(message)
    
    if len(audio_wav_list) > 0:
        sound_length = np.array(audio_len_list)  # (num_audio_clips,)
        sound_clips = collate_audio_wav_list(audio_wav_list)  # (num_audio_clips, max_audio_length)
    else:
        sound_length = None
        sound_clips = None

    assert isinstance(tokenizer, MultimodalTokenizer), "MultimodalTokenizer is required for chat completion request"

    input_ids = tokenizer.tokenize(new_messages)
    actual_prompt = tokenizer.detokenize(input_ids)
    print("--------------------------------")
    print(f"actual_prompt: {actual_prompt}")
    print("--------------------------------")
    req = AVLMInferenceRequest(
        request_id=request_id,
        prompt=actual_prompt,
        prompt_tokens=input_ids,
        sampling_params=sampling_params,
        decoder_seq_length=args.decoder_seq_length,
        num_img_embeddings=0,
        imgs=None,
        num_tiles=None,
        imgs_sizes=None,
        vision_packed_seq_params=None,
        num_sound_embeddings=total_num_sound_embeddings,
        sound_clips=sound_clips,
        sound_length=sound_length,
    )
    return req


def parse_avlm_request_from_prompt(prompt: str, tokenizer, sampling_params: SamplingParams, request_id: int) -> AVLMInferenceRequest:
    """
    Parse an AVLM inference request from a formatted prompt after tokenizer.apply_chat_template().
    Currently only support text-only evaluation tasks
    """
    args = get_args()
    if is_chat_completion_request(prompt):
        return parse_avlm_chat_completion_request_from_prompt(prompt, tokenizer, sampling_params, request_id)

    req = AVLMInferenceRequest(
        request_id=request_id,
        prompt=prompt,
        prompt_tokens=tokenizer.tokenize(prompt),
        sampling_params=sampling_params,
        decoder_seq_length=args.decoder_seq_length,
        num_img_embeddings=0,
        imgs=None,
        num_tiles=None,
        imgs_sizes=None,
        vision_packed_seq_params=None,
        num_sound_embeddings=0,
        sound_clips=None,
        sound_length=None,
    )
    return req


def run_mcore_engine_avlm(
    engine,
    prompts=None,
    temperature=1.0,
    top_k=0,
    top_p=0.0,
    logprobs=True,
    tokens_to_generate=0,
    top_n_logprobs=0,
):
    """Server-compatible version of the MCore Engine, used in
    tools/run_text_generation_server.py."""

    values = [tokens_to_generate, logprobs, top_k, top_p, temperature, top_n_logprobs]
    values_float_tensor = broadcast_float_list(len(values), float_list=values, data_parallel=False)
    tokens_to_generate = int(values_float_tensor[0].item())
    return_output_log_probs = bool(values_float_tensor[1].item())
    top_k = int(values_float_tensor[2].item())
    top_p = values_float_tensor[3].item()
    temperature = values_float_tensor[4].item()
    top_n_logprobs = int(values_float_tensor[5].item())

    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        return_segments=True,
        return_log_probs=return_output_log_probs,
        num_tokens_to_generate=tokens_to_generate,
        top_n_logprobs=top_n_logprobs,
        return_prompt_top_n_logprobs=True
    )

    # broadcast the tokenized prompts to all ranks and then detokenize them back to strings
    context_tokens_tensor, context_length_tensor = tokenize_prompts(
        prompts=prompts, tokens_to_generate=tokens_to_generate, add_BOS=False, data_parallel=False
    )

    tokenized_prompts = []
    for p, l in zip(context_tokens_tensor, context_length_tensor):
        tokenized_prompts.append(p[:l].cpu().numpy().tolist())

    tokenizer = engine.text_generation_controller.tokenizer

    # detect if detokenize supports skip_special_tokens or **kwargs
    sig_params = inspect.signature(tokenizer.detokenize).parameters.values()
    accepts_skip = any(
        p.name == "skip_special_tokens" or p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig_params
    )

    # Detokenize prompts into strings to pass through the engine
    detokenized_prompts = [
        (
            tokenizer.detokenize(p, skip_special_tokens=False)
            if accepts_skip
            else tokenizer.detokenize(p)
        )
        for p in tokenized_prompts
    ]
    
    # drop extra header added by the tokenizer during tokenize_prompts function, if it exists
    detokenized_prompts_cleaned = []
    for i, d_prompt in enumerate(detokenized_prompts):
        # use the first segment of the prompt to find the index of the extra header
        # since sometimes the detokenized prompt differs from the original prompt on white space tokens
        idx = d_prompt.find(prompts[i].split(" ")[0])
        if idx != -1:
            detokenized_prompts_cleaned.append(d_prompt[idx:])
        else:
            detokenized_prompts_cleaned.append(d_prompt)
    detokenized_prompts = detokenized_prompts_cleaned

    requests = []
    for i in range(len(tokenized_prompts)):
        req = parse_avlm_request_from_prompt(detokenized_prompts[i], tokenizer, sampling_params, engine.get_new_request_id())
        requests.append(req)

    print(f"[DEBUG] About to call engine.generate with {len(requests)} requests", flush=True)
    result = engine.generate(inference_requests=requests)
    print(f"[DEBUG] engine.generate completed", flush=True)

    # Clear the inference context state after generation to avoid reusing
    # cached audio/image embeddings from previous requests
    inference_context = engine.text_generation_controller.inference_wrapped_model.inference_context
    if inference_context is not None and hasattr(inference_context, 'key_value_memory_dict'):
        # Clear audio and image token counts that trigger KV cache reuse
        inference_context.key_value_memory_dict.pop("sound_tokens_count", None)
        inference_context.key_value_memory_dict.pop("image_tokens_count", None)

    # Only post-process on first stage.
    if mpu.is_pipeline_first_stage():
        response_dict = {
            "text": [x.prompt + x.generated_text for x in result],
            "generated_text": [x.generated_text for x in result],
            "tokens": [x.prompt_tokens + x.generated_tokens.tolist() for x in result],
        }
        if sampling_params.return_log_probs:
            response_logprobs = [x.prompt_log_probs + x.generated_log_probs for x in result]
            response_dict["logprobs"] = response_logprobs
        if sampling_params.return_segments:
            response_dict["segments"] = [x.segments for x in result]
        if sampling_params.top_n_logprobs > 0:
            # TODO(ksanthanam): Support disabling `return_prompt_top_n_logprobs`
            assert sampling_params.return_prompt_top_n_logprobs
            response_dict["top_n_logprobs"] = [
                x.prompt_top_n_logprobs + x.generated_top_n_logprobs for x in result
            ]

        return response_dict
    return None
