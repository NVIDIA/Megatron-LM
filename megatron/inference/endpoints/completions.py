# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""This endpoint is for mimicking the OpenAI completions API.
See https://platform.openai.com/docs/api-reference/completions/create
"""

import torch
import numpy as np
from megatron.training import get_tokenizer
from megatron.inference.text_generation.mcore_engine_server import run_mcore_engine
from megatron.inference.text_generation.api import generate_and_post_process
from megatron.inference.endpoints.common import send_do_generate, LOCK

from flask import request, jsonify
from flask_restful import Resource


def detokenize(prompt, tok) -> list[str]:
    if isinstance(prompt, str):
        return [prompt]
    elif isinstance(prompt, list):
        if not prompt:  # The list is empty, can't determine its intended type.
            raise ValueError(f"prompt contains no items: {prompt}")
        if all(isinstance(item, str) for item in prompt):
            return prompt
        elif all(isinstance(item, int) for item in prompt):
            return [tok.detokenize(prompt[0])]
        elif all(  # list[list[int]]
            isinstance(item, list) and all(isinstance(subitem, int) for subitem in item)
            for item in prompt
        ):
            return [tok.detokenize(item) for item in prompt]
        else:
            raise ValueError(f"Unknown prompt type: {type(prompt)}")
    else:
        raise ValueError(f"Unknown prompt type: {type(prompt)}")


class MegatronCompletions(Resource):
    def __init__(self, engine, args):
        self.engine = engine
        self.args = args

    def post(self):
        req = request.get_json()
        tokenizer = get_tokenizer()
        prompts = detokenize(req["prompt"], tokenizer)

        # convert the openai-local-completions api to the format
        # expected by the generate_and_post_process function
        local_kwargs = {
            "prompts": prompts,
            "tokens_to_generate": int(req["max_tokens"]),
            "temperature": float(req.get("temperature", 1.0)),
            "top_k_sampling": int(req.get("top_k", 0)),
            "top_p_sampling": float(req.get("top_p", 1.0)),
            "return_topk_logprobs": int(req.get("logprobs", 0)),
            "echo": bool(req.get("echo", False)),
            "random_seed": int(req.get("seed", -1)),
            "best_of": int(req.get("best_of", 1)),
            "num_completions": int(req.get("n", 1)),
            "stop": req.get("stop", [tokenizer.detokenize([tokenizer.eod])]),
            "return_output_log_probs": True,
        }

        if isinstance(local_kwargs["stop"], str):
            local_kwargs["stop"] = [local_kwargs["stop"]]

        if local_kwargs["temperature"] == 0:
            # temperature = 0 is openai api's way of specifying greedy
            # deterministic sampling but actually passing temperature=0
            # is undefined and leads to div by zero, so set top-k = 1
            local_kwargs["top_k_sampling"] = 1
            local_kwargs["top_p_sampling"] = 0

        echo = local_kwargs.pop("echo")
        if (not echo) and (local_kwargs["tokens_to_generate"] == 0):
            return "echo=False not supported when tokens_to_generate=0", 400

        if local_kwargs.pop("best_of") > 1:
            return "best_of > 1 not supported", 400

        if local_kwargs.pop("num_completions") > 1:
            return "num_completions > 1 not supported", 400

        if local_kwargs["tokens_to_generate"] > 0 and local_kwargs["return_topk_logprobs"] > 0:
            return "cannot return top-k unless tokens_to_generate=0 at this time", 400

        if local_kwargs["return_topk_logprobs"] > 10:
            return "return_topk_logprobs > 10 not supported", 400

        stop_until = local_kwargs.pop("stop")

        with LOCK:
            send_do_generate()

            temperature = local_kwargs["temperature"]
            top_k = local_kwargs["top_k_sampling"]
            top_p = local_kwargs["top_p_sampling"]
            tokens_to_generate = local_kwargs["tokens_to_generate"]
            logprobs = local_kwargs["return_output_log_probs"]
            top_n_logprobs = local_kwargs["return_topk_logprobs"]
            response_dict = run_mcore_engine(
                self.engine,
                prompts,
                temperature,
                top_k,
                top_p,
                logprobs,
                tokens_to_generate,
                top_n_logprobs=top_n_logprobs,
                echo=echo,
            )
            result = [
                response_dict["text"],
                response_dict["segments"],
                response_dict.get("logprobs", None),
                response_dict["tokens"],
            ]
            result.append(response_dict.get("top_n_logprobs", None))

        prompts_plus_generations, prompts_plus_generations_segments = result[:2]
        output_log_probs, tokens = result[2:4]
        logprobs_topk = result[4]

        if top_n_logprobs > 0:
            assert logprobs_topk is not None

        if "debug_fname" in req:
            torch.save(
                {
                    "args": local_kwargs,
                    "tokenizer": tokenizer,
                    "prompts_plus_generations": prompts_plus_generations,
                    "prompts_plus_generations_segments": prompts_plus_generations_segments,
                    "output_log_probs": output_log_probs,
                    "tokens": tokens,
                    "logprobs_topk": logprobs_topk,
                },
                f"completions_result_{req['debug_fname']}.pt",
            )

        batch_size = len(tokens)

        results = []
        for batch_idx, (prompt_plus_generation, prompt) in enumerate(
            zip(prompts_plus_generations, prompts)
        ):
            tok_offsets = tokenizer.offsets(tokens[batch_idx], prompt_plus_generation)
            if echo:
                str_trunc_start_idx, tok_idx_start = 0, 0
            else:
                str_trunc_start_idx = len(prompt)
                tok_idx_start = np.searchsorted(tok_offsets, len(prompt))

            # truncate the generation at the first stop token
            trunc_idxs = [
                prompt_plus_generation.find(suffix, str_trunc_start_idx)
                for suffix in stop_until
                if suffix and suffix in prompt_plus_generation
            ]
            str_trunc_end_idx = min(
                filter(lambda x: x != -1, trunc_idxs), default=len(prompt_plus_generation)
            )
            truncated_generation = prompt_plus_generation[str_trunc_start_idx:str_trunc_end_idx]

            # TODO(sasatheesh): handle cases where truncated_generation is not a full token
            tok_idx_end = np.searchsorted(tok_offsets, len(truncated_generation))

            truncated_generation_logprobs = output_log_probs[batch_idx][tok_idx_start:tok_idx_end]
            truncated_generation_tokens = tokens[batch_idx][tok_idx_start:tok_idx_end]
            truncated_generation_topk_logprobs = (
                logprobs_topk[batch_idx][tok_idx_start:tok_idx_end]
                if logprobs_topk is not None
                else None
            )

            truncated_generation_tok_offsets = tok_offsets[tok_idx_start:tok_idx_end]

            results.append(
                {
                    "index": batch_idx,
                    "text": truncated_generation,
                    "logprobs": {
                        "token_logprobs": [None] + truncated_generation_logprobs,
                        "tokens": [
                            tokenizer.detokenize([tk]) for tk in truncated_generation_tokens
                        ],
                        "text_offset": truncated_generation_tok_offsets,
                        "top_logprobs": [None] + truncated_generation_topk_logprobs if truncated_generation_topk_logprobs is not None else None,
                    },
                }
            )

        return jsonify({"choices": results})
