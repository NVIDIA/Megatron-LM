# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import json
import logging
import time
import traceback
import uuid
import warnings

from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.tokenizers.text.parsers import PARSER_MAPPING

logger = logging.getLogger(__name__)

try:
    import orjson

    HAVE_ORJSON = True
except ImportError:
    HAVE_ORJSON = False


try:
    from quart import Blueprint, Response, current_app, jsonify, request

    bp = Blueprint('chat_completions_api', __name__)

    def apply_parsers(text, tools, parsers_list):
        """Runs CPU-intensive text parsing."""
        meta = {}
        for parser in parsers_list:
            if parser not in PARSER_MAPPING:
                raise ValueError(f"Parser {parser} not found in PARSER_MAPPING")
            text, new_info = PARSER_MAPPING[parser].parse(text, tools=tools)
            assert not (
                meta.keys() & new_info.keys()
            ), "Multiple parsers found the same information."
            meta.update(new_info)
        return text, meta

    @bp.route('/chat/completions', methods=['POST'])
    @bp.route('/v1/chat/completions', methods=['POST'])
    async def chat_completions():
        """Handles async POST requests for chat completions."""
        client = current_app.config['client']
        tokenizer = current_app.config['tokenizer']
        parsers = current_app.config['parsers']

        req = await request.get_json()

        # --- 1. Parse Messages ---
        messages = req.get("messages")
        if not messages:
            return Response("Missing 'messages' field", status=400)
        if not isinstance(messages, list):
            return Response("'messages' must be a list", status=400)

        # The OpenAI spec sends tool_call arguments as a JSON string, but
        # Jinja chat templates iterate over them with |items, requiring a dict.
        for msg in messages:
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", tc)
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            fn["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            pass

        try:
            prompt_tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                tools=req.get("tools", None),
                **req.get("chat_template_kwargs", {}),
            )
        except (AttributeError, AssertionError):
            warnings.warn(
                "Tokenizer does not support 'apply_chat_template'. Using tokenize instead."
            )
            prompt_tokens = tokenizer.tokenize(
                "\n".join([message["content"] for message in messages])
            )
        except Exception as e:
            logger.error(f"{traceback.format_exc()}")
            return Response(f"Error processing 'messages': {e}", status=500)

        # --- 2. Parse Sampling Params ---
        try:
            temperature = float(req.get("temperature", 1.0))
            top_p = float(req.get("top_p", 1.0))
            top_k = int(req.get("top_k", 0))
            n = int(req.get("n", 1))  # Number of choices to generate

            if temperature == 0.0:
                top_k = 1
                top_p = 0.0

            # Check for 'logprobs' (bool) and 'top_logprobs' (int)
            return_log_probs = bool(req.get("logprobs", False))
            top_n_logprobs = int(req.get("top_logprobs", 0)) if return_log_probs else 0
            skip_prompt_log_probs = bool(req.get("skip_prompt_log_probs", True))
            add_BOS = bool(req.get("add_BOS", False))

            # The engine only handles add_BOS for string prompts, not pre-tokenized
            # input. Since we pre-tokenize via apply_chat_template, we must handle
            # BOS ourselves, matching the logic in tokenize_prompt().
            if hasattr(tokenizer, 'bos') and tokenizer.bos is not None:
                start_idx = 0
                while start_idx < len(prompt_tokens) and prompt_tokens[start_idx] == tokenizer.bos:
                    start_idx += 1
                if start_idx > 0:
                    prompt_tokens = prompt_tokens[start_idx:]

                if add_BOS:
                    prompt_tokens = [tokenizer.bos] + prompt_tokens

            max_tokens = req.get("max_completion_tokens", None) or req.get("max_tokens", None)

            sampling_params = SamplingParams(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                return_log_probs=return_log_probs,
                top_n_logprobs=top_n_logprobs,
                num_tokens_to_generate=(int(max_tokens) if max_tokens is not None else None),
                skip_prompt_log_probs=skip_prompt_log_probs,
                add_BOS=add_BOS,
            )
        except ValueError as e:
            return Response(f"Invalid sampling parameter: {e}", status=400)

        # --- 3. Send Requests to Engine ---
        tasks = [client.add_request(prompt_tokens, sampling_params) for _ in range(n)]

        if current_app.config['verbose']:
            start_time = time.perf_counter()

        try:
            batch_results = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return Response(f"Error during inference: {e}", status=500)

        if current_app.config['verbose']:
            logging.info(
                f"Batch of {len(tasks)} requests (n={n}) processed in "
                f"{time.perf_counter() - start_time:.2f}s"
            )

        # --- 4. Format OpenAI Response ---
        choices = []
        total_completion_tokens = 0
        prompt_tokens_counts = []

        request_idx = 0
        for record in batch_results:
            result = record.merge().serialize()

            result = {
                k: v[1] if isinstance(v, (list, tuple)) and len(v) == 2 and v[0] == "tensor" else v
                for k, v in result.items()
            }
            prompt_tokens_out = result["prompt_tokens"]
            text_output = result["generated_text"]
            prompt_tokens_count = len(prompt_tokens_out) if prompt_tokens_out is not None else 0
            prompt_tokens_counts.append(prompt_tokens_count)

            logprobs_content = None
            if sampling_params.return_log_probs:
                token_logprobs = result.get('log_probs', [])

                tokens_to_decode = [[tok] for tok in result["generated_tokens"]]
                tokens = list(map(tokenizer.detokenize, tokens_to_decode))

                # Get top_n_logprobs if available
                generated_top_n_logprobs = result.get('generated_top_n_logprobs')

                logprobs_content = []
                for i, (tok, lp) in enumerate(zip(tokens, token_logprobs)):
                    # Build top_logprobs list for this token position
                    top_logprobs_list = []
                    if generated_top_n_logprobs and i < len(generated_top_n_logprobs):
                        top_n_dict = generated_top_n_logprobs[i]
                        for token_str, logprob in top_n_dict.items():
                            top_logprobs_list.append(
                                {
                                    "token": token_str,
                                    "logprob": logprob,
                                    "bytes": list(token_str.encode("utf-8")),
                                }
                            )

                    logprobs_content.append(
                        {
                            "token": tok,
                            "logprob": lp,
                            "bytes": list(tok.encode("utf-8")),
                            "top_logprobs": top_logprobs_list,
                        }
                    )

            metadata = {}
            message_text = text_output

            if parsers:
                message_text, metadata = apply_parsers(
                    message_text, req.get("tools", None), parsers
                )

            message = {"role": "assistant", "content": message_text}
            if "tool_calls" in metadata:
                message["tool_calls"] = metadata["tool_calls"]
            if "reasoning" in metadata:
                message["reasoning"] = metadata["reasoning"]

            # Replicate data in the message field for compatibility.
            message["prompt_token_ids"] = result["prompt_tokens"]
            message["generation_token_ids"] = result["generated_tokens"]
            message["generation_log_probs"] = result.get("generated_log_probs", [])
            return_log_probs = sampling_params.return_log_probs

            gen_length = result.get("generated_length") or len(result.get("generated_tokens", []))
            max_gen = result.get("sampling_params", {})
            if isinstance(max_gen, dict):
                max_gen = max_gen.get("num_tokens_to_generate", None)
            elif hasattr(max_gen, "num_tokens_to_generate"):
                max_gen = max_gen.num_tokens_to_generate
            else:
                max_gen = None
            if metadata.get("tool_calls", []):
                finish_reason = "tool_calls"
            elif max_gen is not None and gen_length >= max_gen:
                finish_reason = "length"
            else:
                finish_reason = "stop"

            choice_data = {
                "index": request_idx,
                "message": message,
                "prompt_token_ids": result["prompt_tokens"],
                "generation_token_ids": result["generated_tokens"],
                "generation_log_probs": result.get("generated_log_probs", []),
                "raw_text": result["prompt"] + result["generated_text"],
                "logprobs": (
                    {"content": logprobs_content} if sampling_params.return_log_probs else None
                ),
                "finish_reason": "tool_calls" if metadata.get("tool_calls", []) else finish_reason,
            }
            choice_data["policy_staleness"] = result["policy_staleness"]
            choice_data["kv_cache_staleness"] = result["kv_cache_staleness"]
            choice_data["num_evictions"] = sum(
                1 for e in result["events"] if e.get("type") == "EVICT"
            )
            if current_app.config['verbose']:
                logging.info(result)

            if result["routing_indices"] is not None:
                choice_data["moe_topk_indices"] = result["routing_indices"]
                if prompt_tokens_count:
                    choice_data["prompt_moe_topk_indices"] = result["routing_indices"][
                        :prompt_tokens_count
                    ]

            choices.append(choice_data)
            total_completion_tokens += len(result["generated_tokens"])
            request_idx += 1

        prompt_token_count = max(prompt_tokens_counts) if prompt_tokens_counts else 0
        response = {
            "id": str(uuid.uuid4()),
            "created": int(time.time()),
            "model": "EMPTY",
            "object": "chat.completion",
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": total_completion_tokens,
                "total_tokens": prompt_token_count + total_completion_tokens,
            },
        }

        if HAVE_ORJSON:
            # Use orjson for faster serialization
            return Response(orjson.dumps(response), mimetype="application/json")
        else:
            return jsonify(response)

except ImportError as e:
    logger.warning(f"Could not import quart: {e}")
