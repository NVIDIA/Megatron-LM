# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import logging
import traceback
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from megatron.core.inference.inference_request import (
    PREFIX_EOS_TOKEN_ID_FIELD,
    PREFIX_EXPANDED_TOKEN_COUNT_FIELD,
    PREFIX_MEDIA_COUNT_FIELD,
    PREFIX_TEMPLATE_TOKEN_IDS_FIELD,
    prepare_multimodal_data,
)
from megatron.core.tokenizers.text.parsers import PARSER_MAPPING

from ..incremental_detokenizer import HuggingFaceFastIncrementalDetokenizer
from ..openai_streaming import StreamingChatParser, openai_stream

# Chat rendering and reply building live in `chat_renderer` so frontends outside this server can
# reuse them; `format_chat_response` is re-exported for importers of its former home.
from .chat_renderer import (  # pylint: disable=unused-import
    ChatRenderer,
    _history_prefix,
    _last_assistant_message,
    _maybe_filter_parallel_tool_calls,
    apply_parsers,
    format_chat_response,
)
from .common import (
    failed_requests_response,
    parse_sampling_params,
    run_inference,
    sampling_params_for_choice,
    submit_requests,
    validate_offload_params,
)

logger = logging.getLogger(__name__)

# pylint: disable=line-too-long


def _replace_prefix_tokens(
    eos_token_ids,
    previous_turn_token_ids,
    retokenized_previous_turn_token_ids,
    current_turn_token_ids,
):
    """Replace the token ids that are associated with the previous turn with the actual tokens
    from the previous generation (rather than the ones from the chat template application)."""

    if not previous_turn_token_ids:
        return current_turn_token_ids

    eos_token_ids = _normalize_eos_token_ids(eos_token_ids)

    # Find the boundary of the current sequence's prompt.
    current_turn_additional_token_ids = _suffix_tokens_after_prefix(
        eos_token_ids, retokenized_previous_turn_token_ids, current_turn_token_ids
    )
    if previous_turn_token_ids[-1] in eos_token_ids:
        # Preserve the exact EOS emitted previously. The rendered suffix begins
        # with its own boundary EOS, which may be a different accepted EOS ID.
        current_turn_additional_token_ids = current_turn_additional_token_ids[1:]

    # Return the previous turn token ids + the current turn token ids
    return previous_turn_token_ids + current_turn_additional_token_ids


def _has_previous_turn_tokens(last_assistant_message):
    """True when the last assistant message carries the token ids of a previous
    Megatron-Inference response, so the endpoint can replace the prefix with the exact prior turn here.
    Dataset-provided conversation history won't have these fields."""
    if last_assistant_message is None:
        return False
    prompt_token_ids = last_assistant_message.get("prompt_token_ids")
    generation_token_ids = last_assistant_message.get("generation_token_ids")
    # Check that we have non-empty prompt or generation tokens.
    return (
        isinstance(prompt_token_ids, list)
        and isinstance(generation_token_ids, list)
        and bool(prompt_token_ids or generation_token_ids)
    )


def _replace_prefix_tokens_metadata(eos_token_ids, template_prefix_token_ids, offload_params):
    """Ship the rendered prior-turn tokens so the engine's RequestPromptPreparer can replace the
    prefix with the exact prior tokens itself (NeMo RL's ``replace_prefix_tokens``)."""
    return {
        **offload_params,
        PREFIX_TEMPLATE_TOKEN_IDS_FIELD: list(template_prefix_token_ids),
        PREFIX_EOS_TOKEN_ID_FIELD: _serialize_eos_token_ids(eos_token_ids),
    }


def _expanded_prefix_stitching_metadata(prefix_media_count, expanded_prefix_token_count):
    """Mark the exact, already-expanded prefix so the engine only expands the tokens after it."""
    return {
        PREFIX_MEDIA_COUNT_FIELD: prefix_media_count,
        PREFIX_EXPANDED_TOKEN_COUNT_FIELD: expanded_prefix_token_count,
    }


def _normalize_eos_token_ids(eos_token_ids):
    """Normalize one or more EOS IDs to a validated set."""
    if type(eos_token_ids) is int:
        eos_token_ids = [eos_token_ids]
    if (
        not isinstance(eos_token_ids, (list, tuple, set, frozenset))
        or not eos_token_ids
        or not all(type(token_id) is int for token_id in eos_token_ids)
    ):
        raise ValueError("EOS token IDs must be a non-empty integer or collection of integers.")
    return frozenset(eos_token_ids)


def _serialize_eos_token_ids(eos_token_ids):
    """Serialize one or more EOS IDs as a JSON-compatible list."""
    return sorted(_normalize_eos_token_ids(eos_token_ids))


def _suffix_tokens_after_prefix(eos_token_ids, template_prefix_token_ids, current_tokens):
    """Return the current-turn suffix, beginning at the prefix's final EOS."""
    # Count the number of EOS tokens in the retokenized prefix.
    eos_token_ids = _normalize_eos_token_ids(eos_token_ids)
    eos_count = sum(token_id in eos_token_ids for token_id in template_prefix_token_ids)
    if eos_count <= 0:
        raise ValueError(
            "Could not locate an EOS-delimited previous turn: the chat template's turn "
            f"terminator is not among the model EOS token IDs {sorted(eos_token_ids)}."
        )

    # Scan current_tokens from beginning to end. Return the suffix after eos_count
    # EOS tokens have been seen.
    # This guards against changes in templating or even upstream modification of
    # the prefix tokens to ensure we retrieve the current turn's suffix / prompt.
    seen_eos = 0
    for position, token_id in enumerate(current_tokens):
        if token_id in eos_token_ids:
            seen_eos += 1
            if seen_eos == eos_count:
                return current_tokens[position:]
    raise ValueError(
        f"Expected {eos_count} EOS token(s) before the new turn, but found only {seen_eos}."
    )


def _contains_model_media_token(token_ids, tokenizer, prompt_config):
    """Whether exact prior model tokens still reference image/video embeddings."""
    if not isinstance(token_ids, list) or not hasattr(tokenizer, "convert_tokens_to_ids"):
        return False

    media_token_ids = set()
    for modality in ("image", "video"):
        spec = prompt_config.get_spec(modality)
        token_id = tokenizer.convert_tokens_to_ids(spec.model_token)
        if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
            media_token_ids.add(int(token_id))
    return any(type(token_id) is int and token_id in media_token_ids for token_id in token_ids)


class _PrefixMode(Enum):
    """Where the exact tokens of the previous turn come from, if anywhere."""

    NONE = "none"
    # The client echoed back the token ids of a previous response on the last assistant message;
    # the prefix is replaced here.
    EXACT = "exact"
    # The client sent offload_params, so the exact tokens live in a store the engine's
    # RequestPromptPreparer can reach; we ship what the preparer needs.
    OFFLOADED = "offloaded"


@dataclass(frozen=True)
class _PrefixSource:
    """The previous-turn source decision together with the last assistant turn it refers to."""

    mode: _PrefixMode
    last_assistant_idx: Optional[int]
    last_assistant_message: Optional[dict]


def _resolve_prefix_source(template_messages, prevent_retokenization, offload_params):
    """Decide the previous-turn source; the two sources are mutually exclusive."""
    last_idx, last_message = _last_assistant_message(template_messages)
    has_previous_turn_tokens = _has_previous_turn_tokens(last_message)
    if has_previous_turn_tokens and offload_params is not None:
        raise ValueError(
            "prompt_token_ids/generation_token_ids on the last assistant message and "
            "'offload_params' are mutually exclusive prefix sources"
        )
    if prevent_retokenization and has_previous_turn_tokens:
        mode = _PrefixMode.EXACT
    elif offload_params is not None and last_message is not None:
        mode = _PrefixMode.OFFLOADED
    else:
        mode = _PrefixMode.NONE
    return _PrefixSource(mode, last_idx, last_message)


async def _stitch_previous_turn(
    renderer,
    options,
    prefix_source,
    prompt_tokens,
    offload_params,
    template_messages,
    template_tools,
    media_slots,
):
    """Replace the re-rendered previous turn with its exact tokens, here or via the engine."""
    last_message = prefix_source.last_assistant_message
    messages_to_last_assistant, previous_media_slots = _history_prefix(
        template_messages, media_slots, prefix_source.last_assistant_idx
    )
    if (
        prefix_source.mode is _PrefixMode.EXACT
        and not previous_media_slots
        and _contains_model_media_token(
            last_message["prompt_token_ids"], renderer.render_tokenizer, renderer.prompt_config
        )
    ):
        raise ValueError(
            "The exact previous prompt contains media tokens, but its image/video "
            "payload is missing from message history. Preserve prior media content "
            "when using prevent_retokenization."
        )
    eos_token_ids = renderer.eos_token_ids()
    warnings.warn(
        "Avoiding prefix retokenization. "
        "This is a patch that ensures subsequent generations are not retokenized "
        "differently than the previous generation. "
        "This may cause unexpected behavior if messages (including system messages) "
        "are altered between generations."
    )
    # The templated tokenization of just the previous generation.
    retokenized_prefix = await renderer.tokenize_async(
        messages_to_last_assistant,
        previous_media_slots,
        tools=template_tools,
        chat_template_kwargs=options.chat_template_kwargs,
        add_generation_prompt=False,
    )

    if prefix_source.mode is _PrefixMode.OFFLOADED:
        offload_params = _replace_prefix_tokens_metadata(
            eos_token_ids, retokenized_prefix, offload_params
        )
        if previous_media_slots:
            # Post-expansion stitching needs the media count to know how many subsequent
            # media tokens to expand.
            offload_params[PREFIX_MEDIA_COUNT_FIELD] = len(previous_media_slots)
        return prompt_tokens, offload_params

    previous_turn_token_ids = (
        last_message["prompt_token_ids"] + last_message["generation_token_ids"]
    )
    prompt_tokens = _replace_prefix_tokens(
        eos_token_ids, previous_turn_token_ids, retokenized_prefix, prompt_tokens
    )
    if previous_media_slots:
        # The previous turn is already expanded; the engine only expands the media tokens after it.
        offload_params = _expanded_prefix_stitching_metadata(
            len(previous_media_slots), len(previous_turn_token_ids)
        )
    return prompt_tokens, offload_params


async def _render_prompt(
    renderer, options, template_messages, template_tools, media_slots, prefix_source, offload_params
):
    """Tokenize the conversation; when a previous-turn source exists, stitch its exact tokens in."""
    prompt_tokens = await renderer.tokenize_async(
        template_messages,
        media_slots,
        tools=template_tools,
        chat_template_kwargs=options.chat_template_kwargs,
        add_generation_prompt=True,
    )
    if prefix_source.mode is _PrefixMode.NONE:
        return prompt_tokens, offload_params
    return await _stitch_previous_turn(
        renderer,
        options,
        prefix_source,
        prompt_tokens,
        offload_params,
        template_messages,
        template_tools,
        media_slots,
    )


def _streaming_chat_parsers(parsers, options, n):
    """One `StreamingChatParser` per choice, or `None` when no parsers are configured."""
    if not parsers:
        return None
    marker_prefixes = (
        tuple(
            marker
            for parser_name in parsers
            for marker in getattr(PARSER_MAPPING[parser_name], "streaming_markers", ())
        )
        if options.tools_requested
        else ()
    )

    def parse_streaming_text(text, finished=False):
        parsed_text, metadata = apply_parsers(
            text,
            options.tools,
            parsers,
            options.tools_requested,
            chat_template_kwargs=options.chat_template_kwargs,
            finished=finished,
        )
        metadata["tool_calls"] = _maybe_filter_parallel_tool_calls(
            metadata.get("tool_calls", []), options.parallel_tool_calls
        )
        return parsed_text, metadata

    return [
        StreamingChatParser(
            parse_streaming_text,
            marker_prefixes=marker_prefixes,
            named_tool_choice=options.is_named_tool_choice,
        )
        for _ in range(n)
    ]


try:
    import orjson

    HAVE_ORJSON = True
except ImportError:
    HAVE_ORJSON = False


try:
    from quart import Blueprint, Response, current_app, jsonify, request

    bp = Blueprint('chat_completions_api', __name__)

    def _streaming_response(
        client,
        tokenizer,
        parsers,
        options,
        prompt_tokens,
        sampling_params,
        n,
        *,
        multi_modal_data,
        offload_params,
    ):
        """SSE response streaming `n` choices of one prompt; 400 when cannot stream."""
        # Streaming currently supports only Hugging Face fast tokenizers.
        try:
            incremental_detokenizers = [
                HuggingFaceFastIncrementalDetokenizer(tokenizer, prompt_tokens) for _ in range(n)
            ]
        except ValueError as error:
            return Response(str(error), status=400)
        streams = [
            client.add_request_streaming(
                prompt_tokens,
                sampling_params_for_choice(sampling_params, index),
                multi_modal_data=multi_modal_data,
                offload_params=offload_params,
            )
            for index in range(n)
        ]
        response = Response(
            openai_stream(
                streams,
                tokenizer,
                incremental_detokenizers,
                chat=True,
                return_log_probs=sampling_params.return_log_probs,
                include_usage=options.include_usage,
                chat_parsers=_streaming_chat_parsers(parsers, options, n),
            ),
            content_type="text/event-stream",
        )
        response.timeout = None
        return response

    @bp.route('/chat/completions', methods=['POST'])
    @bp.route('/v1/chat/completions', methods=['POST'])
    async def chat_completions():
        """Handles async POST requests for chat completions."""
        client = current_app.config['client']
        tokenizer = current_app.config['tokenizer']
        parsers = current_app.config['parsers']
        # Capability checks read the shared tokenizer; rendering runs on the executor's private
        # copy, since HF tokenizers are not thread-safe. Callers that build the app config directly
        # register no copy, so the shared tokenizer is the fallback.
        renderer = ChatRenderer(
            tokenizer,
            parsers,
            current_app.config['multimodal_prompt_config'],
            render_tokenizer=current_app.config.get('tokenizer_copy', tokenizer),
            executor=current_app.config.get('tokenize_executor'),
            chat_template=current_app.config.get("chat_template", None),
            eval_mode=current_app.config.get("eval_mode", False),
            # Remote media responses bypass Quart's request-body limit, so apply the same bound.
            max_media_fetch_bytes=current_app.config.get("MAX_CONTENT_LENGTH"),
        )

        req = await request.get_json()
        offload_params = req.get("offload_params")
        offload_params_error = validate_offload_params(offload_params)
        if offload_params_error is not None:
            return Response(offload_params_error, status=400)
        try:
            options = renderer.parse_options(req)
        except ValueError as error:
            return Response(str(error), status=400)

        # --- 1. Parse Messages ---
        messages = req.get("messages")
        if not messages:
            return Response("Missing 'messages' field", status=400)
        if not isinstance(messages, list):
            return Response("'messages' must be a list", status=400)
        # Extract structured media before template sanitization. Remote image fetches block, so
        # keep this work off the event loop.
        try:
            messages, multi_modal_data, media_slots = await asyncio.to_thread(
                renderer.extract_media, messages
            )
        except ValueError as error:
            return Response(str(error), status=400)
        template_messages, template_tools = renderer.sanitize(messages, media_slots, options.tools)

        try:
            prefix_source = _resolve_prefix_source(
                template_messages, options.prevent_retokenization, offload_params
            )
        except ValueError as error:
            return Response(str(error), status=400)

        try:
            if renderer.has_chat_template(options):
                prompt_tokens, offload_params = await _render_prompt(
                    renderer,
                    options,
                    template_messages,
                    template_tools,
                    media_slots,
                    prefix_source,
                    offload_params,
                )
            else:
                prompt_tokens = renderer.tokenize_without_template(messages, media_slots)
        except ValueError as e:
            logger.error(f"{traceback.format_exc()}")
            return Response(f"Invalid 'messages': {e}", status=400)
        except Exception as e:
            logger.error(f"{traceback.format_exc()}")
            return Response(f"Error processing 'messages': {e}", status=500)

        # --- 2. Parse Sampling Params ---
        try:
            sampling_params, extras = parse_sampling_params(
                req,
                current_app.config,
                tokenizer,
                completions_mode=False,
                return_prompt_tokens=options.return_tokenized_data or options.return_raw_text,
            )
            n = extras["n"]  # Number of choices to generate
            prompt_tokens = renderer.apply_bos_policy(
                prompt_tokens, sampling_params.add_BOS, offload_params
            )
        except (ValueError, TypeError) as e:
            return Response(f"Invalid sampling parameter: {e}", status=400)

        # --- 3. Send Requests to Engine ---
        # Hash and serialize shared media once before fanning one prompt out to n independently
        # sampled choices; coordinator affinity keeps equivalent requests on the engine that owns
        # the cached vision embedding.
        prepared_multimodal_data = prepare_multimodal_data(multi_modal_data)
        if options.stream:
            return _streaming_response(
                client,
                tokenizer,
                parsers,
                options,
                prompt_tokens,
                sampling_params,
                n,
                multi_modal_data=prepared_multimodal_data,
                offload_params=offload_params,
            )

        # add_request_with_id, not add_request: a non-streaming response writes
        # nothing to the socket while generating, so a disconnect is never
        # discovered as a broken pipe. Aborting needs the request ids.
        try:
            request_ids, tasks = submit_requests(
                client,
                (
                    (prompt_tokens, sampling_params_for_choice(sampling_params, index))
                    for index in range(n)
                ),
                multi_modal_data=prepared_multimodal_data,
                offload_params=offload_params,
            )
        except Exception as e:
            logger.error(f"Error submitting request: {e}")
            return Response(f"Error submitting request: {e}", status=500)

        try:
            batch_results = await run_inference(
                client, request_ids, tasks, current_app.config['verbose'], log_label=f"(n={n})"
            )
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return Response(f"Error during inference: {e}", status=500)

        # --- 4. Check for failed requests ---
        failure = failed_requests_response(batch_results, len(prompt_tokens))
        if failure is not None:
            body, status = failure
            return Response(body, status=status)

        # --- 5. Format OpenAI Response ---
        response = renderer.build_response(
            batch_results, sampling_params, options, verbose=current_app.config['verbose']
        )

        if HAVE_ORJSON:
            # Use orjson for faster serialization
            return Response(orjson.dumps(response), mimetype="application/json")
        else:
            return jsonify(response)

except ImportError as e:
    logger.warning(f"Could not import quart: {e}")
