# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import contextlib
import datetime
import json
import logging
import os
import sys

try:
    from flask import Flask, jsonify, request
    from flask_restful import Api, Resource

    HAVE_FLASK = True
except ImportError as e:
    Resource = object

    HAVE_FLASK = False

from megatron.core.inference.text_generation_server.endpoints.common import LOCK, send_do_generate
from megatron.core.inference.text_generation_server.endpoints.completions import MegatronCompletions
from megatron.core.inference.text_generation_server.run_mcore_engine import run_mcore_engine

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)


class MegatronGenerate(Resource):
    """Text generation endpoint."""

    def __init__(self, engine, args):
        self.engine = engine
        self.args = args
        self.verbose = getattr(args, 'inference_flask_server_logging', False)

    def put(self):
        """Handle generation request."""
        if not "prompts" in request.get_json():
            return "prompts argument required", 400

        if "max_len" in request.get_json():
            return "max_len is no longer used.  Replace with tokens_to_generate", 400

        if "sentences" in request.get_json():
            return "sentences is no longer used.  Replace with prompts", 400

        if "beam_width" in request.get_json():
            return "Beam search is no longer supported.", 400

        prompts = request.get_json()["prompts"]
        if not isinstance(prompts, list):
            return "prompts is not a list of strings", 400

        if len(prompts) == 0:
            return "prompts is empty", 400

        if len(prompts) > 128:
            return "Maximum number of prompts is 128", 400

        tokens_to_generate = 64  # Choosing hopefully sane default.  Full sequence is slow
        if "tokens_to_generate" in request.get_json():
            tokens_to_generate = request.get_json()["tokens_to_generate"]
            if not isinstance(tokens_to_generate, int):
                return "tokens_to_generate must be an integer greater than 0"
            if tokens_to_generate < 0:
                return "tokens_to_generate must be an integer greater than or equal to 0"

        logprobs = False
        if "logprobs" in request.get_json():
            logprobs = request.get_json()["logprobs"]
            if not isinstance(logprobs, bool):
                return "logprobs must be a boolean value"

        if tokens_to_generate == 0 and not logprobs:
            return "tokens_to_generate=0 implies logprobs should be True"

        temperature = 1.0
        if "temperature" in request.get_json():
            temperature = request.get_json()["temperature"]
            if not (isinstance(temperature, (int, float))):
                return "temperature must be a positive number less than or equal to 100.0"
            if not (0.0 < temperature <= 100.0):
                return "temperature must be a positive number less than or equal to 100.0"

        top_k = 0
        if "top_k" in request.get_json():
            top_k = request.get_json()["top_k"]
            if not (isinstance(top_k, int)):
                return (
                    "top_k must be an integer equal to or greater than 0 "
                    "and less than or equal to 1000"
                )
            if not (0 <= top_k <= 1000):
                return "top_k must be equal to or greater than 0 and less than or equal to 1000"

        top_p = 0.0
        if "top_p" in request.get_json():
            top_p = request.get_json()["top_p"]
            if not (isinstance(top_p, float)):
                return "top_p must be a positive float less than or equal to 1.0"
            if top_p > 0.0 and top_k > 0.0:
                return "cannot set both top-k and top-p samplings."
            if not (0 <= top_p <= 1.0):
                return "top_p must be less than or equal to 1.0"

        top_p_decay = 0.0
        if "top_p_decay" in request.get_json():
            top_p_decay = request.get_json()["top_p_decay"]
            if not (isinstance(top_p_decay, float)):
                return "top_p_decay must be a positive float less than or equal to 1.0"
            if top_p == 0.0:
                return "top_p_decay cannot be set without top_p"
            if not (0 <= top_p_decay <= 1.0):
                return "top_p_decay must be less than or equal to 1.0"

        top_p_bound = 0.0
        if "top_p_bound" in request.get_json():
            top_p_bound = request.get_json()["top_p_bound"]
            if not (isinstance(top_p_bound, float)):
                return "top_p_bound must be a positive float less than or equal to top_p"
            if top_p == 0.0:
                return "top_p_bound cannot be set without top_p"
            if not (0.0 < top_p_bound <= top_p):
                return "top_p_bound must be greater than 0 and less than top_p"

        add_BOS = False
        if "add_BOS" in request.get_json():
            add_BOS = request.get_json()["add_BOS"]
            if not isinstance(add_BOS, bool):
                return "add_BOS must be a boolean value"

        if any([len(prompt) == 0 for prompt in prompts]) and not add_BOS:
            return "Empty prompts require add_BOS=true"

        stop_on_double_eol = False
        if "stop_on_double_eol" in request.get_json():
            stop_on_double_eol = request.get_json()["stop_on_double_eol"]
            if not isinstance(stop_on_double_eol, bool):
                return "stop_on_double_eol must be a boolean value"

        stop_on_eol = False
        if "stop_on_eol" in request.get_json():
            stop_on_eol = request.get_json()["stop_on_eol"]
            if not isinstance(stop_on_eol, bool):
                return "stop_on_eol must be a boolean value"

        prevent_newline_after_colon = False
        if "prevent_newline_after_colon" in request.get_json():
            prevent_newline_after_colon = request.get_json()["prevent_newline_after_colon"]
            if not isinstance(prevent_newline_after_colon, bool):
                return "prevent_newline_after_colon must be a boolean value"

        random_seed = -1
        if "random_seed" in request.get_json():
            random_seed = request.get_json()["random_seed"]
            if not isinstance(random_seed, int):
                return "random_seed must be integer"
            if random_seed < 0:
                return "random_seed must be a positive integer"

        stop_token = 50256
        if "stop_token" in request.get_json():
            stop_token = request.get_json()["stop_token"]
            if not isinstance(stop_token, int):
                return "stop_token must be an integer"

        length_penalty = 1
        if "length_penalty" in request.get_json():
            length_penalty = request.get_json()["length_penalty"]
            if not isinstance(length_penalty, float):
                return "length_penalty must be a float"

        with LOCK:  # Need to get lock to keep multiple threads from hitting code

            if self.verbose:
                logging.info(f"request IP: {str(request.remote_addr)}")
                logging.info(json.dumps(request.get_json()))
                logging.info(f"start time: {datetime.datetime.now()}")

            # OTel: instrument the request handler with GenAI semantic conventions.
            # https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
            # Gated by the INFERENCE span group so users can selectively disable.
            _otel_tracer = _otel_meter = None
            try:
                from nemo.lens.state import is_span_group_enabled
                if is_span_group_enabled('inference'):
                    from nemo.lens.helpers import span_cm
                    from nemo.lens.propagation import extract_context
                    from opentelemetry import trace as _trace_mod, metrics as _metrics_mod
                    _otel_tracer = _trace_mod.get_tracer('megatron.core')
                    _otel_meter = _metrics_mod.get_meter('megatron.core')
                    _parent_ctx = extract_context(dict(request.headers))
                else:
                    _parent_ctx = None
            except Exception:
                _parent_ctx = None

            # Resolve model identifier used in span name and attributes.
            _model_id = (
                str(getattr(self.args, 'model_type', 'megatron'))
                if self.args is not None else 'megatron'
            )

            # Build GenAI-compliant span attributes.
            # Required: gen_ai.operation.name, gen_ai.provider.name
            # Conditionally required / recommended: model, max_tokens, temperature, top_k/p, seed
            _span_attrs: dict = {
                'gen_ai.operation.name': 'text_completion',
                'gen_ai.provider.name': 'megatron',
                'gen_ai.request.model': _model_id,
                'gen_ai.request.max_tokens': tokens_to_generate,
                'gen_ai.request.temperature': float(temperature),
            }
            if top_k > 0:
                _span_attrs['gen_ai.request.top_k'] = float(top_k)
            if top_p > 0.0:
                _span_attrs['gen_ai.request.top_p'] = float(top_p)
            if random_seed >= 0:
                _span_attrs['gen_ai.request.seed'] = random_seed

            # Span name format: "{gen_ai.operation.name} {gen_ai.request.model}"
            _span_name = f"text_completion {_model_id}"
            _req_start = datetime.datetime.now()

            _request_cm = (
                span_cm(_span_name, tracer=_otel_tracer, **_span_attrs)
                if _otel_tracer is not None else contextlib.nullcontext()
            )

            try:
                with _request_cm as _req_span:
                    send_do_generate()  # Tell other ranks we're doing generate

                    response_dict = run_mcore_engine(
                        self.engine, prompts, temperature, top_k, top_p, logprobs, tokens_to_generate
                    )

                    # Set output token count on span if derivable from response.
                    if _req_span is not None and isinstance(response_dict, dict):
                        _output_tokens = sum(
                            len(v.get('tokens', []))
                            for v in response_dict.values()
                            if isinstance(v, dict)
                        )
                        if _output_tokens:
                            _req_span.set_attribute('gen_ai.usage.output_tokens', _output_tokens)

                _req_duration_s = (datetime.datetime.now() - _req_start).total_seconds()
                if _otel_meter is not None:
                    from nemo.lens.instruments.inference import record_inference_metrics
                    record_inference_metrics(
                        meter=_otel_meter,
                        request_duration_s=_req_duration_s,
                        model=_model_id,
                    )

                return jsonify(response_dict)

            except ValueError as ve:
                return ve.args[0]


class MegatronServer(object):
    """Megatron text generation server."""

    def __init__(self, model, args=None):
        if not HAVE_FLASK:
            raise RuntimeError(f"`flask` and/or `flask_restful` are not installed.")

        self.app = Flask(__name__, static_url_path='')
        api = Api(self.app)
        api.add_resource(MegatronGenerate, '/api', resource_class_args=[model, args])
        api.add_resource(MegatronCompletions, '/completions', resource_class_args=[model, args])

    def run(self, url, port):
        """Run the server."""
        self.app.run(url, threaded=True, debug=False, port=port)
