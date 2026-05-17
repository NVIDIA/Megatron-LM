# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import hashlib
import itertools
import json
import random
import time
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from functools import partial
from typing import Any, List, Optional

import torch
from tqdm import tqdm

from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.contexts.dynamic_context import get_mem_size_str
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.transformer.module import MegatronModule
from megatron.training import get_args


def get_default_sampling_params(termination_id: int = None):
    return SamplingParams(
        temperature=1.0,
        top_k=1,
        top_p=0.0,
        return_log_probs=False,
        num_tokens_to_generate=30,
        termination_id=termination_id,
    )


def get_curr_time() -> float:
    """Get synchronized time across ranks."""
    curr_time = torch.cuda.LongTensor([time.time_ns()])
    if torch.distributed.is_initialized():
        torch.distributed.broadcast(curr_time, src=0)
    return curr_time.item() / 10**9


class Request:
    """Class to hold attributes for a single request.

    A request is initialized with its prompt text. As it is added, processed,
    and completed through the inference engine, the request is populated with its
    start time, end time, and output tokens.

    Args:
        prompt_text (str): Prompt text.
        time_offset (float): Artificial time offset for simulating incoming
            requests. This value is later added to the `base_arrival_time` to
            simulate the requests arrival time.
        tokenizer (Any): Tokenizer for tokenizing the prompt.
    """

    def __init__(
        self,
        prompt_text: str,
        time_offset: float,
        tokenizer: Any,
        sampling_params: SamplingParams = None,
    ):
        self.prompt_text = prompt_text
        self.prompt_tokens = tokenizer.tokenize(prompt_text)
        self.output_text = None
        self.output_tokens = []
        self.time_offset = time_offset
        self.time_arrival = None
        self.time_start = None
        self.time_end = None
        self.ttft = None  # Time-to-first-token in seconds
        self.state = "not-started"
        self.sampling_params: SamplingParams = (
            sampling_params
            if sampling_params is not None
            else get_default_sampling_params(tokenizer.eod)
        )
        self.sampling_params = copy.deepcopy(self.sampling_params)

    def __str__(self) -> str:
        return "state '%s'; toffset %.1e; prompt len %d; output len %d; '%s'" % (
            self.state,
            self.time_offset,
            len(self.prompt_tokens),
            len(self.output_tokens),
            self.prompt_text,
        )


def get_time_offsets(
    seed: int | None,
    incoming_requests_per_step: int,
    incoming_requests_per_sec: float,
    num_requests: int,
) -> list[float]:
    """Get example time offsets."""

    # Time offsets to add all requests at once.
    if incoming_requests_per_step is not None or incoming_requests_per_sec <= 0:
        return [-1] * num_requests

    # if num_requests is not None:
    incoming_requests_duration = num_requests / incoming_requests_per_sec
    incoming_requests_duration *= 2  # extra margin, to accomodate time sampling

    random.seed(seed)

    import simpy  # Guard against this import in test case

    # Generate random time offsets.
    def arrival(r):
        while True:
            yield env.timeout(random.expovariate(r))
            time_offsets.append(env.now)

    time_offsets = []
    env = simpy.Environment()
    env.process(arrival(incoming_requests_per_sec))
    env.run(incoming_requests_duration)

    # Ensure at least a single request.
    if len(time_offsets) == 0:
        time_offsets = [0.0]

    # Ensure first time is 0.
    time_offsets = [to - time_offsets[0] for to in time_offsets]

    # Truncate to num_requests.
    assert len(time_offsets) >= num_requests
    time_offsets = time_offsets[:num_requests]

    return time_offsets


def get_cli_requests(
    args: Namespace, tokenizer: Any, sampling_params: Optional[SamplingParams] = None
) -> list[Request]:

    # Get time offsets.
    t_offsets = get_time_offsets(
        args.seed,
        args.incoming_requests_per_step,
        args.incoming_requests_per_sec,
        len(args.prompts),
    )

    # Init requests.
    requests = [Request(p, t, tokenizer, sampling_params) for p, t in zip(args.prompts, t_offsets)]
    return requests


def get_synthetic_requests(
    args: Namespace, tokenizer: Any, sampling_params: Optional[SamplingParams] = None
) -> list[Request]:
    """Get example requests."""

    # Get time offsets.
    time_offsets = get_time_offsets(
        args.seed,
        args.incoming_requests_per_step,
        args.incoming_requests_per_sec,
        int(args.incoming_requests_per_sec * args.incoming_requests_duration),
    )

    # Build prompts with expected lengths.
    assert (
        len(args.num_tokens_to_prompt) == 2
        and args.num_tokens_to_prompt[1] >= args.num_tokens_to_prompt[0]
    )
    max_prompt_length = args.num_tokens_to_prompt[1]
    max_prompt_text = "hi " * max_prompt_length
    max_prompt_tokens = tokenizer.tokenize(max_prompt_text)
    prompt_lengths = [random.randint(*args.num_tokens_to_prompt) for _ in time_offsets]
    prompt_tokens_list = [max_prompt_tokens[:l] for l in prompt_lengths]
    prompt_texts = [tokenizer.detokenize(tt) for tt in prompt_tokens_list]

    # Init requests.
    assert len(prompt_texts) == len(time_offsets)
    requests = [
        Request(t, o, tokenizer, sampling_params=sampling_params)
        for t, o in zip(prompt_texts, time_offsets)
    ]

    return requests


def get_requests_from_file(
    args: Namespace, tokenizer: Any, sampling_params: Optional[SamplingParams] = None
) -> list[Request]:
    """Get requests from a file."""
    if not args.prompt_file:
        raise ValueError("Prompt file is required to read requests from a file.")

    # Load prompts.
    n_prompts = sum(1 for _ in open(args.prompt_file))
    prompts = []
    if sampling_params is None:
        sampling_params = get_default_sampling_params(tokenizer.eod)
    sampling_params_list = []
    with open(args.prompt_file) as f:
        for line in tqdm(f.readlines(), "read prompt file", total=n_prompts):
            line_dict = json.loads(line)
            prompts.append(line_dict["text"])

            sp = copy.deepcopy(sampling_params)
            if args.num_tokens_from_file:
                sp.num_tokens_to_generate = line_dict["chatgpt_output_token_length"]
            sampling_params_list.append(sp)

            if len(prompts) == args.prompt_file_num_truncate:
                break

    # Get time offsets.
    time_offsets: list[float] = get_time_offsets(
        args.seed, args.incoming_requests_per_step, args.incoming_requests_per_sec, len(prompts)
    )

    # Init requests.
    requests = [
        Request(p, t, tokenizer, sp)
        for p, t, sp in tqdm(
            zip(prompts, time_offsets, sampling_params_list), "init requests", total=len(prompts)
        )
    ]

    return requests


def build_requests(
    args: Namespace, tokenizer: Any, sampling_params: Optional[SamplingParams] = None
) -> list[Request]:
    # Check if we have any prompts (from command line or JSONL)
    if args.prompts:
        if args.prompt_file:
            raise ValueError("Cannot use both --prompts and --prompt-file")
        return get_cli_requests(args, tokenizer, sampling_params)
    elif args.prompt_file:
        return get_requests_from_file(args, tokenizer, sampling_params)
    else:
        return get_synthetic_requests(args, tokenizer, sampling_params)


def get_model_size_str(model):
    n = sum(p.numel() for p in model.parameters())
    for exp, suffix in ((12, "t"), (9, "b"), (6, "m"), (3, "k"), (0, "")):
        nquery = int(10**exp)
        if n > nquery:
            return "%d%s" % (n // nquery, suffix)
    raise Exception("something went wrong.")


def build_dynamic_engine_setup_prefix(
    args: Namespace,
    model: MegatronModule,
    context: DynamicInferenceContext,
    requests: list[DynamicInferenceRequest],
):
    """
    Returns a compact, pipe-separated summary of the dynamic-batching setup.

    Example output:

    `dynamic | cg True | prompts: synth(16 256), n 1024, g 512, t 1.0e+02 5.0e-01 | bf 4, 1.2 [r 1024, t 8192] | gtd 0.50 [r 512] | reqs 100` # pylint: disable=line-too-long

    Args:
        args (Namespace): Command-line arguments for this run.
        context (DynamicInferenceContext): Stores limits such as `max_requests`,
            `max_tokens`, and `gtd_request_count`.
        requests (List[DynamicInferenceRequest]): List of inference requests.

    Returns:
        A configuration string for logging.
    """
    # CUDA graph config
    if args.cuda_graph_impl == "local":
        cg_str = f"graphs {len(context.cuda_graph_batch_dimensions_list)}"
    else:
        cg_str = "--"

    # Unified memory (UVM).
    uvm_str = f"uvm {int(context.unified_memory_level)}"

    # Prompt description
    prompt_src_str = (
        "cli"
        if args.prompts
        else (
            "file"
            if args.prompt_file
            else f"synth({', '.join(map(str, args.num_tokens_to_prompt))})"
        )
    )
    request_str = (
        f"requests: {prompt_src_str}, " f"n {len(requests):d}, g {args.num_tokens_to_generate:d}, "
    )
    request_str += (
        f"dur {args.incoming_requests_duration:.1e} " f"r/sec {args.incoming_requests_per_sec:.1e}"
        if args.incoming_requests_per_step is None
        else f"r/step {args.incoming_requests_per_step}"
    )

    # Buffer limits config
    buffer_limits_str = (
        f"bf: {get_mem_size_str(args.inference_dynamic_batching_buffer_size_gb*1024**3)}, "
        f"{context.kv_block_allocator.active_count} chunks "
        f"[r {context.max_requests}, t {context.max_tokens}]"
    )

    parts = [get_model_size_str(model), "dynamic", cg_str, uvm_str, request_str, buffer_limits_str]

    return " | ".join(parts)


def get_global_peak_memory_stats_bytes() -> dict:
    """Peak allocated CUDA memory aggregated across ranks (MAX), in bytes.

    Uses `torch.cuda.max_memory_allocated()` and assumes peak stats were reset
    before the benchmark run.
    """
    peak_alloc = int(torch.cuda.max_memory_allocated())
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        t = torch.tensor([peak_alloc], device="cuda", dtype=torch.int64)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        peak_alloc = int(t[0].item())
    return {"mem-max-allocated-bytes": peak_alloc}


def escape_str(s: str) -> str:
    return s.replace("\n", "\\n")


def print_unique_prompts_and_outputs(results: List["DynamicInferenceRequest"]) -> None:
    """Print unique prompts and their outputs in legacy gpt_dynamic_inference.py format.

    Reads from the high-level API's ``DynamicInferenceRequest`` records returned
    by ``MegatronLLM.generate`` / ``MegatronAsyncLLM.generate``.
    """
    print("~~~~ Unique prompts + outputs. ~~~~")

    unique_prompt_map = defaultdict(list)
    for idx, req in enumerate(results):
        unique_prompt_map[req.prompt].append(idx)

    for unique_idx, (prompt_text, request_idxs) in enumerate(unique_prompt_map.items()):
        prompt_len = len(results[request_idxs[0]].prompt_tokens)
        print(
            f"\n{unique_idx+1}/{len(unique_prompt_map)}"
            f"[n {len(request_idxs)}, l {prompt_len}] {escape_str(prompt_text)}"
        )

        output_map = defaultdict(list)
        for idx in request_idxs:
            output_map[results[idx].generated_text].append(idx)

        for output_text, output_request_idxs in output_map.items():
            evicted = any(
                event.type.name == "EVICT"
                for idx in output_request_idxs
                for event in results[idx].events
            )
            if output_text is not None:
                o_hash = hashlib.sha256((prompt_text + output_text).encode()).hexdigest()[:6]
                o_len = len(results[output_request_idxs[0]].generated_tokens)
                escaped_output_text = escape_str(output_text)
            else:
                o_hash = "--"
                o_len = 0
                escaped_output_text = "--"
            print(
                f"  >>>> [n {len(output_request_idxs)}, {o_len} tokens, hash {o_hash}"
                f"{', <evicted>' if evicted else ''}] {escaped_output_text}"
            )


def dump_inference_results_to_json(
    args: Namespace,
    results: List["DynamicInferenceRequest"],
    throughputs: List[float],
    peak_mem_stats: dict,
    step_count: int,
    lifetime_prefill_token_count: int,
) -> None:
    """JSON dump of per-request results matching legacy gpt_dynamic_inference.py shape.

    Reads from the high-level API's ``DynamicInferenceRequest`` records.
    Note: ``latency`` is currently always ``None`` in direct mode because the
    low-level engine doesn't populate it on ``DynamicInferenceRequest.merge()``;
    will be populated once that field is wired up upstream.
    """
    if not args.output_path:
        return

    json_results = {}
    for i, req in enumerate(results):
        if i % args.output_every_n_results == 0 or i == len(results) - 1:
            # cuda_graph_request_count_map is only populated by the legacy
            # add_request/step_modern loop and is not surfaced through the
            # high-level API; omitting it here.
            result_dict = {
                "input_prompt": req.prompt,
                "generated_text": req.generated_text,
                "generated_tokens": req.generated_tokens,
                "latency": req.latency,
                "ttft": req.ttft,
                "step_count": step_count,
                "top_n_logprobs": getattr(req, 'generated_top_n_logprobs', None),
                "prompt_top_n_logprobs": getattr(req, 'prompt_top_n_logprobs', None),
            }
            if req.sampling_params.return_log_probs:
                prompt_lp = getattr(req, 'prompt_log_probs', None)
                generated_lp = getattr(req, 'generated_log_probs', None)
                result_dict["prompt_logprobs"] = prompt_lp
                result_dict["generated_logprobs"] = generated_lp
                # Synthesize the legacy "logprobs" field as the concatenation,
                # since DynamicInferenceRequest doesn't carry a single combined list.
                if prompt_lp is not None or generated_lp is not None:
                    result_dict["logprobs"] = (prompt_lp or []) + (generated_lp or [])
                else:
                    result_dict["logprobs"] = None
            if args.output_request_events:
                result_dict["events"] = [e.serialize() for e in req.events]
            json_results[req.request_id] = result_dict

    if args.record_throughput:
        json_results["throughput"] = throughputs
    json_results.update(peak_mem_stats)
    json_results["lifetime_prefill_token_count"] = lifetime_prefill_token_count

    print(f' Saving results to {args.output_path}')
    with open(args.output_path, "w") as fp:
        json.dump(json_results, fp, indent=1)
