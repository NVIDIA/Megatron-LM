# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import random
import time
import torch
from argparse import ArgumentParser, Namespace
from typing import Any, List, Optional
import json

from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.contexts import DynamicInferenceContext


def add_common_inference_args(parser: ArgumentParser) -> ArgumentParser:
    """Common inference arguments."""

    group = parser.add_argument_group(title='Common inference')

    group.add_argument("--temperature", type=float, default=1.0, help='Sampling temperature.')
    group.add_argument("--top_k", type=int, default=1, help='Top k sampling.')
    group.add_argument("--top_p", type=float, default=0.0, help='Top p sampling.')
    group.add_argument(
        "--return-log-probs",
        action='store_true',
        default=False,
        help='Return the log probabilities of the final output tokens',
    )
    group.add_argument(
        "--prompts",
        metavar='N',
        type=str,
        nargs='+',
        help='Input prompts with each prompt within quotes and seperated by space',
    )
    group.add_argument(
        "--num-tokens-to-prompt",
        type=int,
        nargs="+",
        default=[64, 1024],
        help='Number of tokens to use for simulated prompts. This should be a '
        'space-separated pair of integers, and the generated prompt lengths will '
        'be uniformly sampled within this range.',
    )
    group.add_argument(
        "--num-tokens-to-generate",
        type=int,
        default=30,
        help='Number of tokens to generate for each prompt',
    )
    group.add_argument(
        "--top-n-logprobs",
        type=int,
        default=0,
        help='Return the top n logprobs for the generated tokens and their corresponding token as a dictionary',
    )
    group.add_argument(
        "--incoming-requests-per-sec",
        type=float,
        default=100.0,
        help="Simulated number of requests per second.",
    )
    group.add_argument(
        "--incoming-requests-duration",
        type=float,
        default=10.0,
        help="Total amount of time to simulate that requests are "
        "arriving. Multiply this value with "
        "`--incoming-requests-per-sec` to get the approximate "
        "total number of requests.",
    )
    group.add_argument(
        "--model-provider", choices=["mamba", "gpt"], default="gpt", help="Model provider"
    )
    group.add_argument(
        "--prompt-file",
        help='Jsonl file containing input prompts, where each item (i.e., line) '
        'contains the field \'text\' where the value is the prompt. All other '
        'fields within each item are ignored, and may be customized for each '
        'application.',
    )


    return parser


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

    def __init__(self, prompt_text: str, time_offset: float, tokenizer: Any):
        self.prompt_text = prompt_text
        self.prompt_tokens = tokenizer.tokenize(prompt_text)
        self.output_text = None
        self.output_tokens = []
        self.time_offset = time_offset
        self.time_arrival = None
        self.time_start = None
        self.time_end = None
        self.state = "not-started"

    def __str__(self) -> str:
        return "state '%s'; prompt len %d; output len %d; '%s'" % (
            self.state,
            len(self.prompt_tokens),
            len(self.output_tokens),
            self.prompt_text,
        )

def get_time_offsets(
    seed: Optional[int],
    incoming_requests_per_sec: float,
    incoming_requests_duration: float,
) -> List[Request]:
    """Get example time offsets."""

    import simpy  # Guard against this import in test case

    random.seed(seed)

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

    return time_offsets


def get_user_requests(args: Namespace, tokenizer: Any) -> List[Request]:
    requests = [Request(p, -1.0, tokenizer) for p in args.prompts]
    return requests


def get_auto_requests(args: Namespace, tokenizer: Any) -> List[Request]:
    """Get example requests."""

    time_offsets = get_time_offsets(
        args.seed,
        args.incoming_requests_per_sec,
        args.incoming_requests_duration,
    )
    
    requests = [
        Request("hi " * random.randint(*args.num_tokens_to_prompt), t, tokenizer)
        for t in time_offsets
    ]

    return requests

def get_requests_from_file(args: Namespace, tokenizer: Any) -> List[Request]:
    """Get requests from a file."""
    if not args.prompt_file:
        raise ValueError("Prompt file is required to read requests from a file.")

    requests = []
    time_offsets = get_time_offsets(
        args.seed,
        args.incoming_requests_per_sec,
        args.incoming_requests_duration,
    )
    
    with open(args.prompt_file, 'r') as f:
        for i, (line, time_offset) in enumerate(zip(f, time_offsets)):
            item = json.loads(line.strip())
            if 'text' in item:
                requests.append(Request(item['text'], time_offset, tokenizer))
    
    return requests


def build_requests(args: Namespace, tokenizer: Any) -> List[Request]:
    if args.prompts:
        return get_user_requests(args, tokenizer)
    elif args.prompt_file:
        return get_requests_from_file(args, tokenizer)
    else:
        return get_auto_requests(args, tokenizer)


def build_dynamic_engine_setup_prefix(
    args: Namespace, context: DynamicInferenceContext, requests: List[DynamicInferenceRequest]
):
    """
    Returns a compact, pipe-separated summary of the dynamic-batching setup.

    Example output:

    `dynamic | cg True | <auto prompts> (128 256), 512, 1.0e+00, 5.0e-01 | bf 4, 1.2 [r 1024, t 8192] | gtd 0.50 [r 512] | reqs 100` # pylint: disable=line-too-long

    Args:
        args (Namespace): Command-line arguments for this run.
        context (DynamicInferenceContext): Stores limits such as `max_requests`,
            `max_tokens`, and `gtd_request_count`.
        requests (List[DynamicInferenceRequest]): List of inference requests.

    Returns:
        A configuration string for logging.
    """
    # Prompt description
    if args.prompts:
        prompts_str = f"<user prompts, n {len(args.prompts)}>"
    else:
        prompt_lengths = " ".join(map(str, args.num_tokens_to_prompt))
        prompts_str = (
            f"<auto prompts> "
            f"({prompt_lengths}), "
            f"{args.num_tokens_to_generate:d}, "
            f"{args.incoming_requests_duration:.1e}, "
            f"{args.incoming_requests_per_sec:.1e}"
        )

    # CUDA graph config
    cg_str = f"cg {args.enable_cuda_graph}"

    # Buffer limits config
    flw = args.inference_dynamic_batching_buffer_overflow_factor
    flw_str = "no overflow" if flw is None else f"{flw:.1f}"
    buffer_limits_str = (
        f"bf {args.inference_dynamic_batching_buffer_size_gb:.0f}, {flw_str} "
        f"[r {context.max_requests}, t {context.max_tokens}]"
    )

    # Guaranteed request config
    guaranteed_fraction_str = (
        f"gtd {args.inference_dynamic_batching_buffer_guaranteed_fraction:.2f} "
        f"[r {context.gtd_request_count}]"
    )

    parts = [
        "dynamic",
        cg_str,
        prompts_str,
        buffer_limits_str,
        guaranteed_fraction_str,
        f"reqs {len(requests)}",
    ]

    return " | ".join(parts)
