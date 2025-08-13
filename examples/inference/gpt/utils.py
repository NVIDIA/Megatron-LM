# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import json
import random
import time
import torch
from argparse import ArgumentParser, Namespace
from typing import Any
import json
import itertools
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.transformer.module import MegatronModule


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
        "--output-path", type=str, default=None, help="Path to save generations as JSON"
    )
    group.add_argument(
        "--prompt-file",
        help='Jsonl file containing input prompts, where each item (i.e., line) '
        'contains the field \'text\' where the value is the prompt. All other '
        'fields within each item are ignored, and may be customized for each '
        'application.',
    )
    group.add_argument(
        "--random-sample-prompts",
        action="store_true",
        default=False,
        help="Randomly sample prompts from the prompt file based on simulated request arrivals times "
        "rather than inferring using all of them.",
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
        return "state '%s'; toffset %.1e; prompt len %d; output len %d; '%s'" % (
            self.state,
            self.time_offset,
            len(self.prompt_tokens),
            len(self.output_tokens),
            self.prompt_text,
        )

def get_time_offsets(
    seed: int | None,
    incoming_requests_per_sec: float,
    incoming_requests_duration: float,
) -> list[float]:
    """Get example time offsets."""
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

    return time_offsets


def get_user_requests(args: Namespace, tokenizer: Any) -> list[Request]:
    if args.random_sample_prompts:
        # Maybe exclude some prompts after the first as well as including random time offsets
        #  following a Poisson process.
        time_offsets: list[float] = get_time_offsets(
            args.seed,
            args.incoming_requests_per_sec,
            args.incoming_requests_duration,
        )
    else:
        # One request per prompt with a -1 time offset default for each.
        time_offsets = itertools.repeat(-1)
    requests = [Request(p, t, tokenizer) for p,t in zip(args.prompts, time_offsets)]
    return requests


def get_auto_requests(args: Namespace, tokenizer: Any) -> list[Request]:
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

def get_requests_from_file(args: Namespace, tokenizer: Any) -> list[Request]:
    """Get requests from a file."""
    if not args.prompt_file:
        raise ValueError("Prompt file is required to read requests from a file.")

    requests = []
    if args.random_sample_prompts:
        time_offsets: list[float] = get_time_offsets(
            args.seed,
            args.incoming_requests_per_sec,
            args.incoming_requests_duration,
        )
    else:
        # match the behavior of providing a list of --prompts, use -1 for each prompt
        time_offsets = itertools.repeat(-1)

    with open(args.prompt_file, 'r') as f:
        for i, (line, time_offset) in enumerate(zip(f, time_offsets)):
            item = json.loads(line.strip())
            if 'text' in item:
                requests.append(Request(item['text'], time_offset, tokenizer))
    
    return requests


def build_requests(args: Namespace, tokenizer: Any) -> list[Request]:
    # Check if we have any prompts (from command line or JSONL)
    if args.prompts:
        if args.prompt_file:
            raise ValueError("Cannot use both --prompts and --prompt-file")
        return get_user_requests(args, tokenizer)
    elif args.prompt_file:
        return get_requests_from_file(args, tokenizer)
    else:
        return get_auto_requests(args, tokenizer)


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

    `dynamic | cg True | <auto prompts> (128 256), 512, 1.0e+00, 5.0e-01 | bf 4, 1.2 [r 1024, t 8192] | gtd 0.50 [r 512] | reqs 100` # pylint: disable=line-too-long

    Args:
        args (Namespace): Command-line arguments for this run.
        context (DynamicInferenceContext): Stores limits such as `max_requests`,
            `max_tokens`, and `gtd_request_count`.
        requests (List[DynamicInferenceRequest]): List of inference requests.

    Returns:
        A configuration string for logging.
    """
    # CUDA graph config
    if args.enable_cuda_graph:
        cg_str = (
            f"graphs {context.cuda_graph_request_counts[0]}:"
            f"{context.cuda_graph_request_counts[-1]}"
        )
    else:
        cg_str = "--"

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
        get_model_size_str(model),
        "dynamic",
        cg_str,
        prompts_str,
        buffer_limits_str,
        guaranteed_fraction_str,
        f"reqs {len(requests)}",
    ]

    return " | ".join(parts)
