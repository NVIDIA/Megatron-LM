# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import random
import simpy
import time
import torch
from argparse import ArgumentParser, Namespace
from typing import Any, List


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
    group.add_argument("--incoming-requests-per-sec", type=float, default=100.,
                       help="Simulated number of requests per second.")
    group.add_argument("--incoming-requests-duration", type=float, default=10.,
                       help="Total amount of time to simulate that requests are "
                       "arriving. Multiply this value with "
                       "`--incoming-requests-per-sec` to get the approximate "
                       "total number of requests.")

    return parser

def get_curr_time() -> float:
    """Get synchronized time across ranks."""
    curr_time = torch.cuda.LongTensor([time.time_ns()])
    if torch.distributed.is_initialized():
        torch.distributed.broadcast(
            curr_time,
            src=0)
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
    ):
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


def get_user_requests(args: Namespace, tokenizer: Any) -> List[Request]:
    requests = [ Request(p, -1., tokenizer) for p in args.prompts ]
    return requests


def get_auto_requests(args: Namespace, tokenizer: Any) -> List[Request]:
    """Get example requests."""

    random.seed(args.seed)

    # Generate random time offsets.
    def arrival(r):
        while True:
            yield env.timeout(random.expovariate(r))
            time_offsets.append(env.now)

    time_offsets = []
    env = simpy.Environment()
    env.process(arrival(args.incoming_requests_per_sec))
    env.run(args.incoming_requests_duration)

    # Ensure at least a single request.
    if len(time_offsets) == 0:
        time_offsets = [ 0. ]

    # Initialize requests.
    requests = [ Request(
        "hi " * random.randint(*args.num_tokens_to_prompt),
        t,
        tokenizer,
    ) for t in time_offsets ]

    # Round down to multiple of --inference-max-requests, until cuda graphs are
    # fixed with static inference batching.
    # todo: @lmcafee, remove following lines after fix.
    factor = getattr(args, "inference_max_requests", 8)
    rounded_len = factor * (len(requests) // factor)
    requests = requests[:rounded_len]

    return requests


def build_requests(args: Namespace, tokenizer: Any) -> List[Request]:
    if args.prompts:
        return get_user_requests(args, tokenizer)
    else:
        return get_auto_requests(args, tokenizer)
